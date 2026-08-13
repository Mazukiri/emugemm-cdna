// audit_table.cpp - quantify, across a random sample of the published table, how much the survey's
// gains were inflated by measuring the DEFAULT cold.
//
// gen_tune_table times the default first, immediately after data generation, with one warm-up and three
// timed launches -- before the GPU has been driven by the hundreds of launches that follow. The best
// solution is then timed at the end of that sequence, on a warm device. Spot checks showed the default
// recorded up to 19% slower than it runs when both are timed fairly, which inflates gain = default/best.
// The winner's-curse concern turned out to be negligible (<0.5% on the best side), so this ordering
// effect is the whole error.
//
// Method here: re-time default and best INTERLEAVED, median-of-N, after warming both. Report the gain
// distribution against the published one so the correction can be stated in numbers, not adjectives.
//
// Build: hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a audit_table.cpp -o audit_table -lrocblas
// Run:   ./audit_table tune_table.csv [nsample] [seed]
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#define HC(c) do{hipError_t _e=(c); if(_e){printf("HIP %d @%d\n",(int)_e,__LINE__);exit(1);}}while(0)

// A HIP failure inside a timing lambda must not look like a measurement: those lambdas return
// double, so `return 1` hands back 1.0 ms, an ordinary-looking value. HC() is for int-returning
// scopes; timing lambdas use HCD(), whose sentinel loses every comparison it takes part in.
#define TIMED_FAIL 1e30
#define HCD(c) do{hipError_t _e=(c); if(_e){printf("HIP %d @%d\n",(int)_e,__LINE__);fflush(stdout);return TIMED_FAIL;}}while(0)
typedef __hip_bfloat16 bf16;

__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__global__ void gen(float* X,size_t n,unsigned s){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    unsigned a=hashu((unsigned)i^s),b=hashu((unsigned)(i>>32)^(a+0x9e3779b9u));
    float u1=(a>>8)*(1.f/16777216.f),u2=(b>>8)*(1.f/16777216.f);
    if(u1<1e-7f)u1=1e-7f; X[i]=sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); }
__global__ void tobf(const float* X,bf16* Y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)Y[i]=(bf16)X[i]; }
__global__ void tofp16(const float* X,_Float16* Y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)Y[i]=(_Float16)X[i]; }

struct Row { int M,N; long K; char dt[8]; int sol; double bms,dms,gain; };

int main(int argc,char**argv){
    const char* csv = argc>1?argv[1]:"tune_table.csv";
    int NS = argc>2?atoi(argv[2]):40;
    unsigned seed = argc>3?atoi(argv[3]):7u;
    const int REPS=9, ITER=3;

    std::vector<Row> all;
    FILE* f=fopen(csv,"r"); if(!f){printf("no %s\n",csv);return 1;}
    char line[512];
    while(fgets(line,sizeof line,f)){ Row r;
        if(sscanf(line,"%*[^,],%*[^,],%d,%d,%ld,%7[^,],%d,%lf,%lf,%lf",
                  &r.M,&r.N,&r.K,r.dt,&r.sol,&r.bms,&r.dms,&r.gain)==8) all.push_back(r); }
    fclose(f);
    printf("loaded %zu rows from %s\n",all.size(),csv);

    std::mt19937 rng(seed); std::shuffle(all.begin(),all.end(),rng);
    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_pointer_mode(h,rocblas_pointer_mode_host);
    hipEvent_t e0,e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const float f1=1.f,f0=0.f;

    std::vector<double> pub, remeas; std::vector<int> dt_idx;
    printf("\n%6s %6s %8s %5s | %8s %8s | %8s %8s | %7s %7s %8s\n",
           "M","N","K","dt","pub_def","pub_best","re_def","re_best","pub_x","re_x","def_bias");
    int done=0;
    for(auto& r : all){
        if(done>=NS) break;
        size_t na=(size_t)r.M*r.K, nb=(size_t)r.K*r.N, nc=(size_t)r.M*r.N;
        if((na+nb)*6.0+nc*4.0 > 40e9) continue;
        int d = !strcmp(r.dt,"fp32")?0 : !strcmp(r.dt,"bf16")?1 : 2;
        float *fA,*fB,*dC; void *lA,*lB;
        if(hipMalloc(&fA,na*4)!=hipSuccess) continue;
        HC(hipMalloc(&fB,nb*4)); HC(hipMalloc(&dC,nc*4));
        HC(hipMalloc(&lA,na*2)); HC(hipMalloc(&lB,nb*2));
        int t=256;
        gen<<<(na+t-1)/t,t>>>(fA,na,12345u); gen<<<(nb+t-1)/t,t>>>(fB,nb,67890u);
        if(d==1){ tobf<<<(na+t-1)/t,t>>>(fA,(bf16*)lA,na); tobf<<<(nb+t-1)/t,t>>>(fB,(bf16*)lB,nb); }
        if(d==2){ tofp16<<<(na+t-1)/t,t>>>(fA,(_Float16*)lA,na); tofp16<<<(nb+t-1)/t,t>>>(fB,(_Float16*)lB,nb); }
        HC(hipDeviceSynchronize());
        rocblas_datatype ty = d==0?rocblas_datatype_f32_r : d==1?rocblas_datatype_bf16_r : rocblas_datatype_f16_r;
        const void *pA = d==0?(void*)fA:lA, *pB = d==0?(void*)fB:lB;
        auto ex=[&](rocblas_gemm_algo al,int32_t so){
            return rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,r.N,r.M,(rocblas_int)r.K,
              &f1,pB,ty,r.N,pA,ty,(rocblas_int)r.K,&f0,dC,rocblas_datatype_f32_r,r.N,
              dC,rocblas_datatype_f32_r,r.N,rocblas_datatype_f32_r,al,so,0); };
        if(ex(rocblas_gemm_algo_solution_index,r.sol)!=rocblas_status_success){
            hipFree(fA);hipFree(fB);hipFree(dC);hipFree(lA);hipFree(lB); continue; }
        // warm BOTH before timing EITHER -- this is the step gen_tune_table skipped
        for(int i=0;i<4;++i){ ex(rocblas_gemm_algo_standard,0); ex(rocblas_gemm_algo_solution_index,r.sol); }
        HC(hipDeviceSynchronize());
        std::vector<double> vd,vb;
        for(int q=0;q<REPS;++q){
            auto tm=[&](int w)->double{ HCD(hipEventRecord(e0));
                for(int i=0;i<ITER;++i) w? ex(rocblas_gemm_algo_solution_index,r.sol) : ex(rocblas_gemm_algo_standard,0);
                HCD(hipEventRecord(e1)); HCD(hipEventSynchronize(e1));
                float ms=0; HCD(hipEventElapsedTime(&ms,e0,e1)); return ms/ITER; };
            vd.push_back(tm(0)); vb.push_back(tm(1)); }
        std::sort(vd.begin(),vd.end()); std::sort(vb.begin(),vb.end());
        double D=vd[REPS/2], B=vb[REPS/2];
        printf("%6d %6d %8ld %5s | %8.3f %8.3f | %8.3f %8.3f | %6.3fx %6.3fx %7.1f%%\n",
               r.M,r.N,r.K,r.dt, r.dms,r.bms, D,B, r.gain, D/B, 100.0*(r.dms/D-1.0));
        fflush(stdout);
        pub.push_back(r.gain); remeas.push_back(D/B); dt_idx.push_back(d);
        hipFree(fA);hipFree(fB);hipFree(dC);hipFree(lA);hipFree(lB);
        ++done;
    }
    auto stats=[&](std::vector<double> v,const char* nm){
        if(v.empty()) return; std::sort(v.begin(),v.end()); int n=v.size();
        int over10=0; for(double x:v) if(x>1.10) ++over10;
        double s=0; for(double x:v) s+=x;
        printf("  %-10s n=%d  mean %.3f  median %.3f  p90 %.3f  max %.3f  >1.10x: %.0f%%\n",
               nm,n,s/n,v[n/2],v[(int)(n*0.9)],v[n-1],100.0*over10/n); };
    printf("\n=== gain distribution, same %zu shapes ===\n",pub.size());
    stats(pub,"published"); stats(remeas,"re-measured");
    double sb=0; for(size_t i=0;i<pub.size();++i) sb+=pub[i]/remeas[i];
    printf("  mean inflation factor published/re-measured = %.3f\n", sb/pub.size());
    return 0;
}
