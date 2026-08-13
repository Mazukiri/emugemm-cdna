// backend_v2.cpp - backend.cpp brought up to the METHOD_V2 measurement rules.
//
// backend.cpp times the rocBLAS default path under ROCBLAS_USE_HIPBLASLT unset/0/1. Its numbers are
// the ones the reply quotes for "does switching backend fix it" (fp32 1.039, fp16 1.034, bf16 0.162),
// Those numbers are ratios between two arms measured the same way, so the older method's biases
// should cancel. That is an argument rather than a measurement, and re-measuring is cheap, so this
// program exists to settle it: paired per shape, the two methods agree to within 3%.
//
// Four differences from backend.cpp:
//   --rotate N   N buffer sets cycled per launch, so iteration 2 onward does not read from cache.
//                The achieved count is recorded (`nrot`) because memory pressure can force fewer.
//   --minms MS   repetition count sized so the timed window spans at least MS. Fixed reps=5
//                compresses ratios toward 1 by ~5%, since launch overhead adds to both arms in
//                absolute terms.
//   uniform fill Hashed [-1,1] instead of memset(0x3c). A constant fill is defensible at fixed
//                frequency and biases a power-limited part at narrow precisions.
//   clk_drift    A 2048^3 reference GEMM before and after each shape. Reported, never filtered on -
//                drift tracks shape size, so filtering by it filters by size in disguise.
//
// The column layout of backend.csv is preserved and extended, so the old and new files can be
// concatenated and compared per shape.
//
// Build: hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a backend_v2.cpp -o backend_v2 -lrocblas
// Run:   ./backend_v2 shapes.txt out.csv <label> [--rotate N] [--minms MS] [--legacy-fill]
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
#define HC(c) do{hipError_t _e=(c); if(_e){printf("HIP %d @%d\n",(int)_e,__LINE__);fflush(stdout);return 1;}}while(0)

// A HIP call that fails inside a timing lambda must not look like a measurement. Those lambdas
// return double, so a plain `return 1` hands back 1.0 ms - a plausible-looking timing for a
// failure. HC() is used only where the return type is int; the timing lambdas use HCD(), which
// returns TIMED_FAIL, and every caller already treats values above 1e29 as "did not run".
#define TIMED_FAIL 1e30
#define HCD(c) do{hipError_t _e=(c); if(_e){printf("HIP %d @%d\n",(int)_e,__LINE__);fflush(stdout);return TIMED_FAIL;}}while(0)
typedef __hip_bfloat16 bf16;

// identical to tune_trans.cpp so both harnesses feed the hardware the same data
__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__global__ void fill_f32(float* X,size_t n,unsigned seed){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    X[i]=(float)((double)(hashu((unsigned)i^seed)%2000001)/1000000.0-1.0); }
__global__ void fill_bf16(bf16* X,size_t n,unsigned seed){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    X[i]=(bf16)(float)((double)(hashu((unsigned)i^seed)%2000001)/1000000.0-1.0); }
__global__ void fill_f16(_Float16* X,size_t n,unsigned seed){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    X[i]=(_Float16)(float)((double)(hashu((unsigned)i^seed)%2000001)/1000000.0-1.0); }

int main(int argc,char**argv){
    const char* inf  = argc>1?argv[1]:"shapes.txt";
    const char* outf = argc>2?argv[2]:"backend_v2.csv";
    const char* lab  = argc>3?argv[3]:"unset";
    int rot=4, legacy_fill=0; double minms=10.0;
    for(int i=4;i<argc;++i){
        if(!strcmp(argv[i],"--rotate") && i+1<argc) rot=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--minms") && i+1<argc) minms=atof(argv[++i]);
        else if(!strcmp(argv[i],"--legacy-fill")) legacy_fill=1;
    }
    if(rot<1) rot=1;
    const char* env=getenv("ROCBLAS_USE_HIPBLASLT");
    printf("label=%s  ROCBLAS_USE_HIPBLASLT=%s  rotate=%d minms=%.1f fill=%s\n",
           lab, env?env:"(unset)", rot, minms, legacy_fill?"legacy-memset":"uniform[-1,1]");
    fflush(stdout);

    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_pointer_mode(h,rocblas_pointer_mode_host);
    hipEvent_t e0,e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const float f1=1.f,f0=0.f;

    // ---- clock reference, same construction as tune_trans.cpp ---------------------------------
    const int RN=2048; float *rA,*rB,*rC;
    HC(hipMalloc(&rA,(size_t)RN*RN*4)); HC(hipMalloc(&rB,(size_t)RN*RN*4)); HC(hipMalloc(&rC,(size_t)RN*RN*4));
    { int t=256; size_t n=(size_t)RN*RN;
      fill_f32<<<(n+t-1)/t,t>>>(rA,n,1u); fill_f32<<<(n+t-1)/t,t>>>(rB,n,2u);
      HC(hipMemset(rC,0,n*4)); HC(hipDeviceSynchronize()); }
    auto ref_ms=[&]()->double{
        auto go=[&]{ return rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,RN,RN,RN,
            &f1,rB,rocblas_datatype_f32_r,RN,rA,rocblas_datatype_f32_r,RN,&f0,
            rC,rocblas_datatype_f32_r,RN,rC,rocblas_datatype_f32_r,RN,
            rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0); };
        go(); HCD(hipDeviceSynchronize());
        HCD(hipEventRecord(e0)); for(int i=0;i<3;++i) go();
        HCD(hipEventRecord(e1)); HCD(hipEventSynchronize(e1));
        float ms=0; HCD(hipEventElapsedTime(&ms,e0,e1)); return ms/3; };
    { double prev=ref_ms(); int stable=0;
      for(int i=0;i<40 && stable<3;++i){ double cur=ref_ms();
          if(fabs(cur-prev)/prev < 0.02) ++stable; else stable=0; prev=cur; }
      printf("clock settled: reference GEMM %.3f ms\n",prev); fflush(stdout); }

    FILE* fi=fopen(inf,"r"); if(!fi){printf("no %s\n",inf);return 1;}
    FILE* fo=fopen(outf,"a");
    if(ftell(fo)==0) fprintf(fo,"label,M,N,K,dtype,opA,opB,beta,def_ms,tflops,nrot,reps,clk_drift\n");
    char line[256];
    while(fgets(line,sizeof line,fi)){
        int M,N,beta_i=0; long K; char dn[16],ca,cb; double og=0;
        int nf=sscanf(line,"%d,%d,%ld,%15[^,],%c,%c,%d,%lf",&M,&N,&K,dn,&ca,&cb,&beta_i,&og);
        if(nf<6) continue;                       // beta and the trailing gain column are optional
        int d=!strcmp(dn,"fp32")?0:!strcmp(dn,"bf16")?1:2;
        int ta=(ca=='T'||ca=='W'), tb=(cb=='T'||cb=='W');
        rocblas_operation opA=ta?rocblas_operation_transpose:rocblas_operation_none;
        rocblas_operation opB=tb?rocblas_operation_transpose:rocblas_operation_none;
        size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N, esz=d==0?4:2;
        if((double)na*esz+(double)nb*esz+(double)nc*4 > 28e9) continue;

        double clk_a=ref_ms();

        std::vector<void*> vA,vB; std::vector<float*> vC; int nrot=0;
        for(int c=0;c<rot;++c){
            void *a=nullptr,*b=nullptr; float* cc=nullptr;
            if(hipMalloc(&a,na*esz)!=hipSuccess) break;
            if(hipMalloc(&b,nb*esz)!=hipSuccess){ hipFree(a); break; }
            if(hipMalloc(&cc,nc*4)!=hipSuccess){ hipFree(a); hipFree(b); break; }
            if(legacy_fill){ HC(hipMemset(a,0x3c,na*esz)); HC(hipMemset(b,0x3c,nb*esz)); }
            else{ int t=256; unsigned s1=12345u+c*7919u, s2=67890u+c*104729u;
                if(d==0){ fill_f32<<<(na+t-1)/t,t>>>((float*)a,na,s1); fill_f32<<<(nb+t-1)/t,t>>>((float*)b,nb,s2); }
                else if(d==1){ fill_bf16<<<(na+t-1)/t,t>>>((bf16*)a,na,s1); fill_bf16<<<(nb+t-1)/t,t>>>((bf16*)b,nb,s2); }
                else { fill_f16<<<(na+t-1)/t,t>>>((_Float16*)a,na,s1); fill_f16<<<(nb+t-1)/t,t>>>((_Float16*)b,nb,s2); } }
            HC(hipMemset(cc,0,nc*4));
            vA.push_back(a); vB.push_back(b); vC.push_back(cc); ++nrot;
        }
        if(nrot==0) continue;
        HC(hipDeviceSynchronize());
        auto freeall=[&]{ for(int c=0;c<nrot;++c){ hipFree(vA[c]); hipFree(vB[c]); hipFree(vC[c]); } };
        int rr=0;
        rocblas_datatype ty=d==0?rocblas_datatype_f32_r:d==1?rocblas_datatype_bf16_r:rocblas_datatype_f16_r;
        rocblas_int ldb=tb?(rocblas_int)K:N, lda=ta?(rocblas_int)M:(rocblas_int)K;
        const float* beta=beta_i?&f1:&f0;
        auto run=[&](){
            int i=rr++ % nrot;
            return rocblas_gemm_ex(h,opB,opA,N,M,(rocblas_int)K,&f1,vB[i],ty,ldb,vA[i],ty,lda,beta,
                vC[i],rocblas_datatype_f32_r,N,vC[i],rocblas_datatype_f32_r,N,
                rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0); };
        if(run()!=rocblas_status_success){ freeall(); continue; }
        for(int w=0;w<8;++w) run(); HC(hipDeviceSynchronize());

        // one cheap estimate sizes the timed window
        HC(hipEventRecord(e0)); for(int i=0;i<3;++i) run();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float est=0; HC(hipEventElapsedTime(&est,e0,e1)); double t1=est/3;
        int reps=5;
        if(minms>0 && t1>0) reps=(int)std::min(2000.0,std::max(5.0,ceil(minms/t1)));

        std::vector<double> v;
        for(int r=0;r<9;++r){
            HC(hipEventRecord(e0)); for(int i=0;i<reps;++i) run();
            HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
            float ms=0; HC(hipEventElapsedTime(&ms,e0,e1)); v.push_back(ms/reps); }
        std::sort(v.begin(),v.end()); double D=v[4];

        double clk_b=ref_ms(), drift=(clk_a>0)?clk_b/clk_a:1.0;
        double tf=2.0*M*N*(double)K/(D*1e-3)/1e12;
        fprintf(fo,"%s,%d,%d,%ld,%s,%c,%c,%d,%.5f,%.2f,%d,%d,%.4f\n",
                lab,M,N,K,dn,ca,cb,beta_i,D,tf,nrot,reps,drift);
        fflush(fo);
        printf("  %6dx%-6dx%-6ld %-5s %c%c b%d  %9.3f ms  %7.2f TF  reps %4d%s\n",
               M,N,K,dn,ca,cb,beta_i,D,tf,reps,(drift>1.03||drift<0.97)?"  CLKDRIFT":"");
        fflush(stdout);
        freeall();
    }
    fclose(fi); fclose(fo);
    printf("backend_v2[%s] done\n",lab);
    return 0;
}
