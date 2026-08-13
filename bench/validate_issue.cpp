// validate_issue.cpp - audit the claim I already filed publicly as ROCm/rocm-libraries#9985.
//
// Two methodological weaknesses in gen_tune_table.cpp that could inflate the reported gains:
//
//  (1) WINNER'S CURSE. Stage 1 times every solution ONCE and keeps the top 16; stage 2 re-times those 5x
//      and reports the MINIMUM. A minimum over many noisy draws is biased low. The default, meanwhile, is
//      timed once with 3 reps and taken at face value. So "gain = default/best" is measured with a
//      pessimistic numerator and an optimistic denominator. Here we re-measure both INTERLEAVED with
//      median-of-N, which removes both the selection bias and any thermal ordering effect.
//
//  (2) WRONG API? The survey timed rocblas_gemm_ex with rocblas_gemm_algo_standard. Most users call
//      rocblas_sgemm. If rocblas_sgemm dispatches differently and lands on a better kernel, the issue is
//      about a path few people take, and the framing is wrong. So we time it too.
//
// Build: hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a validate_issue.cpp -o validate_issue -lrocblas
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

struct Case { int M,N; long K; const char* dt; int sol; double rec_def, rec_best; };

int main(int argc,char**argv){
    int REPS = argc>1?atoi(argv[1]):11;      // interleaved samples, median
    int ITER = argc>2?atoi(argv[2]):3;       // launches per sample

    // the exact rows quoted in the issue, with the solution index and timings the survey recorded
    std::vector<Case> cs = {
        {12288,12288,12288,"fp32",-606628012,100.001,100.038},
        { 4096, 4096,16384,"fp32",-606628017, 29.772, 14.890},
        { 1024,32768,16384,"fp32",-606628017,133.043, 29.720},
        { 2048,32768,  512,"bf16",-606605310,  3.108,  0.814},
        { 8192,28672, 8192,"bf16",-606605357, 45.883, 27.923},
    };

    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_pointer_mode(h,rocblas_pointer_mode_host);
    hipEvent_t e0,e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const float f1=1.f,f0=0.f;

    printf("Re-audit of ROCm/rocm-libraries#9985.  %d interleaved samples, %d launches each, median.\n\n",REPS,ITER);
    printf("%6s %6s %7s %5s | %9s %9s %7s | %9s %9s %7s | %8s\n",
           "M","N","K","dt","surv_def","surv_best","surv_x","re_def","re_best","re_x","sgemm");
    for(auto& c : cs){
        size_t na=(size_t)c.M*c.K, nb=(size_t)c.K*c.N, nc=(size_t)c.M*c.N;
        bool isf32 = !strcmp(c.dt,"fp32");
        float *fA,*fB,*dC; bf16 *bA,*bB;
        HC(hipMalloc(&fA,na*4)); HC(hipMalloc(&fB,nb*4)); HC(hipMalloc(&dC,nc*4));
        HC(hipMalloc(&bA,na*2)); HC(hipMalloc(&bB,nb*2));
        int t=256;
        gen<<<(na+t-1)/t,t>>>(fA,na,12345u); gen<<<(nb+t-1)/t,t>>>(fB,nb,67890u);
        tobf<<<(na+t-1)/t,t>>>(fA,bA,na); tobf<<<(nb+t-1)/t,t>>>(fB,bB,nb);
        HC(hipDeviceSynchronize());
        rocblas_datatype ty = isf32?rocblas_datatype_f32_r:rocblas_datatype_bf16_r;
        const void *pA = isf32?(void*)fA:(void*)bA, *pB = isf32?(void*)fB:(void*)bB;

        auto ex=[&](rocblas_gemm_algo al,int32_t so){
            rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,c.N,c.M,(rocblas_int)c.K,
              &f1,pB,ty,c.N,pA,ty,(rocblas_int)c.K,&f0,dC,rocblas_datatype_f32_r,c.N,
              dC,rocblas_datatype_f32_r,c.N,rocblas_datatype_f32_r,al,so,0); };
        auto sgemm=[&](){ if(!isf32) return;
            rocblas_sgemm(h,rocblas_operation_none,rocblas_operation_none,c.N,c.M,(rocblas_int)c.K,
              &f1,fB,c.N,fA,(rocblas_int)c.K,&f0,dC,c.N); };

        // warm all three paths before any timing, so none of them pays the cold-start
        for(int i=0;i<3;++i){ ex(rocblas_gemm_algo_standard,0); ex(rocblas_gemm_algo_solution_index,c.sol); sgemm(); }
        HC(hipDeviceSynchronize());

        std::vector<double> vd,vb,vs;
        for(int r=0;r<REPS;++r){                       // INTERLEAVED: no variant owns the cold or hot slot
            auto tm=[&](int which)->double{
                HCD(hipEventRecord(e0));
                for(int i=0;i<ITER;++i){ if(which==0) ex(rocblas_gemm_algo_standard,0);
                                         else if(which==1) ex(rocblas_gemm_algo_solution_index,c.sol);
                                         else sgemm(); }
                HCD(hipEventRecord(e1)); HCD(hipEventSynchronize(e1));
                float ms=0; HCD(hipEventElapsedTime(&ms,e0,e1)); return ms/ITER; };
            vd.push_back(tm(0)); vb.push_back(tm(1)); if(isf32) vs.push_back(tm(2));
        }
        auto med=[](std::vector<double>& v){ if(v.empty()) return 0.0; std::sort(v.begin(),v.end()); return v[v.size()/2]; };
        double d=med(vd), b=med(vb), s=med(vs);
        printf("%6d %6d %7ld %5s | %9.3f %9.3f %6.2fx | %9.3f %9.3f %6.2fx | %8s\n",
               c.M,c.N,c.K,c.dt, c.rec_def,c.rec_best,c.rec_def/c.rec_best,
               d,b,d/b, isf32?"":"n/a");
        if(isf32) printf("%6s %6s %7s %5s | %9s %9s %7s | %9s %9s %7s | %8.3f  (sgemm/gemm_ex_default = %.3fx)\n",
                         "","","","","","","","","","",s,d/s);
        fflush(stdout);
        hipFree(fA);hipFree(fB);hipFree(dC);hipFree(bA);hipFree(bB);
    }
    printf("\nsurv_* = what gen_tune_table recorded (stage-1 min-of-many then stage-2 min-of-16).\n");
    printf("re_*   = re-measured here, interleaved, median-of-%d. If re_x < surv_x the survey overstated.\n",REPS);
    printf("sgemm  = rocblas_sgemm, the API most users actually call.\n");
    return 0;
}
