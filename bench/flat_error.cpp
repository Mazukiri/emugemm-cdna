// flat_error.cpp - Round 9 Phase 1: can we make the emulation's error TRULY independent of K?
//
// Round 8 found bf16x3's error is nearly flat in K (4.45e-6 @2048 -> 6.42e-6 @262144) because it is
// bounded by REPRESENTATION error (2 bf16 slices ~ 2^-16), while native fp32 error is ACCUMULATION
// rounding and grows as K^0.4999. They cross at K ~ 76500.
// But bf16x3 is not perfectly flat: it also accumulates in fp32, so its total is ~ sqrt(floor^2 + c^2*K).
// If we kill that accumulation term, bf16x3 holds 4.45e-6 at EVERY K and the statement becomes clean:
// "the emulation's error does not depend on K; fp32's does."
//
// Method: split K into `c` chunks, run each chunk's GEMM into an fp32 partial, accumulate the partials
// in FP64, round once to fp32 at the end (so it is still an honest SGEMM replacement).
// Accumulation error drops from ~sqrt(K) to ~sqrt(K/c) plus a c-term fp64 sum (negligible).
//
// APPLIED TO BOTH SIDES. Round 8's hardest-won lesson was that tuning/helping only your own side is
// self-deception, so fp32 gets exactly the same chunked-fp64 treatment. If fp32 benefits MORE and the
// crossover disappears, that is the result and it gets reported as such.
//
// Build: hipcc -O3 --offload-arch=gfx90a flat_error.cpp -o flat_error -lrocblas
// Run:   ./flat_error [K] [M]        (default K=262144, M=N=4096)
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>

#define HIP_CHECK(c) do{hipError_t _he=(c); if(_he){fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(_he),__LINE__);exit(1);}}while(0)
#define RB_CHECK(c)  do{rocblas_status _s=(c); if(_s!=rocblas_status_success){fprintf(stderr,"rocBLAS %d @%d\n",(int)_s,__LINE__);exit(1);}}while(0)
typedef __hip_bfloat16 bf16;

__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__global__ void gen_normal(float* X,size_t n,unsigned seed){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    unsigned a=hashu((unsigned)i^seed), b=hashu((unsigned)(i>>32)^(a+0x9e3779b9u));
    float u1=(a>>8)*(1.0f/16777216.0f),u2=(b>>8)*(1.0f/16777216.0f);
    if(u1<1e-7f)u1=1e-7f; X[i]=sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); }
__global__ void to_f64(const float* X,double* Y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)Y[i]=(double)X[i]; }
__global__ void split_bf(const float* X,bf16* S0,bf16* S1,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=X[i]; bf16 s0=(bf16)x; S0[i]=s0; S1[i]=(bf16)(x-(float)s0); }
// the only genuinely new kernel in this phase
__global__ void accum_f64(double* acc,const float* part,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) acc[i]+=(double)part[i]; }
__global__ void zero_f64(double* acc,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) acc[i]=0.0; }
// round the fp64 accumulator back to fp32: the deliverable is still an SGEMM, not a DGEMM
__global__ void f64_to_f32(const double* acc,float* out,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=(float)acc[i]; }

int main(int argc,char**argv){
    long K = argc>1?atol(argv[1]):262144;
    int  M = argc>2?atoi(argv[2]):4096, N=M;

    rocblas_handle rb; RB_CHECK(rocblas_create_handle(&rb));
    RB_CHECK(rocblas_set_pointer_mode(rb,rocblas_pointer_mode_host));

    size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N;
    float *fA,*fB,*dPart,*dOut; double *dA64,*dB64,*dCref,*dAcc; bf16 *A0,*A1,*B0,*B1;
    HIP_CHECK(hipMalloc(&fA,na*4));   HIP_CHECK(hipMalloc(&fB,nb*4));
    HIP_CHECK(hipMalloc(&dA64,na*8)); HIP_CHECK(hipMalloc(&dB64,nb*8));
    HIP_CHECK(hipMalloc(&dPart,nc*4));HIP_CHECK(hipMalloc(&dOut,nc*4));
    HIP_CHECK(hipMalloc(&dAcc,nc*8)); HIP_CHECK(hipMalloc(&dCref,nc*8));
    HIP_CHECK(hipMalloc(&A0,na*2)); HIP_CHECK(hipMalloc(&A1,na*2));
    HIP_CHECK(hipMalloc(&B0,nb*2)); HIP_CHECK(hipMalloc(&B1,nb*2));
    printf("M=N=%d K=%ld   ~%.1f GB allocated\n",M,K,(na*14.0+nb*14.0+nc*24.0)/1e9);

    int t=256;
    gen_normal<<<(na+t-1)/t,t>>>(fA,na,12345u); gen_normal<<<(nb+t-1)/t,t>>>(fB,nb,67890u);
    to_f64<<<(na+t-1)/t,t>>>(fA,dA64,na);       to_f64<<<(nb+t-1)/t,t>>>(fB,dB64,nb);
    split_bf<<<(na+t-1)/t,t>>>(fA,A0,A1,na);    split_bf<<<(nb+t-1)/t,t>>>(fB,B0,B1,nb);
    HIP_CHECK(hipDeviceSynchronize());

    const float f1=1.f,f0=0.f; const double d1=1.0,d0=0.0;
    // row-major C=A*B via column-major swap: m=N, n=M, k=K; first operand B (ld=N), second A (ld=K)
    RB_CHECK(rocblas_gemm_ex(rb,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
        &d1,dB64,rocblas_datatype_f64_r,N,dA64,rocblas_datatype_f64_r,(rocblas_int)K,&d0,
        dCref,rocblas_datatype_f64_r,N,dCref,rocblas_datatype_f64_r,N,
        rocblas_datatype_f64_r,rocblas_gemm_algo_standard,0,0));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hC(nc); std::vector<double> hR(nc);
    HIP_CHECK(hipMemcpy(hR.data(),dCref,nc*8,hipMemcpyDeviceToHost));
    auto err_of_dOut=[&]()->double{ HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hC.data(),dOut,nc*4,hipMemcpyDeviceToHost));
        double num=0,den=0; for(size_t i=0;i<nc;++i){ double d=(double)hC[i]-hR[i]; num+=d*d; den+=hR[i]*hR[i]; }
        return std::sqrt(num/den); };

    // one chunk of the K dimension, into dPart (fp32), for either dtype
    auto chunk_fp32=[&](long k0,long kc){
        RB_CHECK(rocblas_gemm_ex(rb,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)kc,
            &f1, fB+(size_t)k0*N, rocblas_datatype_f32_r, N,
                 fA+k0,           rocblas_datatype_f32_r, (rocblas_int)K,
            &f0, dPart,rocblas_datatype_f32_r,N, dPart,rocblas_datatype_f32_r,N,
            rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)); };
    auto chunk_bf16x3=[&](long k0,long kc){
        struct Term { bf16 *b, *a; const float* beta; };
        Term ts[3] = { {B0+(size_t)k0*N, A0+k0, &f0},
                       {B0+(size_t)k0*N, A1+k0, &f1},
                       {B1+(size_t)k0*N, A0+k0, &f1} };
        for(auto& tm : ts)
            RB_CHECK(rocblas_gemm_ex(rb,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)kc,
                &f1, tm.b, rocblas_datatype_bf16_r, N,
                     tm.a, rocblas_datatype_bf16_r, (rocblas_int)K,
                tm.beta, dPart,rocblas_datatype_f32_r,N, dPart,rocblas_datatype_f32_r,N,
                rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)); };

    size_t cb=(nc+t-1)/t;
    auto run_chunked=[&](int c,std::function<void(long,long)> chunk){
        zero_f64<<<cb,t>>>(dAcc,nc);
        long kc=(K+c-1)/c;
        for(long k0=0;k0<K;k0+=kc){ long len=std::min(kc,K-k0);
            chunk(k0,len); accum_f64<<<cb,t>>>(dAcc,dPart,nc); }
        f64_to_f32<<<cb,t>>>(dAcc,dOut,nc); };

    hipEvent_t s,e; HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&e));
    auto timeit=[&](std::function<void()> fn)->double{
        fn(); HIP_CHECK(hipDeviceSynchronize());
        std::vector<double> v;
        for(int r=0;r<3;++r){ HIP_CHECK(hipEventRecord(s)); fn();
            HIP_CHECK(hipEventRecord(e)); HIP_CHECK(hipEventSynchronize(e));
            float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,s,e)); v.push_back(ms); }
        std::sort(v.begin(),v.end()); return v[1]; };

    double fl=2.0*M*N*(double)K;
    printf("\n  fp64 chunk-reduction sweep. c = number of K-chunks summed in fp64.\n");
    printf("  %4s | %11s %9s %8s | %11s %9s %8s\n","c","fp32 err","ms","TF","bf16x3 err","ms","TF");
    int cs[]={1,4,16,64,256};
    for(int c : cs){
        if((long)c>K) continue;
        run_chunked(c,chunk_fp32);    double e32=err_of_dOut();
        run_chunked(c,chunk_bf16x3);  double eb3=err_of_dOut();
        double t32=timeit([&]{run_chunked(c,chunk_fp32);});
        double tb3=timeit([&]{run_chunked(c,chunk_bf16x3);});
        printf("  %4d | %11.4e %9.2f %8.2f | %11.4e %9.2f %8.2f\n",
               c,e32,t32,fl/(t32*1e9), eb3,tb3,fl/(tb3*1e9));
        fflush(stdout);
    }
    printf("\n  Reference: unchunked Round-8 numbers at K=262144 were fp32 9.162e-06, bf16x3 6.417e-06.\n");
    printf("  Read the c=1 row as the regression check, and the trend across c as the answer.\n");
    return 0;
}
