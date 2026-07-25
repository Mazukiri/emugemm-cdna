// emu_adversarial.cpp - the acceptance test for the whole rho line of work.
//
// emugemm's contract is "ask for max_rel_err and either get it, or be told it is unreachable". Round 9
// Phase 5 showed that contract is BROKEN on cancelling data: the model assumed ||C|| ~ sqrt(K)||a||||b||,
// so under cancellation it would promise 1e-5 and deliver far worse. Round 10 measured the fix:
// err scales as rho, rho is measurable up front for ~1.6%, and rho and K are orthogonal axes.
//
// This test does the only thing that matters: feed emugemm data it was NOT calibrated on, at cancellation
// levels spanning five orders, and check the contract holds. A pass is either
//     measured error <= requested target,      or
//     the plan explicitly said TARGET UNREACHABLE.
// Silently returning a worse answer than promised is the failure this whole exercise exists to prevent.
//
// Matrices are family 2 from rho_sweep.cpp (A = [G|G], B = [H; -H + d*G2]): the bulk annihilates so rho
// is tunable, and every stored value is a generic N(0,1) draw -- no values sitting exactly on
// representable bf16 numbers, which is what made family 1 lie.
//
// Build: hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a emu_adversarial.cpp emugemm.cpp -o emu_adversarial -lrocblas
#include "emugemm.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#define HC(c) do{hipError_t _e=(c); if(_e){printf("HIP %d @%d\n",(int)_e,__LINE__);return 1;}}while(0)
__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__device__ __forceinline__ float unif(size_t i,unsigned s){ return (hashu((unsigned)i^s)>>8)*(1.0f/16777216.0f); }
__device__ __forceinline__ float gauss(size_t i,unsigned s){
    float u1=unif(i,s),u2=unif(i,s+0x9e37u); if(u1<1e-7f)u1=1e-7f;
    return sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); }
__global__ void gA(float* A,double* A64,size_t n,long K){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    long r=(long)(i/K), c=(long)(i%K), half=K/2;
    float v=gauss((size_t)r*half+(c%half),31337u); A[i]=v; A64[i]=(double)v; }
__global__ void gB(float* B,double* B64,size_t n,long N,long K,float d){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    long k=(long)(i/N), c=(long)(i%N), half=K/2;
    float base=gauss((size_t)(k%half)*N+c,71993u);
    float v=(k<half)?base:(-base+d*gauss(i,55555u)); B[i]=v; B64[i]=(double)v; }

static const char* SN[]={"AUTO","NATIVE_FP32","FP32_CHUNKED","BF16X3","FP16X3","BF16X6"};

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):2048, N=M; long K=argc>2?atol(argv[2]):16384;
    const char* tbl=argc>3?argv[3]:"tune_table.csv";
    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_pointer_mode(h,rocblas_pointer_mode_host);
    emugemm_init(tbl);
    size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N;
    float *A,*B,*C; double *A64,*B64,*R;
    HC(hipMalloc(&A,na*4)); HC(hipMalloc(&B,nb*4)); HC(hipMalloc(&C,nc*4));
    HC(hipMalloc(&A64,na*8)); HC(hipMalloc(&B64,nb*8)); HC(hipMalloc(&R,nc*8));
    std::vector<float> hC(nc); std::vector<double> hR(nc);
    const double d1=1.0,d0=0.0; int t=256;

    printf("adversarial contract test   M=N=%d K=%ld   family 2 (generic values, tunable cancellation)\n",M,K);
    printf("  %9s %10s %14s %6s %12s %12s  %s\n",
           "rho_meas","target","scheme","chunks","predicted","measured","contract");
    int fail=0;
    for(double rho=1e2; rho<=1.1e6; rho*=10.0) for(double tgt : {1e-5,1e-4}){
        float d=(float)(0.6366*sqrt(2.0*K)/rho);
        gA<<<(na+t-1)/t,t>>>(A,A64,na,K); gB<<<(nb+t-1)/t,t>>>(B,B64,nb,N,K,d);
        HC(hipDeviceSynchronize());
        rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
          &d1,B64,rocblas_datatype_f64_r,N,A64,rocblas_datatype_f64_r,(rocblas_int)K,&d0,
          R,rocblas_datatype_f64_r,N,R,rocblas_datatype_f64_r,N,
          rocblas_datatype_f64_r,rocblas_gemm_algo_standard,0,0);
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(hR.data(),R,nc*8,hipMemcpyDeviceToHost));

        emu_request_t rq{tgt,EMU_SCHEME_AUTO}; emu_hints_t hn{-1.f,-1.f,0,-1.f,64}; emu_plan_t p{};
        emugemm_sgemm(h,M,N,K,A,B,C,&rq,&hn,&p);
        HC(hipDeviceSynchronize()); HC(hipMemcpy(hC.data(),C,nc*4,hipMemcpyDeviceToHost));
        double num=0,den=0; for(size_t i=0;i<nc;++i){double e=(double)hC[i]-hR[i];num+=e*e;den+=hR[i]*hR[i];}
        double meas=sqrt(num/den);
        bool unreachable = strstr(p.reason,"UNREACHABLE")!=nullptr;
        bool ok = (meas<=tgt) || unreachable;
        if(!ok) ++fail;
        printf("  %9.2e %10.0e %14s %6d %12.3e %12.3e  %s\n", hn.rho,tgt,SN[p.chosen],p.chunks,
               p.predicted_err,meas, unreachable?"declined (ok)":(ok?"met":"** BROKEN **"));
    }
    printf("\n  contract violations: %d\n",fail);
    printf("  A violation means emugemm promised an accuracy it did not deliver -- the exact failure the\n"
           "  rho probe was added to prevent. 'declined' is a pass: refusing is allowed, lying is not.\n");
    emugemm_shutdown();
    return fail?1:0;
}
