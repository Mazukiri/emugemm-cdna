// error_model.cpp - Phase 5: does the error model hold on data that is not N(0,1)?
//
// Every accuracy number in Rounds 7-9 was measured on independent N(0,1) entries. That is the friendliest
// case for native fp32: its rounding errors are zero-mean and independent, so they cancel as sqrt(K) and
// the fitted exponent came out 0.4999. Real matrices are not that polite. This sweep re-fits
//     err(scheme, K) = a * K^p
// per distribution, and reports where (if anywhere) each emulation scheme crosses native fp32.
//
// Distributions:
//   0 normal       N(0,1)                     the baseline everything so far used
//   1 positive     U(0,1)                     no cancellation; ||C|| grows like K, not sqrt(K)
//   2 lognormal    10^U(-3,3)                 wide dynamic range within one matrix
//   3 cancelling   near-orthogonal rows/cols  ||C|| collapses, so relative error is amplified
//
// Ground truth is rocBLAS DGEMM on device (MI250 has full-rate fp64 matrix cores; validated against a CPU
// fp64 loop in Round 8 to 1.355e-15).
//
// Build: hipcc -O3 --offload-arch=gfx90a error_model.cpp -o error_model -lrocblas
// Run:   ./error_model [Kmax] [M]
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define HIP_CHECK(c) do{hipError_t _e=(c); if(_e){fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(_e),__LINE__);exit(1);}}while(0)
#define RB(c) do{rocblas_status _s=(c); if(_s){fprintf(stderr,"rocBLAS %d @%d\n",(int)_s,__LINE__);exit(1);}}while(0)
typedef __hip_bfloat16 bf16;

__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__device__ __forceinline__ float unif(size_t i,unsigned s){
    return (hashu((unsigned)i^s)>>8)*(1.0f/16777216.0f); }
__device__ __forceinline__ float gauss(size_t i,unsigned s){
    float u1=unif(i,s), u2=unif(i,s+0x9e37u); if(u1<1e-7f)u1=1e-7f;
    return sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); }

// isB distinguishes the two operands so the cancelling case can put the alternating sign on one side only
__global__ void gen(float* X,size_t n,long lead,unsigned seed,int mode,int isB){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    long k = isB ? (long)(i/lead) : (long)(i%lead);      // index along the contracted dimension
    switch(mode){
        case 0: X[i]=gauss(i,seed); break;
        case 1: X[i]=unif(i,seed); break;
        case 2: X[i]=(gauss(i,seed)>0?1.f:-1.f)*powf(10.f,6.f*unif(i,seed+11u)-3.f); break;
        default:{ float base=1.f+1e-3f*gauss(i,seed);     // near-orthogonal: alternating sign on B only
                  X[i]= isB ? ((k&1)?-base:base) : base; } }
}
__global__ void to_f64(const float* X,double* Y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)Y[i]=(double)X[i]; }
__global__ void split_bf(const float* X,bf16* S0,bf16* S1,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=X[i]; bf16 s0=(bf16)x; S0[i]=s0; S1[i]=(bf16)(x-(float)s0); }
__global__ void split_f16(const float* X,_Float16* H,_Float16* L,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=X[i]; _Float16 h=(_Float16)x; H[i]=h; L[i]=(_Float16)((x-(float)h)*2048.f); }

int main(int argc,char**argv){
    long Kmax=argc>1?atol(argv[1]):65536;  int M=argc>2?atoi(argv[2]):2048, N=M;
    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_pointer_mode(h,rocblas_pointer_mode_host);
    size_t maxel=(size_t)M*Kmax, nc=(size_t)M*N;
    float *fA,*fB,*dC; double *dA64,*dB64,*dR; bf16 *A0,*A1,*B0,*B1; _Float16 *Ah,*Al,*Bh,*Bl;
    HIP_CHECK(hipMalloc(&fA,maxel*4)); HIP_CHECK(hipMalloc(&fB,maxel*4));
    HIP_CHECK(hipMalloc(&dA64,maxel*8));HIP_CHECK(hipMalloc(&dB64,maxel*8));
    HIP_CHECK(hipMalloc(&dC,nc*4));    HIP_CHECK(hipMalloc(&dR,nc*8));
    for(bf16**p:{&A0,&A1,&B0,&B1}) HIP_CHECK(hipMalloc(p,maxel*2));
    for(_Float16**p:{&Ah,&Al,&Bh,&Bl}) HIP_CHECK(hipMalloc(p,maxel*2));

    const float f1=1.f,f0=0.f,fi=1.f/2048.f; const double d1=1.0,d0=0.0;
    std::vector<float> hC(nc); std::vector<double> hR(nc);
    const char* DN[]={"normal N(0,1)","positive U(0,1)","lognormal 1e-3..1e3","cancelling (near-orth)"};

    printf("M=N=%d, ground truth = device DGEMM\n",M);
    for(int mode=0;mode<4;++mode){
        printf("\n=== %s ===\n%9s %12s %12s %12s   %s\n","K","fp32","bf16x3","fp16x3",DN[mode],"");
        std::vector<double> lk,l32,lb3;
        for(long K=1024;K<=Kmax;K*=4){
            size_t nel=(size_t)M*K; int t=256; size_t nb=(nel+t-1)/t;
            gen<<<nb,t>>>(fA,nel,K,1234u,mode,0);       // A is M x K, contracted index is the column
            gen<<<nb,t>>>(fB,nel,N,5678u,mode,1);       // B is K x N, contracted index is the row
            to_f64<<<nb,t>>>(fA,dA64,nel); to_f64<<<nb,t>>>(fB,dB64,nel);
            split_bf<<<nb,t>>>(fA,A0,A1,nel); split_bf<<<nb,t>>>(fB,B0,B1,nel);
            split_f16<<<nb,t>>>(fA,Ah,Al,nel); split_f16<<<nb,t>>>(fB,Bh,Bl,nel);
            HIP_CHECK(hipDeviceSynchronize());
            RB(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
               &d1,dB64,rocblas_datatype_f64_r,N,dA64,rocblas_datatype_f64_r,(rocblas_int)K,&d0,
               dR,rocblas_datatype_f64_r,N,dR,rocblas_datatype_f64_r,N,
               rocblas_datatype_f64_r,rocblas_gemm_algo_standard,0,0));
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(hR.data(),dR,nc*8,hipMemcpyDeviceToHost));
            auto err=[&]()->double{ HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(hC.data(),dC,nc*4,hipMemcpyDeviceToHost));
                double num=0,den=0; for(size_t i=0;i<nc;++i){double d=(double)hC[i]-hR[i];num+=d*d;den+=hR[i]*hR[i];}
                return sqrt(num/den); };
            auto G=[&](const void*Bp,const void*Ap,rocblas_datatype ty,const float*al,const float*be){
                RB(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
                   al,Bp,ty,N,Ap,ty,(rocblas_int)K,be,dC,rocblas_datatype_f32_r,N,dC,rocblas_datatype_f32_r,N,
                   rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)); };
            G(fB,fA,rocblas_datatype_f32_r,&f1,&f0);                      double e32=err();
            G(B0,A0,rocblas_datatype_bf16_r,&f1,&f0); G(B0,A1,rocblas_datatype_bf16_r,&f1,&f1);
            G(B1,A0,rocblas_datatype_bf16_r,&f1,&f1);                     double eb3=err();
            G(Bh,Ah,rocblas_datatype_f16_r,&f1,&f0); G(Bh,Al,rocblas_datatype_f16_r,&fi,&f1);
            G(Bl,Ah,rocblas_datatype_f16_r,&fi,&f1);                      double e16=err();
            printf("%9ld %12.4e %12.4e %12.4e\n",K,e32,eb3,e16); fflush(stdout);
            if(std::isfinite(e32)&&e32>0){ lk.push_back(log((double)K)); l32.push_back(log(e32));
                                           lb3.push_back(log(eb3)); }
        }
        auto fit=[&](std::vector<double>& ly)->double{ int n=lk.size(); if(n<2) return NAN;
            double sx=0,sy=0,sxx=0,sxy=0;
            for(int i=0;i<n;++i){sx+=lk[i];sy+=ly[i];sxx+=lk[i]*lk[i];sxy+=lk[i]*ly[i];}
            return (n*sxy-sx*sy)/(n*sxx-sx*sx); };
        double p32=fit(l32), pb3=fit(lb3);
        printf("  fit: err_fp32 ~ K^%.4f   err_bf16x3 ~ K^%.4f\n",p32,pb3);
        printf("  %s\n", (p32>0.7)? "  -> fp32 grows FASTER than sqrt(K): crossover arrives EARLIER than on N(0,1)"
                        : (p32>0.3)? "  -> fp32 grows ~sqrt(K), as on N(0,1)"
                                   : "  -> fp32 error is flat here; the sqrt(K) story does not apply");
    }
    return 0;
}
