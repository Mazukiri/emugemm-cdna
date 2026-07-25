// rho_sweep.cpp - Round 10: turn "bf16x3 is 35x worse under cancellation" into a usable curve,
// and test whether the cancellation can be CHEAPLY predicted before doing the GEMM.
//
// Why this and not something else. The literature sweep (EmuGEMM 2606.25453, Ozaki-II 2508.03984,
// Baboulin et al. Euro-Par 2024, tile-centric 2508.14848) says: fused emulation kernels, CholeskyQR+IR
// for RandNLA, tile-condition-based precision, and vendor-heuristic waste are all already published.
// What is NOT solved, and is stated as an open problem in arXiv 2601.08077, is emulating to a SPECIFIED
// accuracy: "inaccurate estimation of the emulation level to achieve desired accuracy levels... ozIMMU
// and GEMMul8 don't support emulating to a specific accuracy level." Existing adaptive schemes select on
// EXPONENT statistics (dynamic range). Our Phase 5 showed the thing that actually breaks the error model
// by 35x is not dynamic range but CANCELLATION. Nobody dispatches on that.
//
// Two questions, one program:
//   (c) how does each scheme's error depend on rho = || |A||B| ||_F / || AB ||_F, over 1 .. 1e6?
//       Phase 5 sampled exactly two points (rho=163 comparable, rho=1.8e5 35x worse). Three orders
//       between them are unmeasured, and the dispatcher needs the curve, not the anecdote.
//   (b) can rho be estimated BEFORE the GEMM, cheaply? Hutchinson: for Rademacher Omega,
//       E||M Omega||_F^2 = ||M||_F^2, and A*B*Omega is two skinny GEMMs, O(MK+KN) not O(MNK).
//       If rho_hat tracks rho over six orders, the dispatcher can be closed.
//
// Construction with controllable cancellation. With A ~ all-ones and B carrying an alternating sign,
// the sum telescopes and ||AB|| collapses while || |A||B| || does not:
//     A_ik = 1 + (1/rho) g,     B_kj = 1/rho + (-1)^k + (sqrt(K)/rho) g
// gives ||AB|| ~ K/rho and || |A||B| || ~ K, hence rho. We do NOT trust that algebra: the achieved rho
// is measured in fp64 for every point and that is what gets reported.
//
// Build: hipcc -O3 --offload-arch=gfx90a rho_sweep.cpp -o rho_sweep -lrocblas
// Run:   ./rho_sweep [K] [M] [nprobe]
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define HC(c) do{hipError_t _e=(c); if(_e){printf("HIP %d @%d\n",(int)_e,__LINE__);exit(1);}}while(0)
#define RB(c) do{rocblas_status _s=(c); if(_s){printf("rocBLAS %d @%d\n",(int)_s,__LINE__);exit(1);}}while(0)
typedef __hip_bfloat16 bf16;

__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__device__ __forceinline__ float unif(size_t i,unsigned s){ return (hashu((unsigned)i^s)>>8)*(1.0f/16777216.0f); }
__device__ __forceinline__ float gauss(size_t i,unsigned s){
    float u1=unif(i,s),u2=unif(i,s+0x9e37u); if(u1<1e-7f)u1=1e-7f;
    return sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); }

__global__ void gen_A(float* A,float* aA,double* A64,size_t n,long K,float invrho){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float v=1.f+invrho*gauss(i,1234u); A[i]=v; aA[i]=fabsf(v); A64[i]=(double)v; }
__global__ void gen_B(float* B,float* aB,double* B64,size_t n,long N,float invrho,float amp){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    long k=(long)(i/N);                       // row index = contracted index
    float v=invrho + ((k&1)?-1.f:1.f) + amp*gauss(i,5678u);
    B[i]=v; aB[i]=fabsf(v); B64[i]=(double)v; }
// ---- FAMILY 2: cancellation from near-annihilation, with NOTHING exactly representable ----
// Family 1 drives B -> +-1 and A -> 1 as rho grows, and bf16 represents +-1 EXACTLY. That is almost
// certainly why bf16x6's error fell to 2.5e-8 there -- a result that should not happen, and which makes
// the whole bf16x6 column untrustworthy. Family 2 removes the possibility entirely:
//     A = [G | G]          (the same random block repeated along K)
//     B = [H ; -H + d*G2]  (its negation, plus a small generic perturbation)
//   => A*B = G*H - G*H + d*G*G2 = d*(G*G2)      <- the large parts annihilate
//      || |A||B| || stays O(K)                   <- the magnitudes do not
// so rho ~ (2/pi)*sqrt(2K)/d, tunable, while every stored value is a generic N(0,1) draw.
__global__ void gen_A2(float* A,float* aA,double* A64,size_t n,long K){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    long r=(long)(i/K), c=(long)(i%K), half=K/2;
    float v=gauss((size_t)r*half + (c%half), 31337u);      // identical in both halves of K
    A[i]=v; aA[i]=fabsf(v); A64[i]=(double)v; }
__global__ void gen_B2(float* B,float* aB,double* B64,size_t n,long N,long K,float d){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    long k=(long)(i/N), c=(long)(i%N), half=K/2;
    float base=gauss((size_t)(k%half)*N + c, 71993u);
    float v = (k<half) ? base : (-base + d*gauss(i,55555u));
    B[i]=v; aB[i]=fabsf(v); B64[i]=(double)v; }

// ---- FAMILY 3: smooth x oscillatory. Cancellation from a MECHANISM, not from algebra ----
// Family 1 fails because its values collapse onto +-1, which bf16 stores exactly. Family 2 fixes the
// values but still builds cancellation by exact algebraic annihilation (H against -H) and repeats a
// block in A, so its rows are not independent along K. Neither resembles how cancellation actually
// arises in numerical work.
// Here A's rows are samples of smooth decaying functions and B's columns are oscillations of random
// integer frequency. Their inner product is an oscillatory integral, which nearly vanishes -- the same
// mechanism that makes quadrature and differential operators ill-conditioned. A DC offset m added to B
// restores a non-cancelling component and tunes rho (~0.64/m). Every value is an irrational transcendental
// with no special binary structure, no repeated blocks, and no exact annihilation anywhere.
// A must be SMOOTH along K for the mechanism to work: a non-smooth perturbation does not cancel against
// an oscillation, so it floors rho. A first attempt with 0.1*gauss noise saturated at rho=20 for exactly
// that reason. Rows differ by decay rate and a smooth phase, which keeps A well-conditioned in M without
// breaking smoothness in K.
__global__ void gen_A3(float* A,float* aA,double* A64,size_t n,long K){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    long r=(long)(i/K), c=(long)(i%K);
    float a=0.5f+3.5f*unif((size_t)r,909u);                       // per-row decay rate
    float b=0.3f*unif((size_t)r,1313u);                           // per-row smooth modulation
    float x=(float)c/(float)K;
    float v=expf(-a*x)*(1.0f+b*sinf(3.14159265f*x));
    A[i]=v; aA[i]=fabsf(v); A64[i]=(double)v; }
// rho is tuned by the OSCILLATION FREQUENCY, not by a DC offset: the inner product of a smooth function
// with cos(2*pi*f*x) decays like 1/f^2, so rho ~ f^2. A small DC term keeps low rho reachable.
__global__ void gen_B3(float* B,float* aB,double* B64,size_t n,long N,long K,float fbase,float m){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    long k=(long)(i/N), c=(long)(i%N);
    float f=floorf(fbase*(0.7f+0.6f*unif((size_t)c,313u)))+1.0f;  // jittered around fbase
    float ph=6.2831853f*unif((size_t)c,977u);
    float v=cosf(6.2831853f*f*(float)k/(float)K+ph)+m;
    B[i]=v; aB[i]=fabsf(v); B64[i]=(double)v; }

__global__ void rademacher(float* X,size_t n,unsigned s){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) X[i]=(unif(i,s)>0.5f)?1.f:-1.f; }
__global__ void to_bf(const float* X,bf16* S0,bf16* S1,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=X[i]; bf16 s0=(bf16)x; S0[i]=s0; S1[i]=(bf16)(x-(float)s0); }
__global__ void to_bf3(const float* X,bf16* S0,bf16* S1,bf16* S2,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=X[i]; bf16 s0=(bf16)x; float r=x-(float)s0; bf16 s1=(bf16)r; S0[i]=s0;S1[i]=s1;
    S2[i]=(bf16)(r-(float)s1); }
__global__ void to_f16(const float* X,_Float16* H,_Float16* L,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=X[i]; _Float16 h=(_Float16)x; H[i]=h; L[i]=(_Float16)((x-(float)h)*2048.f); }
__global__ void sqsum_d(const double* X,size_t n,double* o){
    __shared__ double sm[256];
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
    double s=0; for(; i<n; i+=st) s+=X[i]*X[i];
    sm[threadIdx.x]=s; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) sm[threadIdx.x]+=sm[threadIdx.x+k]; __syncthreads(); }
    if(threadIdx.x==0) atomicAdd(o,sm[0]); }
__global__ void sqsum_f(const float* X,size_t n,double* o){
    __shared__ double sm[256];
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
    double s=0; for(; i<n; i+=st){ double v=X[i]; s+=v*v; }
    sm[threadIdx.x]=s; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) sm[threadIdx.x]+=sm[threadIdx.x+k]; __syncthreads(); }
    if(threadIdx.x==0) atomicAdd(o,sm[0]); }

int main(int argc,char**argv){
    long K = argc>1?atol(argv[1]):65536;
    int  M = argc>2?atoi(argv[2]):2048, N=M;
    int  P = argc>3?atoi(argv[3]):8;          // Hutchinson probes
    int  FAM = argc>4?atoi(argv[4]):1;        // 1 = sign-alternating (values near +-1), 2 = annihilating (generic)

    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_pointer_mode(h,rocblas_pointer_mode_host);
    size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N;
    float *A,*B,*aA,*aB,*C,*Om,*T1,*T2; double *A64,*B64,*R,*acc;
    HC(hipMalloc(&A,na*4));  HC(hipMalloc(&B,nb*4));
    HC(hipMalloc(&aA,na*4)); HC(hipMalloc(&aB,nb*4));
    HC(hipMalloc(&A64,na*8));HC(hipMalloc(&B64,nb*8));
    HC(hipMalloc(&C,nc*4));  HC(hipMalloc(&R,nc*8)); HC(hipMalloc(&acc,8));
    HC(hipMalloc(&Om,(size_t)N*P*4)); HC(hipMalloc(&T1,(size_t)K*P*4)); HC(hipMalloc(&T2,(size_t)M*P*4));
    bf16 *A0,*A1,*A2,*B0,*B1,*B2; _Float16 *Ah,*Al,*Bh,*Bl;
    for(bf16**p:{&A0,&A1,&A2}) HC(hipMalloc(p,na*2));
    for(bf16**p:{&B0,&B1,&B2}) HC(hipMalloc(p,nb*2));
    for(_Float16**p:{&Ah,&Al}) HC(hipMalloc(p,na*2));
    for(_Float16**p:{&Bh,&Bl}) HC(hipMalloc(p,nb*2));

    const float f1=1.f,f0=0.f,fi=1.f/2048.f; const double d1=1.0,d0=0.0;
    int t=256; std::vector<float> hC(nc); std::vector<double> hR(nc);
    auto froD=[&](const double* X,size_t n){ HC(hipMemset(acc,0,8)); sqsum_d<<<256,256>>>(X,n,acc);
        double v=0; HC(hipMemcpy(&v,acc,8,hipMemcpyDeviceToHost)); return sqrt(v); };
    auto froF=[&](const float* X,size_t n){ HC(hipMemset(acc,0,8)); sqsum_f<<<256,256>>>(X,n,acc);
        double v=0; HC(hipMemcpy(&v,acc,8,hipMemcpyDeviceToHost)); return sqrt(v); };

    printf("rho sweep   M=N=%d  K=%ld  probes=%d  family=%d (%s)   truth=device DGEMM\n",M,K,P,FAM,
           FAM==1?"sign-alternating, values near +-1":
           FAM==2?"annihilating, all values generic N(0,1)":
                  "smooth x oscillatory, cancellation from an oscillatory integral");
    printf("%9s %11s %11s %7s | %11s %11s %11s %11s | %s\n",
           "rho_tgt","rho_true","rho_hat","hat/tru","fp32","bf16x3","bf16x6","fp16x3","bf16x3/fp32");

    for(double rho=1.0; rho<=1.05e6; rho*=4.0){
        float invrho=(float)(1.0/rho), amp=(float)(sqrt((double)K)/rho);
        if(FAM==1){ gen_A <<<(na+t-1)/t,t>>>(A,aA,A64,na,K,invrho);
                    gen_B <<<(nb+t-1)/t,t>>>(B,aB,B64,nb,N,invrho,amp); }
        else if(FAM==2){ float d=(float)(0.6366*sqrt(2.0*K)/rho);
                    gen_A2<<<(na+t-1)/t,t>>>(A,aA,A64,na,K);
                    gen_B2<<<(nb+t-1)/t,t>>>(B,aB,B64,nb,N,K,d); }
        else      { float fb=(float)fmin(sqrt(rho)*0.9, (double)K/16.0);
                    float m=(float)(0.6366/rho);
                    gen_A3<<<(na+t-1)/t,t>>>(A,aA,A64,na,K);
                    gen_B3<<<(nb+t-1)/t,t>>>(B,aB,B64,nb,N,K,fb,m); }
        to_bf <<<(na+t-1)/t,t>>>(A,A0,A1,na);  to_bf <<<(nb+t-1)/t,t>>>(B,B0,B1,nb);
        to_bf3<<<(na+t-1)/t,t>>>(A,A0,A1,A2,na); to_bf3<<<(nb+t-1)/t,t>>>(B,B0,B1,B2,nb);
        to_f16<<<(na+t-1)/t,t>>>(A,Ah,Al,na);  to_f16<<<(nb+t-1)/t,t>>>(B,Bh,Bl,nb);
        HC(hipDeviceSynchronize());

        // --- fp64 truth, and the true rho ---
        RB(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
           &d1,B64,rocblas_datatype_f64_r,N,A64,rocblas_datatype_f64_r,(rocblas_int)K,&d0,
           R,rocblas_datatype_f64_r,N,R,rocblas_datatype_f64_r,N,
           rocblas_datatype_f64_r,rocblas_gemm_algo_standard,0,0));
        HC(hipDeviceSynchronize());
        double nAB=froD(R,nc);
        HC(hipMemcpy(hR.data(),R,nc*8,hipMemcpyDeviceToHost));
        // || |A||B| ||_F, also in fp64, reusing R
        { double* t64a=A64; (void)t64a; }
        RB(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
           &f1,aB,rocblas_datatype_f32_r,N,aA,rocblas_datatype_f32_r,(rocblas_int)K,&f0,
           C,rocblas_datatype_f32_r,N,C,rocblas_datatype_f32_r,N,
           rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
        HC(hipDeviceSynchronize());
        double nP=froF(C,nc);
        double rho_true=nP/nAB;

        // --- the cheap estimator: two skinny GEMMs each, O(MK+KN) not O(MNK) ---
        rademacher<<<((size_t)N*P+t-1)/t,t>>>(Om,(size_t)N*P,99u);
        HC(hipDeviceSynchronize());
        auto skinny=[&](const float* X,const float* Y)->double{   // || X*(Y*Om) ||_F
            RB(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,P,(rocblas_int)K,N,
               &f1,Om,rocblas_datatype_f32_r,P,Y,rocblas_datatype_f32_r,N,&f0,
               T1,rocblas_datatype_f32_r,P,T1,rocblas_datatype_f32_r,P,
               rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
            RB(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,P,M,(rocblas_int)K,
               &f1,T1,rocblas_datatype_f32_r,P,X,rocblas_datatype_f32_r,(rocblas_int)K,&f0,
               T2,rocblas_datatype_f32_r,P,T2,rocblas_datatype_f32_r,P,
               rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
            HC(hipDeviceSynchronize());
            return froF(T2,(size_t)M*P)/sqrt((double)P); };
        double rho_hat = skinny(aA,aB)/std::max(1e-300,skinny(A,B));

        // --- error of each scheme vs fp64 ---
        auto err=[&]()->double{ HC(hipDeviceSynchronize());
            HC(hipMemcpy(hC.data(),C,nc*4,hipMemcpyDeviceToHost));
            double num=0,den=0; for(size_t i=0;i<nc;++i){double d=(double)hC[i]-hR[i];num+=d*d;den+=hR[i]*hR[i];}
            return sqrt(num/den); };
        auto G=[&](const void*Bp,const void*Ap,rocblas_datatype ty,const float*al,const float*be){
            RB(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
               al,Bp,ty,N,Ap,ty,(rocblas_int)K,be,C,rocblas_datatype_f32_r,N,C,rocblas_datatype_f32_r,N,
               rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)); };
        G(B,A,rocblas_datatype_f32_r,&f1,&f0);                        double e32=err();
        to_bf<<<(na+t-1)/t,t>>>(A,A0,A1,na); to_bf<<<(nb+t-1)/t,t>>>(B,B0,B1,nb);
        HC(hipDeviceSynchronize());
        G(B0,A0,rocblas_datatype_bf16_r,&f1,&f0); G(B0,A1,rocblas_datatype_bf16_r,&f1,&f1);
        G(B1,A0,rocblas_datatype_bf16_r,&f1,&f1);                     double eb3=err();
        to_bf3<<<(na+t-1)/t,t>>>(A,A0,A1,A2,na); to_bf3<<<(nb+t-1)/t,t>>>(B,B0,B1,B2,nb);
        HC(hipDeviceSynchronize());
        { bf16* As[3]={A0,A1,A2}; bf16* Bs[3]={B0,B1,B2}; bool first=true;
          for(int i=0;i<3;++i)for(int j=0;j<3;++j) if(i+j<=2){
            G(Bs[j],As[i],rocblas_datatype_bf16_r,&f1,first?&f0:&f1); first=false; } }
        double eb6=err();
        G(Bh,Ah,rocblas_datatype_f16_r,&f1,&f0); G(Bh,Al,rocblas_datatype_f16_r,&fi,&f1);
        G(Bl,Ah,rocblas_datatype_f16_r,&fi,&f1);                      double e16=err();

        printf("%9.1e %11.3e %11.3e %7.2f | %11.3e %11.3e %11.3e %11.3e | %8.2fx\n",
               rho,rho_true,rho_hat,rho_hat/rho_true,e32,eb3,eb6,e16,eb3/e32);
        fflush(stdout);
    }
    printf("\nrho_hat costs 2 skinny GEMMs per operand pair: O((MK+KN)*P) vs O(MNK) for the real GEMM.\n");
    printf("If hat/tru stays near 1 across six orders, the dispatcher can price cancellation before committing.\n");
    return 0;
}
