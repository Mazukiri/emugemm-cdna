// randsvd2.cpp - Phase 4 done properly: randomized SVD with CholeskyQR2 instead of Householder QR.
//
// randsvd.cpp measured only 1.018x end-to-end, and the reason was in its own numbers: 6192 ms total of
// which the GEMMs were ~300 ms. rocSOLVER's Householder geqrf+orgqr on a 262144x2048 panel runs at ~1
// TFLOP/s (3% of peak) and ate 5.9 s. The pipeline was orthogonalisation-bound, so a 1.2x GEMM is worth
// 0.8%. That is an honest negative result about THAT pipeline -- and also a sign the pipeline was wrong:
// Householder QR is not what anyone uses for tall-skinny panels on a GPU, precisely because of this.
//
// CholeskyQR2 is the standard GPU choice and is GEMM-dominated:
//     G = Y^T Y   (l x l, K = m)   <- big-K GEMM
//     R = chol(G) (l x l, tiny)
//     Q = Y R^-1  (trsm)
//     repeat once for stability (that is the "2")
// so three of the four heavy operations become GEMMs with K = m, exactly the regime Round 8/9 mapped.
// Reported alongside the Householder number, not instead of it.
//
// Layout: everything is column-major (rocSOLVER/rocBLAS convention). A column-major MxN matrix with ld=M
// is byte-identical to a row-major NxM one, so column-major C=A*B is emugemm(M=n, N=m, K=k, B, A, C).
// The Gram matrix genuinely needs Y^T, which no argument swap can produce, so it is materialised once by
// a transpose kernel -- O(m*l), negligible against O(m*l^2).
//
// Build: hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a randsvd2.cpp emugemm.cpp -o randsvd2 -lrocblas -lrocsolver
// Run:   ./randsvd2 [m] [n] [l] [tune_table.csv]
#include "emugemm.h"
#include <rocsolver/rocsolver.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define HIP_CHECK(c) do{hipError_t _e=(c); if(_e){fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(_e),__LINE__);exit(1);}}while(0)

__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__device__ __forceinline__ float nrm(size_t i,unsigned s){
    unsigned a=hashu((unsigned)i^s), b=hashu((unsigned)(i>>32)^(a+0x9e3779b9u));
    float u1=(a>>8)*(1.0f/16777216.0f),u2=(b>>8)*(1.0f/16777216.0f);
    if(u1<1e-7f)u1=1e-7f; return sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); }

// A column-major m x n, decaying spectrum so a rank-l approximation is meaningful
__global__ void gen_A(float* A,int m,int n,float tau){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)m*n) return;
    int j=idx/m;  A[idx]=nrm(idx,4242u)*expf(-(float)j/tau); }
__global__ void gen_rand(float* X,size_t n,unsigned s){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) X[i]=nrm(i,s); }
// transpose column-major m x l  ->  column-major l x m
__global__ void transp(const float* X,float* Y,int m,int l){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)m*l) return;
    int i=idx%m, j=idx/m; Y[(size_t)i*l+j]=X[idx]; }
__global__ void sqsum(const float* X,size_t n,double* out){
    __shared__ double sm[256];
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
    double s=0; for(; i<n; i+=st){ double v=X[i]; s+=v*v; }
    sm[threadIdx.x]=s; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) sm[threadIdx.x]+=sm[threadIdx.x+k]; __syncthreads(); }
    if(threadIdx.x==0) atomicAdd(out,sm[0]); }
static double fro2(const float* X,size_t n){
    double* d; HIP_CHECK(hipMalloc(&d,8)); HIP_CHECK(hipMemset(d,0,8));
    sqsum<<<256,256>>>(X,n,d);
    double h=0; HIP_CHECK(hipMemcpy(&h,d,8,hipMemcpyDeviceToHost)); HIP_CHECK(hipFree(d)); return h; }

static rocblas_handle H;
static bool  g_emu; static double g_tgt;
// column-major C(m x n) = A(m x k) * B(k x n), routed through emugemm or plain rocBLAS
static void gemm_cm(int m,int n,long k,const float* A,const float* B,float* C,emu_plan_t* p){
    // The baseline goes through emugemm with the scheme FORCED to native fp32, so it gets the same
    // per-shape tuned solution lookup the emulated path gets. Calling rocblas_gemm_ex with
    // algo_standard here would hand the emulation a free 1.35x (the mean tuning gain measured in
    // Phase 2) and inflate every speedup in this benchmark.
    emu_request_t rq{ g_emu? g_tgt : 1e30, g_emu? EMU_SCHEME_AUTO : EMU_SCHEME_NATIVE_FP32 };
    emu_hints_t hn{-1.f,-1.f,0};
    emugemm_sgemm(H,n,m,k,B,A,C,&rq,&hn,p);
}

int main(int argc,char**argv){
    int m=argc>1?atoi(argv[1]):262144, n=argc>2?atoi(argv[2]):4096, l=argc>3?atoi(argv[3]):2048;
    const char* tbl=argc>4?argv[4]:"tune_table.csv";
    rocblas_create_handle(&H); rocblas_set_pointer_mode(H,rocblas_pointer_mode_host);
    emugemm_init(tbl);

    size_t nA=(size_t)m*n, nY=(size_t)m*l, nG=(size_t)l*l, nB=(size_t)l*n, nO=(size_t)n*l;
    float *A,*Om,*Y,*Yt,*G,*B; rocblas_int* info;
    HIP_CHECK(hipMalloc(&A,nA*4)); HIP_CHECK(hipMalloc(&Om,nO*4)); HIP_CHECK(hipMalloc(&Y,nY*4));
    HIP_CHECK(hipMalloc(&Yt,nY*4));HIP_CHECK(hipMalloc(&G,nG*4));  HIP_CHECK(hipMalloc(&B,nB*4));
    HIP_CHECK(hipMalloc(&info,4));
    printf("randomized SVD + CholeskyQR2   m=%d n=%d rank=%d   ~%.1f GB\n",m,n,l,(nA+3*nY+nG+nB+nO)*4.0/1e9);

    int T=256;
    gen_A<<<(nA+T-1)/T,T>>>(A,m,n,(float)n/6.f);
    gen_rand<<<(nO+T-1)/T,T>>>(Om,nO,777u);
    HIP_CHECK(hipDeviceSynchronize());
    double A_f2=fro2(A,nA);

    hipEvent_t e0,e1,e2,e3; for(auto p:{&e0,&e1,&e2,&e3}) HIP_CHECK(hipEventCreate(p));
    const float one=1.f;

    auto run=[&](bool emu,double tgt,double* t_tot,double* t_gemm,double* resid,emu_plan_t* pg){
        g_emu=emu; g_tgt=tgt;
        HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipEventRecord(e0));
        gemm_cm(m,l,(long)n,A,Om,Y,pg);                       // Y = A*Omega        (K=n)
        HIP_CHECK(hipEventRecord(e1));
        for(int pass=0;pass<2;++pass){                        // CholeskyQR2
            transp<<<(nY+T-1)/T,T>>>(Y,Yt,m,l);
            gemm_cm(l,l,(long)m,Yt,Y,G,pg);                   // G = Y^T Y          (K=m)
            rocsolver_spotrf(H,rocblas_fill_upper,l,G,l,info);
            rocblas_strsm(H,rocblas_side_right,rocblas_fill_upper,rocblas_operation_none,
                          rocblas_diagonal_non_unit,m,l,&one,G,l,Y,m);   // Y <- Y R^-1
        }
        HIP_CHECK(hipEventRecord(e2));
        transp<<<(nY+T-1)/T,T>>>(Y,Yt,m,l);
        gemm_cm(l,n,(long)m,Yt,A,B,pg);                       // B = Q^T A          (K=m)
        HIP_CHECK(hipEventRecord(e3)); HIP_CHECK(hipEventSynchronize(e3));
        float a=0,b=0,c=0; HIP_CHECK(hipEventElapsedTime(&a,e0,e1));
        HIP_CHECK(hipEventElapsedTime(&b,e1,e2)); HIP_CHECK(hipEventElapsedTime(&c,e2,e3));
        *t_tot=a+b+c; *t_gemm=a+c;
        *resid=sqrt(std::max(0.0,A_f2-fro2(B,nB))/A_f2); };

    static const char* SN[]={"AUTO","NATIVE","FP32_CHUNKED","BF16X3","FP16X3","BF16X6"};
    double tt,tg,rs; emu_plan_t pg{};
    printf("\n  %-24s %10s %10s %18s\n","variant","total ms","GEMM ms","rel residual");
    run(false,0,&tt,&tg,&rs,&pg); double t0=tt,r0=rs;
    printf("  %-24s %10.2f %10.2f %18.10e\n","native fp32",tt,tg,rs);
    for(double tgt : {1e-5,1e-6}){
        run(true,tgt,&tt,&tg,&rs,&pg);
        printf("  %-14s %-9.0e %10.2f %10.2f %18.10e   last scheme %s\n","emugemm",tgt,tt,tg,rs,SN[pg.chosen]);
        printf("      -> %.3fx end-to-end,  residual %s native (%+.4f%%)\n",
               t0/tt, rs<=r0?"<=":">", 100.0*(rs-r0)/r0); }
    printf("\n  Compare with randsvd.cpp (Householder QR): 6192 ms total, of which ~300 ms GEMM -> 1.018x.\n");
    emugemm_shutdown(); return 0;
}
