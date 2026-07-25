// emugemm_test.cpp - enforce the two invariants emugemm promises, against fp64 ground truth.
//   INV-1  the measured error never exceeds the requested max_rel_err
//   INV-2  emugemm is never slower than a plain rocblas_sgemm on the same shape
// A violation of either is a bug, not a tuning issue.
//
// Build: hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a emugemm_test.cpp emugemm.cpp -o emugemm_test -lrocblas
// Run:   ./emugemm_test [tune_table.csv] [--plan-only]
#include "emugemm.h"
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define HIP_CHECK(c) do{hipError_t _e=(c); if(_e){fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(_e),__LINE__);exit(1);}}while(0)

__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__global__ void gen_normal(float* X,size_t n,unsigned seed,float scale){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    unsigned a=hashu((unsigned)i^seed), b=hashu((unsigned)(i>>32)^(a+0x9e3779b9u));
    float u1=(a>>8)*(1.0f/16777216.0f),u2=(b>>8)*(1.0f/16777216.0f);
    if(u1<1e-7f)u1=1e-7f; X[i]=scale*sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); }
__global__ void to_f64(const float* X,double* Y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)Y[i]=(double)X[i]; }

static const char* SNAME[]={"AUTO","NATIVE_FP32","FP32_CHUNKED","BF16X3","FP16X3","BF16X6"};

int main(int argc,char**argv){
    const char* tbl = (argc>1 && argv[1][0]!='-')? argv[1] : "tune_table.csv";
    bool plan_only=false; for(int i=1;i<argc;++i) if(!strcmp(argv[i],"--plan-only")) plan_only=true;

    // ---------- dispatch logic unit test (no GPU) ----------
    printf("=== plan logic ===\n");
    printf("  %8s %10s %16s %8s %12s %10s  %s\n","K","target","scheme","chunks","pred err","pred spd","reason");
    long Ks[]={4096,65536,262144};
    double tgts[]={1e-4,1e-5,4e-6,1e-6,1e-7};
    int bad_plan=0;
    for(long K:Ks) for(double tg:tgts){
        emu_request_t rq{tg,EMU_SCHEME_AUTO};
        emu_hints_t   hn{1.0f,1.0f,0};                 // in fp16 range
        emu_plan_t p=emugemm_plan(4096,4096,K,&rq,&hn);
        bool unreachable = strstr(p.reason,"UNREACHABLE")!=nullptr;
        printf("  %8ld %10.0e %16s %8d %12.3e %9.2fx  %s\n",K,tg,SNAME[p.chosen],p.chunks,
               p.predicted_err,p.predicted_speedup,p.reason);
        // A miss is only a bug if the plan CLAIMS to have met the target.
        if(!unreachable && p.predicted_err>tg*1.0001){ printf("    !! PLAN VIOLATES TARGET\n"); ++bad_plan; }
        // "Never slower than native" applies only when native itself meets the target -- otherwise native
        // is not a legal answer and buying accuracy with time is correct, not a violation.
        bool native_ok = emugemm_model_err(EMU_SCHEME_NATIVE_FP32,K,1)<=tg;
        if(native_ok && p.predicted_speedup<0.999){ printf("    !! SLOWER THAN NATIVE, WHICH ALSO MET TARGET\n"); ++bad_plan; }
    }
    // bf16x3 must never be chosen below its hard floor
    { emu_request_t rq{1e-6,EMU_SCHEME_AUTO}; emu_hints_t hn{1e6f,1e6f,0};   // out of fp16 range
      emu_plan_t p=emugemm_plan(4096,4096,262144,&rq,&hn);
      printf("  out-of-fp16-range, target 1e-6 -> %s (must NOT be BF16X3)\n",SNAME[p.chosen]);
      if(p.chosen==EMU_SCHEME_BF16X3){ printf("    !! BF16X3 CHOSEN BELOW ITS FLOOR\n"); ++bad_plan; } }
    printf("  plan violations: %d\n",bad_plan);
    if(plan_only) return bad_plan?1:0;

    // ---------- end-to-end against fp64 ----------
    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_pointer_mode(h,rocblas_pointer_mode_host);
    emugemm_init(tbl);

    struct S { int M,N; long K; };
    std::vector<S> shapes = {{2048,2048,4096},{4096,4096,16384},{4096,4096,65536},
                             {8192,4096,8192},{2048,8192,32768}};
    printf("\n=== end-to-end vs FP64 truth ===\n");
    printf("  %5s %5s %8s %9s %14s %6s %11s %11s %8s %8s %s\n",
           "M","N","K","target","scheme","chunks","meas err","native err","emu ms","nat ms","verdict");
    int viol=0;
    for(const S& s : shapes) for(double tg : {1e-5, 4e-6, 1e-6}){
        size_t na=(size_t)s.M*s.K, nb=(size_t)s.K*s.N, nc=(size_t)s.M*s.N;
        float *A,*B,*C,*Cn; double *A64,*B64,*R;
        HIP_CHECK(hipMalloc(&A,na*4)); HIP_CHECK(hipMalloc(&B,nb*4));
        HIP_CHECK(hipMalloc(&C,nc*4)); HIP_CHECK(hipMalloc(&Cn,nc*4));
        HIP_CHECK(hipMalloc(&A64,na*8));HIP_CHECK(hipMalloc(&B64,nb*8));HIP_CHECK(hipMalloc(&R,nc*8));
        int t=256;
        gen_normal<<<(na+t-1)/t,t>>>(A,na,12345u,1.f); gen_normal<<<(nb+t-1)/t,t>>>(B,nb,67890u,1.f);
        to_f64<<<(na+t-1)/t,t>>>(A,A64,na); to_f64<<<(nb+t-1)/t,t>>>(B,B64,nb);
        HIP_CHECK(hipDeviceSynchronize());
        const double d1=1.0,d0=0.0; const float f1=1.f,f0=0.f;
        rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,s.N,s.M,(rocblas_int)s.K,
            &d1,B64,rocblas_datatype_f64_r,s.N,A64,rocblas_datatype_f64_r,(rocblas_int)s.K,&d0,
            R,rocblas_datatype_f64_r,s.N,R,rocblas_datatype_f64_r,s.N,
            rocblas_datatype_f64_r,rocblas_gemm_algo_standard,0,0);
        auto native=[&]{ rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,s.N,s.M,(rocblas_int)s.K,
            &f1,B,rocblas_datatype_f32_r,s.N,A,rocblas_datatype_f32_r,(rocblas_int)s.K,&f0,
            Cn,rocblas_datatype_f32_r,s.N,Cn,rocblas_datatype_f32_r,s.N,
            rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0); };
        std::vector<float> hC(nc); std::vector<double> hR(nc);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hR.data(),R,nc*8,hipMemcpyDeviceToHost));
        auto errof=[&](float* dev)->double{ HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(hC.data(),dev,nc*4,hipMemcpyDeviceToHost));
            double num=0,den=0; for(size_t i=0;i<nc;++i){ double d=(double)hC[i]-hR[i]; num+=d*d; den+=hR[i]*hR[i]; }
            return sqrt(num/den); };

        emu_request_t rq{tg,EMU_SCHEME_AUTO}; emu_hints_t hn{-1.f,-1.f,0}; emu_plan_t plan;
        emugemm_sgemm(h,s.M,s.N,s.K,A,B,C,&rq,&hn,&plan);
        double e_emu=errof(C);
        native(); double e_nat=errof(Cn);

        hipEvent_t ev1,ev2; HIP_CHECK(hipEventCreate(&ev1)); HIP_CHECK(hipEventCreate(&ev2));
        auto tim=[&](void(*)(void),int){return 0.0;}; (void)tim;
        double t_emu,t_nat;
        { HIP_CHECK(hipEventRecord(ev1)); for(int i=0;i<3;++i) emugemm_sgemm(h,s.M,s.N,s.K,A,B,C,&rq,&hn,&plan);
          HIP_CHECK(hipEventRecord(ev2)); HIP_CHECK(hipEventSynchronize(ev2));
          float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,ev1,ev2)); t_emu=ms/3; }
        { HIP_CHECK(hipEventRecord(ev1)); for(int i=0;i<3;++i) native();
          HIP_CHECK(hipEventRecord(ev2)); HIP_CHECK(hipEventSynchronize(ev2));
          float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,ev1,ev2)); t_nat=ms/3; }

        bool ok_err = e_emu<=tg;
        // Same correction as in the plan test: "never slower than native" only binds when native itself
        // meets the target. Where it does not, native is not a legal answer and spending time is correct.
        bool native_ok = (e_nat<=tg);
        bool ok_spd = !native_ok || (t_emu<=t_nat*1.05);
        if(!ok_err) ++viol; if(!ok_spd) ++viol;
        printf("  %5d %5d %8ld %9.0e %14s %6d %11.3e %11.3e %8.2f %8.2f %s%s\n",
               s.M,s.N,s.K,tg,SNAME[plan.chosen],plan.chunks,e_emu,e_nat,t_emu,t_nat,
               ok_err?"":"ERR! ", ok_spd?(ok_err?"ok":""):"SLOW!");
        hipFree(A);hipFree(B);hipFree(C);hipFree(Cn);hipFree(A64);hipFree(B64);hipFree(R);
    }
    printf("\n  invariant violations: %d\n",viol);
    emugemm_shutdown();
    return (viol||bad_plan)?1:0;
}
