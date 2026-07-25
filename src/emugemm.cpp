// emugemm.cpp - see emugemm.h for the measured model this implements.
#include "emugemm.h"
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

typedef __hip_bfloat16 bf16;
#define HC(c) do{hipError_t _e=(c); if(_e) return rocblas_status_internal_error;}while(0)

// ---------------- measured model constants (Round 7-9, MI250 gfx90a) ----------------
static const double A_FP32   = 1.79e-8;   // default rocBLAS fp32 kernel accumulation constant
static const double A_LOWP   = 8.95e-9;   // every bf16/fp16 Tensile path measured at exactly half of it
static const double FLOOR_BF16X3 = 4.4355e-6;
static const double FLOOR_FP16X3 = 3.6e-7;
static const double FLOOR_BF16X6 = 0.0;
static const double SPD_BF16X3 = 1.21, SPD_FP16X3 = 1.19, SPD_BF16X6 = 0.57;
// fp16's representable range; outside it the fp16 path produces inf/NaN (measured Round 7)
static const float  FP16_MAX = 65504.0f;
// Chunking cost. NOT a smooth formula: measured at M=N=4096,K=262144 the wall time went
//   c=1 338.6 ms | c=4 333.9 | c=16 643.0 | c=64 441.3 | c=256 601.6
// i.e. NON-MONOTONIC, because each chunk depth K/c makes rocBLAS reselect a different kernel. So this is
// a deliberately CONSERVATIVE envelope over those measurements, not a fit. It is calibrated at exactly one
// shape; Phase 5 must measure it across shapes before this is trusted anywhere else.
static double chunk_cost(int c){
    if(c<=1) return 1.00;
    if(c<=4) return 1.05;
    if(c<=16) return 1.95;
    if(c<=64) return 1.35;
    if(c<=256) return 1.85;
    return 3.0;                 // beyond 256 each chunk is short enough to be memory-bound: assume bad
}
// Past this, chunks get so shallow that the GEMM stops being compute-bound and the model above is not
// evidence-backed at all. Refuse rather than silently under-deliver.
static const int MAX_CHUNKS = 256;

double emugemm_model_err(emu_scheme_t s, long K, int c){
    if(c<1) c=1; double kc=(double)K/c, fl=0, a=A_LOWP;
    switch(s){
        case EMU_SCHEME_NATIVE_FP32: case EMU_SCHEME_FP32_CHUNKED: fl=0;               a=A_FP32; break;
        case EMU_SCHEME_BF16X3:      fl=FLOOR_BF16X3; break;
        case EMU_SCHEME_FP16X3:      fl=FLOOR_FP16X3; break;
        case EMU_SCHEME_BF16X6:      fl=FLOOR_BF16X6; break;
        default: return NAN;
    }
    return sqrt(fl*fl + a*a*kc);
}
// rho for iid data: measured 0.64*sqrt(K) (Round 9 Phase 5 measured 163 at K=65536 -> 0.637*sqrt(K)).
// The old model err = sqrt(floor^2 + a^2 K/c) was calibrated on iid data, so sqrt(K) was rho in disguise.
static double rho_ref(long K){ return 0.64*sqrt((double)K); }

double emugemm_model_err_rho(emu_scheme_t s, long K, int c, double rho){
    double e = emugemm_model_err(s,K,c);
    if(!(rho>0)) return e;                       // unknown rho: fall back to the iid model
    return e * (rho/rho_ref(K));
}

double emugemm_model_speed(emu_scheme_t s, int c){
    double base;
    switch(s){
        case EMU_SCHEME_NATIVE_FP32: base=1.0; break;
        case EMU_SCHEME_FP32_CHUNKED:base=1.0; break;
        case EMU_SCHEME_BF16X3:      base=SPD_BF16X3; break;
        case EMU_SCHEME_FP16X3:      base=SPD_FP16X3; break;
        case EMU_SCHEME_BF16X6:      base=SPD_BF16X6; break;
        default: return 0;
    }
    return base/chunk_cost(c);
}

// ---------------- workspace ----------------
// The invariant test caught emugemm being SLOWER than native on small shapes (2048x2048x4096: 1.39 ms vs
// 1.03 ms). Cause: hipMalloc/hipFree of the split buffers on EVERY call, which costs more than a ~1 ms
// GEMM saves. Buffers are now allocated once, grown on demand, and freed only at shutdown.
static void*  g_ws[6] = {nullptr};
static size_t g_ws_sz[6] = {0};
static void* ws_get(int i,size_t bytes){
    if(g_ws_sz[i]>=bytes) return g_ws[i];
    if(g_ws[i]) hipFree(g_ws[i]);
    if(hipMalloc(&g_ws[i],bytes)!=hipSuccess){ g_ws[i]=nullptr; g_ws_sz[i]=0; return nullptr; }
    g_ws_sz[i]=bytes; return g_ws[i];
}
static void ws_free_all(){ for(int i=0;i<6;++i){ if(g_ws[i]) hipFree(g_ws[i]); g_ws[i]=nullptr; g_ws_sz[i]=0; } }

// Overhead the FLOP model does not see: the absmax scan (reads A and B once) and the split kernels
// (read A,B once, write two low-precision copies each). Effective HBM bandwidth measured on this part is
// ~1.3 TB/s; the constant below is deliberately pessimistic so the planner errs toward native.
static const double HBM_BPS = 1.1e12;
static const double FP32_TFLOPS = 37.0e12;
static double overhead_seconds(long M,long N,long K,emu_scheme_t s){
    if(s==EMU_SCHEME_NATIVE_FP32) return 0.0;
    double elems=(double)M*K+(double)K*N;
    if(s==EMU_SCHEME_FP32_CHUNKED) return elems*4.0/HBM_BPS;      // scan only, no split
    return elems*(4.0 /*scan*/ + 4.0 /*read*/ + 4.0 /*write 2x fp16*/)/HBM_BPS;
}

// ---------------- offline solution table ----------------
struct TuneRow { int M,N; long K; int dt; int sol; };   /* dt: 0=fp32 1=bf16 2=fp16 */
static std::vector<TuneRow> g_tune;
static std::string g_arch;
static bool g_ready=false;

rocblas_status emugemm_init(const char* path){
    g_tune.clear(); g_ready=false;
    hipDeviceProp_t p; if(hipGetDeviceProperties(&p,0)!=hipSuccess) return rocblas_status_internal_error;
    std::string live = p.gcnArchName;
    FILE* f=fopen(path,"r");
    if(!f){ fprintf(stderr,"emugemm: no tune table at %s -- running untuned (expect up to 1.7x slower bf16)\n",path);
            g_ready=true; return rocblas_status_success; }
    char line[512]; int rejected=0;
    while(fgets(line,sizeof line,f)){
        char arch[96],rocv[96],dt[16]; int M,N,sol; long K; double bms,dms,gain;
        if(sscanf(line,"%95[^,],%95[^,],%d,%d,%ld,%15[^,],%d,%lf,%lf,%lf",
                  arch,rocv,&M,&N,&K,dt,&sol,&bms,&dms,&gain)!=10) continue;
        // A solution index from another arch would silently select a wrong kernel. Refuse it.
        if(live.find(arch)==std::string::npos && strncmp(arch,live.c_str(),6)!=0){ ++rejected; continue; }
        int dcode = (strcmp(dt,"bf16")==0)?1 : (strcmp(dt,"fp16")==0)?2 : 0;
        g_tune.push_back({M,N,K,dcode,sol});
    }
    fclose(f);
    if(rejected) fprintf(stderr,"emugemm: rejected %d rows from a different arch than %s\n",rejected,live.c_str());
    fprintf(stderr,"emugemm: loaded %zu tuned entries for %s\n",g_tune.size(),live.c_str());
    g_arch=live; g_ready=true; return rocblas_status_success;
}
void emugemm_shutdown(void){ g_tune.clear(); g_ready=false; ws_free_all(); }

// nearest shape in log space; kernel choice tracks ratios more than absolute size
static int lookup_sol(int M,int N,long K,int dt){
    double best=1e30; int sol=0;
    for(const TuneRow& r : g_tune){
        if(r.dt!=dt) continue;
        double d = fabs(log2((double)r.M/M)) + fabs(log2((double)r.N/N)) + fabs(log2((double)r.K/K));
        if(d<best){ best=d; sol=r.sol; }
    }
    return (best<1.0)? sol : 0;   // too far away to trust: fall back to the library heuristic
}

// ---------------- kernels ----------------
__global__ void k_split_bf(const float* X,bf16* S0,bf16* S1,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=X[i]; bf16 s0=(bf16)x; S0[i]=s0; S1[i]=(bf16)(x-(float)s0); }
__global__ void k_split_f16(const float* X,_Float16* H,_Float16* L,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=X[i]; _Float16 h=(_Float16)x; H[i]=h; L[i]=(_Float16)((x-(float)h)*2048.f); }
__global__ void k_accum_f64(double* acc,const float* part,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) acc[i]+=(double)part[i]; }
__global__ void k_zero_f64(double* acc,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) acc[i]=0.0; }
__global__ void k_f64_f32(const double* acc,float* out,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=(float)acc[i]; }
__global__ void k_absmax(const float* X,size_t n,float* out){
    __shared__ float sm[256];
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t stride=(size_t)gridDim.x*blockDim.x;
    float m=0; for(; i<n; i+=stride) m=fmaxf(m,fabsf(X[i]));
    sm[threadIdx.x]=m; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+s]); __syncthreads(); }
    if(threadIdx.x==0) atomicMax((int*)out,__float_as_int(sm[0]));   // values are >=0, so int order == float order
}
static float device_absmax(const float* X,size_t n){
    float* d=(float*)ws_get(5,4); if(!d) return -1.f;
    float z=0; hipMemcpy(d,&z,4,hipMemcpyHostToDevice);
    k_absmax<<<256,256>>>(X,n,d);
    float h=0; hipMemcpy(&h,d,4,hipMemcpyDeviceToHost); return h;
}

// ---------------- cancellation probe ----------------
__global__ void k_abs(const float* X,float* Y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) Y[i]=fabsf(X[i]); }
__global__ void k_rademacher(float* X,size_t n,unsigned s){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    unsigned h=(unsigned)i^s; h^=h>>16; h*=0x7feb352du; h^=h>>15; h*=0x846ca68bu; h^=h>>16;
    X[i]=(h&1u)?1.f:-1.f; }
__global__ void k_sqsum(const float* X,size_t n,double* o){
    __shared__ double sm[256];
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
    double s=0; for(; i<n; i+=st){ double v=X[i]; s+=v*v; }
    sm[threadIdx.x]=s; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) sm[threadIdx.x]+=sm[threadIdx.x+k]; __syncthreads(); }
    if(threadIdx.x==0) atomicAdd(o,sm[0]); }

double emugemm_estimate_rho(rocblas_handle h,int M,int N,long K,const float* A,const float* B,int P){
    if(P<=0) P=64;
    size_t na=(size_t)M*K, nb=(size_t)K*N;
    float* aA=(float*)ws_get(0,na*4); float* aB=(float*)ws_get(1,nb*4);
    float* Om=(float*)ws_get(2,(size_t)N*P*4);
    float* T1=(float*)ws_get(3,(size_t)K*P*4); float* T2=(float*)ws_get(4,(size_t)M*P*4);
    double* acc=(double*)ws_get(5,8);
    if(!aA||!aB||!Om||!T1||!T2||!acc) return -1.0;
    int TPB=256;
    k_abs<<<(na+TPB-1)/TPB,TPB>>>(A,aA,na);
    k_abs<<<(nb+TPB-1)/TPB,TPB>>>(B,aB,nb);
    k_rademacher<<<((size_t)N*P+TPB-1)/TPB,TPB>>>(Om,(size_t)N*P,0x5eedu);
    const float f1=1.f,f0=0.f;
    auto skinny=[&](const float* X,const float* Y)->double{     // || X*(Y*Omega) ||_F / sqrt(P)
        if(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,P,(rocblas_int)K,N,
             &f1,Om,rocblas_datatype_f32_r,P,Y,rocblas_datatype_f32_r,N,&f0,
             T1,rocblas_datatype_f32_r,P,T1,rocblas_datatype_f32_r,P,
             rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)!=rocblas_status_success) return -1.0;
        if(rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,P,M,(rocblas_int)K,
             &f1,T1,rocblas_datatype_f32_r,P,X,rocblas_datatype_f32_r,(rocblas_int)K,&f0,
             T2,rocblas_datatype_f32_r,P,T2,rocblas_datatype_f32_r,P,
             rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)!=rocblas_status_success) return -1.0;
        hipMemset(acc,0,8); k_sqsum<<<256,256>>>(T2,(size_t)M*P,acc);
        double v=0; hipMemcpy(&v,acc,8,hipMemcpyDeviceToHost);
        return sqrt(v/(double)P); };
    double num=skinny(aA,aB), den=skinny(A,B);
    if(num<0||den<=0) return -1.0;
    return num/den;
}

// ---------------- planning ----------------
emu_plan_t emugemm_plan(int M,int N,long K,const emu_request_t* req,const emu_hints_t* hints){
    emu_plan_t p; p.chunks=1; p.chosen=EMU_SCHEME_NATIVE_FP32; p.reason="default";
    double tgt = (req && req->max_rel_err>0)? req->max_rel_err : 1e-5;
    double rho = (hints && hints->rho>0)? hints->rho : -1.0;   // -1 => iid assumption

    if(req && req->force_scheme!=EMU_SCHEME_AUTO){
        p.chosen=(emu_scheme_t)req->force_scheme; p.reason="forced";
        p.predicted_err=emugemm_model_err_rho(p.chosen,K,1,rho);
        p.predicted_speedup=emugemm_model_speed(p.chosen,1); return p;
    }

    bool fp16_safe = hints && hints->a_absmax>0 && hints->b_absmax>0 &&
                     hints->a_absmax<FP16_MAX && hints->b_absmax<FP16_MAX &&
                     hints->a_absmax>6e-5f   && hints->b_absmax>6e-5f;

    // Candidates, fastest first. Pick the first that MEETS the target -- never trade accuracy for speed
    // silently, and never pick a scheme whose floor is above the target (bf16x3's floor is hard).
    struct Cand { emu_scheme_t s; int c; };
    std::vector<Cand> cands;
    if(fp16_safe) cands.push_back({EMU_SCHEME_FP16X3,1});
    cands.push_back({EMU_SCHEME_BF16X3,1});
    cands.push_back({EMU_SCHEME_NATIVE_FP32,1});
    // c needed so that a*sqrt(K/c) <= tgt, i.e. c >= (a^2 K)/tgt^2. Solve it instead of scanning, then
    // round up to a power of two.
    { double need=(A_FP32*A_FP32*(double)K)/(tgt*tgt);
      for(int c=2;c<=MAX_CHUNKS;c*=2) if(c>=need*0.5) cands.push_back({EMU_SCHEME_FP32_CHUNKED,c}); }

    // Speed must be judged END TO END, including the scan and split passes. Comparing raw GEMM rates is
    // what made the library lose to native on small shapes in the first invariant run.
    double t_native = 2.0*(double)M*N*K/FP32_TFLOPS;
    auto eff_speedup=[&](emu_scheme_t s,int c)->double{
        double gemm = t_native/emugemm_model_speed(s,c);
        return t_native/(gemm + overhead_seconds(M,N,K,s));
    };
    bool found=false; Cand best{EMU_SCHEME_NATIVE_FP32,1}; double best_spd=-1;
    for(const Cand& cd : cands){
        double e=emugemm_model_err_rho(cd.s,K,cd.c,rho);
        if(e>tgt) continue;                       // includes bf16x3's hard-floor rejection
        double sp=eff_speedup(cd.s,cd.c);
        if(sp>best_spd){ best_spd=sp; best=cd; found=true; }
    }
    if(!found){
        // Nothing on offer reaches the target. Say so; do NOT quietly return the closest miss.
        p.chosen=EMU_SCHEME_FP32_CHUNKED; p.chunks=MAX_CHUNKS;
        p.predicted_err=emugemm_model_err_rho(EMU_SCHEME_FP32_CHUNKED,K,MAX_CHUNKS,rho);
        p.predicted_speedup=emugemm_model_speed(EMU_SCHEME_FP32_CHUNKED,MAX_CHUNKS);
        p.reason="TARGET UNREACHABLE: even fp32 with the maximum chunk count misses it";
        return p;
    }
    p.chosen=best.s; p.chunks=best.c;
    p.predicted_err=emugemm_model_err_rho(best.s,K,best.c,rho);
    p.predicted_speedup=eff_speedup(best.s,best.c);
    // "Never slower than native" only makes sense when native is itself a legal answer. If the target is
    // below what native fp32 delivers, native is not an option and paying more time is the only way.
    bool native_ok = emugemm_model_err_rho(EMU_SCHEME_NATIVE_FP32,K,1,rho)<=tgt;
    if(p.predicted_speedup<1.0 && native_ok){
        p.chosen=EMU_SCHEME_NATIVE_FP32; p.chunks=1;
        p.predicted_err=emugemm_model_err_rho(EMU_SCHEME_NATIVE_FP32,K,1,rho);
        p.predicted_speedup=1.0; p.reason="emulation would be slower than native";
    } else {
        p.reason = (p.chosen==EMU_SCHEME_FP16X3) ? "fp16 range safe, lowest floor at top speed" :
                   (p.chosen==EMU_SCHEME_BF16X3) ? "robust, target above bf16x3 floor" :
                   (p.chosen==EMU_SCHEME_FP32_CHUNKED) ? "target below bf16x3 floor: fp32 + fp64 chunk sum" :
                                                          "native fp32 suffices";
    }
    return p;
}

// ---------------- execution ----------------
rocblas_status emugemm_sgemm(rocblas_handle h,int M,int N,long K,
                             const float* A,const float* B,float* C,
                             const emu_request_t* req,emu_hints_t* hints,emu_plan_t* out_plan){
    emu_hints_t local = hints? *hints : emu_hints_t{-1.f,-1.f,0,-1.f,0};
    if(local.a_absmax<0) local.a_absmax=device_absmax(A,(size_t)M*K);
    if(local.b_absmax<0) local.b_absmax=device_absmax(B,(size_t)K*N);
    // Measure cancellation before committing. This is the DATA axis: it does not change WHICH scheme
    // wins (K decides that) -- it decides whether the winner actually meets the requested accuracy.
    if(local.rho<0) local.rho=(float)emugemm_estimate_rho(h,M,N,K,A,B,local.rho_probes);
    if(hints){ hints->a_absmax=local.a_absmax; hints->b_absmax=local.b_absmax; hints->rho=local.rho; }

    emu_plan_t plan=emugemm_plan(M,N,K,req,&local);
    if(out_plan) *out_plan=plan;

    const float f1=1.f,f0=0.f,finv=1.f/2048.f;
    size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N;
    int TPB=256; size_t gc=(nc+TPB-1)/TPB;

    auto gemm=[&](const void*Bp,const void*Ap,rocblas_datatype ty,long k0,long kc,
                  const float*al,const float*be,float* Cp,int sol)->rocblas_status{
        return rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)kc,
            al,
            (const char*)Bp + (size_t)k0*N*(ty==rocblas_datatype_f32_r?4:2), ty, N,
            (const char*)Ap + (size_t)k0  *(ty==rocblas_datatype_f32_r?4:2), ty, (rocblas_int)K,
            be, Cp,rocblas_datatype_f32_r,N, Cp,rocblas_datatype_f32_r,N,
            rocblas_datatype_f32_r,
            sol? rocblas_gemm_algo_solution_index : rocblas_gemm_algo_standard, sol, 0); };

    if(plan.chosen==EMU_SCHEME_NATIVE_FP32){
        int sol=lookup_sol(M,N,K,0);
        return gemm(B,A,rocblas_datatype_f32_r,0,K,&f1,&f0,C,sol);
    }
    if(plan.chosen==EMU_SCHEME_FP32_CHUNKED){
        int sol=lookup_sol(M,N,K/plan.chunks,0);
        float*  part=(float*) ws_get(0,nc*4);
        double* acc =(double*)ws_get(1,nc*8);
        if(!part||!acc) return rocblas_status_memory_error;
        k_zero_f64<<<gc,TPB>>>(acc,nc);
        long kc=(K+plan.chunks-1)/plan.chunks;
        for(long k0=0;k0<K;k0+=kc){ long len=std::min(kc,K-k0);
            rocblas_status st=gemm(B,A,rocblas_datatype_f32_r,k0,len,&f1,&f0,part,sol);
            if(st!=rocblas_status_success) return st;
            k_accum_f64<<<gc,TPB>>>(acc,part,nc); }
        k_f64_f32<<<gc,TPB>>>(acc,C,nc);
        return rocblas_status_success;
    }
    if(plan.chosen==EMU_SCHEME_BF16X3 || plan.chosen==EMU_SCHEME_BF16X6){
        bf16 *A0=(bf16*)ws_get(0,na*2), *A1=(bf16*)ws_get(1,na*2);
        bf16 *B0=(bf16*)ws_get(2,nb*2), *B1=(bf16*)ws_get(3,nb*2);
        if(!A0||!A1||!B0||!B1) return rocblas_status_memory_error;
        k_split_bf<<<(na+TPB-1)/TPB,TPB>>>(A,A0,A1,na);
        k_split_bf<<<(nb+TPB-1)/TPB,TPB>>>(B,B0,B1,nb);
        int sol=lookup_sol(M,N,K,1);
        rocblas_status st;
        st=gemm(B0,A0,rocblas_datatype_bf16_r,0,K,&f1,&f0,C,sol);
        if(st==rocblas_status_success) st=gemm(B0,A1,rocblas_datatype_bf16_r,0,K,&f1,&f1,C,sol);
        if(st==rocblas_status_success) st=gemm(B1,A0,rocblas_datatype_bf16_r,0,K,&f1,&f1,C,sol);
        return st;
    }
    if(plan.chosen==EMU_SCHEME_FP16X3){
        _Float16 *Ah=(_Float16*)ws_get(0,na*2), *Al=(_Float16*)ws_get(1,na*2);
        _Float16 *Bh=(_Float16*)ws_get(2,nb*2), *Bl=(_Float16*)ws_get(3,nb*2);
        if(!Ah||!Al||!Bh||!Bl) return rocblas_status_memory_error;
        k_split_f16<<<(na+TPB-1)/TPB,TPB>>>(A,Ah,Al,na);
        k_split_f16<<<(nb+TPB-1)/TPB,TPB>>>(B,Bh,Bl,nb);
        int sol=lookup_sol(M,N,K,2);
        rocblas_status st;
        st=gemm(Bh,Ah,rocblas_datatype_f16_r,0,K,&f1,  &f0,C,sol);
        if(st==rocblas_status_success) st=gemm(Bh,Al,rocblas_datatype_f16_r,0,K,&finv,&f1,C,sol);
        if(st==rocblas_status_success) st=gemm(Bl,Ah,rocblas_datatype_f16_r,0,K,&finv,&f1,C,sol);
        return st;
    }
    return rocblas_status_not_implemented;
}
