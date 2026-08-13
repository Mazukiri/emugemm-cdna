// tune_trans.cpp - issue #9985 surveyed ONLY transA=N,transB=N. Tensile tunes each transpose
// combination in a SEPARATE logic file, so the published "61-79% of shapes" says nothing about
// TN/NT/TT. This measures default-vs-best-available per transpose. NN is kept as a control: if the
// tool is sound it must reproduce the published NN numbers.
//
// ============================================================================================
// MEASUREMENT RULES (2026-08-12). Each was adopted after reading how the field benchmarks these
// parts, and each is *checked at runtime* rather than merely asserted in a comment. Sources:
// CUTLASS GEMM performance-measurement guidelines; Julich JSC MI250 evaluation; ROCm compute/memory
// partition-mode documentation.
//
// 1. WARM BOTH ARMS BEFORE TIMING EITHER (the original rule, kept). An earlier version timed the
//    default on a cold device and the winners on a warm one: 2.4% mean, 23% max inflation of gain.
//
// 2. BUFFER ROTATION  --rotate N
//    CUTLASS: allocate duplicate buffers totalling >= 2x L2 and rotate pointers per iteration, or
//    iteration 2 onward reads from cache. This is not cosmetic for a CROSS-ARCHITECTURE comparison:
//    MI300X has 256 MB of Infinity Cache, one MI250 GCD has 8 MB of L2, 80% of our shape population
//    fits in the former and 47% in the latter. Our own measured gap splits along exactly that line
//    (1.44 cache-resident vs 1.11 streaming on gfx942). --rotate 1 reproduces the old behaviour so
//    earlier rounds stay comparable; the achieved count is written to the CSV as `nrot`, because
//    memory pressure can silently force fewer copies than requested.
//
// 3. CACHE-RESIDENCY DETECTOR  --hbm GBS
//    Rotation is a mitigation, not a proof. The check: compute the achieved bandwidth of the best
//    kernel, bytes/time. If that exceeds the part's HBM peak, the data cannot have come from HBM, so
//    the row was cache-served whatever the rotation count says. Written as `cache_flag` and
//    `ach_gbs`. This needs no performance counters and cannot be fooled by a rotation that failed.
//
// 4. TIME-BASED REPETITION  --minms MS
//    CUTLASS asks for >=1000 iterations on large problems; a fixed 5 is far too few for a 20 us
//    kernel, where launch overhead and timer granularity dominate. Repetitions are now chosen so
//    each timed window spans at least MS milliseconds, clamped to [5, 2000] launches.
//
// 5. UNIFORM [-1,1] FILL  (was memset(0x3c))
//    The old fill made every element the same bf16/fp16 bit pattern. That is defensible for a
//    fixed-frequency measurement and indefensible for anything power-sensitive, since narrow-
//    precision power draw depends strongly on operand distribution. Filling with uniform [-1,1]
//    costs one kernel per buffer and removes the objection.
//
// 6. CLOCK-DRIFT DETECTOR  (always on)
//    We cannot set a fixed frequency without host admin rights, so instead we detect when the clock
//    moved. A small reference GEMM is timed at the start and at the end of each shape; the ratio of
//    the two is written as `clk_drift`. ScalarLM measured MI300X dropping to 1.136 GHz under a 750 W
//    cap during BF16 GEMM, so this is a real effect on rented hardware. Report `clk_drift` alongside
//    the results; do not filter on it, since drift correlates with shape size and filtering by it
//    filters by size.
//
// 7. PER-SHAPE BUDGET  --budget SEC
//    Cost is nsol * t_default and the log-uniform sampler draws nsol up to ~600 with t_default up to
//    tens of ms; one draw stalled a dry run for >90 s. Over-budget shapes are SKIPPED AND COUNTED -
//    a truncation rule that is not reported is a truncation rule that biases the sample.
// ============================================================================================
//
//   hipcc -O3 --offload-arch=gfx942 tune_trans.cpp -o tune_trans -lrocblas
//   ./tune_trans shapes.txt out.csv [--budget SEC] [--rotate N] [--minms MS] [--hbm GBS] [--legacy-fill]
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
#define HC(c) do{hipError_t e=(c); if(e){printf("HIP %d @%d\n",(int)e,__LINE__);return 1;}}while(0)
typedef __hip_bfloat16 bf16;

// Uniform [-1,1] from a hashed index: deterministic, reproducible, no host->device copy.
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
    const char* inf=argc>1?argv[1]:"shapes.txt";
    const char* outf=argc>2?argv[2]:"trans.csv";
    double budget=1e30, minms=0.0, hbm=0.0; long nskip=0; int rot=1, legacy_fill=0;
    for(int i=3;i<argc;++i){
        if(!strcmp(argv[i],"--budget") && i+1<argc) budget=atof(argv[++i]);
        else if(!strcmp(argv[i],"--rotate") && i+1<argc) rot=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--minms")  && i+1<argc) minms=atof(argv[++i]);
        else if(!strcmp(argv[i],"--hbm")    && i+1<argc) hbm=atof(argv[++i]);
        else if(!strcmp(argv[i],"--legacy-fill")) legacy_fill=1;
    }
    if(rot<1) rot=1;
    // Default HBM peak by architecture, so the cache detector works without being told. These are
    // spec figures and only ever used as a threshold, never as a normaliser.
    if(hbm<=0){
        hipDeviceProp_t p;
        if(hipGetDeviceProperties(&p,0)==hipSuccess){
            if(strstr(p.gcnArchName,"gfx942")) hbm=5300.0;      // MI300X
            else if(strstr(p.gcnArchName,"gfx90a")) hbm=1600.0; // one MI250 GCD
        }
    }
    printf("config: rotate=%d minms=%.1f hbm_peak=%.0f GB/s fill=%s\n",
           rot,minms,hbm,legacy_fill?"legacy-memset":"uniform[-1,1]");

    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_pointer_mode(h,rocblas_pointer_mode_host);
    hipEvent_t e0,e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    FILE* fi=fopen(inf,"r"); if(!fi){printf("no %s\n",inf);return 1;}
    FILE* fo=fopen(outf,"a");
    if(ftell(fo)==0) fprintf(fo,"M,N,K,dtype,opA,opB,nsol,best_sol,best_ms,def_ms,gain,"
                                "nrot,reps,ach_gbs,cache_flag,clk_drift\n");
    const float f1=1.f,f0=0.f;

    // ---- reference GEMM for the clock-drift detector -------------------------------------------
    // Small enough to cost ~1 ms, large enough that launch overhead does not dominate. Its timing is
    // a proxy for the current clock; we never report it as a performance number.
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
        go(); HC(hipDeviceSynchronize());
        HC(hipEventRecord(e0)); for(int i=0;i<3;++i) go();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms=0; HC(hipEventElapsedTime(&ms,e0,e1)); return ms/3; };

    // ---- clock stabilisation: spin the reference until consecutive windows agree to 2% ----------
    // CUTLASS: "run warmups until power stabilizes (~3 seconds observed for clock settling)". We
    // measure the settling instead of assuming a duration.
    { double prev=ref_ms(); int stable=0;
      for(int i=0;i<40 && stable<3;++i){
          double cur=ref_ms();
          if(fabs(cur-prev)/prev < 0.02) ++stable; else stable=0;
          prev=cur; }
      printf("clock settled: reference GEMM %.3f ms (%.1f TF)\n",
             prev, 2.0*RN*RN*RN/(prev*1e-3)/1e12); fflush(stdout); }

    char line[256];
    while(fgets(line,sizeof line,fi)){
        int M,N; long K; char dn[16],ca,cb;
        if(sscanf(line,"%d,%d,%ld,%15[^,],%c,%c",&M,&N,&K,dn,&ca,&cb)!=6) continue;
        int d=!strcmp(dn,"fp32")?0:!strcmp(dn,"bf16")?1:2;
        int ta=(ca==87||ca==84), tb=(cb==87||cb==84);
        rocblas_operation opA=ta?rocblas_operation_transpose:rocblas_operation_none;
        rocblas_operation opB=tb?rocblas_operation_transpose:rocblas_operation_none;
        size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N, esz=d==0?4:2;
        if(na*esz+nb*esz+nc*4 > 28e9) continue;

        double clk_a = ref_ms();     // clock probe BEFORE this shape

        std::vector<void*> vA, vB; std::vector<float*> vC; int nrot=0;
        for(int c=0;c<rot;++c){
            void *a=nullptr,*b=nullptr; float* cc=nullptr;
            if(hipMalloc(&a,na*esz)!=hipSuccess) break;
            if(hipMalloc(&b,nb*esz)!=hipSuccess){ hipFree(a); break; }
            if(hipMalloc(&cc,nc*4)!=hipSuccess){ hipFree(a); hipFree(b); break; }
            if(legacy_fill){
                HC(hipMemset(a,0x3c,na*esz)); HC(hipMemset(b,0x3c,nb*esz));
            }else{
                int t=256; unsigned s1=12345u+c*7919u, s2=67890u+c*104729u;
                if(d==0){ fill_f32<<<(na+t-1)/t,t>>>((float*)a,na,s1); fill_f32<<<(nb+t-1)/t,t>>>((float*)b,nb,s2); }
                else if(d==1){ fill_bf16<<<(na+t-1)/t,t>>>((bf16*)a,na,s1); fill_bf16<<<(nb+t-1)/t,t>>>((bf16*)b,nb,s2); }
                else { fill_f16<<<(na+t-1)/t,t>>>((_Float16*)a,na,s1); fill_f16<<<(nb+t-1)/t,t>>>((_Float16*)b,nb,s2); }
            }
            HC(hipMemset(cc,0,nc*4));
            vA.push_back(a); vB.push_back(b); vC.push_back(cc); ++nrot;
        }
        if(nrot==0) continue;
        HC(hipDeviceSynchronize());
        auto freeall=[&]{ for(int c=0;c<nrot;++c){ hipFree(vA[c]); hipFree(vB[c]); hipFree(vC[c]); } };
        int rr=0;
        void *pA=vA[0], *pB=vB[0]; float* C=vC[0];
        rocblas_datatype ty=d==0?rocblas_datatype_f32_r:d==1?rocblas_datatype_bf16_r:rocblas_datatype_f16_r;
        rocblas_int ldb=tb?(rocblas_int)K:N, lda=ta?(rocblas_int)M:(rocblas_int)K;
        auto run=[&](rocblas_gemm_algo al,int32_t so){
            int i = rr++ % nrot;                       // rotate BEFORE each launch
            return rocblas_gemm_ex(h,opB,opA,N,M,(rocblas_int)K,&f1,vB[i],ty,ldb,vA[i],ty,lda,&f0,
                vC[i],rocblas_datatype_f32_r,N,vC[i],rocblas_datatype_f32_r,N,
                rocblas_datatype_f32_r,al,so,0); };
        auto timeit=[&](rocblas_gemm_algo al,int32_t so,int reps)->double{
            if(run(al,so)!=rocblas_status_success) return 1e30;
            HC(hipDeviceSynchronize()); HC(hipEventRecord(e0));
            for(int i=0;i<reps;++i) run(al,so);
            HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
            float ms=0; HC(hipEventElapsedTime(&ms,e0,e1)); return ms/reps; };
        if(run(rocblas_gemm_algo_standard,0)!=rocblas_status_success){
            printf("  %dx%dx%ld %s %c%c: unsupported\n",M,N,K,dn,ca,cb);
            freeall(); continue; }
        rocblas_int nsol=0;
        rocblas_gemm_ex_get_solutions(h,opB,opA,N,M,(rocblas_int)K,&f1,pB,ty,ldb,pA,ty,lda,&f0,
            C,rocblas_datatype_f32_r,N,C,rocblas_datatype_f32_r,N,rocblas_datatype_f32_r,
            rocblas_gemm_algo_solution_index,0,nullptr,&nsol);
        if(nsol<=0){ printf("  %dx%dx%ld %s %c%c: no solutions\n",M,N,K,dn,ca,cb);
                     freeall(); continue; }

        // one cheap estimate serves both the budget guard and the repetition count
        for(int w=0;w<2;++w) run(rocblas_gemm_algo_standard,0);
        double tdef=timeit(rocblas_gemm_algo_standard,0,3);
        if(budget<1e29){
            double est=nsol*tdef/1000.0;
            if(est>budget){
                ++nskip;
                printf("  %dx%dx%ld %s %c%c: SKIP est %.0fs > budget %.0fs (%d sols x %.2f ms)\n",
                       M,N,K,dn,ca,cb,est,budget,(int)nsol,tdef); fflush(stdout);
                freeall(); continue; } }
        // reps so a timed window spans at least minms; clamped so a 5 us kernel does not run 10^6
        // times and a 50 ms kernel does not blow the budget.
        int reps = 5;
        if(minms>0 && tdef>0){
            double want = minms/tdef;
            reps = (int)std::min(2000.0, std::max(5.0, ceil(want)));
        }

        std::vector<rocblas_int> sv(nsol);
        rocblas_gemm_ex_get_solutions(h,opB,opA,N,M,(rocblas_int)K,&f1,pB,ty,ldb,pA,ty,lda,&f0,
            C,rocblas_datatype_f32_r,N,C,rocblas_datatype_f32_r,N,rocblas_datatype_f32_r,
            rocblas_gemm_algo_solution_index,0,sv.data(),&nsol);
        // WARM every candidate (and the default) before timing anything
        for(int w=0;w<4;++w) run(rocblas_gemm_algo_standard,0);
        for(int i=0;i<nsol;++i) run(rocblas_gemm_algo_solution_index,sv[i]);
        HC(hipDeviceSynchronize());
        std::vector<std::pair<double,int32_t>> r1; r1.reserve(nsol);
        for(int i=0;i<nsol;++i){ double ms=timeit(rocblas_gemm_algo_solution_index,sv[i],1);
            if(ms<1e29) r1.push_back({ms,sv[i]}); }
        if(r1.empty()){ freeall(); continue; }
        std::sort(r1.begin(),r1.end());
        double bt=1e30; int32_t bi=r1[0].second; double dt=1e30;
        for(size_t i=0;i<r1.size() && i<8;++i){                       // interleave default with candidates
            double dm=timeit(rocblas_gemm_algo_standard,0,reps); if(dm<dt) dt=dm;
            double ms=timeit(rocblas_gemm_algo_solution_index,r1[i].second,reps);
            if(ms<bt){ bt=ms; bi=r1[i].second; } }

        double clk_b = ref_ms();                       // clock probe AFTER
        double drift = (clk_a>0)? clk_b/clk_a : 1.0;   // >1 means the machine got slower
        // Cache detector: bytes that a cold run would have to move, over the best kernel's time.
        double bytes = (double)na*esz + (double)nb*esz + (double)nc*4;
        double ach_gbs = bytes / (bt*1e-3) / 1e9;
        int cache_flag = (hbm>0 && ach_gbs > hbm) ? 1 : 0;

        fprintf(fo,"%d,%d,%ld,%s,%c,%c,%d,%d,%.5f,%.5f,%.4f,%d,%d,%.1f,%d,%.4f\n",
                M,N,K,dn,ca,cb,(int)nsol,(int)bi,bt,dt,dt/bt,nrot,reps,ach_gbs,cache_flag,drift);
        fflush(fo);
        printf("  %6dx%-6dx%-6ld %s %c%c: %4d sols  best %8.3f  def %8.3f  gain %5.2fx  "
               "reps %4d  %6.0f GB/s%s%s\n",
               M,N,K,dn,ca,cb,(int)nsol,bt,dt,dt/bt,reps,ach_gbs,
               cache_flag?"  CACHED":"", (drift>1.03||drift<0.97)?"  CLKDRIFT":""); fflush(stdout);
        freeall();
    }
    fclose(fi); fclose(fo);
    hipFree(rA); hipFree(rB); hipFree(rC);
    printf("done (%ld shapes skipped over budget)\n",nskip); return 0;
}
