// three_way2.cpp - the four-arm comparison, with the two defects of three_way.cpp fixed.
//
// three_way.cpp answered "is rocBLAS behind hipBLASLt?" well enough, but it cannot carry the
// vendor comparison the paper needs, for two reasons found on 2026-08-12/13:
//
//  1. THE CANDIDATE POOL WAS CAPPED. It asked hipblasLtMatmulAlgoGetHeuristic for NALGO=64 and
//     called the best of what came back "hipBLASLt's ceiling". On `three_way_400_gfx90a.csv`,
//     48% of rows came back with exactly 64 - the ceiling was binding, not the library's opinion.
//     A paired re-run of the same 39 shapes with NALGO=256 (`tw256_gfx90a.csv`) moves the measured
//     hipBLASLt gap from 1.1053 to 1.1415 and finds a kernel >2% faster on 15 of 39 shapes, up to
//     5.57x. So "best of the shortlist" is a function of the shortlist length, and comparing it to
//     the NVIDIA arm - which enumerates cuBLASLt's configuration space exhaustively via
//     AlgoGetIds -> capability query -> cross product -> AlgoCheck, 43-512 candidates per shape -
//     compares two caps rather than two heuristics. hipblaslt_ext::getAllAlgos + matmulIsAlgoSupported
//     is the structural equivalent of that recipe, so arm 5 uses it and the cap becomes a reported
//     number (`n_lt_all`) instead of a hidden parameter.
//
//  2. IT USED THE PRE-v2 TIMING. No buffer rotation, no time-based repetition, memset fill. The
//     rocBLAS arms of the paper were re-measured under METHOD_V2 and the hipBLASLt arms were not,
//     so the two sides of the same table were produced by different harnesses. reps=5-equivalent
//     timing compresses every ratio by ~5% (measured, `launch-overhead` note), which is the same
//     order as the vendor difference being claimed - not acceptable in the arm that decides it.
//
// Five arms, all timed in one interleaved loop after all five are warm:
//   1 rb_def     rocBLAS, default heuristic                  what a user gets today
//   2 rb_best    rocBLAS, best solution index                 already in the library, unreachable
//   3 lt_h0      hipBLASLt, heuristic candidate #0            what a hipBLASLt user gets today
//   4 lt_besth   hipBLASLt, best of the heuristic shortlist   comparable to old three_way (lower bound)
//   5 lt_all     hipBLASLt, best of ALL supported algos       the true analogue of the NVIDIA arm
//
// Arm 4 is kept deliberately: it is the only way to state how much of the published gap was the cap.
//
// Build: hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx942 three_way2.cpp -o three_way2 -lrocblas -lhipblaslt
// Run:   ./three_way2 shapes.txt out.csv [--rotate N] [--minms MS] [--budget SEC] [--maxalgo N] [--legacy-fill]
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#if __has_include(<hipblaslt/hipblaslt-ext.hpp>)
#include <hipblaslt/hipblaslt-ext.hpp>
#define HAVE_LT_EXT 1
#endif
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

// Uniform [-1,1] from a hashed index - identical to tune_trans.cpp so the two harnesses feed the
// hardware the same data. A constant fill is defensible for fixed-frequency timing but biases a
// power-limited part at narrow precisions, and MI300X is power-limited.
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
    const char* inf = argc>1?argv[1]:"shapes.txt";
    const char* outf= argc>2?argv[2]:"three_way2.csv";
    int rot=4, maxalgo=1024, legacy_fill=0, heurN=256;
    double minms=10.0, budget=1e30; long nskip=0;
    for(int i=3;i<argc;++i){
        if(!strcmp(argv[i],"--rotate")  && i+1<argc) rot=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--minms")   && i+1<argc) minms=atof(argv[++i]);
        else if(!strcmp(argv[i],"--budget")  && i+1<argc) budget=atof(argv[++i]);
        else if(!strcmp(argv[i],"--maxalgo") && i+1<argc) maxalgo=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--heur")    && i+1<argc) heurN=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--legacy-fill")) legacy_fill=1;
    }
    if(rot<1) rot=1;
#ifdef HAVE_LT_EXT
    const int have_ext=1;
#else
    const int have_ext=0;
#endif
    printf("config: rotate=%d minms=%.1f heur=%d maxalgo=%d ext_enum=%s fill=%s\n",
           rot,minms,heurN,maxalgo,have_ext?"yes":"NO(heuristic only)",
           legacy_fill?"legacy-memset":"uniform[-1,1]"); fflush(stdout);

    rocblas_handle rb; rocblas_create_handle(&rb); rocblas_set_pointer_mode(rb,rocblas_pointer_mode_host);
    hipblasLtHandle_t lt; hipblasLtCreate(&lt);
    void* ws=nullptr; size_t wsBytes=512ull<<20; HC(hipMalloc(&ws,wsBytes));
    hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsBytes,sizeof(wsBytes));
    hipEvent_t e0,e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const float f1=1.f,f0=0.f;

    // ---- clock-drift reference, same construction as tune_trans.cpp ----------------------------
    const int RN=2048; float *rA,*rB,*rC;
    HC(hipMalloc(&rA,(size_t)RN*RN*4)); HC(hipMalloc(&rB,(size_t)RN*RN*4)); HC(hipMalloc(&rC,(size_t)RN*RN*4));
    { int t=256; size_t n=(size_t)RN*RN;
      fill_f32<<<(n+t-1)/t,t>>>(rA,n,1u); fill_f32<<<(n+t-1)/t,t>>>(rB,n,2u);
      HC(hipMemset(rC,0,n*4)); HC(hipDeviceSynchronize()); }
    auto ref_ms=[&]()->double{
        auto go=[&]{ return rocblas_gemm_ex(rb,rocblas_operation_none,rocblas_operation_none,RN,RN,RN,
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
    if(ftell(fo)==0) fprintf(fo,"M,N,K,dtype,opA,opB,beta,rb_def_ms,rb_best_ms,lt_h0_ms,lt_besth_ms,lt_all_ms,"
                                "tf_rb_def,tf_rb_best,tf_lt_h0,tf_lt_besth,tf_lt_all,"
                                "n_rb_sol,n_lt_heur,n_lt_all,n_lt_ok,nrot,reps,clk_drift\n");
    char line[256];
    while(fgets(line,sizeof line,fi)){
        int M,N,beta_i=0; long K; char dn[16],ca,cb; double og=0;
        int nf=sscanf(line,"%d,%d,%ld,%15[^,],%c,%c,%d,%lf",&M,&N,&K,dn,&ca,&cb,&beta_i,&og);
        if(nf<6) continue;                       // beta and the old gain column are optional
        int d=!strcmp(dn,"fp32")?0:!strcmp(dn,"bf16")?1:2;
        int ta=(ca=='T'||ca=='W'), tb=(cb=='T'||cb=='W');
        size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N, esz=d==0?4:2;
        if((double)na*esz+(double)nb*esz+(double)nc*4 > 28e9) continue;

        double clk_a=ref_ms();

        // ---- rotated buffer sets, shared by both libraries ------------------------------------
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

        // ---- rocBLAS side (row-major C=A*B as col-major m=N,n=M,k=K, operands swapped) --------
        rocblas_operation opA=ta?rocblas_operation_transpose:rocblas_operation_none;
        rocblas_operation opB=tb?rocblas_operation_transpose:rocblas_operation_none;
        rocblas_datatype ty=d==0?rocblas_datatype_f32_r:d==1?rocblas_datatype_bf16_r:rocblas_datatype_f16_r;
        rocblas_int ldb=tb?(rocblas_int)K:N, lda=ta?(rocblas_int)M:(rocblas_int)K;
        const float* beta=beta_i?&f1:&f0;
        auto rrun=[&](rocblas_gemm_algo al,int32_t so){
            int i=rr++ % nrot;
            return rocblas_gemm_ex(rb,opB,opA,N,M,(rocblas_int)K,&f1,vB[i],ty,ldb,vA[i],ty,lda,beta,
                vC[i],rocblas_datatype_f32_r,N,vC[i],rocblas_datatype_f32_r,N,
                rocblas_datatype_f32_r,al,so,0); };
        auto rtm=[&](rocblas_gemm_algo al,int32_t so,int reps)->double{
            if(rrun(al,so)!=rocblas_status_success) return 1e30;
            HCD(hipDeviceSynchronize()); HCD(hipEventRecord(e0));
            for(int i=0;i<reps;++i) rrun(al,so);
            HCD(hipEventRecord(e1)); HCD(hipEventSynchronize(e1));
            float ms=0; HCD(hipEventElapsedTime(&ms,e0,e1)); return ms/reps; };

        if(rrun(rocblas_gemm_algo_standard,0)!=rocblas_status_success){ freeall(); continue; }
        for(int w=0;w<2;++w) rrun(rocblas_gemm_algo_standard,0);
        double tdef=rtm(rocblas_gemm_algo_standard,0,3);
        // reps for a timed window of at least minms - the ~5% ratio compression fix
        int reps=5;
        if(minms>0 && tdef>0) reps=(int)std::min(2000.0,std::max(5.0,ceil(minms/tdef)));

        double rb_best=-1; int32_t bi=0; rocblas_int nsol=0;
        rocblas_gemm_ex_get_solutions(rb,opB,opA,N,M,(rocblas_int)K,&f1,vB[0],ty,ldb,vA[0],ty,lda,beta,
            vC[0],rocblas_datatype_f32_r,N,vC[0],rocblas_datatype_f32_r,N,rocblas_datatype_f32_r,
            rocblas_gemm_algo_solution_index,0,nullptr,&nsol);
        if(budget<1e29 && tdef>0){
            double est=(nsol+ (have_ext?300:64))*tdef/1000.0;   // screening dominates; both libraries
            if(est>budget){ ++nskip;
                printf("  %dx%dx%ld %s %c%c: SKIP est %.0fs > budget %.0fs\n",M,N,K,dn,ca,cb,est,budget);
                fflush(stdout); freeall(); continue; } }
        if(nsol>0){
            std::vector<rocblas_int> sv(nsol);
            rocblas_gemm_ex_get_solutions(rb,opB,opA,N,M,(rocblas_int)K,&f1,vB[0],ty,ldb,vA[0],ty,lda,beta,
                vC[0],rocblas_datatype_f32_r,N,vC[0],rocblas_datatype_f32_r,N,rocblas_datatype_f32_r,
                rocblas_gemm_algo_solution_index,0,sv.data(),&nsol);
            for(int w=0;w<5;++w) rrun(rocblas_gemm_algo_standard,0); HC(hipDeviceSynchronize());
            std::vector<std::pair<double,rocblas_int>> scr;
            for(int i=0;i<nsol;++i){ if(rrun(rocblas_gemm_algo_solution_index,sv[i])!=rocblas_status_success) continue;
                scr.push_back({rtm(rocblas_gemm_algo_solution_index,sv[i],1),sv[i]}); }
            std::sort(scr.begin(),scr.end());
            double b2=1e30;
            for(size_t j=0;j<scr.size()&&j<10;++j){
                for(int w=0;w<3;++w) rrun(rocblas_gemm_algo_solution_index,scr[j].second); HC(hipDeviceSynchronize());
                std::vector<double> v; for(int r=0;r<5;++r) v.push_back(rtm(rocblas_gemm_algo_solution_index,scr[j].second,reps));
                std::sort(v.begin(),v.end()); if(v[2]<b2){ b2=v[2]; bi=scr[j].second; } }
            // index 0 is a legal rocBLAS argument meaning "pick for me", so falling through on an
            // all-failed screen would report the default again and label it the tuned result
            if(b2<1e29) rb_best=b2;
        }

        // ---- hipBLASLt side, identical col-major problem ---------------------------------------
        hipDataType dt = d==0?HIP_R_32F : d==1?HIP_R_16BF : HIP_R_16F;
        hipblasLtMatmulDesc_t desc=nullptr;
        hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F);
        hipblasOperation_t oT=HIPBLAS_OP_T, oN=HIPBLAS_OP_N;
        hipblasOperation_t ltA = tb?oT:oN, ltB = ta?oT:oN;     // swapped, like the rocBLAS call
        hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ltA,sizeof(ltA));
        hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&ltB,sizeof(ltB));
        hipblasLtMatrixLayout_t lB=nullptr,lA=nullptr,lC=nullptr;
        hipblasLtMatrixLayoutCreate(&lB,dt, tb?(int64_t)K:(int64_t)N, tb?(int64_t)N:(int64_t)K, tb?(int64_t)K:(int64_t)N);
        hipblasLtMatrixLayoutCreate(&lA,dt, ta?(int64_t)M:(int64_t)K, ta?(int64_t)K:(int64_t)M, ta?(int64_t)M:(int64_t)K);
        hipblasLtMatrixLayoutCreate(&lC,HIP_R_32F,(int64_t)N,(int64_t)M,(int64_t)N);

        // arm 3/4 pool: the heuristic shortlist, asked for `heurN` so the cap is visible in n_lt_heur
        std::vector<hipblasLtMatmulHeuristicResult_t> hr(heurN); int got=0;
        hipblasLtMatmulAlgoGetHeuristic(lt,desc,lB,lA,lC,lC,pref,heurN,hr.data(),&got);

        // arm 5 pool: every algo the library has for this type signature, filtered to the ones that
        // actually support THIS problem. This is the recipe the CUDA harness uses, so the two
        // vendors are finally measured against pools of the same kind.
        std::vector<hipblasLtMatmulHeuristicResult_t> all;
        int n_all=0;
#ifdef HAVE_LT_EXT
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> raw;
            if(hipblaslt_ext::getAllAlgos(lt, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                   ltA, ltB, dt, dt, HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F, raw)==HIPBLAS_STATUS_SUCCESS){
                for(size_t i=0;i<raw.size() && (int)all.size()<maxalgo;++i){
                    size_t need=0;
                    if(hipblaslt_ext::matmulIsAlgoSupported(lt,desc,&f1,lB,lA,beta,lC,lC,raw[i].algo,need)
                       ==HIPBLAS_STATUS_SUCCESS && need<=wsBytes) all.push_back(raw[i]);
                }
                n_all=(int)all.size();
            }
        }
#endif
        auto lrun=[&](hipblasLtMatmulAlgo_t* alg){
            int i=rr++ % nrot;
            return hipblasLtMatmul(lt,desc,&f1,vB[i],lB,vA[i],lA,beta,vC[i],lC,vC[i],lC,alg,ws,wsBytes,0); };
        auto ltm=[&](hipblasLtMatmulAlgo_t* alg,int r)->double{
            if(lrun(alg)!=HIPBLAS_STATUS_SUCCESS) return 1e30;
            HCD(hipDeviceSynchronize()); HCD(hipEventRecord(e0));
            for(int i=0;i<r;++i) lrun(alg);
            HCD(hipEventRecord(e1)); HCD(hipEventSynchronize(e1));
            float ms=0; HCD(hipEventElapsedTime(&ms,e0,e1)); return ms/r; };

        // screen a pool once each, refine the top 10 - same two-stage rule as the rocBLAS arm, so a
        // difference between the arms cannot come from a difference in search effort
        auto refine=[&](std::vector<hipblasLtMatmulHeuristicResult_t>& pool,
                        hipblasLtMatmulAlgo_t* winner)->double{
            std::vector<std::pair<double,int>> s;
            for(size_t i=0;i<pool.size();++i){
                if(lrun(&pool[i].algo)!=HIPBLAS_STATUS_SUCCESS) continue;
                HCD(hipDeviceSynchronize()); s.push_back({ltm(&pool[i].algo,1),(int)i}); }
            if(s.empty()) return -1;
            std::sort(s.begin(),s.end());
            double b=1e30; int bidx=-1;
            for(size_t j=0;j<s.size()&&j<10;++j){
                for(int w=0;w<3;++w) lrun(&pool[s[j].second].algo); HCD(hipDeviceSynchronize());
                std::vector<double> v; for(int r=0;r<5;++r) v.push_back(ltm(&pool[s[j].second].algo,reps));
                std::sort(v.begin(),v.end()); if(v[2]<b){ b=v[2]; bidx=s[j].second; } }
            if(bidx<0||b>1e29) return -1;
            if(winner) *winner=pool[bidx].algo;
            return b; };

        double lt_h0=-1, lt_besth=-1, lt_all=-1;
        hipblasLtMatmulAlgo_t algH{}, algA{}; bool okH=false, okA=false;
        if(got>0){ hr.resize(got); lt_besth=refine(hr,&algH); okH=(lt_besth>0); }
        if(n_all>0){ lt_all=refine(all,&algA); okA=(lt_all>0); }
        bool ok0 = (got>0) && (lrun(&hr[0].algo)==HIPBLAS_STATUS_SUCCESS);

        // ---- final head-to-head: warm everything, then alternate, median of 9 -------------------
        // The searches above leave the device hot at different points of the ramp, so screening
        // times are used only to pick indices and never reported. Correction #1 on the issue was
        // exactly this bug, measured once already.
        {
            for(int w=0;w<4;++w){
                rrun(rocblas_gemm_algo_standard,0);
                if(rb_best>0) rrun(rocblas_gemm_algo_solution_index,bi);
                if(ok0) lrun(&hr[0].algo);
                if(okH) lrun(&algH);
                if(okA) lrun(&algA); }
            HC(hipDeviceSynchronize());
            std::vector<double> vd,vb,v0,vH,vAl;
            for(int r=0;r<9;++r){
                vd.push_back(rtm(rocblas_gemm_algo_standard,0,reps));
                if(rb_best>0) vb.push_back(rtm(rocblas_gemm_algo_solution_index,bi,reps));
                if(ok0) v0.push_back(ltm(&hr[0].algo,reps));
                if(okH) vH.push_back(ltm(&algH,reps));
                if(okA) vAl.push_back(ltm(&algA,reps)); }
            auto med=[&](std::vector<double>& v)->double{
                if(v.empty()) return -1; std::sort(v.begin(),v.end()); return v[v.size()/2]; };
            tdef=med(vd); rb_best=med(vb); lt_h0=med(v0); lt_besth=med(vH); lt_all=med(vAl);
        }

        double clk_b=ref_ms(), drift=(clk_a>0)?clk_b/clk_a:1.0;
        double flop=2.0*M*N*(double)K;
        auto TF=[&](double ms){ return ms>0? flop/(ms*1e-3)/1e12 : -1.0; };
        fprintf(fo,"%d,%d,%ld,%s,%c,%c,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d,%d,%.4f\n",
                M,N,K,dn,ca,cb,beta_i,tdef,rb_best,lt_h0,lt_besth,lt_all,
                TF(tdef),TF(rb_best),TF(lt_h0),TF(lt_besth),TF(lt_all),
                (int)nsol,got,n_all,(int)(okH?1:0)+(int)(okA?2:0),nrot,reps,drift);
        fflush(fo);
        printf("%6dx%-6dx%-6ld %-5s %c%c | rb %7.2f/%7.2f  lt %7.2f/%7.2f/%7.2f TF | sol %3d heur %3d all %4d reps %4d%s\n",
               M,N,K,dn,ca,cb,TF(tdef),TF(rb_best),TF(lt_h0),TF(lt_besth),TF(lt_all),
               (int)nsol,got,n_all,reps,(drift>1.03||drift<0.97)?"  CLKDRIFT":""); fflush(stdout);

        hipblasLtMatrixLayoutDestroy(lB); hipblasLtMatrixLayoutDestroy(lA); hipblasLtMatrixLayoutDestroy(lC);
        hipblasLtMatmulDescDestroy(desc);
        freeall();
    }
    fclose(fi); fclose(fo);
    printf("three_way2 done, %ld shapes skipped by budget\n",nskip);
    return 0;
}
