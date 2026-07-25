// gen_tune_table.cpp - Round 9 Phase 2: offline per-shape solution tuning table.
//
// Round 8 measured that rocBLAS's DEFAULT bf16 kernel choice wastes 1.17x-1.71x depending on shape
// (83.9 vs 137.8 TF at M=8192 N=28672 K=8192). rocblas_gemm_ex_get_solutions can find the good one, but
// enumerating hundreds of solutions costs minutes, so it must be done OFFLINE and looked up at runtime.
//
// Design notes:
//  * Two-stage tuning: time every solution ONCE, keep the top 16, then re-time those properly. Cuts the
//    cost ~3x versus timing all solutions 3x, with the same winner in practice.
//  * Resumable: results are appended to a CSV and already-done (M,N,K,dtype) rows are skipped on restart.
//    A multi-hour job that cannot resume is a multi-hour job you will lose.
//  * Shardable: --shard i/n processes only shapes with index%n==i, so 8 GCDs can split the grid.
//  * Per-shape time budget: a single 32768x32768x65536 GEMM is ~4.7 s, so tuning it would take >1 h alone.
//    Shapes whose estimated stage-1 cost exceeds the budget are SKIPPED and logged, not silently dropped.
//  * Solution indices are NOT portable across ROCm versions or architectures (rocBLAS docs), so the arch
//    and ROCm version are written into the file and must be checked at load time.
//
// Build: hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a gen_tune_table.cpp -o gen_tune_table -lrocblas
// Run:   ./gen_tune_table out.csv [--shard i/n] [--budget SECONDS]
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <set>
#include <vector>
#include <algorithm>

#define HIP_CHECK(c) do{hipError_t _he=(c); if(_he){fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(_he),__LINE__);exit(1);}}while(0)
typedef __hip_bfloat16 bf16;

__device__ __forceinline__ unsigned hashu(unsigned x){
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
__global__ void gen_normal(float* X,size_t n,unsigned seed){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    unsigned a=hashu((unsigned)i^seed), b=hashu((unsigned)(i>>32)^(a+0x9e3779b9u));
    float u1=(a>>8)*(1.0f/16777216.0f),u2=(b>>8)*(1.0f/16777216.0f);
    if(u1<1e-7f)u1=1e-7f; X[i]=sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); }
__global__ void to_bf16(const float* X,bf16* Y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)Y[i]=(bf16)X[i]; }
__global__ void to_fp16(const float* X,_Float16* Y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)Y[i]=(_Float16)X[i]; }

struct Shape { int M,N; long K; };

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s out.csv [--shard i/n] [--budget SEC]\n",argv[0]); return 1; }
    const char* out=argv[1];
    int shard=0,nshard=1; double budget=120.0;
    for(int i=2;i<argc;++i){
        if(!strcmp(argv[i],"--shard")&&i+1<argc) sscanf(argv[++i],"%d/%d",&shard,&nshard);
        else if(!strcmp(argv[i],"--budget")&&i+1<argc) budget=atof(argv[++i]);
    }

    // ---- what is already done (resume) ----
    std::set<std::string> done;
    if(FILE* f=fopen(out,"r")){ char line[512];
        while(fgets(line,sizeof line,f)){ int M,N; long K; char dt[16];
            if(sscanf(line,"%*[^,],%*[^,],%d,%d,%ld,%15[^,]",&M,&N,&K,dt)==4){
                char key[128]; snprintf(key,sizeof key,"%d,%d,%ld,%s",M,N,K,dt); done.insert(key); } }
        fclose(f); }
    bool fresh = done.empty();

    // ---- shape grid: powers of two plus real LLM / RandNLA points ----
    std::vector<Shape> shapes;
    int   MNs[] = {1024,2048,4096,8192,16384,32768};
    long  Ks[]  = {512,1024,2048,4096,8192,16384,32768,65536};
    for(int m : MNs) for(int n : MNs) for(long k : Ks) shapes.push_back({m,n,k});
    // shapes this project has actually benchmarked, so the table covers them exactly
    shapes.push_back({8192,28672,8192}); shapes.push_back({16384,8192,8192});
    shapes.push_back({12288,12288,12288}); shapes.push_back({4096,4096,131072});
    shapes.push_back({4096,4096,262144}); shapes.push_back({28672,8192,8192});

    rocblas_handle rb; rocblas_create_handle(&rb); rocblas_set_pointer_mode(rb,rocblas_pointer_mode_host);
    char arch[64]="gfx90a"; { hipDeviceProp_t p; if(hipGetDeviceProperties(&p,0)==hipSuccess)
        snprintf(arch,sizeof arch,"%s",p.gcnArchName); }
    int rmaj=0,rmin=0,rpat=0; rocblas_get_version_string_size(nullptr);
    char rocv[64]; { size_t sz=64; rocblas_get_version_string(rocv,sz); }
    (void)rmaj;(void)rmin;(void)rpat;

    FILE* fo=fopen(out,"a");
    if(fresh) fprintf(fo,"arch,rocm,M,N,K,dtype,best_sol,best_ms,default_ms,gain\n");
    fflush(fo);
    printf("shard %d/%d  budget %.0fs/shape  arch=%s rocblas=%s  %zu shapes, %zu already done\n",
           shard,nshard,budget,arch,rocv,shapes.size(),done.size());

    const float f1=1.f,f0=0.f;
    hipEvent_t s,e; HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&e));

    for(size_t si=0; si<shapes.size(); ++si){
        if((int)(si%nshard)!=shard) continue;
        Shape sp=shapes[si];
        int M=sp.M,N=sp.N; long K=sp.K;
        size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N;
        double bytes=(na+nb)*4.0+nc*4.0+(na+nb)*2.0;
        if(bytes>55e9){ printf("  skip %dx%dx%ld: %.1f GB > 55 GB\n",M,N,K,bytes/1e9); continue; }

        float *fA,*fB,*dC; bf16 *bA,*bB; _Float16 *hA,*hB;
        if(hipMalloc(&fA,na*4)!=hipSuccess){ printf("  skip %dx%dx%ld: alloc\n",M,N,K); continue; }
        HIP_CHECK(hipMalloc(&fB,nb*4)); HIP_CHECK(hipMalloc(&dC,nc*4));
        HIP_CHECK(hipMalloc(&bA,na*2)); HIP_CHECK(hipMalloc(&bB,nb*2));
        HIP_CHECK(hipMalloc(&hA,na*2)); HIP_CHECK(hipMalloc(&hB,nb*2));
        int t=256;
        gen_normal<<<(na+t-1)/t,t>>>(fA,na,12345u); gen_normal<<<(nb+t-1)/t,t>>>(fB,nb,67890u);
        to_bf16<<<(na+t-1)/t,t>>>(fA,bA,na); to_bf16<<<(nb+t-1)/t,t>>>(fB,bB,nb);
        to_fp16<<<(na+t-1)/t,t>>>(fA,hA,na); to_fp16<<<(nb+t-1)/t,t>>>(fB,hB,nb);
        HIP_CHECK(hipDeviceSynchronize());

        // fp16 was missing from the first table build, and the invariant test caught the consequence:
        // emugemm picked the fp16 path on a skinny shape and lost to native, because the fp16 path was
        // running rocBLAS's untuned default -- exactly the 2-4x collapse this table exists to fix.
        for(int d=0; d<3; ++d){
            const char* dname = (d==0)?"fp32":(d==1)?"bf16":"fp16";
            char key[128]; snprintf(key,sizeof key,"%d,%d,%ld,%s",M,N,K,dname);
            if(done.count(key)) continue;
            rocblas_datatype ty = (d==0)?rocblas_datatype_f32_r:(d==1)?rocblas_datatype_bf16_r:rocblas_datatype_f16_r;
            const void *pA = (d==0)?(void*)fA:(d==1)?(void*)bA:(void*)hA;
            const void *pB = (d==0)?(void*)fB:(d==1)?(void*)bB:(void*)hB;

            auto run=[&](rocblas_gemm_algo al,int32_t so){
                return rocblas_gemm_ex(rb,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
                    &f1,pB,ty,N,pA,ty,(rocblas_int)K,&f0,dC,rocblas_datatype_f32_r,N,dC,rocblas_datatype_f32_r,N,
                    rocblas_datatype_f32_r,al,so,0); };
            auto time1=[&](rocblas_gemm_algo al,int32_t so,int reps)->double{
                if(run(al,so)!=rocblas_status_success) return 1e30;
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipEventRecord(s)); for(int i=0;i<reps;++i) run(al,so);
                HIP_CHECK(hipEventRecord(e)); HIP_CHECK(hipEventSynchronize(e));
                float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,s,e)); return ms/reps; };

            // Warm the default AND the candidates before timing anything: an earlier version timed the
            // default first on a cold device and the winners last on a warm one, inflating gain by ~2.4%
            // on average and up to 23% on individual shapes. See bench/audit_table.cpp.
            for(int w=0;w<4;++w) run(rocblas_gemm_algo_standard,0);
            HIP_CHECK(hipDeviceSynchronize());
            double t_def=time1(rocblas_gemm_algo_standard,0,5);
            rocblas_int nsol=0;
            if(rocblas_gemm_ex_get_solutions(rb,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
                 &f1,pB,ty,N,pA,ty,(rocblas_int)K,&f0,dC,rocblas_datatype_f32_r,N,dC,rocblas_datatype_f32_r,N,
                 rocblas_datatype_f32_r,rocblas_gemm_algo_solution_index,0,nullptr,&nsol)!=rocblas_status_success
               || nsol<=0){ printf("  %dx%dx%ld %s: no solutions\n",M,N,K,dname); continue; }
            // budget guard: stage 1 costs about nsol single runs
            double est = nsol*t_def/1000.0;
            if(est>budget){ printf("  skip %dx%dx%ld %s: est %.0fs > budget %.0fs (%d sols x %.1f ms)\n",
                                   M,N,K,dname,est,budget,(int)nsol,t_def); fflush(stdout); continue; }
            std::vector<rocblas_int> sv(nsol);
            rocblas_gemm_ex_get_solutions(rb,rocblas_operation_none,rocblas_operation_none,N,M,(rocblas_int)K,
                 &f1,pB,ty,N,pA,ty,(rocblas_int)K,&f0,dC,rocblas_datatype_f32_r,N,dC,rocblas_datatype_f32_r,N,
                 rocblas_datatype_f32_r,rocblas_gemm_algo_solution_index,0,sv.data(),&nsol);

            // stage 1: one run each
            std::vector<std::pair<double,int32_t>> r1; r1.reserve(nsol);
            for(int i=0;i<nsol;++i){ double ms=time1(rocblas_gemm_algo_solution_index,sv[i],1);
                if(ms<1e29) r1.push_back({ms,sv[i]}); }
            if(r1.empty()){ printf("  %dx%dx%ld %s: none ran\n",M,N,K,dname); continue; }
            std::sort(r1.begin(),r1.end());
            // stage 2: re-time the survivors properly
            double bt=1e30; int32_t bi=r1[0].second;
            for(size_t i=0;i<r1.size() && i<16;++i){
                double ms=time1(rocblas_gemm_algo_solution_index,r1[i].second,5);
                if(ms<bt){ bt=ms; bi=r1[i].second; } }

            fprintf(fo,"%s,%s,%d,%d,%ld,%s,%d,%.5f,%.5f,%.4f\n",arch,rocv,M,N,K,dname,(int)bi,bt,t_def,t_def/bt);
            fflush(fo);
            printf("  %6d %6d %7ld %5s: %4d sols  best %.3f ms  default %.3f ms  gain %.3fx\n",
                   M,N,K,dname,(int)nsol,bt,t_def,t_def/bt); fflush(stdout);
        }
        hipFree(fA);hipFree(fB);hipFree(dC);hipFree(bA);hipFree(bB);hipFree(hA);hipFree(hB);
    }
    fclose(fo);
    printf("shard %d/%d done\n",shard,nshard);
    return 0;
}
