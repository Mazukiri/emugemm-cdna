// mfma_peak2.cpp - true matrix-core rate for EVERY precision that matters, measured with one instrument.
// Fixes two gaps in mfma_peak.cpp: (1) it never measured fp32-matrix (the DENOMINATOR of every economic
// claim in this project) or bf16 (the numerator of the robust variant); (2) it timed kernels back-to-back
// in a fixed order, so thermal drift biased whichever ran last.
// Here: all kernels interleaved round-robin, median of REPS, so drift hits every kernel equally.
//
// Decisive question: is bf16 == fp16 in HARDWARE on gfx90a? If yes, rocBLAS's slow bf16 is a LIBRARY gap
// (fixable by hipBLASLt) and the robust bf16 emulator deserves a rematch. If no, bf16 is dead on CDNA2.
//
// Build: hipcc -O3 --offload-arch=gfx90a mfma_peak2.cpp -o mfma_peak2 && ./mfma_peak2
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define HIP_CHECK(c) do{hipError_t _hce=(c); if(_hce){printf("HIP err %s @%d\n",hipGetErrorString(_hce),__LINE__);exit(1);}}while(0)

typedef float    f32x4 __attribute__((ext_vector_type(4)));
typedef double   f64x4 __attribute__((ext_vector_type(4)));
typedef int      i32x4 __attribute__((ext_vector_type(4)));
typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
typedef short    s16x4 __attribute__((ext_vector_type(4)));   // bf16 pairs are passed as i16 vectors
typedef short    s16x2 __attribute__((ext_vector_type(2)));

constexpr int NACC = 8, LOOP = 20000, BLOCKS = 104*8, THREADS = 256;
constexpr int REPS = 7;

#define BODY(MFMA, AVAL, BVAL, ACCT)                                    \
    ACCT acc[NACC];                                                     \
    _Pragma("unroll") for(int i=0;i<NACC;i++) acc[i]=ACCT{0,0,0,0};     \
    for(int k=0;k<LOOP;k++)                                             \
        _Pragma("unroll") for(int i=0;i<NACC;i++) acc[i]=MFMA(AVAL,BVAL,acc[i],0,0,0);

__global__ void peak_f32(float* out){
    float a=1.f,b=1.f;
    BODY(__builtin_amdgcn_mfma_f32_16x16x4f32, a, b, f32x4)
    float s=0; for(int i=0;i<NACC;i++) s+=acc[i][0];
    if(s<-1) out[threadIdx.x]=s;
}
// fp64 matrix core. If this comes out ~= fp32 matrix on CDNA2, then EVERY fp64-emulation scheme
// (from fp32, fp16 or int8) needs >=3 slices at <=1x the rate -> strictly worse than native fp64.
// That closes the whole fp64 direction with a measurement instead of a spec sheet.
__global__ void peak_f64(double* out){
    double a=1.0,b=1.0;
    BODY(__builtin_amdgcn_mfma_f64_16x16x4f64, a, b, f64x4)
    double s=0; for(int i=0;i<NACC;i++) s+=acc[i][0];
    if(s<-1) out[threadIdx.x]=s;
}
__global__ void peak_f16(float* out){
    f16x4 a={1,1,1,1}, b={1,1,1,1};
    BODY(__builtin_amdgcn_mfma_f32_16x16x16f16, a, b, f32x4)
    float s=0; for(int i=0;i<NACC;i++) s+=acc[i][0];
    if(s<-1) out[threadIdx.x]=s;
}
__global__ void peak_bf16_1k(float* out){
    s16x4 a={0x3f80,0x3f80,0x3f80,0x3f80}, b={0x3f80,0x3f80,0x3f80,0x3f80}; // bf16 1.0
    BODY(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k, a, b, f32x4)
    float s=0; for(int i=0;i<NACC;i++) s+=acc[i][0];
    if(s<-1) out[threadIdx.x]=s;
}
__global__ void peak_bf16_old(float* out){
    s16x2 a={0x3f80,0x3f80}, b={0x3f80,0x3f80};
    BODY(__builtin_amdgcn_mfma_f32_16x16x8bf16, a, b, f32x4)
    float s=0; for(int i=0;i<NACC;i++) s+=acc[i][0];
    if(s<-1) out[threadIdx.x]=s;
}
__global__ void peak_i8(int* out){
    int a=0x01010101, b=0x01010101;
    BODY(__builtin_amdgcn_mfma_i32_16x16x16i8, a, b, i32x4)
    int s=0; for(int i=0;i<NACC;i++) s+=acc[i][0];
    if(s<-1) out[threadIdx.x]=s;
}

enum { K_F64=0, K_F32, K_F16, K_BF16_1K, K_BF16_OLD, K_I8, NKERN };
static const char* NAMES[NKERN] = {
    "fp64   mfma_16x16x4f64", "fp32   mfma_16x16x4f32", "fp16   mfma_16x16x16f16",
    "bf16   mfma_16x16x16bf16_1k", "bf16   mfma_16x16x8bf16 (legacy)", "int8   mfma_16x16x16i8" };
// MACs per mfma per wave: M*N*K ; flops = 2*MACs
static const double FPM[NKERN] = { 2.0*16*16*4, 2.0*16*16*4, 2.0*16*16*16, 2.0*16*16*16, 2.0*16*16*8, 2.0*16*16*16 };

static void launch(int k, void* d){
    switch(k){
        case K_F64:      peak_f64    <<<BLOCKS,THREADS>>>((double*)d); break;
        case K_F32:      peak_f32    <<<BLOCKS,THREADS>>>((float*)d); break;
        case K_F16:      peak_f16    <<<BLOCKS,THREADS>>>((float*)d); break;
        case K_BF16_1K:  peak_bf16_1k<<<BLOCKS,THREADS>>>((float*)d); break;
        case K_BF16_OLD: peak_bf16_old<<<BLOCKS,THREADS>>>((float*)d); break;
        case K_I8:       peak_i8     <<<BLOCKS,THREADS>>>((int*)d);   break;
    }
}

int main(){
    void* d; HIP_CHECK(hipMalloc(&d, THREADS*8));   // 8 B/lane: fp64 kernel writes doubles
    hipEvent_t s,e; HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&e));
    const long long waves = (long long)BLOCKS*(THREADS/64);

    for(int k=0;k<NKERN;++k){ launch(k,d); }            // warm up all of them first
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::vector<double>> ms(NKERN);
    for(int r=0;r<REPS;++r)                             // INTERLEAVED: no kernel owns the cold or hot slot
        for(int k=0;k<NKERN;++k){
            HIP_CHECK(hipEventRecord(s)); launch(k,d); HIP_CHECK(hipEventRecord(e));
            HIP_CHECK(hipEventSynchronize(e));
            float t=0; HIP_CHECK(hipEventElapsedTime(&t,s,e)); ms[k].push_back(t);
        }

    double rate[NKERN], med[NKERN], spread[NKERN];
    for(int k=0;k<NKERN;++k){                       // compute ALL rates before printing any ratio,
        std::sort(ms[k].begin(), ms[k].end());      // otherwise row 0 divides by an unfilled rate[K_F32]
        med[k] = ms[k][REPS/2];
        spread[k] = 100.0*(ms[k].back()-ms[k].front())/med[k];
        rate[k] = (double)waves*NACC*LOOP*FPM[k]/(med[k]*1e-3)/1e12;
    }
    printf("gfx90a matrix-core peak (NACC=%d, LOOP=%d, %d blocks x %d thr, median of %d interleaved reps)\n\n",
           NACC, LOOP, BLOCKS, THREADS, REPS);
    printf("  %-34s %10s %12s %10s %10s\n","instruction","median ms","spread %","TFLOP/s","vs fp32");
    for(int k=0;k<NKERN;++k)
        printf("  %-34s %10.3f %11.1f%% %10.1f %9.2fx\n", NAMES[k], med[k], spread[k], rate[k], rate[k]/rate[K_F32]);

    printf("\n  --- the numbers this project actually runs on ---\n");
    printf("  bf16_1k / fp16          = %.3f   %s\n", rate[K_BF16_1K]/rate[K_F16],
           rate[K_BF16_1K] > 0.95*rate[K_F16] ? "bf16 == fp16 in HW -> rocBLAS's slow bf16 is a LIBRARY gap"
                                              : "bf16 < fp16 in HW  -> hardware limit, bf16 emu is capped here");
    printf("  bf16_1k / bf16_legacy   = %.3f   %s\n", rate[K_BF16_1K]/rate[K_BF16_OLD],
           rate[K_BF16_1K] > 1.5*rate[K_BF16_OLD] ? "legacy is ~half rate: a library on the wrong opcode pays 2x" : "both similar");
    printf("  int8 / fp16             = %.3f\n", rate[K_I8]/rate[K_F16]);
    printf("\n  Break-even product counts (S < R to beat native fp32):\n");
    printf("    fp16 : R = %.2f  -> 3 products = %.2fx ceiling\n", rate[K_F16]/rate[K_F32], rate[K_F16]/rate[K_F32]/3.0);
    printf("    bf16 : R = %.2f  -> 3 products = %.2fx | 6 products = %.2fx | 8 products = %.2fx\n",
           rate[K_BF16_1K]/rate[K_F32], rate[K_BF16_1K]/rate[K_F32]/3.0,
           rate[K_BF16_1K]/rate[K_F32]/6.0, rate[K_BF16_1K]/rate[K_F32]/8.0);

    printf("\n  --- can we emulate FP64 on this chip? (denominator = native fp64) ---\n");
    double r32=rate[K_F32]/rate[K_F64], r16=rate[K_F16]/rate[K_F64], ri8=rate[K_I8]/rate[K_F64];
    printf("    fp32/fp64 = %.2f   fp16/fp64 = %.2f   int8/fp64 = %.2f\n", r32, r16, ri8);
    printf("    fp64-from-fp32 needs >=3 slices -> %.2fx   %s\n", r32/3.0, r32/3.0>1.0?"viable":"LOSES");
    printf("    fp64-from-fp16 needs ~7 slices  -> %.2fx   %s\n", r16/7.0, r16/7.0>1.0?"viable":"LOSES");
    printf("    fp64-from-int8 (Ozaki) ~9 moduli-> %.2fx   %s\n", ri8/9.0, ri8/9.0>1.0?"viable":"LOSES");
    return 0;
}
