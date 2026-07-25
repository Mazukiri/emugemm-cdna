// emugemm.h - emulated-precision SGEMM for AMD CDNA, with measurement-driven auto-dispatch.
//
// Round 7-9 measured, on MI250 (gfx90a), every number this dispatcher relies on. The unified error model
// that came out of Round 9 Phase 1 is:
//
//     err(scheme, K, c) = sqrt( floor_scheme^2 + a_scheme^2 * (K/c) )
//
// where c is the number of K-chunks summed in fp64 (c=1 = ordinary GEMM), and error depends only on the
// accumulation chain length K/c -- verified to 4 significant figures across K=65536 and K=262144.
//
//   scheme       floor       a          speed vs native fp32   robust over full fp32 exponent range?
//   fp32         0           1.79e-8    1.00x                  yes
//   bf16x3       4.4355e-6   8.95e-9    1.20-1.22x             YES
//   fp16x3       3.6e-7      8.95e-9    ~1.19x                 no (needs row/col scaling; see notes)
//   bf16x6       ~0          8.95e-9    0.57x                  yes -- but DOMINATED, see below
//
// Two consequences that drive the whole dispatcher:
//
//  1. bf16x3 has a HARD FLOOR at 4.4355e-6. No amount of time buys accuracy below it. Any target under
//     that must use fp32 with chunked-fp64 accumulation instead. This is a correctness constraint, not a
//     preference.
//  2. bf16x6 is strictly dominated and is never selected. It costs 1.75x native fp32 time to reach the
//     same error that fp32 with c=4 reaches at ~1.0x. (Measured K=262144: bf16x6 4.637e-6 @ ~594 ms vs
//     fp32 c=4 4.579e-6 @ 334 ms.) It is kept in the enum only so benchmarks can still request it.
//
// All bf16/fp16 paths share a = 8.95e-9, exactly half of the default fp32 kernel's 1.79e-8, because the
// low-precision Tensile kernels use a better accumulation structure than rocBLAS's default fp32 kernel.
// That is a library artifact, not a property of the number formats -- do not describe it as one.
#pragma once
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EMU_SCHEME_AUTO = 0,
    EMU_SCHEME_NATIVE_FP32,
    EMU_SCHEME_FP32_CHUNKED,
    EMU_SCHEME_BF16X3,
    EMU_SCHEME_FP16X3,
    EMU_SCHEME_BF16X6       /* dominated; benchmarks only */
} emu_scheme_t;

typedef struct {
    double max_rel_err;   /* required normwise Frobenius relative error, e.g. 1e-5 */
    int    force_scheme;  /* emu_scheme_t; EMU_SCHEME_AUTO to let the dispatcher choose */
} emu_request_t;

typedef struct {
    float  a_absmax, b_absmax;  /* < 0 means "unknown, please scan" (an O(MK+KN) pass) */
    int    a_is_reused;         /* A is a reused weight matrix: its split may be cached */
    /* Cancellation ratio rho = || |A||B| ||_F / || AB ||_F. < 0 means "unknown, please estimate".
       Round 10 established that rho and K are ORTHOGONAL axes:
         err(scheme,K,c,rho)  =  rho/rho_ref(K) * err_iid(scheme,K,c),   rho_ref(K) = 0.64*sqrt(K)
       rho scales EVERY scheme's error by the same factor, so it does NOT decide which scheme wins --
       K does. rho decides whether the winner actually meets the requested accuracy. Two different jobs.
       Estimated by Hutchinson with `rho_probes` Rademacher vectors: two skinny GEMMs per operand pair,
       cost 2*P*K*(M+N) versus 2*M*N*K, i.e. ~2P/M. P=64 gives ~10% accuracy, P=256 gives ~1%. */
    float  rho;
    int    rho_probes;          /* 0 -> default 64 */
} emu_hints_t;

typedef struct {
    emu_scheme_t chosen;
    int          chunks;          /* c actually used */
    double       predicted_err;   /* from the model above */
    double       predicted_speedup;
    const char*  reason;
} emu_plan_t;

/* Load the offline per-shape solution table produced by gen_tune_table.cpp.
   Solution indices are NOT portable across ROCm versions or GPU architectures (rocBLAS docs), so the
   arch string in the file is checked against the live device and the table is REJECTED on mismatch --
   a stale table silently selecting wrong kernels would be worse than no table at all. */
rocblas_status emugemm_init(const char* tune_csv_path);
void           emugemm_shutdown(void);

/* Decide only -- no GPU work. Exposed so the choice can be inspected and tested. */
emu_plan_t emugemm_plan(int M, int N, long K, const emu_request_t* req, const emu_hints_t* hints);

/* Row-major C = A*B, A is MxK (lda=K), B is KxN (ldb=N), C is MxN (ldc=N).
   Invariants this function must never violate:
     * never slower than a plain rocblas_sgemm on the same shape (falls back if the model says so)
     * never returns a result whose predicted error exceeds req->max_rel_err */
rocblas_status emugemm_sgemm(rocblas_handle handle,
                             int M, int N, long K,
                             const float* A, const float* B, float* C,
                             const emu_request_t* req, emu_hints_t* hints,
                             emu_plan_t* out_plan);

/* Model helpers, exported for the error-model validation in Phase 5. */
double emugemm_model_err(emu_scheme_t s, long K, int chunks);          /* iid data (rho = rho_ref) */
double emugemm_model_err_rho(emu_scheme_t s, long K, int chunks, double rho);
/* Estimate rho on device. Returns < 0 on failure. O((MK+KN)*P), not O(MNK). */
double emugemm_estimate_rho(rocblas_handle h, int M, int N, long K,
                            const float* A, const float* B, int probes);
double emugemm_model_speed(emu_scheme_t s, int chunks);

#ifdef __cplusplus
}
#endif
