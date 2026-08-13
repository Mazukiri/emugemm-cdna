# Round 9 (2026-07-25) — Phase 1: a refuted hypothesis, and a correction to Round 8

## ⚠ Important correction to `RESULTS_ROUND8.md`

Round 8's headline conclusion — *"beyond K ≈ 76 500, bf16×3 is more accurate than native fp32"* —
holds **only against rocBLAS's default fp32 GEMM**. It does **not** hold when fp32 is given a better
accumulation strategy. Details below.

---

## Experiment (`flat_error.cpp`)

**Hypothesis going in:** bf16×3's error is nearly flat in K because it is bounded by **representation**
error (two bf16 slices ≈ 2⁻¹⁶), but not perfectly flat because it also accumulates in fp32. Splitting
K into `c` chunks and **summing the chunks in fp64** should remove the accumulation term, hold 4.45e-6
at every K, and turn the crossover into a clean statement.

**Applied to BOTH sides** — the rule carried over from Round 8: tuning only one side is self-deception.
M=N=4096, fp64 reference on the GPU, final result rounded back to fp32 so it remains a real SGEMM.

### K = 262144

| c | fp32 err | ms | bf16×3 err | ms |
|---|---|---|---|---|
| 1 | 9.1619e-06 | 338.6 | **6.4167e-06** | **285.8** |
| 4 | 4.5794e-06 | 333.9 | 4.9905e-06 | 331.0 |
| 16 | 2.2906e-06 | 643.0 | 4.5661e-06 | 504.6 |
| 64 | 1.1457e-06 | 441.3 | 4.4615e-06 | 334.1 |
| 256 | **5.7339e-07** | 601.6 | 4.4397e-06 | 479.3 |

### K = 65536

| c | fp32 err | ms | bf16×3 err | ms |
|---|---|---|---|---|
| 1 | 4.5809e-06 | 83.3 | 4.9894e-06 | 70.3 |
| 4 | 2.2898e-06 | 144.8 | 4.5660e-06 | 111.3 |
| 16 | 1.1452e-06 | 121.3 | 4.4619e-06 | 83.9 |
| 64 | 5.7338e-07 | 135.7 | 4.4397e-06 | 98.4 |
| 256 | 2.8805e-07 | 134.9 | 4.4355e-06 | 197.6 |

Re-run twice: errors agree to **five digits**, times to within 1%.

---

## Three results

### 1. The representation floor is confirmed exactly

bf16×3 converges to **4.4355e-06** as `c` grows — matching the value measured at K=2048 (4.448e-06).
The first half of the hypothesis is exactly right: **bf16×3 has a hard floor it cannot go below, no
matter how much time is spent.**

### 2. But fp32 has no floor, so the crossover disappears

fp32 inputs are already exact in fp32, so there is **no representation error to be bounded by**.
Chunking drives the error down without limit: 9.16e-06 → 5.73e-07, falling as `1/√c`. At c=256, fp32
is **7.7× more accurate** than bf16×3.

⇒ **Given the same accumulation strategy, fp32 beats bf16×3 at every accuracy level below 4.4e-6.**

Chunking only the bf16×3 side would have produced the report *"the emulation error is now flat and the
crossover holds at every K"* — a conclusion that is entirely wrong. This is precisely what the
"apply to both sides" rule exists to prevent.

### 3. A clean law: error depends only on `K/c`

| | K=65536, c=1 | K=262144, c=4 |
|---|---|---|
| `K/c` | 65536 | 65536 |
| fp32 | 4.5809e-06 | 4.5794e-06 |
| bf16×3 | 4.9894e-06 | 4.9905e-06 |

Agreement to four digits. **Error is a function of the accumulation chain length `K/c`, not of K.**
The model:

```
err_fp32(K,c)   = a·√(K/c)                          a ≈ 1.79e-8
err_bf16×3(K,c) = √( floor² + (a/2)²·(K/c) )        floor ≈ 4.4355e-6
```

The `a/2` coefficient for bf16×3 (derived from `√(4.9894e-6² − 4.4355e-6²) = 2.285e-6 ≈ 4.5809e-6 / 2`)
matches a Round 8 finding: rocBLAS's most accurate fp32 solution has **exactly half** the error of the
default. ⇒ **the bf16 kernel in use has a better accumulation structure than the default fp32 kernel.**
A substantial part of Round 8's "win" therefore comes from **the kernel's accumulation strategy, not
from the number format.**

---

## The correct statement, replacing Round 8's headline

Pareto frontier (time, error) at K=262144 — the non-dominated points:

| point | ms | err | |
|---|---|---|---|
| **bf16×3, c=1** | **285.8** | 6.42e-06 | fastest, unmatched |
| **bf16×3, c=64** | 334.1 | **4.46e-06** | best at the ~334 ms mark (fp32 c=4: 333.9 ms / 4.58e-06) |
| fp32, c=64 | 441.3 | 1.15e-06 | |
| fp32, c=256 | 601.6 | **5.73e-07** | most accurate |

- **bf16×3 owns the fast end** and the ~4.5e-6 accuracy tier: at 286 ms nothing competes, and at
  334 ms it still matches or beats chunked fp32.
- **fp32 owns everything below 4.4e-6.** bf16×3 can never go below its floor.
- **Round 8's crossover holds only against rocBLAS's default fp32.** With chunking plus fp64 reduction
  — a cheap and standard technique — fp32 catches up at ~1.17× the time and pulls ahead after that.

**What survives from Round 8:** bf16×3 is still **1.20–1.22× faster than native fp32 at the same
accuracy tier** (Phase C), robust across the exponent range, and the fastest point on the Pareto
frontier. What is lost is the unconditional claim *"more accurate than fp32"*.

---

# Phase 2 — Offline tuning table, and a second correction to Round 8

`gen_tune_table.cpp`: 294 shapes × 2 dtypes, enumerating and timing **every** rocBLAS solution in two
stages (one pass over all → keep the top 16 → re-time those five times), resumable, sharded across
8 GCDs. 48 shapes were skipped by the time-budget guard (logged, not silently dropped).

| | mean gain | max gain | default already optimal (<1.02×) |
|---|---|---|---|
| **bf16** | **1.384×** | 3.773× at 2048×32768×512 | **3%** of 287 shapes |
| **fp32** | **1.349×** | 4.395× at 1024×32768×16384 | **9%** of 253 shapes |
| **fp16** | **1.304×** | 3.570× at 2048×32768×512 | **10%** of 247 shapes |

*(fp16 was added in a second run — see the invariant tests below; its absence caused a real violation.
The resume logic meant the second run only had to compute fp16.)*
**Total: 787 rows.** Across all three types, **only 3–10% of shapes have a default solution within 2%
of optimal.**

### ⚠ Correction to Round 8 section C

Round 8 concluded *"rocBLAS's fp32 heuristic is fine, only the bf16 heuristic is broken"* — based on
**three shapes**. Over 253 shapes, **fp32 is nearly as bad as bf16**. Those three shapes happened to
be well-behaved ones.

**Solo verification** (no GPU contention, to rule out noise from running 8 shards in parallel):

| shape | dtype | default | best | solo gain | grid gain |
|---|---|---|---|---|---|
| 1024×32768×16384 | fp32 | 133.0 ms = **8.26 TF** | 29.7 ms = **37.00 TF** | **4.48×** | 4.395 |
| 2048×32768×512 | bf16 | 3.11 ms = 22.11 TF | 0.81 ms = 84.46 TF | **3.82×** | 3.773 |
| 4096×4096×16384 | fp32 | 29.77 ms = 18.47 TF | 14.89 ms = 36.92 TF | **2.00×** | |
| 4096×4096×16384 | bf16 | 5.50 ms = 99.92 TF | 4.15 ms = 132.40 TF | 1.35× | |
| 12288×12288×12288 | fp32 | 100.00 ms = 37.11 TF | 100.04 ms = 37.09 TF | **1.00×** | matches Round 8 |

Solo agrees with the grid to ~2%. **The gains are real.** The correct statement: rocBLAS's heuristic
is **good on large square shapes and collapses by 2–4.5× on skinny or large-K shapes**, for *both*
fp32 and bf16.

**A consequence running the other way:** `crossover_test` runs the **default** solution at M=N=4096
with large K — exactly the regime where the default collapses — so its TFLOP/s column (ranging
19–35 TF) reflects kernel choice, not the scheme. That column is marked not-for-quotation. The
**accuracy** results are unaffected: Round 8 verified that the default solution's error matches the
fastest solution's (6.4817e-06 vs 6.4818e-06).

---

# Phase 3 — The `emugemm` library

`emugemm/{emugemm.h, emugemm.cpp, emugemm_test.cpp}`. A unified model, fitted to the Round 7–9
measurements:

```
err(scheme, K, c) = √( floor² + a²·(K/c) )
  fp32     floor 0          a 1.79e-8   speed 1.00×
  bf16×3   floor 4.4355e-6  a 8.95e-9   1.21×   robust across the range
  fp16×3   floor 3.6e-7     a 8.95e-9   1.19×   needs fp16 range
  bf16×6   floor ~0         a 8.95e-9   0.57×
```

Every low-precision path shares `a = 8.95e-9` = **exactly half** the constant of the default fp32
kernel, i.e. the bf16/fp16 kernels accumulate better. This is a **library property, not a property of
the number format**.

**Two consequences fall directly out of the model:**

1. **bf16×6 is strictly dominated and should never be chosen.** It costs 1.75× native time to reach an
   error that `fp32 + chunk(4)` reaches at ~1.0× (K=262144: 4.637e-6 @594 ms vs 4.579e-6 @334 ms).
2. **fp16×3 has a floor 12× lower than bf16×3 at the same speed** ⇒ when the data fits the fp16 range,
   fp16×3 is the right choice, not bf16×3.

### The invariant tests found three real bugs

1. **Silent under-delivery.** At a target of 1e-7 no candidate qualifies, but the planner returned its
   initialisation value instead of signalling failure. Fixed: `TARGET UNREACHABLE`.
2. **Slower than native on small shapes.** `hipMalloc`/`hipFree` of the split buffers on **every call**
   cost more than the GEMM saved (2048×2048×4096: 1.39 ms vs 1.03 ms). Fixed: a persistent workspace,
   plus scan/split cost in the model (`overhead_seconds`). Small shapes now correctly fall back to
   native.
3. **fp16 missing from the tuning table.** The fp16 path ran the default solution — precisely where
   Phase 2 had just shown the default collapses 2–4× — so it lost to native on skinny shapes
   (39.10 vs 32.22 ms) despite the model promising 1.19×. Fixed by adding fp16 to the table; the same
   case now runs **28.31 ms = 1.14× faster than native**.

A fourth defect was in the invariant itself: *"never slower than native"* is meaningless when the
target is below what native achieves — there native is not a valid option, so spending more time is
correct.

**Final result: 0 violations across 15 cases** (5 shapes × 3 targets), and **no accuracy violation in
any run** — the model predicted 4.472e-6, measurement gave 4.455e-6.

**Confidence in the chunk cost model was lowered:** the measurements are **non-monotonic** (c=16 →
643 ms but c=64 → 441 ms) because the kernel is re-selected with chunk depth. The smooth formula was
replaced by a **conservative envelope**, and the code records that it is calibrated on **exactly one
shape** — Phase 5 must re-measure before it is trusted elsewhere.

---

# Phase 5 — The error model across data distributions (`error_model.cpp`)

Every accuracy number in Rounds 7–9 was measured on **i.i.d. N(0,1)**, the case friendliest to fp32.
Phase 5 refits `err ~ a·K^p` per distribution. M=N=2048, reference = DGEMM on the GPU.

| distribution | fp32 `p` | bf16×3 `p` | @K=65536: bf16×3 vs fp32 |
|---|---|---|---|
| normal N(0,1) | 0.5249 | **0.0146** (flat) | 4.71e-6 vs 4.59e-6 — near the crossover |
| **positive U(0,1)** | 0.5506 | 0.2909 | **8.4e-7 vs 3.4e-6 → bf16×3 4× BETTER** |
| lognormal 1e-3…1e3 | 0.5047 | 0.0173 | 4.34e-6 vs 3.71e-6 |
| **cancelling (near-orthogonal)** | **0.1471** | 0.0055 | **1.65e-3 vs 4.66e-5 → bf16×3 35× WORSE** |

### The prediction going in was wrong

Phase 5 began with the hypothesis: *"correlated data makes fp32's accumulation error grow as O(K)
rather than √K, so the crossover arrives earlier — 76 500 is a conservative upper bound."* That is
wrong. On cancelling data fp32's `p` **falls to 0.147**, it does not rise.

What actually happens: when rows and columns are near-orthogonal, **‖C‖ collapses**, so the *relative*
error of every scheme inflates — but **emulation inflates far more**, because its representation error
is a fixed fraction of |a|·|b| rather than of ‖C‖. **Cancellation hurts emulation more than it hurts
fp32.**

### ⚠ The most important limit found in this round

**The model `err = √(floor² + a²K/c)` holds only for well-conditioned data.** On a strongly cancelling
problem bf16×3 gives **1.65e-3** — **35×** fp32's error and **370×** its own error on normal data. A
dispatcher trusting the model would select bf16×3 for a 1e-5 target and deliver 1.65e-3.
⇒ **This must be stated as an API assumption, not left implicit.** See "Open items".

### The other side, also worth having

On **all-positive** data (no cancellation) bf16×3 is **4× more accurate than fp32** at K=65536, and
its `p` is only 0.29 against fp32's 0.55. Because ‖C‖ grows like K while the error grows like √K, the
relative error *decreases* with K. This is the best regime for emulation, and it is a common one:
non-negative matrices, convolutions, counts, histograms.

---

# Phase 4 — RandNLA application: one negative result and one positive

### (a) Householder QR — a negative result that explains itself

`randsvd.cpp`, m=262144 n=4096 rank=2048:

| | total ms | relative residual |
|---|---|---|
| native fp32 | 6192.10 | 1.21709548e-01 |
| emugemm target 1e-5 | 6085.08 (**1.018×**) | 1.21709793e-01 |

Only 1.8% faster, and the reason is in the numbers: of 6192 ms total, GEMM is only ~300 ms —
**rocSOLVER `geqrf`+`orgqr` on a 262144×2048 panel runs at ~1 TFLOP/s (3% of peak) and consumes 5.9
seconds.** The pipeline is orthogonalisation-bound, so a 1.2× GEMM speedup buys 0.8%.

**The correct conclusion is not "emulation is useless for RandNLA"** but: *emulation only pays when
the pipeline is genuinely GEMM-bound; for tall-skinny panels that requires abandoning Householder QR.*

### (b) CholeskyQR2 — the standard GPU choice

`randsvd2.cpp` replaces Householder with CholeskyQR2, turning three of the four heavy operations into
GEMMs with **K = m**: `G = YᵀY` (K=m) → `chol(G)` (ℓ×ℓ, tiny) → `Q = YR⁻¹` (trsm) → one refinement
iteration → `B = QᵀA` (K=m).

| | total ms | GEMM ms | relative residual | |
|---|---|---|---|---|
| native fp32 (**tuned**) | 1312.15 | 598.43 | 1.2148000602e-01 | |
| emugemm target 1e-5 | 1101.87 | 566.38 | 1.2148003416e-01 | **1.191×**, residual identical to 7 digits |
| emugemm target 1e-6 | 991.02 | 468.24 | 1.2171402813e-01 | **1.324×**, residual +0.19% |

**Replacing Householder with CholeskyQR2 makes the whole pipeline 4.7× faster** (6192 → 1312 ms) and
turns it GEMM-bound (598/1312 = 46%). Only then does emulation have room to help: **1.018× → 1.19–1.32×**.

> ⚠ **An inflated number was caught before publication.** The first version ran the "native" path
> through `rocblas_gemm_ex` with `algo_standard` — **untuned** — while `emugemm` consulted the tuning
> table, and Phase 2 had just measured the default collapsing by 1.35× on average. Fixed: the baseline
> now goes through `emugemm` with `force_scheme=NATIVE_FP32`, so it gets **the same table lookup**.
> Results moved 1.204/1.305 → **1.191/1.324**. The change was small here, but the procedure that
> caught it — tuning both sides through the same path — is what should be relied on, not the size of
> this particular correction.

**Phase 4 conclusion:** emulation pays **only when the pipeline is genuinely GEMM-bound**. For
tall-skinny panels that means abandoning Householder QR first — and that single change is worth
several times the entire emulation speedup. A modest but correct conclusion: **fix the real bottleneck
before optimising GEMM.**

---

# Round 9 summary

| phase | result |
|---|---|
| 1 | Hypothesis **refuted**: given fp32 the same accumulation strategy, the crossover disappears. The bf16×3 floor of **4.4355e-6** is confirmed exactly. The law `err = f(K/c)` matches to four digits. |
| 2 | Tuning table of **787 rows across 3 types**. The default is optimal on only **3–10%** of shapes; mean gain **1.30–1.38×**, peak **4.4×**. Corrects one Round 8 conclusion. |
| 3 | The `emugemm` library plus a dispatcher driven by the measured model. Invariant tests found **3 real bugs**; final state **0 violations**. |
| 4 | RandNLA: Householder QR **1.018×** (QR-bound) → CholeskyQR2 **1.19–1.32×** (GEMM-bound), with a fairly tuned baseline. |
| 5 | The model **holds only for well-conditioned data**. Cancelling data: bf16×3 is **35× worse** than fp32. All-positive data: bf16×3 is **4× better**. |

## Open items, recorded so they are not mistaken for done

1. **`emugemm` is not safe for ill-conditioned problems.** The error model assumes ‖C‖ ~ √K·‖a‖‖b‖. On
   strongly cancelling data it is wrong by **35×** in the dangerous direction (delivering 1.65e-3 when
   promising 1e-5). This needs a field in `emu_request_t` for the caller to declare conditioning, or a
   cheap estimator, **before any real use**.
2. **`chunk_cost` is calibrated on exactly one shape** and the underlying measurements are
   non-monotonic. It is currently a conservative envelope.
3. **The tuning table is not portable** across ROCm versions or architectures (the architecture is
   checked at load time and a mismatch is rejected).
4. **48 shapes were skipped** by the budget guard; very large shapes fall back to nearest-shape lookup.
5. **MI300/MI355 untested.** Every constant here is for gfx90a.
