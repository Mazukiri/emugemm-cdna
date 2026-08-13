# Round 8 (2026-07-25) — the accuracy crossover, and closing the shape gap

Continues `RESULTS_2026-07-25.md` (Round 7). Hardware: one node of 8× GCD gfx90a (MI250), 104 CU and
68.7 GB per GCD → **549 GB total**. Container `ducmai-dev`, ROCm 7.2.3.
New files: `mfma_peak2.cpp` (+fp64), `crossover_test.cpp`, `rocblas_tune.cpp`, `emu_tuned.cpp`,
`node_sweep.sh`.

---

## A. The hardware economics table is complete — emulating FP64 is closed

| instruction | TFLOP/s | vs fp32 |
|---|---|---|
| **fp64** `mfma_16x16x4f64` | **41.7** | 0.98× |
| fp32 `mfma_16x16x4f32` | 42.5 | 1.00× |
| fp16 `mfma_16x16x16f16` | 169.4 | 3.99× |
| bf16 `mfma_16x16x16bf16_1k` | 169.2 | 3.98× |
| bf16 `mfma_16x16x8bf16` (legacy) | 85.1 | 2.00× |
| int8 `mfma_16x16x16i8` | 170.2 | 4.01× |

**CDNA2 has full-rate fp64 matrix throughput (equal to fp32).** The consequence, measured rather than
inferred from the datasheet:

| fp64 emulation scheme | ceiling | |
|---|---|---|
| from fp32, ≥3 slices | 0.34× | **loses** |
| from fp16, ~7 slices | 0.58× | **loses** |
| Ozaki int8, ~9 moduli | 0.45× | **loses** |

⇒ On MI250 **no route** to emulating fp64 is profitable. The direction is closed by measurement, the
same way `mfma_peak.cpp` closed the int8 direction in Round 3.

---

## B. The accuracy crossover

> **⚠ Corrected by Round 9 — read `RESULTS_ROUND9.md` before quoting anything from this section.**
> What follows holds **only against rocBLAS's default fp32 GEMM**. Round 9 (Phase 1) gave fp32 the
> same accumulation strategy (split K into chunks, sum the chunks in fp64) and **the crossover
> disappears**: fp32 has no representation floor, so its error keeps falling as `1/√c` (down to
> 5.73e-07), while bf16×3 hits a hard floor at **4.4355e-06** and stops. What survives: bf16×3 is
> still **1.20–1.22× faster** at the same accuracy tier and is the fastest point on the Pareto
> frontier. What does not: the unconditional claim *"more accurate than fp32"*.

**Setup:** M=N=4096, sweeping K = 2048 → 262144. **Reference = rocBLAS DGEMM run on the GPU itself**
(feasible because fp64 is full-rate; reference error ~6e-14, eight orders below the measurement
scale). **The reference was cross-checked against an fp64 CPU loop at K=2048: agreement 1.355e-15.**

The two bold bf16×3 entries are the rows where it overtakes fp32.

| K | fp32 | **bf16×3** | bf16×6 | fp16×3 |
|---|---|---|---|---|
| 2 048 | 8.105e-07 | 4.448e-06 | 3.320e-07 | 5.413e-07 |
| 4 096 | 1.145e-06 | 4.460e-06 | 4.961e-07 | 6.782e-07 |
| 8 192 | 1.620e-06 | 4.494e-06 | 7.390e-07 | 8.877e-07 |
| 16 384 | 2.290e-06 | 4.566e-06 | 1.090e-06 | 1.200e-06 |
| 32 768 | 3.240e-06 | 4.712e-06 | 1.589e-06 | 1.660e-06 |
| 65 536 | 4.581e-06 | 4.989e-06 | 2.289e-06 | 2.318e-06 |
| **131 072** | 6.482e-06 | **5.511e-06** | 3.273e-06 | 3.259e-06 |
| **262 144** | 9.162e-06 | **6.417e-06** | 4.637e-06 | 4.596e-06 |

**Agreement with theory:** fitting `err_fp32 = 1.791e-08 · K^0.4999` over the eight points recovers
the rounding-accumulation exponent **p = 0.5** to four digits. **Measured crossover at K ≈ 76 500**
(predicted before the run: 62 000, 25% off — reasonable for an extrapolation from two points).

> **⚠ The speed column of `crossover_test` must not be quoted.** It runs the **default** solution for
> every variant, and section C below shows the default can be off by up to 1.7×. At this large-K shape
> fp32 ranges 19–35 TF on kernel choice alone. **The speed numbers of record come from `emu_tuned`,
> which tunes both sides.** Accuracy is unaffected: bf16 error is identical across all 388 solutions
> (spread 1.00×), and the default fp32 error coincides with that of the fastest solution.

**Why bf16×3 is the only flat variant:** its error is bounded by **representation** error (two bf16
slices ⇒ ~2⁻¹⁶), independent of K. fp32, bf16×6 and fp16×3 all have representation error *below* the
accumulation threshold, so all three are dominated by **accumulation error ∝ √K**. At large enough K,
the constant beats the growing term.

### Confound check (planned before the run)

The concern: fp32 and low precision might use different accumulation strategies (split-K), which
would make the "win" a kernel-selection artifact rather than a property of the number format.
Enumerating **every** rocBLAS solution at K=131072:

| | solutions | error spread | note |
|---|---|---|---|
| fp32 | 562 | **2.00×** | most accurate 3.240e-06 **but only 4.07 TF** |
| bf16 | 388 | **1.00×** | every solution gives identical error |

- bf16 error is **independent** of kernel choice, so the bf16×3 conclusion holds.
- fp32 **does** have one solution twice as accurate (3.24e-6), but it runs at **4.07 TF — 9× slower**.
  For comparison: measured fp64 matrix peak is **41.7 TF**; even assuming a conservatively low library
  efficiency of 60%, DGEMM would still reach ~25 TF, i.e. **6× faster** than that fp32 solution *and*
  nine orders more accurate. (Achieved DGEMM speed was not measured directly — this is an argument
  from peak, safe given the 6× margin.) That "accurate" fp32 solution is strictly dominated and is not
  a realistic choice.
- ⇒ Against the fp32 a user actually gets (default = fastest = 6.48e-6), the crossover stands.

**Correct statement of the result:** *at equal or better speed, bf16×3 is more accurate than native
fp32 for K ≳ 76 000; obtaining more accurate fp32 costs a 9× slowdown, at which point native fp64 is
better on every axis.*

### Limits of this result

- **The data is i.i.d. N(0,1).** Under that distribution fp32's accumulation error grows as **√K**
  because rounding errors cancel like a random walk. For **same-sign or correlated** data the growth
  is **O(K)** rather than √K, so **the crossover arrives earlier**. N(0,1) is therefore the case most
  favourable to fp32 — this is a conservative upper bound, not a cherry-pick.
- bf16×3's error is flat because it is bounded by **representation** error, which is stable across
  data distributions; but the Frobenius norm is normalised by ‖C‖, so with strong output cancellation
  both curves shift upward. **Not measured** for distributions other than N(0,1) at large K.
- K ≈ 76 000 is **very large** for a single ML GEMM (hidden sizes are typically ≤ 30k). This regime is
  realistic for **HPC, long convolutions, batch accumulation**, not a typical linear layer.

---

## C. Closing the shape gap — 0.71× becomes 1.21×

Round 7 left a failure behind: bf16×3 reached only 0.74× on the MLP shape because rocBLAS's default
bf16 kernel collapsed to 82 TF. `rocblas_gemm_ex_get_solutions` allows enumerating and timing each
solution individually.

**Diagnosis (`rocblas_tune`, M=8192 N=28672 K=8192, bf16):**

| | ms | TFLOP/s |
|---|---|---|
| default | 45.88 | 83.9 (reproduces Round 7's 82.0) |
| **best of 388 solutions** | 27.92 | **137.8 = 81.5% of peak** |

The default heuristic leaves **1.64×** on the table. At the square K=131072 shape it leaves **1.58×**
(76.9 → 121.3 TF), so this is a **broad property of the rocBLAS bf16 heuristic**, not specific to the
MLP shape.

**End-to-end verification (`emu_tuned`) on all three shapes — the fp32 baseline is tuned too, for
fairness:**

| shape | fp32 default | **fp32 tuned (baseline)** | bf16×3 default | **bf16×3 tuned** | result |
|---|---|---|---|---|---|
| M=8192 N=28672 K=8192 (MLP) | 21.57 | **37.52** | 26.48 (0.71×) | **45.33** | **1.21×** |
| M=16384 N=8192 K=8192 | 36.94 | **37.03** | 37.72 (1.02×) | **44.26** | **1.20×** |
| M=N=K=12288 (square) | 37.06 | **37.02** | 37.44 (1.01×) | **45.06** | **1.22×** |

Error is unchanged by tuning (bf16×3: 4.495e-06 / 4.495e-06 / 4.530e-06).

**Two conclusions:**

1. **bf16×3 gives 1.20–1.22× on every shape tried** — strikingly uniform, once the bf16 solution is
   tuned. Round 7's shape gap was **entirely a kernel-selection artifact**, not a technical limit.
2. **The rocBLAS heuristic is weak for bf16 and strong for fp32.** Tuning lifts bf16×3 by 1.71× /
   1.17× / 1.20×, while fp32 barely moves on the two squarish shapes (37.06→37.02; 36.94→37.03) and
   jumps only on the MLP shape (21.57→37.52). The problem is on the **bf16 path**, consistent with
   section E.

---

## D. Whole node — does the advantage survive with all 8 GCDs saturated?

The question is concrete: the 8 GCDs are **4 dual-die cards sharing a power budget**, and Round 7
measured bf16×3 drawing **11% more power** than fp32. So there is a specific reason to suspect the
advantage erodes under full load. `node_sweep.sh`, N=8192³, one independent process per GCD, **both
sides tuned**.

| | fp32 | bf16×3 |
|---|---|---|
| 1 GCD alone | 37.02 TF | 43.96 TF (**1.19×**) |
| **8 GCDs, total** | **278.4 TF** | **325.1 TF** (**1.17×**) |
| % of linear scaling | **94.0%** | **92.4%** |

**The advantage survives: 1.19× → 1.17×.** Both lose 6–8% under full load, and **bf16×3 loses slightly
more (92.4% vs 94.0%)** — consistent with Round 7's power measurement (bf16×3 draws 11% more, so it
reaches the power ceiling sooner). Two independent measurements point at the same mechanism.

**Methodological caveat:** each process tunes **while the other seven are running**, so solution
selection is measured under contention. This shows in the spread of bf16×3 across GCDs (37.6–44.3 TF)
against fp32 (32.8–36.1 TF). The aggregate number is usable; **per-GCD** numbers should not be quoted
individually.

**Not done:** demonstrating the largest single GEMM the 549 GB allows. This sweep uses only ~3.3 GB
per GCD, so memory is **not** the binding constraint here. Doing it needs a lighter harness —
`emu_tuned` times ~950 solutions, which at N=32768 would take ~90 minutes per GCD. Recorded as **not
measured**, not as measured.

---

## E. hipBLASLt bf16 — the hypothesis was refuted; the mechanism is still unidentified

Round 8 began with the hypothesis that *hipBLASLt falls back to non-MFMA kernels for bf16 on gfx90a*,
based on the symptom (25% of peak, 10 algorithms offered) and AMD's documented note that *"Not all
problem sizes may select MFMA-based kernels"*. Reading the shipped Tensile library directly shows the
hypothesis is **wrong**.

**Two mechanisms ruled out:**

1. **Not a fallback to non-MFMA.** Extracting actual kernel names from
   `/opt/rocm/lib/hipblaslt/library/`:
   `Cijk_Ailk_Bljk_BBS_BH_..._MT64x16x32_MI16x16x1_...ISA90a...`
   In one bf16 NN library file: **1912 kernels, all 1912 carry `_MI` (MFMA)**. There is no non-MFMA
   kernel to fall back to.
2. **Not a shortage of good kernels.** The bf16 macro-tile distribution matches fp16 and includes
   large tiles (`MT256x64x32`, `MT96x160x64`, `MT128x96x64`, …). bf16 is **not** kernel-poor.

**What still holds (reproducible, measured repeatedly):**

| | best measured | % of 169.1 TF |
|---|---|---|
| hipBLASLt bf16 (NN, gfx90a) | 41.8 TF | **25%** |
| rocBLAS bf16 (same hardware, also MFMA) | 137.8 TF | **81.5%** |
| hipBLASLt fp16 | 137.0 TF | 81% |

and hipBLASLt returns only **10 candidate algorithms** for bf16 NN, while fp16 and fp32 both hit the
requested cap of 48.

**Conclusion at the right strength:** the gap is **real, large (3.3×) and reproducible**, and it is
**not** caused by missing MFMA kernels — but **the mechanism is unidentified**. It lies in hipBLASLt's
solution-selection layer for bf16 NN on gfx90a, which is not observable from here
(`HIPBLASLT_LOG_LEVEL=4` prints the library path but not the selected solution; the container has
neither `hipblaslt-bench` nor `rocblas-gemm-tune`).

⇒ **No issue filed.** The *reproducible symptom* plus the ruled-out mechanisms could be reported, but
**no cause may be claimed**. The immediately usable finding: **use rocBLAS for bf16, hipBLASLt for
fp16.**

---

## Method notes from Round 8

1. **Piping a build command through `| head` kills the compiler with SIGPIPE** — the build "fails"
   with no error message at all. Do not pipe compiler output through `head`.
2. **Log progress in long loops.** The first `rocblas_tune` printed only at the end and ran 12 minutes
   in silence, making "running" indistinguishable from "hung".
3. **Tune both sides.** On the MLP shape, tuning lifts fp32 from 21.6 to 37.5 TF. Tuning only the
   proposed method would have manufactured a false 2.1× win.
