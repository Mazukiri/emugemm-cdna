# emugemm-cdna

**Emulated-FP32 GEMM for AMD CDNA, with an accuracy contract it actually keeps.**

Split each FP32 input into BF16 or FP16 slices, multiply on the matrix cores, and reassemble — the
standard error-corrected emulation idea (Markidis 2018; Ootomo & Yokota 2022; Ozaki). What is here that
is not elsewhere: it runs on **AMD CDNA**, where the published work does not, and you can ask it for a
specific accuracy and it will either deliver that accuracy or **tell you it cannot** — including on
badly-conditioned data, where a naive error model silently under-delivers by orders of magnitude.

Measured on MI250 (gfx90a, 1 GCD, 104 CU, ROCm 7.2.3).

| | result |
|---|---|
| **bf16×3 vs native FP32** | **1.20–1.22×** on every shape tested, after per-shape solution tuning of **both sides** |
| **Robustness** | full FP32 exponent range (bf16 keeps FP32's 8-bit exponent; the FP16 path does not) |
| **Accuracy floor** | **4.4355e-6** — a hard limit; no amount of time buys accuracy below it |
| **Contract** | 0 violations across 15 well-conditioned cases and 10 adversarial ones (ρ up to 1.1e6) |

> **Arriving from [ROCm/rocm-libraries#9985](https://github.com/ROCm/rocm-libraries/issues/9985)?**
> That investigation started here — the tuning table below showed rocBLAS's default solution losing to
> another solution in the same library — and it now has its own directory, [`issue9985/`](issue9985/),
> with data for both gfx90a and gfx942, the measurement harnesses, and the dispatch evidence. Its
> [`issue9985/src/cmparch3.py`](issue9985/src/cmparch3.py) reproduces the mechanism with no GPU at
> all. This repository's own `src/` is the emulation library and is unrelated.

---

## Quick start

```bash
hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a \
      -Isrc bench/emugemm_test.cpp src/emugemm.cpp -o emugemm_test -lrocblas
./emugemm_test data/tune_table.csv
```

```c
#include "emugemm.h"
emugemm_init("data/tune_table.csv");

emu_request_t req  = { .max_rel_err = 1e-5, .force_scheme = EMU_SCHEME_AUTO };
emu_hints_t   hint = { -1.f, -1.f, 0, -1.f, 64 };   // <0 = measure it for me
emu_plan_t    plan;

emugemm_sgemm(handle, M, N, K, A, B, C, &req, &hint, &plan);
// plan.chosen, plan.predicted_err, plan.reason
// If plan.reason contains "TARGET UNREACHABLE", the request could not be met -- the result is the
// closest available, and you were told. It never silently under-delivers.
```

---

## What this does **not** do

Put high, not buried, because these are the things that would bite you:

- **bf16×3 cannot go below 4.4355e-6, ever.** That is a representation floor, not a tuning parameter.
  Requests below it are routed to `FP32_CHUNKED` (chunk K, accumulate the chunks in FP64) or refused.
- **The tuning table is not portable.** Solution indices are specific to the ROCm version *and* the GPU
  architecture. `emugemm_init` checks the arch string and rejects mismatched rows rather than silently
  selecting the wrong kernel. Regenerate with `bench/gen_tune_table.cpp` on your machine.
- **Only gfx90a (MI250) has been tested.** Every constant in the error model was measured there.
  CDNA3/CDNA4 will have different ratios and the conclusions may invert — see *Limits* below.
- **`chunk_cost` is calibrated at one shape.** The measured chunking cost is non-monotonic (kernel
  reselection at different chunk depths), so the planner uses a deliberately conservative envelope.
- **48 shapes were skipped** by the tuning time budget; very large shapes fall back to nearest-shape lookup.
- **The ρ correction is pessimistic for native fp32 under distributed cancellation.** It can make the
  planner escalate or decline when plain fp32 would have sufficed. Safe direction, but it costs
  efficiency — see *What ρ does and does not determine*.
- Not a drop-in `rocblas_sgemm` replacement. Row-major, non-transposed, `alpha=1, beta=0` only.

---

## The error model

Two orthogonal axes, both measured:

```
err(scheme, K, c, ρ)  =  ρ/ρ_ref(K) · sqrt( floor² + a²·(K/c) )        ρ_ref(K) = 0.64·√K
                         └─── data ──┘   └──── algorithm ────┘
```

| scheme | floor | a | speed | robust range |
|---|---|---|---|---|
| native fp32 | 0 | 1.79e-8 | 1.00× | yes |
| **bf16×3** | 4.4355e-6 | 8.95e-9 | **1.21×** | **yes** |
| fp16×3 | 3.6e-7 | 8.95e-9 | 1.19× | no — needs row/col scaling |
| bf16×6 | ~0 | 8.95e-9 | 0.57× | yes, but **dominated** |
| fp32 + chunk(c) | 0 | 1.79e-8 | <1× | yes |

- **`c` is the number of K-chunks summed in FP64.** Error depends only on the accumulation chain length
  `K/c` — verified to 4 significant figures across K=65536 and K=262144.
- **All low-precision paths share `a = 8.95e-9`, exactly half the default fp32 kernel's constant.**
  This is a *library* property (the bf16/fp16 Tensile kernels accumulate better than the default fp32
  kernel), **not** a property of the number formats. Do not describe it as one.
- **`bf16×6` is strictly dominated and never selected.** It costs ~1.75× native fp32 time to reach the
  same error `fp32 + chunk(4)` reaches at ~1.0×. Kept in the enum for benchmarking only.
- **ρ = ‖|A||B|‖_F / ‖AB‖_F** is the cancellation ratio: how far the true product shrinks below the
  magnitude of the terms being summed. `K` decides *which* scheme wins; ρ decides whether the winner
  meets your request. See the caveat immediately below — ρ is not the whole story.

### What ρ does and does not determine

Tested on three structurally different synthetic families. **Representation-bound schemes obey `err ∝ ρ`;
accumulation-bound ones do not, because their error depends on the *structure* of the cancellation.**

| | family 2 — algebraic annihilation | family 3 — oscillatory integral |
|---|---|---|
| how cancellation arises | `A=[G\|G]`, `B=[H; −H+δG₂]`; the bulk annihilates | smooth rows against oscillating columns; ρ tuned by frequency |
| **fp32** | `1.989e-8 · ρ` | **flat at ~4.4e-6** across ρ = 1.2 → 3019 |
| **bf16×3** | `≈3.3–4.6e-8 · ρ` | **`≈3.4e-8 · ρ`** — same constant, different mechanism |

Representation error is a fixed fraction of `|a||b|`, so it always scales with `‖|A||B|‖ = ρ‖C‖` whatever
the summation order. Accumulation error instead depends on how large the partial sums grow *before* they
cancel: family 2 cancels late (sums balloon, then cancel), family 3 cancels in a distributed way (the
oscillation keeps every partial sum small).

**Consequence for this library, stated plainly:** the planner scales *every* scheme by `ρ/ρ_ref`, which is
correct for bf16×3 and **pessimistic for fp32** on family-3-like data. It may therefore escalate to a more
expensive scheme, or decline a request, when plain fp32 would in fact have been accurate enough. That
costs efficiency and never correctness — it cannot cause an under-delivery — but it is a real limitation,
not a rounding detail.

### Measuring ρ cheaply

Hutchinson with Rademacher probes: `A(BΩ)` and `|A|(|B|Ω)`, two skinny GEMMs each.
Cost `2·P·K·(M+N)` versus `2MNK`, i.e. `2P/M` for square problems.

| probes | ρ̂/ρ | cost at M=N=8192 |
|---|---|---|
| 8 | 1.56–1.64 | 0.2% |
| **64** (default) | **1.10** | **1.6%** |
| 256 | 0.99–1.00 | 6.3% |

10% accuracy is ample: the decision boundaries are orders of magnitude apart.

---

## How the numbers were checked

Every claim here survived a specific attempt to break it:

- **Ground truth is FP64 `DGEMM` on the device**, not a CPU loop — MI250 has full-rate FP64 matrix cores
  (measured 41.7 TF vs 42.5 TF for FP32), so a reference at K=262144 costs ~0.2 s. Validated against a
  CPU FP64 loop at K=2048: agreement to **1.355e-15**.
- **Both sides are tuned.** The baseline goes through the same per-shape solution lookup as the emulated
  path. At `M=8192 N=28672 K=8192` this lifts the fp32 baseline from 21.6 to 37.5 TF — reporting against
  the untuned baseline would have turned a true 1.21× into a fake 2.1×.
- **Timing is interleaved A/B/A/B with median-of-N**, never one block then the other, because the
  emulated path draws ~11% more power and a fixed order lets thermal drift favour whoever runs first.
- **Accuracy and speed are measured in the same run, on the same data, by the same implementation.**
- **The library is tested adversarially**, not only on the data it was calibrated on
  (`bench/emu_adversarial.cpp`).

---

## Things I got wrong

Kept deliberately. These cost real time to find and each one nearly became a published result.

**1. A 31× accuracy cliff that did not exist — and then a model that was only half right.** Sweeping the
cancellation ratio produced a clean non-monotonic curve: bf16×3 peaking at 31× worse than fp32 around
ρ≈10⁴, then recovering. It was an artifact of my own matrix construction — as ρ grew, the entries
approached ±1, and **bf16 represents ±1 exactly**. The tell was a number that had no right to exist:
bf16×6's error *improving* with ρ, down to 2.5e-8. A second family with generic N(0,1) values showed no
cliff at all, and a clean `err = c·ρ` for every scheme.

That second conclusion was also incomplete. A **third** family, inducing cancellation through an
oscillatory integral rather than by construction, shows `err = c·ρ` holds **only for the
representation-bound scheme**: bf16×3 obeys it with the same `c ≈ 3.4e-8` across both mechanisms, while
fp32's error is *flat* in ρ where family 2 had it growing linearly. Three families, three different
answers; only two things survived all of them (§ *What ρ does and does not determine*).

*A synthetic matrix family is a hypothesis about your data, not evidence. One is worthless, two can
disagree, three told me which parts were real.*

**2. "Emulation is more accurate than FP32 at large K" — true, then false.** bf16×3's error is flat in K
while native FP32's grows as K^0.4999, and they cross at K≈76,500. Real, reproducible — and it evaporates
the moment you give FP32 the same treatment: chunk K and accumulate the chunks in FP64, and FP32 keeps
improving as 1/√c while bf16×3 stops at its floor. The crossover holds only against rocBLAS's *default*
FP32 GEMM. Had I applied the chunking only to my own side, I would have published the exact opposite
conclusion.

**3. A 2.1× speedup that was really 1.21×.** The baseline was running rocBLAS's default kernel while the
emulated path used the tuned solution table. Same data, same hardware, 74% overclaim.

**4. A mechanism I asserted without reading the source.** I claimed hipBLASLt was falling back to
non-MFMA kernels for bf16. Reading the shipped Tensile libraries showed **1912 of 1912 bf16 kernels are
MFMA-based**. The performance gap is real and reproducible; the mechanism is still unknown, and the
README says so rather than guessing.

**5. Comparing medians across different shape populations — three times in one day, while verifying
the rocBLAS report.** Each time it produced a confident wrong verdict: an architecture comparison run
over two different sweeps, and twice a "the numbers moved" conclusion drawn from files that shared
zero shapes. Paired per shape, nothing had moved. That the error recurred inside the verification of a
report *about* that exact error is the reason it is listed here. The fix is mechanical: print the size
of the intersection before comparing anything, and refuse to proceed if it is zero. Details, along
with two figures this pass corrected (a density ratio overstated by 40% by counting table entries
instead of distinct keys, and a correlation of 0.371 published as 0.412), are in
[`issue9985/evidence/verification_2026-08-13.md`](issue9985/evidence/verification_2026-08-13.md).

---

## Repository layout

```
src/emugemm.{h,cpp}        the library: model, dispatcher, ρ probe, tuned-solution lookup
bench/emugemm_test.cpp     invariant suite (never slower than native; never exceeds declared error)
bench/emu_adversarial.cpp  contract test on ill-conditioned data, ρ from 1e2 to 1.1e6
bench/rho_sweep.cpp        cancellation sweep, three independent matrix families (--family 1/2/3)
bench/gen_tune_table.cpp   offline per-shape solution tuner: resumable, shardable across GCDs
bench/audit_table.cpp      re-measures the tuning table without the ordering bias
bench/validate_issue.cpp   spot-check of the published figures, plus rocblas_sgemm vs gemm_ex
bench/mfma_peak2.cpp       raw matrix-core rates: fp64/fp32/fp16/bf16(both opcodes)/int8
bench/flat_error.cpp       chunked-FP64 accumulation study
bench/error_model.cpp      error vs data distribution
bench/randsvd2.cpp         randomized SVD with CholeskyQR2, an end-to-end application
data/tune_table.csv        820 tuned solution indices (gfx90a, ROCm 7.2.3), the warm-up-corrected
                           re-run — regenerate for your own setup
docs/                      full experiment logs, including every failed experiment and why
issue9985/                 the rocBLAS kernel-selection investigation: data for gfx90a and gfx942,
                           harnesses, and the dispatch evidence behind ROCm/rocm-libraries#9985
```

---

## Limits, and what would change the picture

- **The speed ceiling on MI250 is `R/3 = 1.33×`** where `R = 3.98` is the measured bf16:fp32 matrix-core
  ratio. At 1.20–1.22× this is **92% of the ceiling** — there is no meaningful speed left on this chip.
  Three products is the accuracy floor for ~16 significand bits, and 3× the FLOPs is 3× the FLOPs no
  matter how you package it (K-stacking merges launches, not work).
- **FP64 emulation is dead on CDNA2.** Measured: FP64 matrix = 41.7 TF = 0.98× FP32. Any scheme needs
  ≥3 slices, so ≤0.58× at best. Closed by measurement, not by a datasheet.
- **CDNA3/CDNA4 could invert several conclusions**, since the whole economic argument rests on the
  measured bf16:fp32 ratio. Run `bench/mfma_peak2.cpp` first on new hardware — it decides everything else.
- **`c_bf16×3` is not perfectly constant in ρ** (falls from 4.6e-8 to 3.3e-8 at ρ=1e6). Its representation
  floor does not scale with ρ the way the accumulation term does; an additive correction is unfitted.

---

## Relation to prior work

The emulation technique is **not** new, and neither is the K-crossover phenomenon. What is here is the
AMD/CDNA data point, the ρ-based dispatch, and a contract that holds.

| work | what it does | overlap |
|---|---|---|
| Markidis et al. 2018; Ootomo & Yokota 2022 | origin of FP32-from-FP16 with error correction | the technique itself |
| Ozaki scheme; ozIMMU; GEMMul8 | integer/multi-slice emulation, mainly for FP64 | alternative decomposition |
| SGEMM-cube / H2SGEMM (arXiv 2507.23387) | FP32-from-FP16 on Ascend NPUs | reports the same K-crossover |
| EmuGEMM (arXiv 2606.25453) | fused Ozaki kernels, 83% of INT8 peak | Hopper/Blackwell only; lists other architectures as future work |
| Ozaki-II INT8 (arXiv 2508.03984) | SGEMM/DGEMM via INT8 moduli | moduli count is *user-specified and fixed* |
| Baboulin et al., Euro-Par 2024 | mixed-precision randomized low-rank: tensor cores + CholeskyQR + IR | our `randsvd2` re-derives this design |
| Fasi, Higham, Lopez, Mary, Mikaitis, SISC 2023 | error analysis of multiword matrix multiplication | the theory for this splitting |
| cuBLAS `32F_EMULATED_16BFX9`; AMD `HIPBLAS_COMPUTE_32F_FAST_16BF` (gfx950) | shipped emulation | the 1-product and 9-product ends; the middle is unoccupied |

The specific gap this addresses is stated as open in arXiv 2601.08077: *"inaccurate estimation of the
emulation level to achieve desired accuracy levels… ozIMMU and GEMMul8 don't support emulating to a
specific accuracy level."* Existing adaptive schemes select on **exponent statistics**. This one selects
on **cancellation**, which is what actually breaks the error model.

---

## License

MIT. See [LICENSE](LICENSE).

---

## A correction I had to make to my own published numbers

After filing [ROCm/rocm-libraries#9985](https://github.com/ROCm/rocm-libraries/issues/9985) I audited my
own measurement and found an ordering bias: `gen_tune_table.cpp` timed the *default* first, on a cold
device, and the winning solution last, after hundreds of launches had warmed it. That inflates
`gain = default / best`.

Re-measured with both variants warmed and then timed interleaved (`bench/audit_table.cpp`,
`bench/validate_issue.cpp`), over a random sample of 40 rows:

| | as published | re-measured |
|---|---|---|
| mean gain | 1.464 | 1.432 |
| median gain | 1.254 | 1.247 |
| shapes losing >10% | 80% | 75% |

Aggregate inflation is **2.4%**. The tool now warms every candidate before timing any of them.

Two things I *suspected* were wrong turned out not to be, and I checked rather than assumed: the
minimum-over-16 selection biases the best time by less than 0.5%, and `rocblas_sgemm` agrees with
`rocblas_gemm_ex`+`algo_standard` to within 1.2%, so the finding applies to the API people actually call.

### …and a correction to the correction

That correction also claimed one headline example fell from 2.00× to **1.63×**. That figure came from a
single re-measurement with the default at 24.128 ms; two full sweeps put it near 28.1 ms. Re-measured
alone on an idle GCD, warmed, interleaved, median-of-9:

| | default | best solution | gain |
|---|---|---|---|
| `4096×4096×16384` fp32 | 28.72 ms (19.14 TF) | 14.86 ms (36.99 TF) | **1.933×** |

**The original 2.00× was very nearly right; the 1.63× was the outlier** — a noisy quantity quoted to
three digits from one sample, which is the mistake the correction was written to fix.

The sweep is also sharded across 8 GCDs, which can distort GEMM timing badly (`hipEventElapsedTime`
charges host-side stalls to the kernel; on sub-millisecond shapes I measured a mean 12.3× inflation).
All 39 shapes quoted here were therefore re-measured on an idle GCD: mean(isolated/sharded) = **0.9945**,
mean absolute deviation 3.1%, 2 of 39 differing by more than 10%. No systematic inflation — this grid's
shapes are long enough that a fixed stall is negligible.

The two exceptions are named rather than averaged away: `8192×16384×32768` bf16 measured **2.636×**
sharded and **1.580×** isolated (40% apart), and `8192×28672×8192` bf16 went the other way, 1.508× →
1.661×. The first is the largest shape in the table (8.8e12 flops, ~2.1 GB of operands) and is the one
row here where 8-way HBM pressure plausibly bit. Worth knowing if you reproduce this on a shared node.
