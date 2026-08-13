# Data and tools for ROCm/rocm-libraries#9985

Every figure quoted in the issue thread, with the CSV it came from, so it can be recomputed rather
than taken on trust. Two architectures throughout: **gfx90a** (MI250, one GCD, ROCm 7.2.3) and
**gfx942** (MI300X, rented pod).

## The two questions asked in the thread

| file | what it is |
|---|---|
| `data/q2_families.csv` | 746 rows — 150 skinny-M (M ≤ 16, N,K ≥ 1024) and 596 large-K (K ≥ 8192, min(M,N) ≥ 128), each with the winning solution index in **both** encodings, candidate count, both timings, gain |
| `data/q2_override.csv` | the same rows already converted for `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH`, in `rocblas-gemm-tune` column order |
| `evidence/q1_logic_file_dispatch.tsv` | which Tensile logic file rocBLAS actually loads, per transpose and dtype, from `TENSILE_DB=0x40` |
| `evidence/q2_override_dispatch.tsv` | 30-row check that the override changes the dispatched macro tile — verified by kernel name, never by timing |

**The index conversion is the part that fails silently.** `rocblas_gemm_ex_get_solutions()` returns
rocBLAS-encoded indices, negative for Tensile-backed solutions. The override parser wants the raw
Tensile index, positive and 1-based (`MasterSolutionLibrary.hpp` does `getSolutionByIndex(value - 1)`):

    file_value = -rocblas_index - c_rocblas_solutions_reserved     # reserved == 10

Feed the value through unconverted and rocBLAS prints one warning line, then keeps the default
kernel. `src/answer_q2.py` does the conversion and the coordinate swap in one place.

## Which file backs which claim

| claim in the thread | gfx90a | gfx942 |
|---|---|---|
| the sweep the headline medians come from | `data/gfx90a/fullspace_v2_gfx90a.csv` (7009) | `data/gfx942/fullspace_v2_gfx942.csv` (4776) |
| five ROCm releases pick the same kernel | `data/gfx90a/rand_{c63,c64,r700,714}.csv` + `B_n13.csv`, kernel names in `knames.csv` | `data/gfx942/rand_*.csv` (5 × 239), names in `knames_gfx942.csv` |
| real-workload cost, call-count weighted | `data/gfx90a/workload/*_gain.csv` + `*_counts.csv` | `data/gfx942/wl_gfx942.csv` |
| the override fixes it, end to end | `data/gfx90a/workload/*_ab.csv` (3 models, plain vs override vs hipBLASLt) | — |
| `ROCBLAS_USE_HIPBLASLT` A/B | `A_n13.csv`, `full_fp32.csv` (original method); `backend_*_v2.csv` (re-measured) | — |
| selection bias: no search, fresh process, median | `data/gfx90a/unb.csv` (60) | `data/gfx942/unbiased_gfx942.csv` (60) |
| levers that do **not** help | `metric.csv` (1200 × 3), `pred.csv` (450 × 8), `determinism.csv` (39), `batched.csv` (231 × 4) | — |
| the fp16 TT band, isolated and repeated | `data/gfx90a/band_{tt,nn}_r{1,2,3}.csv` | — |

## Tools

| file | notes |
|---|---|
| `src/tune_trans.cpp` | the sweep harness. Rotated buffers (`--rotate`), time-based repetition (`--minms`), uniform [-1,1] fill, and a clock probe before and after each shape written to the CSV as `clk_drift` |
| `src/three_way2.cpp` | four-arm comparison including **exhaustive** hipBLASLt enumeration via `hipblaslt_ext::getAllAlgos`, because "best of the heuristic shortlist" turned out to be a function of the shortlist length |
| `src/backend_v2.cpp` | `ROCBLAS_USE_HIPBLASLT` A/B on the default path only |
| `src/cmparch3.py` | counts tuned points in the shipped `.dat` files. **No GPU needed** — install ROCm and run it |
| `src/answer_q2.py` | builds the two Q2 tables, including the index conversion and the coordinate swap |
| `src/analyze_abc.py` | the verification pass described in `evidence/verification_2026-08-13.md` |

## Three things worth reading before reusing any of this

**Coordinate swap.** The harnesses compute row-major `C = A·B` by calling column-major rocBLAS with
the operands swapped: `gemm_ex(opB, opA, N, M, K, pB, ldb, pA, lda, …)`. So rocBLAS `M` is the
harness's `N`, and rocBLAS `transA` is the harness's `opB`. Measured consequence: harness `NT` lands
on Tensile `Alik_Bljk` and harness `TN` on `Ailk_Bjlk` — the opposite of what the labels suggest.
`NN` and `TT` are symmetric under the swap. `q2_families.csv` carries both coordinate systems in
separate columns for this reason.

**Counting tuned points.** The `.dat` tables contain duplicate keys, unevenly across architectures:
the gfx90a fp32 table holds 9874 entries over 4888 **distinct** (M,N,K) points, while gfx942's 415
entries are 345 distinct. Counting entries overstates the gfx90a/gfx942 ratio by about 40 %.
`cmparch3.py` reports both.

**Two measurement methods are present, and they are labelled.** Files with `nrot`, `reps` and
`clk_drift` columns use the current method (rotated buffers, ≥10 ms timed windows, uniform fill).
Files without them predate it (single buffer, five repetitions, constant fill). Every figure from the
older method was re-measured on the same shapes; paired per shape, nothing moved by more than 3 %
(`evidence/verification_2026-08-13.md`). Comparing a median from one shape population against a
constant from another is the mistake this whole report is about — the populations differ between
files, so pair per shape before concluding anything moved.
