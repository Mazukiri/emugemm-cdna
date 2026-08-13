# Data and tools for ROCm/rocm-libraries#9985

Everything the issue thread refers to, so the numbers can be checked rather than taken on trust.
Two architectures: **gfx90a** (MI250, one GCD, ROCm 7.2.3) and **gfx942** (MI300X, rented pod).

## The two questions answered in the thread

| file | what it is |
|---|---|
| `data/q2_families.csv` | 746 rows — 150 skinny-M (M ≤ 16, N,K ≥ 1024) and 596 large-K (K ≥ 8192, min(M,N) ≥ 128), each with the winning solution index in **both** encodings, candidate count, both timings, gain |
| `data/q2_override.csv` | the same rows already converted for `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH`, in `rocblas-gemm-tune` column order |
| `evidence/q1_logic_file_dispatch.tsv` | which Tensile logic file rocBLAS actually loads, per transpose and dtype, from `TENSILE_DB=0x40` |
| `evidence/q2_override_dispatch.tsv` | 30-row check that the override changes the dispatched macro tile — verified by kernel name, never by timing |

**The index conversion is the part that fails silently.** `rocblas_gemm_ex_get_solutions()` returns
rocBLAS-encoded indices, negative for Tensile-backed solutions. The override file parser wants the
raw Tensile index, positive and 1-based (`MasterSolutionLibrary.hpp` does
`getSolutionByIndex(value - 1)`). The conversion is

    file_value = -rocblas_index - c_rocblas_solutions_reserved     # reserved == 10

Feed the value through unconverted and rocBLAS prints one warning line, then keeps the default
kernel. `src/answer_q2.py` does the conversion and the coordinate swap in one place.

## Supporting data

| file | n | what it shows |
|---|---|---|
| `data/fullspace_v2_gfx942.csv` | 4776 | log-uniform sweep on MI300X, current measurement method |
| `data/rand_{6.3.4,6.4.4,7.0.3,7.1.1_native,7.2.3}.csv` | 5 × 239 | the same 239 shapes under five ROCm releases on **one** MI300X pod |
| `data/knames_gfx942.csv` | 48 | dispatched kernel names per release — 12/12 shapes pick the same kernel across four releases |
| `data/wl_gfx942.csv` | 338 | GEMM shapes captured from 11 runs of 7 real models, measured on MI300X |
| `data/unbiased_gfx942.csv` | 60 | the selection-bias control: no search, fresh process, interleaved, randomised order, median not minimum |

## Tools

| file | notes |
|---|---|
| `src/tune_trans.cpp` | the sweep harness. Rotated buffers (`--rotate`), time-based repetition (`--minms`), uniform [-1,1] fill, and a clock probe before and after each shape written to the CSV as `clk_drift` |
| `src/three_way2.cpp` | four-arm comparison including **exhaustive** hipBLASLt enumeration via `hipblaslt_ext::getAllAlgos`, because "best of the heuristic shortlist" turned out to be a function of the shortlist length |
| `src/backend_v2.cpp` | `ROCBLAS_USE_HIPBLASLT` A/B on the default path only |
| `src/cmparch3.py` | counts tuned points in the shipped `.dat` files. **No GPU needed** — install ROCm and run it |
| `src/answer_q2.py` | builds the two Q2 tables, including the index conversion and the coordinate swap |
| `src/analyze_abc.py` | the verification pass described in `evidence/verification_2026-08-13.md` |

## Two things worth reading before reusing any of this

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

`evidence/verification_2026-08-13.md` lists every figure that was re-derived, which three moved, and
what was corrected.
