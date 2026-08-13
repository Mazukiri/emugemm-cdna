# Column reference

Every CSV in `data/` uses columns from this list. Shapes are in **harness coordinates** unless a
column says `rocblas_`; see the coordinate note at the end.

## Shape and problem

| column | meaning |
|---|---|
| `M`, `N`, `K` | GEMM dimensions, `C[M×N] = A[M×K] · B[K×N]` |
| `dtype` | input type: `fp32`, `bf16`, `fp16`. Output and compute are fp32 throughout |
| `opA`, `opB` | `N` or `T` — whether A and B are transposed |
| `beta` | 0 or 1 |
| `batch` | batch count (batched GEMM only) |

## Timing and result

| column | meaning |
|---|---|
| `def_ms` | milliseconds for the **default** heuristic, i.e. `rocblas_gemm_algo_standard` |
| `best_ms` | milliseconds for the fastest solution found by enumerating all candidates |
| `gain` | `def_ms / best_ms`. Above 1 means the default is slower than a solution in the same library |
| `tflops` | achieved 2·M·N·K / time |
| `nsol` | how many solutions `rocblas_gemm_ex_get_solutions` returned for this problem |
| `best_sol` | the winning solution index, **rocBLAS-encoded** (negative for Tensile-backed solutions — see the conversion note in the README) |

## Measurement diagnostics

These columns are written by every version of the harness, so their *presence* does not indicate the
method. **`nrot` is what distinguishes them:** 4 in the current method, 1 in the earlier one, since
the default is `--rotate 1`. Files here with `nrot=4` were measured the current way.

| column | meaning |
|---|---|
| `nrot` | buffer sets actually allocated for rotation. Requested 4; memory pressure can force fewer, so the achieved count is recorded rather than assumed |
| `reps` | kernel launches per timed window, chosen so the window spans at least 10 ms |
| `ach_gbs` | achieved bandwidth, GB/s |
| `cache_flag` | 1 if `ach_gbs` exceeds the HBM peak, which would mean the row was served from cache despite rotation. 0 everywhere in the published data |
| `clk_drift` | a 2048³ reference GEMM timed after the shape divided by the same before it. Below 1 means clocks rose during the measurement. **Report it, do not filter on it** — drift correlates with shape size, so filtering by drift filters by size |

## Per-file columns

| column | file | meaning |
|---|---|---|
| `label` | the A/B files | which arm the row was measured under; see below |
| `capture` | `workload_*.csv` | which model run the shape came from |
| `call_count` | `workload_counts.csv` | how many times that shape was issued during the capture |
| `version` | `gfx942/kernel_names.csv` | the ROCm release the row was measured under |
| `kernel`, `def_kernel`, `best_kernel` | the kernel-name files | the dispatched Tensile kernel name |
| `metric` | `performance_metric.csv` | argument to `rocblas_set_performance_metric()`: 0 default, 1 and 2 the alternatives |
| `pad`, `cneqd` | `tensile_predicates.csv` | leading-dimension padding, and whether C and D are distinct pointers. Both are Tensile predicates that could in principle disqualify fast kernels |
| `n_all`, `n_noatomic` | `determinism.csv` | candidate count with and without `rocblas_atomics_not_allowed` |
| `def_repro`, `best_repro` | `determinism.csv` | 1 if the arm produced bit-identical output across five runs |
| `ms_def`, `ms_best`, `ms_best_noatomic` | `determinism.csv` | default, best, and best-with-atomics-forbidden. The ratio `ms_best_noatomic / ms_best` is the figure to use: median 0.9997, worst 1.010 |
| `seq` | `unbiased.csv` | position in the randomised measurement order, recorded so drift across the sweep can be checked (first half 1.0037, second half 1.0043) |
| `sol` | `unbiased.csv` | the solution index handed to the re-measurement. It is not searched for again — that is the point of the control |
| `new_gain_min` | `unbiased.csv` | the re-measured gain using the minimum instead of the median, showing the choice costs 0.03% |
| `best_in_noatomic_set` | `determinism.csv` | **misleading name**: it records whether two searches picked the same solution index, not set membership. Use `ms_best_noatomic / ms_best` instead |
| `old_gain`, `new_gain` | `unbiased.csv` | gain as originally measured, and re-measured in a fresh process with no selection |
| `def_med`, `best_med`, `def_min`, `best_min` | `unbiased.csv` | median and minimum of the re-measurement, to show that reporting the minimum changes nothing (0.03%) |

## `q2_families.csv` — the answer to question 2

One row per shape, carrying the same problem in both coordinate systems so the mapping is explicit
rather than something a reader has to derive.

| column | meaning |
|---|---|
| `family` | `skinny-M` (M ≤ 16, N and K ≥ 1024) or `large-K` (K ≥ 8192, min(M,N) ≥ 128) |
| `harness_M`, `harness_N`, `K`, `harness_opA`, `harness_opB` | the problem as the harness states it: row-major `C = A·B` |
| `rocblas_M`, `rocblas_N`, `rocblas_transA`, `rocblas_transB` | the same problem as rocBLAS receives it, operands swapped. `rocblas_M` = `harness_N`, `rocblas_transA` = `harness_opB`. `K` is the same in both |
| `n_solutions` | candidates returned by `rocblas_gemm_ex_get_solutions` |
| `rocblas_solution_index` | the winner, rocBLAS-encoded. Negative means a Tensile-backed solution |
| `tensile_index_for_override` | the same winner converted for the override file: `-index - 10`. It would read `hipBLASLt-backed` for a winner that is not a Tensile solution, since a Tensile override table cannot express one — but all 746 winners here are Tensile-backed, so every row is usable as it stands |
| `default_ms`, `best_ms`, `gain` | as elsewhere |

## `q2_override.csv` — the same rows in `rocblas-gemm-tune` format

Entirely in rocBLAS coordinates, because that is what the override parser reads. Column order and
names follow the Programmer's Guide so the file can be used without editing.

| column | meaning |
|---|---|
| `transA`, `transB` | rocBLAS operations, i.e. the harness's `opB` and `opA` |
| `M`, `N`, `K`, `batch_count` | problem size in rocBLAS coordinates; `batch_count` is 1 throughout |
| `alpha`, `beta` | 1 and 0 throughout |
| `lda`, `ldb`, `ldc` | leading dimensions for the swapped operand order |
| `input_type`, `output_type`, `compute_type` | `f32_r` / `bf16_r` / `f16_r` input; output and compute are `f32_r` throughout |
| `solution_index` | the raw Tensile index, positive and 1-based — already converted, so the file works as it stands |

## Arm labels

| label | meaning |
|---|---|
| `unset`, `hipblaslt` | `ROCBLAS_USE_HIPBLASLT` unset versus `=1`, on the default path only |
| `tensile` | `ROCBLAS_USE_HIPBLASLT=0`, forcing the Tensile backend |
| `hipblaslt_b` | `=1` plus `ROCBLAS_USE_HIPBLASLT_BATCHED=1` |
| `plain`, `override` | default dispatch versus `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` pointed at a generated table |
| `..._n13`, `..._n18` | the same arm measured on **two different nodes** — independent replicates, not two settings |
| `..._r1`, `..._r2` | two repeats in separate processes on one node |

## Repeated records

Eighteen records across six files measure a shape that appeared twice in the input list: 4 each in
the three larger `hipblaslt_ab_*` files, 2 in `hipblaslt_ab_fp32_v2.csv`, 1 in
`gfx942/fullspace.csv`, 3 in `data/tune_table.csv` at the repository root. They are independent
repeats, not duplicated rows, and they are left in rather than deduplicated.

This matters only if you build a dictionary keyed by shape, because the row you keep then depends on
whether you assign or `setdefault`. Checked both ways: every published median is identical to six
decimal places either way. The analysis scripts here keep the first.

Do not confuse this with the deliberate repeats: `hipblaslt_ab_fp32.csv` carries `_r1` and `_r2`
labels, two full passes over the whole population in separate processes, which is how the reported
run-to-run agreement of 1.007 was measured.

## Coordinate note

The harnesses compute row-major `C = A·B` by calling column-major rocBLAS with the operands swapped:

    gemm_ex(opB, opA, N, M, K, pB, ldb, pA, lda, ...)

So rocBLAS's `M` is the harness's `N`, and rocBLAS's `transA` is the harness's `opB`. Measured
consequence: harness `NT` dispatches to Tensile `Alik_Bljk` and harness `TN` to `Ailk_Bjlk` — the
opposite of what the labels suggest. `NN` and `TT` are symmetric under the swap.

`q2_families.csv` is the one file that carries both coordinate systems, in columns prefixed
`harness_` and `rocblas_`. `q2_override.csv` is entirely in rocBLAS coordinates, because that is what
the override file format requires.
