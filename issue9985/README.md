# Data and tools for ROCm/rocm-libraries#9985

Every figure quoted in the issue thread, with the file it came from, so it can be recomputed rather
than taken on trust. Two architectures throughout: **gfx90a** (MI250, one GCD, ROCm 7.2.3) and
**gfx942** (MI300X, rented pod).

Column definitions for every CSV are in [COLUMNS.md](COLUMNS.md).

## The two questions asked in the thread

| file | what it is |
|---|---|
| `data/q2_families.csv` | 746 rows — 150 skinny-M (M ≤ 16, N,K ≥ 1024) and 596 large-K (K ≥ 8192, min(M,N) ≥ 128), each with the winning solution index in **both** encodings, candidate count, both timings, gain |
| `data/q2_override.csv` | the same rows already converted for `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH`, in `rocblas-gemm-tune` column order |
| `evidence/q1_logic_file_dispatch.tsv` | which Tensile logic file rocBLAS loads, per transpose and dtype, from `TENSILE_DB=0x40` |
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
| the sweep the headline medians come from | `data/gfx90a/fullspace.csv` (7009) | `data/gfx942/fullspace.csv` (4776) |
| five ROCm releases pick the same kernel | `data/gfx90a/versions/rocm_*.csv` (5 × 240) and `kernel_names.csv` | `data/gfx942/versions/rocm_*.csv` (5 × 239) and `kernel_names.csv` |
| real-workload cost, call-count weighted | `data/gfx90a/workload_gain.csv` + `workload_counts.csv` (338 shapes, 11 captures of 7 models) | `data/gfx942/workload.csv` — same 338, joinable on `capture` + shape |
| the override fixes it, end to end | `data/gfx90a/workload_override_ab.csv` (3 captures × 3 arms) | — |
| `ROCBLAS_USE_HIPBLASLT` A/B | `hipblaslt_ab_{fp32,bf16fp16}.csv`, and `*_v2.csv` re-measured under the current method | — |
| selection bias: no search, fresh process, median | `data/gfx90a/unbiased.csv` (60) | `data/gfx942/unbiased.csv` (60) |
| levers that do **not** help | `performance_metric.csv` (1200 × 3), `tensile_predicates.csv` (450 × 8), `determinism.csv` (39), `batched_gemm.csv` (231 × 4) | — |
| the fp16 TT band, isolated and repeated | `fp16_{tt,nn}_band_run{1,2,3}.csv` | — |

## Tools

| file | notes |
|---|---|
| `src/tune_trans.cpp` | the sweep harness: rotated buffers (`--rotate`), time-based repetition (`--minms`), uniform [-1,1] fill, and a clock probe before and after each shape |
| `src/three_way2.cpp` | four-arm comparison including **exhaustive** hipBLASLt enumeration via `hipblaslt_ext::getAllAlgos`, because "best of the heuristic shortlist" turned out to be a function of the shortlist length |
| `src/backend_v2.cpp` | `ROCBLAS_USE_HIPBLASLT` A/B on the default path only |
| `src/cmparch3.py` | counts tuned points in the shipped `.dat` files. **No GPU needed** — install ROCm and run it |
| `src/answer_q2.py` | builds the two Q2 tables, including the index conversion and the coordinate swap |
| `src/analyze_abc.py` | the verification pass described in `evidence/verification_2026-08-13.md` |

## Reproducing

**Without a GPU.** `pip install msgpack`, then:

```
python3 src/cmparch3.py [/opt/rocm/lib/rocblas/library]
```

prints the tuned-point table from an installed rocBLAS, no hardware required.

**With one GPU.** Any CSV under `data/` doubles as a shape list — the harness reads the first six
columns and skips lines it cannot parse, including the header:

```
hipcc -O3 -Wno-deprecated-declarations --offload-arch=gfx90a \
      src/tune_trans.cpp -o tune_trans -lrocblas

cut -d, -f1-6 data/gfx90a/fullspace.csv > shapes.txt      # 7009 shapes, hours
head -200 shapes.txt > small.txt                          # or a subset first
./tune_trans small.txt out.csv --rotate 4 --minms 10
```

Two things about the output are deliberate and easy to trip over:

- **`out.csv` is opened for append.** Re-running adds rows rather than replacing them, which is what
  makes an interrupted sweep resumable; delete the file for a clean run.
- **Shapes whose enumeration would exceed `--budget` seconds are skipped**, with the count printed at
  the end. A skipped shape is simply absent from the CSV, so compare row counts against the input if
  that matters.

**Pass the flags.** `--rotate 4 --minms 10` are the settings the published data uses; the defaults are
`--rotate 1 --minms 0`, which is the earlier method. The distinction is not visible in the output —
the `nrot` and `reps` columns are written either way, just with `nrot=1` — so a run without flags
produces a file that looks current and was measured the old way. `nrot` is the column that tells them
apart.

**Checking the override.** Point `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` at `data/q2_override.csv` and
verify by kernel name with `TENSILE_DB=0x8000`. Never verify it by timing: a silently ignored
override and a working-but-useless one are identical on a stopwatch.

## Two things worth knowing before reusing this

**Two measurement methods are present, and `nrot` is what tells them apart.** Files with an `nrot`
column were produced by the current harness, and every one of them has `nrot=4`: four rotated buffer
sets, timed windows of at least 10 ms, uniform [-1,1] fill. Files without the column predate it —
single buffer, five repetitions, constant fill. Do not read the column's *presence* as the marker,
because the harness writes it in either configuration; `nrot=1` would mean the earlier method run
through the current binary.

Every figure from the older method was re-measured on the same shapes; paired per shape, nothing
moved by more than 3% (`evidence/verification_2026-08-13.md`). The shape populations differ between
files, so pair per shape before concluding that anything changed — comparing a median from one
population against a constant from another is the mistake this whole report is about.

**Counting tuned points.** The `.dat` tables contain duplicate keys, unevenly across architectures:
the gfx90a fp32 table holds 9874 entries over 4888 **distinct** (M,N,K) points, while gfx942's 415
entries are 345 distinct. Counting entries overstates the gfx90a/gfx942 ratio by about 40%.
`cmparch3.py` reports both.
