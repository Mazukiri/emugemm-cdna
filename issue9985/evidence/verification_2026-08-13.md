# Re-verification of every carried-over number, 2026-08-13

Each item below was recomputed from data rather than copied forward. Three moved; two reproduced
exactly. Commands and sources are given so the next person does not have to guess the choices.

## Reproduced exactly

**§5 version matrix** — from `mi250b/rand_{c63,c64,r700,714}.csv` and `B_n13.csv`, 240 shapes each:

| ROCm | median | >10% | max | mean nsol | paired vs 7.2.3 |
|---|---|---|---|---|---|
| 6.3.0 | 1.251 | 66% | 10.64 | 262 | 1.0006 |
| 6.4.0 | 1.238 | 67% | 10.71 | 262 | 1.0014 |
| 7.0.0 | 1.257 | 66% | 10.74 | 247 | 1.0006 |
| 7.2.3 | 1.258 | 66% | 10.76 | 247 | — |
| 7.14 | 1.270 | 67% | 19.26 | 247 | 1.0027 |

**§9 audit** — from the 820-row `tune_table.csv`: fp32 mean 1.325 / median 1.154 / 62% / 9%;
bf16 1.372 / 1.187 / 73% / 10%; fp16 1.283 / 1.135 / 61% / 11%.

## Corrected

**§3 grid density was counting table entries, not distinct points.** Entries / distinct keys:

| arch | fp32 | bf16 | fp16 |
|---|---|---|---|
| gfx908 | 5472 / 5098 | 850 / 694 | 2583 / 2382 |
| gfx90a | 9874 / 4888 | 3346 / 2499 | 5002 / 2862 |
| gfx942 | 415 / 345 | 210 / 210 | 213 / 213 |
| gfx950 | 341 / 274 | 35 / 35 | 51 / 51 |

gfx90a/gfx942 falls from 23.8x to **14.2x** (fp32), 15.9x to 11.9x (bf16), 23.5x to 13.4x (fp16).
The gfx90a tables duplicate keys about 2x; gfx942's barely at all, so the duplication was inflating
one side of the comparison. Ordering unchanged.

**§3 Spearman is +0.371, not 0.412.** 253 fp32 shapes, log-distance to the nearest tuned point in
rocBLAS coordinates against measured gain. Nearest quartile median 1.093 / 44% losing >10%; farthest
1.489 / 72%. In harness coordinates it is +0.384 — the difference is small, but the choice has to be
stated. Choosing `CU104` or generic makes no difference at all: the CU104 grid is a **strict subset**
(4859 of 4888 distinct keys) and the 29 extra points are tiny cubes, never a nearest neighbour here.

**§7 sample sizes were rows quoted as shapes.** Metric sweep: 1200 shapes x 3 settings. Predicate
sweep: 450 shapes x 8 variants. Atomics: 39 shapes, not 37. Conclusions unchanged — metric median
0.9993 with candidate counts differing on 0 shapes; predicates leave the mean candidate count at
249.4 across all eight variants with 0 of 3150 rows losing a candidate; atomics median 0.9997, worst
1.010, 0/39 worse than 5%.

**§7 batched** — `USE_HIPBLASLT_BATCHED=1` vs `USE_HIPBLASLT=1` alone is 1.0011 paired over 231
shapes. Paired `unset/hipblaslt` on the batched path: bf16 0.521, fp32 0.919, fp16 1.103 — so
hipBLASLt is slower for fp32 here while it is faster for fp32 on the non-batched path.

## Not carried into the reply, and should not be reused

`comment2_FINAL.md` §6 contrasts "large square" (mean 1.051, 20% losing >10%) with "skinny or
large-K" (1.505, 76%). The first cell is **M == N == K exactly, n = 5**. The complement of that
definition is 1.330 / 62% over 248 shapes, so the contrast as written is between a 5-shape cell and
an unstated subset. Do not reuse it in the reply or the paper without redefining both sides.

## Corrected after publication, 2026-08-14

The cross-architecture table in the thread's workload section quoted **31.6%** for gfx90a where the
section heading two paragraphs earlier said **31.9%** — the same quantity, two numbers. The 31.6%
came from an ad-hoc join that keyed on shape alone; because 37 of the 338 shapes occur in more than
one capture, that collapses them and attaches the wrong call count.

Joining on `(capture, shape)`, which matches 338/338, and computed twice by independent routes — once
from the per-capture source files, once from the merged files published here:

| | gfx90a | gfx942 |
|---|---|---|
| GEMM time recoverable, call-count weighted | **31.90%** | **36.79%** |

The thread now reads 31.9% and 36.8%. The gfx942 figure moved by 0.1 points for the same reason: its
file had no `capture` column until this pass, so the join could not be exact.

The leave-one-capture-out range quoted alongside it, 30.4%–34.5%, is confirmed unchanged: the extremes
are dropping llama-1b-3 and dropping gemma-3-12b respectively.

## Fresh-eyes pass over the posted comment, 2026-08-14

Every quantitative claim in the thread was re-derived from the files published here. Twenty-two
figures in sections 1, 2, 4, 5 and 6 matched exactly. Four things did not, and were corrected in the
thread:

1. **31.6% versus 31.9%** for the same quantity, twenty lines apart — recorded above.
2. **Section 1 counted table entries while section 3 counted distinct keys.** Both now use distinct
   keys, which moves the standout figure from 53× to 40× and surfaces two facts the entry counts
   hid: the pure-bf16 table has 8 distinct points for the fallback transpose, and the CU104 tables
   are smaller than the generic ones in entries while carrying nearly the same distinct count.
3. **"0 of 240 shapes"** where the file has 239 rows, in two places.
4. **A version-matrix claim with no file behind it** — "no version differs by more than 0.3%" — now
   names its source and its worst case (6.3.4, 0.26%).

Checks that passed and are worth recording as passed: the 24/5/1/0 override tally re-derives from
the raw macro-tile columns without trusting the status column each row carries; the Q1 dispatch log
supports its claim on all twelve rows; section 5's kernel-name check is 12/12 against three releases;
tables are column-balanced, section numbering is contiguous, every §-reference points at a section
that exists, and every repository link resolves.
