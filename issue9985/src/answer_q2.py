#!/usr/bin/env python3
"""Answer @NaveenElumalaiAMD's question 2 on ROCm/rocm-libraries#9985 from data already measured.

He asked to extend the skinny-M and large-K families with "the winning solution index per row",
suggesting ROCBLAS_LAYER=4 -> yaml -> rocblas-gemm-tune. That tool is not installed in the cluster
container, but the round already recorded the winning index for every shape: `tune_trans` calls
rocblas_gemm_ex_get_solutions(), times every candidate, and writes the winner as `best_sol`.

Two conversions stand between that column and something he can paste into a build, and both are
silent when wrong, which is why this is a script and not a spreadsheet:

  1. INDEX ENCODING. get_solutions() returns the rocBLAS-encoded index, which is NEGATIVE when the
     winner is a Tensile solution. The override parser wants the raw Tensile index, positive and
     1-based: MasterSolutionLibrary.hpp does getSolutionByIndex(value - 1). Inverting
     map_index_tensile_to_rocblas gives  file_value = -rocblas_index - 10,  where 10 is
     c_rocblas_solutions_reserved (definitions.hpp) - an internal constant that appears in no
     documentation. Feed the raw negative value in and rocBLAS prints one warning line and then
     silently keeps the default kernel.

  2. COORDINATES. This harness computes row-major C = A*B by calling column-major rocBLAS with the
     operands swapped: gemm_ex(opB, opA, N, M, K, pB, ldb, pA, lda, ...). So an override key in
     harness coordinates matches nothing. rocBLAS M <- harness N, rocBLAS transA <- harness opB.

Outputs two files:
  q2_families.csv   one row per shape: shape in BOTH coordinate systems, gain, both index forms
  q2_override.csv   ready to feed to ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH, all four transposes

Usage: python3 answer_q2.py [outdir]
"""
import csv, glob, os, statistics as st, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
RESERVED = 10                      # c_rocblas_solutions_reserved
TY = {"fp32": "f32_r", "bf16": "bf16_r", "fp16": "f16_r"}

# The gfx90a sweep, which carries buffer rotation and time-based repetition — the measurement method
# the reported gains have to stand behind.
SRC = os.path.join(DATA, "gfx90a", "fullspace.csv")
rows = []
if True:
    f = SRC
    for r in csv.DictReader(open(f)):
        try:
            rows.append(dict(M=int(r["M"]), N=int(r["N"]), K=int(r["K"]), dt=r["dtype"],
                             oa=r["opA"], ob=r["opB"], nsol=int(r["nsol"]),
                             sol=int(r["best_sol"]), gain=float(r["gain"]),
                             best_ms=float(r["best_ms"]), def_ms=float(r["def_ms"]),
                             src=os.path.basename(f)))
        except (ValueError, KeyError):
            pass

# de-duplicate: the shards overlap at their boundaries, and a shape counted twice would weight the
# family medians without changing any single number, which is the hardest kind of error to notice
seen, uniq = set(), []
for r in rows:
    k = (r["M"], r["N"], r["K"], r["dt"], r["oa"], r["ob"])
    if k in seen:
        continue
    seen.add(k); uniq.append(r)
print(f"v2 fullspace: {len(rows)} rows, {len(uniq)} unique shapes ({len(rows)-len(uniq)} duplicates dropped)")

FAM = [
    ("skinny-M", lambda r: r["M"] <= 16 and r["N"] >= 1024 and r["K"] >= 1024),
    ("large-K",  lambda r: r["K"] >= 8192 and min(r["M"], r["N"]) >= 128),
]

def tensile_index(sol):
    """rocBLAS-encoded -> raw Tensile index for the override file. Positive means the winner was a
    hipBLASLt solution, which the Tensile override table cannot express; report, never convert."""
    return (-sol - RESERVED) if sol < 0 else None

os.makedirs(OUT, exist_ok=True)
fam_rows = []
for name, sel in FAM:
    sub = [r for r in uniq if sel(r)]
    sub.sort(key=lambda r: -r["gain"])
    g = [r["gain"] for r in sub]
    nhip = sum(1 for r in sub if r["sol"] >= 0)
    print(f"{name:9s} n={len(sub):4d}  median gain {st.median(g):.3f}  "
          f">10% on {100*sum(1 for x in g if x > 1.1)/len(g):.0f}%  max {max(g):.2f}  "
          f"winner is hipBLASLt on {nhip}")
    for r in sub:
        r["family"] = name
        fam_rows.append(r)

with open(os.path.join(OUT, "q2_families.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["family", "harness_M", "harness_N", "K", "dtype", "harness_opA", "harness_opB",
                "rocblas_M", "rocblas_N", "rocblas_transA", "rocblas_transB",
                "n_solutions", "rocblas_solution_index", "tensile_index_for_override",
                "default_ms", "best_ms", "gain"])
    for r in fam_rows:
        ti = tensile_index(r["sol"])
        w.writerow([r["family"], r["M"], r["N"], r["K"], r["dt"], r["oa"], r["ob"],
                    r["N"], r["M"], r["ob"], r["oa"],
                    r["nsol"], r["sol"], ti if ti is not None else "hipBLASLt-backed",
                    f"{r['def_ms']:.5f}", f"{r['best_ms']:.5f}", f"{r['gain']:.4f}"])

hdr = ["transA", "transB", "M", "N", "batch_count", "K", "alpha", "beta",
       "lda", "ldb", "ldc", "input_type", "output_type", "compute_type", "solution_index"]
with open(os.path.join(OUT, "q2_override.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr); n = 0
    for r in fam_rows:
        ti = tensile_index(r["sol"])
        if ti is None:
            continue
        ta, tb = (r["oa"] == "T"), (r["ob"] == "T")
        w.writerow([("T" if tb else "N"), ("T" if ta else "N"),
                    r["N"], r["M"], 1, r["K"], 1, 0,
                    (r["K"] if tb else r["N"]),     # lda <- the first matrix passed (pB)
                    (r["M"] if ta else r["K"]),     # ldb <- the second (pA)
                    r["N"],
                    TY[r["dt"]], "f32_r", "f32_r", ti])
        n += 1
print(f"\nwrote {OUT}/q2_families.csv ({len(fam_rows)} rows) and {OUT}/q2_override.csv ({n} rows)")

print("""
These indices are chosen by timing. Confirming that they are also DISPATCHED takes one command per
shape on a machine with the library:

  TENSILE_DB=0x8000 ./kname <M> <N> <K> <dtype> <opA> <opB> 0 <rocblas_solution_index>
  TENSILE_DB=0x8000 ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=data/q2_override.csv ./kname ...

The override took effect only if the macro tile in phase 1 changes to the phase-2 winner. A silently
ignored override and a working-but-useless one are identical on a stopwatch. Results of that check
over a 30-row sample: evidence/q2_override_dispatch.tsv.""")
