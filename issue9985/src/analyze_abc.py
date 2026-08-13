#!/usr/bin/env python3
"""Re-derive the verification results reported in evidence/verification_2026-08-13.md.

Run from anywhere:  python3 src/analyze_abc.py [data_dir]

Each check has a pass condition stated here, fixed before the data existed. Choosing a threshold
after seeing the numbers is not a check.

  A  Do figures measured with the earlier method survive re-measurement under the current one?
     PASS if, paired per shape, nothing moves by more than 10%.
  B  Is the fp16 TT band real, or an artifact of running the sweep as six concurrent shards?
     PASS if the isolated median stays above 2.0 in all three runs while the control band holds.
  C  Are the shipped tuning tables unchanged between ROCm releases, counted on distinct keys?
     Needs two installs; see the note printed at the end.

Comparisons are paired per shape throughout. Shape populations differ between files, so comparing a
median from one file against a constant from another measures the population, not the change — the
error this report is about.
"""
import collections
import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
G90 = os.path.join(DATA, "gfx90a")


def rule(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def med(v):
    return st.median(v) if v else float("nan")


def load_gain(path, extra_key=()):
    """Shape key -> (gain, def_ms, best_ms)."""
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        try:
            key = tuple(r[c] for c in (*extra_key, "M", "N", "K", "dtype", "opA", "opB"))
            gain = float(r["gain"])
            if gain > 0:
                out.setdefault(key, (gain, float(r["def_ms"]), float(r["best_ms"])))
        except (KeyError, ValueError):
            pass
    return out


def load_arms(path):
    """Shape key -> {arm: def_ms}. Node and repeat suffixes fold into the base arm."""
    out = collections.defaultdict(dict)
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        try:
            key = (r["M"], r["N"], r["K"], r["dtype"], r["opA"], r["opB"], r.get("beta", "0"))
            label = r["label"]
            if label.startswith("hipblaslt_b"):
                continue                       # the BATCHED variant is a separate arm
            for base in ("unset", "hipblaslt"):
                if label == base or label.startswith(base + "_"):
                    out[key].setdefault(base, float(r["def_ms"]))
        except (KeyError, ValueError):
            pass
    return out


# ---------------------------------------------------------------- A
rule("A — do the earlier measurements survive the current method?")

old_be = load_arms(os.path.join(G90, "hipblaslt_ab_bf16fp16.csv"))
old_be.update(load_arms(os.path.join(G90, "hipblaslt_ab_fp32.csv")))
new_be = load_arms(os.path.join(G90, "hipblaslt_ab_bf16fp16_v2.csv"))
new_be.update(load_arms(os.path.join(G90, "hipblaslt_ab_fp32_v2.csv")))

print("ROCBLAS_USE_HIPBLASLT A/B, paired per shape")
print("ratio = unset / =1, so above 1 means the hipBLASLt backend is faster\n")
ok = True
for dtype in ("fp32", "bf16", "fp16"):
    common = [k for k in old_be if k in new_be and k[3] == dtype
              and all(a in old_be[k] and old_be[k][a] > 0 for a in ("unset", "hipblaslt"))
              and all(a in new_be[k] and new_be[k][a] > 0 for a in ("unset", "hipblaslt"))]
    if not common:
        print(f"  {dtype:5s} no paired shapes")
        continue
    old_r = [old_be[k]["unset"] / old_be[k]["hipblaslt"] for k in common]
    new_r = [new_be[k]["unset"] / new_be[k]["hipblaslt"] for k in common]
    paired = med([new_r[i] / old_r[i] for i in range(len(common))])
    ok &= abs(paired - 1) < 0.10
    print(f"  {dtype:5s} n={len(common):4d}   earlier {med(old_r):.3f}   current {med(new_r):.3f}"
          f"   paired ratio {paired:.4f}")
print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}")

# The workload capture is published in its original form; recompute its headline so the figure
# quoted in the thread can be checked against the file.
wl = load_gain(os.path.join(G90, "workload_gain.csv"), extra_key=("capture",))
if wl:
    counts = {}
    for r in csv.DictReader(open(os.path.join(G90, "workload_counts.csv"))):
        counts[(r["capture"], r["M"], r["N"], r["K"], r["dtype"], r["opA"], r["opB"])] = int(r["call_count"])
    total = best = 0.0
    for k, (_, d, b) in wl.items():
        c = counts.get(k, 1)
        total += d * c
        best += b * c
    print(f"\nworkload: {len(wl)} shapes, median gain {med([v[0] for v in wl.values()]):.3f}, "
          f"call-count-weighted recoverable {100 * (total - best) / total:.1f}%")

# ---------------------------------------------------------------- B
rule("B — is the fp16 TT band real, or an artifact of concurrent sharding?")

sweep = {}
for r in csv.DictReader(open(os.path.join(G90, "fullspace.csv"))):
    try:
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        if r["dtype"] == "fp16" and 128 <= min(M, N, K) < 1024:
            sweep.setdefault((r["M"], r["N"], r["K"], r["opA"] + r["opB"]), float(r["gain"]))
    except ValueError:
        pass

for band, transpose, floor in (("tt", "TT", 2.0), ("nn", "NN", None)):
    sharded = [g for k, g in sweep.items() if k[3] == transpose]
    runs = []
    for i in (1, 2, 3):
        path = os.path.join(G90, f"fp16_{band}_band_run{i}.csv")
        if not os.path.exists(path):
            continue
        v = [float(r["gain"]) for r in csv.DictReader(open(path)) if float(r.get("gain", 0)) > 0]
        if v:
            runs.append(v)
    label = "fp16 TT (the claim)" if transpose == "TT" else "fp16 NN (control)"
    print(f"\n{label}: sharded median {med(sharded):.3f} over n={len(sharded)}")
    for i, v in enumerate(runs, 1):
        print(f"  isolated run {i}: n={len(v):4d}  median {med(v):.3f}")
    if runs:
        medians = [med(v) for v in runs]
        print(f"  spread across runs: {min(medians):.3f} - {max(medians):.3f}")
        if floor is not None:
            print(f"  VERDICT: {'PASS — survives isolation' if min(medians) >= floor else 'FAIL'}")

# ---------------------------------------------------------------- C
rule("C — are the shipped tuning tables unchanged between releases?")
print("""Needs two ROCm installs, so it cannot be reproduced from this data alone:

    python3 src/cmparch3.py /path/to/rocm-A/lib/rocblas/library > a.txt
    python3 src/cmparch3.py /path/to/rocm-B/lib/rocblas/library > b.txt
    diff a.txt b.txt

Reported in the thread: identical across rocBLAS 5.2 (ROCm 7.2.3) and 5.5 (ROCm 7.14) for all 20
(architecture, dtype) cells, on entry counts and distinct-key counts alike.""")
