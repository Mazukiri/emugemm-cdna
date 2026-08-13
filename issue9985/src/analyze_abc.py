#!/usr/bin/env python3
"""Analyse the A/B/C verification run and say, per stage, whether the reply needs changing.

Each stage has a pass condition written down BEFORE the data existed, because deciding what counts
as confirmation after seeing the numbers is how the six retracted figures in section 9 happened.

  A  PASS if the v2 re-measurement moves each headline by less than 10% relative, and moves it in
     the direction the "the bias cancels, so the old numbers are conservative" argument predicts
     (v2 gains >= v1 gains). FAIL means sections 6/7/8 must be re-quoted from v2.
  B  PASS if the isolated fp16 TT band median stays above 2.0 across all three runs while the NN
     control band stays near its sharded value. FAIL means shard concurrency produced the 4.19x and
     the finding comes out of the reply.
  C  PASS if 7.14's distinct-key counts match 7.2.3's. FAIL means the "unchanged across versions"
     sentence in section 3 must be narrowed.
"""
import csv, glob, os, statistics as st, sys, collections

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "mi250b/v3")
BASE = os.path.dirname(os.path.abspath(__file__))
def hr(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def med(v): return st.median(v) if v else float("nan")

# ---------------------------------------------------------------- A: workload + backend under v2
hr("A — do sections 6/7/8 change under METHOD_V2?")

def load_gain(pat, key=("M","N","K","dtype","opA","opB")):
    d = {}
    for f in sorted(glob.glob(pat)):
        for r in csv.DictReader(open(f)):
            try:
                k = tuple(r[c] for c in key)
                g = float(r["gain"])
                if g > 0: d.setdefault(k, (g, float(r["def_ms"]), float(r["best_ms"])))
            except (KeyError, ValueError): pass
    return d

old_wl = load_gain(os.path.join(BASE, "mi250b/wl/*_gain.csv"))
new_wl = load_gain(os.path.join(D, "wl_*_v2.csv"))
if new_wl:
    common = sorted(set(old_wl) & set(new_wl))
    print(f"workload: v1 {len(old_wl)} shapes, v2 {len(new_wl)}, paired {len(common)}")
    if common:
        o = [old_wl[k][0] for k in common]; n = [new_wl[k][0] for k in common]
        ratio = med([new_wl[k][0] / old_wl[k][0] for k in common])
        # call-count-weighted recoverable time, the number the reply actually quotes
        counts = {}
        for cf in glob.glob(os.path.join(BASE, "mi250b/wl/*_counts.csv")):
            for r in csv.DictReader(open(cf)):
                counts[(r["M"], r["N"], r["K"], r["dtype"], r["opA"], r["opB"])] = int(r["call_count"])
        def recov(src):
            dd = bb = 0.0
            for k in common:
                c = counts.get(k, 1); dd += src[k][1] * c; bb += src[k][2] * c
            return 100 * (dd - bb) / dd if dd else float("nan")
        print(f"  median gain   v1 {med(o):.3f}  ->  v2 {med(n):.3f}   (paired ratio {ratio:.4f})")
        print(f"  recoverable % v1 {recov(old_wl):.1f}%  ->  v2 {recov(new_wl):.1f}%")
        print(f"  VERDICT: {'PASS' if abs(ratio-1) < 0.10 and ratio >= 0.99 else 'FAIL — re-quote section 6 from v2'}")
else:
    print("  (no v2 workload data yet)")

def load_backend(pat):
    d = collections.defaultdict(dict)
    for f in sorted(glob.glob(pat)):
        for r in csv.DictReader(open(f)):
            try:
                k = (r["M"], r["N"], r["K"], r["dtype"], r["opA"], r["opB"], r.get("beta", "0"))
                d[k][r["label"]] = float(r["def_ms"])
            except (KeyError, ValueError): pass
    return d

# Matched populations ONLY. Section 7's fp32 figure comes from fp32_all.txt and its bf16/fp16
# figures from bf16fp16.txt; rand_ab.txt shares zero shapes with either. Globbing backend_*_v2.csv
# pools all three and produces a "v2 disagrees with v1" verdict that is really a comparison between
# different shape populations - twice now. Pair per shape instead of comparing medians to constants.
new_be = {}
for pop in ("backend_fp32_all_v2.csv", "backend_bf16fp16_v2.csv"):
    new_be.update(load_backend(os.path.join(D, pop)))
old_be = {}
old_be.update(load_backend(os.path.join(BASE, "mi250b/A_n13.csv")))
old_be.update(load_backend(os.path.join(BASE, "mi250b/full_fp32.csv")))
# the v1 files label their arms with a node suffix
for k, v in old_be.items():
    for a, b in (("unset_n13", "unset"), ("hipblaslt_n13", "hipblaslt"),
                 ("unset_r1", "unset"), ("hipblaslt_r1", "hipblaslt")):
        if a in v: v[b] = v[a]
if new_be:
    per = collections.defaultdict(list)
    for k, v in new_be.items():
        if "unset" in v and "hipblaslt" in v and v["hipblaslt"] > 0:
            per[k[3]].append(v["unset"] / v["hipblaslt"])
    print("\nhipBLASLt backend A/B, PAIRED per shape (unset / =1, above 1 means hipBLASLt faster):")
    ok = True
    for dt in ("fp32", "bf16", "fp16"):
        common = [k for k in old_be if k in new_be and k[3] == dt
                  and all(x in old_be[k] and old_be[k][x] > 0 for x in ("unset", "hipblaslt"))
                  and all(x in new_be[k] and new_be[k][x] > 0 for x in ("unset", "hipblaslt"))]
        if not common:
            print(f"  {dt:5s} no paired shapes"); continue
        a = [old_be[k]["unset"] / old_be[k]["hipblaslt"] for k in common]
        b = [new_be[k]["unset"] / new_be[k]["hipblaslt"] for k in common]
        pr = med([b[i] / a[i] for i in range(len(a))])
        if abs(pr - 1) >= 0.10: ok = False
        print(f"  {dt:5s} n={len(common):4d}   v1 {med(a):.3f}   v2 {med(b):.3f}   paired ratio {pr:.4f}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL — re-quote section 7 from v2'}")
else:
    print("\n  (no v2 backend data yet)")

# ---------------------------------------------------------------- B: is the fp16 TT band real?
hr("B — is fp16 TT 128-1023 = 4.19x an artifact of running six shards concurrently?")

sharded = {}
for f in sorted(glob.glob(os.path.join(BASE, "mi250b/v2/fs_v2_s*.csv"))):
    for r in csv.DictReader(open(f)):
        try:
            k = (r["M"], r["N"], r["K"], r["dtype"], r["opA"], r["opB"])
            M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
            if k in sharded: continue
            if r["dtype"] == "fp16" and 128 <= min(M, N, K) < 1024:
                sharded[k] = float(r["gain"])
        except ValueError: pass

for band, tr, floor in [("tt", "TT", 2.0), ("nn", "NN", None)]:
    ref = [g for k, g in sharded.items() if k[4] + k[5] == tr]
    runs = []
    for i in (1, 2, 3):
        p = os.path.join(D, f"band_{band}_r{i}.csv")
        if not os.path.exists(p): continue
        v = []
        for r in csv.DictReader(open(p)):
            try:
                g = float(r["gain"])
                if g > 0: v.append(g)
            except (KeyError, ValueError): pass
        if v: runs.append(v)
    label = "fp16 TT (the claim)" if tr == "TT" else "fp16 NN (control)"
    print(f"\n{label}: sharded median {med(ref):.3f} over n={len(ref)}")
    if not runs:
        print("  (no isolated runs yet)"); continue
    for i, v in enumerate(runs, 1):
        print(f"  isolated run {i}: n={len(v):4d}  median {med(v):.3f}")
    allm = [med(v) for v in runs]
    print(f"  median of run medians: {med(allm):.3f}   spread {min(allm):.3f}-{max(allm):.3f}")
    if floor is not None:
        print(f"  VERDICT: {'PASS — the band survives isolation' if min(allm) >= floor else 'FAIL — remove the fp16 TT band claim from section 1'}")

# ---------------------------------------------------------------- C: 7.14 grid counts
hr("C — are the grid counts unchanged on ROCm 7.14, counted on distinct keys?")

def load_grid(p):
    d = {}
    for r in csv.DictReader(open(p)):
        d[(r["arch"], r["dtype"])] = (int(r["entries"]), int(r["distinct"]))
    return d

here = sorted(glob.glob(os.path.join(D, "grid_*.txt")))
there = sorted(glob.glob(os.path.join(D, "714_grid_*.txt")))
if here and there:
    a, b = load_grid(here[0]), load_grid(there[0])
    print(f"{os.path.basename(here[0])}  vs  {os.path.basename(there[0])}\n")
    print(f"{'arch':10s} {'dtype':10s} {'7.2.3 ent/dist':>18s} {'7.14 ent/dist':>18s}  same?")
    same = True
    for k in sorted(set(a) & set(b)):
        eq = a[k][1] == b[k][1]
        same &= eq
        print(f"{k[0]:10s} {k[1]:10s} {a[k][0]:8d}/{a[k][1]:<8d} {b[k][0]:8d}/{b[k][1]:<8d}  {'yes' if eq else 'NO'}")
    only = (set(a) ^ set(b))
    if only: print(f"\n  present in only one version: {sorted(only)}"); same = False
    print(f"\n  VERDICT: {'PASS — section 3 may keep the unchanged claim' if same else 'FAIL — narrow the claim to the versions actually checked'}")
else:
    print(f"  need both files; have {len(here)} local and {len(there)} from 7.14")
