#!/usr/bin/env python3
"""Count the tuned points in rocBLAS's shipped Tensile logic files. No GPU required.

rocBLAS selects a kernel by nearest neighbour, Euclidean distance, over a table of tuned
(M, N, K, batch) points held per (architecture, dtype, transpose). Those tables ship inside the
`.dat` files as msgpack, so their density is readable from an installed package without running
anything on a GPU.

Reports two counts per cell, because they differ and the difference matters:

  entries   rows in the table as shipped
  distinct  unique (M, N, K) keys

The tables contain duplicate keys, unevenly across architectures: the gfx90a fp32 table holds
9874 entries over 4888 distinct points, while gfx942's 415 entries are 345 distinct. Comparing
architectures on entry counts therefore overstates the gfx90a/gfx942 ratio by roughly 40%. Distinct
keys are the number that matters for a nearest-neighbour lookup.

Type codes are the ones rocBLAS actually dispatches. A GEMM with fp32 output and fp32 compute goes
through `Type_BS_HPA` / `Type_HS_HPA` for bf16 / fp16 inputs, not `Type_BB` / `Type_HH`; both are
listed so the distinction is visible rather than assumed.

Two views, because the issue thread quotes both:

  default            one row per architecture, the NN transpose, largest variant per dtype
  --by-transpose     one architecture, all four transposes, which is where the fallback tables show
                     up: on gfx90a the TT table for bf16 holds 48 tuned points against its siblings'
                     ~2600, because no CU104 table ships for that combination

Usage:  pip install msgpack
        python3 cmparch3.py [library_dir]
        python3 cmparch3.py [library_dir] --by-transpose [gfx90a]
"""
import collections
import glob
import os
import re
import sys

import msgpack

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BY_TRANSPOSE = "--by-transpose" in sys.argv
ROOT = ARGS[0] if ARGS else "/opt/rocm/lib/rocblas/library"
ONE_ARCH = ARGS[1] if len(ARGS) > 1 else "gfx90a"
TRANSPOSES = ["Ailk_Bljk", "Alik_Bljk", "Ailk_Bjlk", "Alik_Bjlk"]
TYPES = {"SS": "fp32", "BB_HPA": "bf16", "HH_HPA": "fp16",
         "BS_HPA": "bf16->fp32", "HS_HPA": "fp16->fp32"}
ARCH_NAMES = {"gfx908": "MI100", "gfx90a": "MI250", "gfx942": "MI300X", "gfx950": "MI355X"}


def tuned_points(path):
    """The nearest-neighbour table inside one .dat, as a list of key tuples."""
    data = msgpack.unpackb(open(path, "rb").read(), strict_map_key=False, raw=False)
    for row in data.get("library", {}).get("rows", []):
        lib = row.get("library", {})
        if lib.get("distance") == "Euclidean":
            return [tuple(entry["key"]) for entry in lib.get("table", [])]
    return []


def by_transpose():
    """All four transposes for one architecture, both variants of each table."""
    print(f"Tuned points per (dtype, transpose) on {ONE_ARCH}")
    print(f"library: {ROOT}")
    print("cells are  entries / distinct keys;  '-' means the table is not shipped\n")
    header = f"{'dtype':14s}{'variant':10s}" + "".join(f"{t:>20s}" for t in TRANSPOSES)
    print(header)
    print("-" * len(header))
    for type_code, dtype in TYPES.items():
        for tag, variant in ((f"CU104_{ONE_ARCH}", "CU104"), (ONE_ARCH, "generic")):
            cells = []
            for tr in TRANSPOSES:
                path = f"{ROOT}/TensileLibrary_Type_{type_code}_Contraction_l_{tr}_Cijk_Dijk_{tag}.dat"
                if not os.path.exists(path):
                    cells.append("-")
                    continue
                pts = tuned_points(path)
                cells.append(f"{len(pts)} / {len(set(pts))}" if pts else "-")
            if any(c != "-" for c in cells):
                print(f"{dtype:14s}{variant:10s}" + "".join(f"{c:>20s}" for c in cells))


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"no such directory: {ROOT}\n"
                 f"pass the rocBLAS library directory as the first argument")

    if BY_TRANSPOSE:
        by_transpose()
        return

    # One variant per (arch, dtype): the largest, since a dtype can ship several (HPA, Fp16Alt, ...).
    # The NN transpose only; within one architecture the four transposes differ by up to 53x, so a
    # single variant is a comparison point, not a description of the architecture.
    agg = collections.defaultdict(dict)
    pattern = f"{ROOT}/**/TensileLibrary_Type_*_Ailk_Bljk_Cijk_Dijk_*.dat"
    for path in sorted(glob.glob(pattern, recursive=True)):
        base = os.path.basename(path)
        if "fallback" in base or "Experimental" in base:
            continue
        m = re.match(r"TensileLibrary_Type_([A-Za-z0-9_]+?)_Contraction_.*?(gfx9[0-9a-z]+)\.dat$", base)
        if not m:
            continue
        type_code, arch = m.group(1), m.group(2)
        if type_code not in TYPES:
            continue
        points = tuned_points(path)
        if not points:
            continue
        counts = (len(points), len(set(points)))
        previous = agg[arch].get(TYPES[type_code])
        if previous is None or counts[0] > previous[0]:
            agg[arch][TYPES[type_code]] = counts

    if not agg:
        sys.exit(f"no Tensile logic files matched under {ROOT}")

    dtypes = ["fp32", "bf16", "fp16", "bf16->fp32", "fp16->fp32"]
    W = 20
    header = f"{'architecture':22s}" + "".join(f"{d:>{W}s}" for d in dtypes)
    print("Tuned points per (architecture, dtype), NN transpose, largest variant")
    print(f"library: {ROOT}")
    print("cells are  entries / distinct keys\n")
    print(header)
    print("-" * len(header))
    for arch in sorted(agg, key=lambda a: list(ARCH_NAMES).index(a) if a in ARCH_NAMES else 99):
        label = f"{arch} ({ARCH_NAMES[arch]})" if arch in ARCH_NAMES else arch
        cells = []
        for d in dtypes:
            v = agg[arch].get(d)
            cells.append(f"{v[0]} / {v[1]}" if v else "-")
        print(f"{label:22s}" + "".join(f"{c:>{W}s}" for c in cells))


if __name__ == "__main__":
    main()
