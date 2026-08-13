# Grid density per (architecture, dtype), using the real Tensile type codes.
import msgpack, glob, os, re, collections

ROOT = "/opt/rocm/lib/rocblas/library"
WANT = {"SS": "fp32", "BB_HPA": "bf16", "HH_HPA": "fp16", "HS_HPA": "fp16->fp32", "BS_HPA": "bf16->fp32"}


def grid(path):
    d = msgpack.unpackb(open(path, "rb").read(), strict_map_key=False, raw=False)
    for row in d.get("library", {}).get("rows", []):
        lib = row.get("library", {})
        if lib.get("distance") == "Euclidean":
            return [tuple(e["key"]) for e in lib.get("table", [])]
    return []


agg = collections.defaultdict(dict)
for p in sorted(glob.glob(f"{ROOT}/**/TensileLibrary_Type_*_Ailk_Bljk_Cijk_Dijk_*.dat", recursive=True)):
    b = os.path.basename(p)
    if "fallback" in b or "Experimental" in b:
        continue
    m = re.match(r"TensileLibrary_Type_([A-Za-z0-9_]+?)_Contraction_.*?(gfx9[0-9a-z]+)\.dat$", b)
    if not m:
        continue
    ty, arch = m.group(1), m.group(2)
    if ty not in WANT:
        continue
    pts = grid(p)
    if not pts:
        continue
    ki = len(pts[0]) - 1
    cur = (len(pts), sum(1 for k in pts if k[ki] >= 8192))
    prev = agg[arch].get(WANT[ty])
    if prev is None or cur[0] > prev[0]:
        agg[arch][WANT[ty]] = cur

names = {"gfx908": "MI100", "gfx90a": "MI250", "gfx942": "MI300X", "gfx950": "MI355X"}
dts = ["fp32", "bf16", "fp16"]
print("Số điểm hiệu chỉnh, bảng NN, biến thể lớn nhất mỗi kiến trúc (ROCm 7.2.3)\n")
print(f"{'kiến trúc':20s} " + " ".join(f"{d:>20s}" for d in dts))
print("-" * 84)
for a in ["gfx908", "gfx90a", "gfx942", "gfx950"]:
    if a not in agg:
        continue
    cells = []
    for d in dts:
        v = agg[a].get(d)
        cells.append(f"{v[0]:6d}  ({v[1]:4d} K≥8k)" if v else "—")
    print(f"{a+' ('+names[a]+')':20s} " + " ".join(f"{c:>20s}" for c in cells))
