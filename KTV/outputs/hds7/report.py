"""report.py -- viet lai outputs/hds7/KET_QUA.txt tu moi file du doan (goi sau moi job + moi 30 phut).
Chay tu thu muc KTV: python outputs/hds7/report.py
"""
import datetime
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from tools.hds_report import acc, load, paired  # noqa: E402

O = "outputs/hds7"
N = {"mlvu": 2174, "videomme": 2700, "lvbench": 1549}
DIRS = {"mlvu": ["outputs/mlvu_dev/pred"],
        "videomme": [f"outputs/{d}/videomme" for d in ["hds7", "hds6", "hds5", "hds4", "hds3", "hds2", "hds"]],
        "lvbench": [f"outputs/{d}/lvbench" for d in ["hds7", "hds6", "hds5", "hds4", "hds3", "hds2", "hds"]]}
DESC = {
    "dense": "100% token (khong nen)", "uniform": "Uniform", "holitom": "HoliTom",
    "H3": "KL, seed 0", "H3s1": "KL, seed 1", "H3s2": "KL, seed 2",
    "H3ce": "KL + CE-gate 0.2, seed 0 (mac dinh)", "L4": "KL + CE-gate 0.2, seed 1", "L4s2": "KL + CE-gate 0.2, seed 2",
    "L1": "KL + CE-gate 0.5", "L2": "KL + CE-gate 1.0", "L3": "KL + CE KHONG gate 0.2", "CEo": "Chi CE caption (bo KL)",
    "W1": "cua so +-1 (CE-gate 0.2)", "W2": "cua so +-5 (CE-gate 0.2)",
    "H3noGate": "H3 bo cong (1-r)", "H3noDF": "H3 bo deform", "H3noSM": "H3 bo self-mask", "H12": "H3 + ghep 6 clip (video dai)",
    "Dx8": "C*, 1/8 data (320 video)", "Dq": "C*, 1/4 data (640 video)", "Dh": "C*, 1/2 data (1.280 video)",
    "D2x": "C*, 1,7x data (4.342 video)", "H3cv": "H3ce + 50% Vript (682 video dai)", "V1": "C* + 50% Vript",
    "S2": "C*, seed khac (2)", "S3": "C*, seed khac (3)", **{f"S{k}": f"C*, seed khac ({k})" for k in range(4, 30)},
    "K40": "chi KL o 40%, seed 0", "K5": "chi KL o 5%, seed 0", "K5s1": "chi KL o 5%, seed 1",
    "K25": "chi KL o 25%, seed 0", "K25s1": "chi KL o 25%, seed 1", "K2s1": "chi KL o 2%, seed 1", "H6": "chi KL o 2%, seed 0",
    "Q1": "chi KL + 50% mau hoi-dap (KL dap an)", "Q1s1": "Q1 seed 1", "K40s1": "chi KL o 40%, seed 1", "X40s1": "C* o muc 40%, seed 1",
    "H10ce": "KL + CE-gate 0.2 o 40%, cua so +-1 (hds3)", "H10ce5": "KL + CE-gate 0.5 o 40%, cua so +-1",
    "NGce": "C* bo cong (1-r), train tu dau", "D2x8k": "C*, 1,7x data, 8000 buoc", "L2s1": "KL + CE-gate 1.0, seed 1",
    "SM10": "chi KL, giai doan 1 xuoi soft + giam tau", "SM10s1": "SM10, seed 1", "SM10s2": "SM10, seed 2", "SM5": "SM10 o muc 5%", "SM25": "SM10 o muc 25%", "SMg10s1": "SMg10, seed 1", "SMg5": "SMg10 o muc 5%", "SMg10": "chi KL, giai doan 1 gop token soft (ca test)",
    "BX10": "chi KL + chon hop co hoc (cach 1)", "SG10": "chi KL + ngan sach theo doan (cach 2)",
    "BX10s1": "BX10 seed 1", "SG10s1": "SG10 seed 1", "BX5": "chon hop co hoc o 5%", "SG5": "ngan sach theo doan o 5%",
    "D20": "chi KL, chi 20 video", "D80": "chi KL, chi 80 video",
    **{f"K5s{k}": f"chi KL o 5%, seed {k}" for k in range(2, 30)}, **{f"K2s{k}": f"chi KL o 2%, seed {k}" for k in range(2, 30)},
    **{f"K25s{k}": f"chi KL o 25%, seed {k}" for k in range(2, 30)}, **{f"K40s{k}": f"chi KL o 40%, seed {k}" for k in range(2, 30)},
    "X2": "C* o muc 2%", "X5": "C* o muc 5%", "X25": "C* o muc 25%", "X40": "C* o muc 40%",
}
RC = {"SM5": "005", "SMg5": "005", "SM25": "025", **{f"K5s{k}": "005" for k in range(2, 30)}, **{f"K2s{k}": "002" for k in range(2, 30)},
      **{f"K25s{k}": "025" for k in range(2, 30)}, **{f"K40s{k}": "040" for k in range(2, 30)}, "BX5": "005", "SG5": "005",
      "X2": "002", "X5": "005", "X25": "025", "X40": "040", "X40s1": "040", "K40": "040", "K40s1": "040", "K5": "005", "K5s1": "005", "K25": "025", "K25s1": "025", "K2s1": "002", "H6": "002", "H10ce": "040", "H10ce5": "040"}


def var(name):
    if re.search(r"x(25|40)$", name):   # ablation o muc 25% / 40% (vd. PN0x25, BK2x40)
        return f"{name}_r0{name[-2:]}"
    return name if name == "dense" or re.search(r"_r\d{3}$", name) else f"{name}_r{RC.get(name, '010')}"


def find(b, v):
    for d in DIRS[b]:
        p = f"{d}/{v}.json"
        if os.path.exists(p):
            return p
    if b == "videomme":
        p = f"outputs/preds/pred_videomme_f128_{v}.json"
        if os.path.exists(p):
            return p
    return None


CACHE = {}


def res(b, name):
    v = var(name)
    if (b, v) not in CACHE:
        p = find(b, v)
        CACHE[(b, v)] = load(p, False) if p else None
    return CACHE[(b, v)]


def cell(b, name):
    d = res(b, name)
    if not d:
        return f"{'-':>13}"
    s = f"{acc(d):6.2f}"
    return f"{s:>13}" if len(d) >= N[b] else f"{s + f' ({len(d)})':>13}"


def full(b, name):
    d = res(b, name)
    return acc(d) if d and len(d) >= N[b] else None


def cmp(name, ref):
    a, r = res("mlvu", name), res("mlvu", ref)
    if not a or not r or name == ref or len(a) < N["mlvu"] or name.split("_r")[0] in ("dense", "uniform", "holitom"):
        return ""
    dd, z, _ = paired(a, r)
    return f"{dd:+.2f} (z={z:+.2f})"


HDR = f"{'cau hinh':<13} {'mo ta':<36} {'MLVU dev':>13} {'VideoMME':>13} {'LVBench':>13} {'TB VM+LVB':>9}   MLVU dev so voi"


def row(name, ref="H3ce"):
    base = name.split("_r")[0]
    ref = f"uniform_r{RC[name]}" if name in RC else ref
    key = next((k for k in sorted(DESC, key=len, reverse=True) if base == k or base.startswith(k + "_")), base)
    vm, lv = full("videomme", name), full("lvbench", name)
    avg = f"{(vm + lv) / 2:9.2f}" if vm is not None and lv is not None else f"{'-':>9}"
    c = cmp(name, ref)
    return (f"{name:<13} {DESC.get(key, ''):<36.36} {cell('mlvu', name)} {cell('videomme', name)} {cell('lvbench', name)} {avg}"
            + (f"   {ref} {c}" if c else ""))


def mean_row(label, names):
    out = [label]
    for b in ["mlvu", "videomme", "lvbench"]:
        v = [full(b, n) for n in names if full(b, n) is not None]
        if len(v) >= 2:
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
            out.append(f"{b} {m:.2f} +- {sd:.2f} (n={len(v)})")
    return "  " + " | ".join(out) if len(out) > 1 else None


def section(title, names, ref="H3ce", means=()):
    L = [f"== {title} ==", HDR]
    L += [row(n, ref) for n in names if any(res(b, n) for b in N)] or ["(chua co ket qua)"]
    for label, ns in means:
        r = mean_row(label, ns)
        if r:
            L.append(r)
    return L + [""]


def tail(p, k=4):
    return open(p).read().strip().splitlines()[-k:] if os.path.exists(p) else []


cstar = open(f"{O}/CSTAR.txt").read().split("\n") if os.path.exists(f"{O}/CSTAR.txt") else None
C = cstar[0] if cstar else None
L = [f"KET QUA DOT HDS7 (plan 3 ngay 30/10 -> 2/11) -- cap nhat {datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))):%d/%m %H:%M} gio VN",
     "Qwen2.5-VL-7B, 128 frame. MLVU dev 2.174 cau (dung de CHON cau hinh), VideoMME 2.700, LVBench 1.549 (chi bao cao).",
     "So trong ngoac = so cau da cham (chua xong). z = McNemar so cap tung cau tren MLVU dev, |z| > 1,96 la co y nghia.",
     "Nhieu giua cac seed o 10% ~1,2 diem -> chenh < 1 diem coi nhu hoa.", ""]
L += ["C* (chon tu dong tren MLVU dev): " + (f"{C}  [{cstar[1]}]" if C else "chua chon (cho xong giai doan 1)")]
if os.path.exists(f"{O}/CSTAR_ly_do.txt"):
    L += ["  " + x for x in open(f"{O}/CSTAR_ly_do.txt").read().strip().splitlines()]
L += [""]
for g in (0, 1):
    L += [f"Tien do GPU{g} (5 dong cuoi STATUS_gpu{g}.txt):"] + ["  " + x for x in tail(f"{O}/STATUS_gpu{g}.txt", 5)]
L += [""]
L += section("Moc 10%", ["dense", "uniform_r010", "holitom_r010"])
L += section("1. Ablation loss (10%)", ["H3", "H3s1", "H3s2", "H3ce", "L4", "L4s2", "L1", "L2", "L2s1", "L3", "CEo", "Q1", "Q1s1"],
             means=[("KL (trung binh cac seed):", ["H3", "H3s1", "H3s2"]),
                    ("KL + CE-gate 0.2 (trung binh cac seed):", ["H3ce", "L4", "L4s2"])])
L += section("2. Cua so thoi gian giai doan 2 (10%, CE-gate 0.2)", ["W1", "H3ce", "W2"])
L += section("3. Ablation thanh phan (10%, so voi H3 = KL)", ["H3", "H12", "H3noDF", "H3noSM", "H3noGate"], ref="H3")
L += section("3b. Bo cong tren C* (train tu dau, so voi H3ce)", ["H3ce", "NGce"])
data = ["Dx8", "Dq", "Dh"] + ([C] if C else []) + ["D2x", "D2x8k", "H3cv", "V1"]
L += section("4. Luong / loai data (10%, cung 4000 buoc; 1x = 2.569 video)", data, ref=C or "H3ce")
seeds = ([C] if C else []) + (["L4"] if C == "H3ce" else []) + [f"S{k}" for k in range(2, 30)]
L += section("5. C* nhieu seed (10%)", seeds, ref=C or "H3ce", means=[("C* (trung binh cac seed):", seeds)])
tr = []
for x, r in [("X2", "002"), ("X5", "005"), ("X25", "025"), ("X40", "040")]:
    tr += [f"uniform_r{r}", f"holitom_r{r}", x]
L += section("6. Mot bo tham so cho moi muc nen (C* o 2/5/25/40%)", tr, ref="uniform_r010")
L += section("7. Muc 40%: chi KL vs KL + CE-gate (cua so +-3, 2 seed moi loai)", ["dense", "uniform_r040", "holitom_r040", "K40", "K40s1", "X40", "X40s1", "H10ce", "H10ce5"], ref="dense",
             means=[("Chi KL 40%:", ["K40", "K40s1"]), ("KL + CE-gate 0.2 40% (C*):", ["X40", "X40s1"])])
kl = []
for x in ["002", "005", "025", "040"]:
    kl += [f"uniform_r{x}", f"holitom_r{x}"] + {"002": ["H6"] + [f"K2s{k}" for k in range(1, 30)], "005": ["K5"] + [f"K5s{k}" for k in range(1, 30)],
               "025": ["K25"] + [f"K25s{k}" for k in range(1, 30)], "040": ["K40"] + [f"K40s{k}" for k in range(1, 30)]}[x]
L += section("8. Bang chinh CHI KL o moi muc nen (10% xem muc 1: H3, H3s1, H3s2)", kl, ref="uniform_r010")
L += section("9b. Phan bo thich ung o 5% (so voi Uniform 5%)", ["uniform_r005", "holitom_r005", "K5", "K5s1", "BX5", "SG5", "SM5", "SMg5"], ref="uniform_r005")
L += section("9. Phan bo token thich ung (10%, chi KL; so voi H3 = chi KL seed 0)", ["H3", "H3s1", "H3s2", "BX10", "BX10s1", "SG10", "SG10s1", "D20", "D80", "SM10", "SM10s1", "SM10s2", "SMg10", "SMg10s1"], ref="H3",
             means=[("Chi KL (trung binh 3 seed):", ["H3", "H3s1", "H3s2"])])
if os.path.exists(f"{O}/Q1_gate.txt"):
    L += ["Q1 (KL tren mau hoi-dap): " + open(f"{O}/Q1_gate.txt").read().strip(), ""]
L += ["Nguon: outputs/mlvu_dev/pred, outputs/hds*/videomme, outputs/hds*/lvbench, outputs/preds. Plan: outputs/hds7/PLAN.txt."]
open(f"{O}/KET_QUA.txt", "w").write("\n".join(L) + "\n")

# day so lieu moi len bang Ablation study (web public + local); loi mang thi bo qua, lan sau thu lai
try:
    import subprocess
    subprocess.run([sys.executable, f"{O}/push_ablation.py"], timeout=120, stdout=open(f"{O}/logs/push_ablation.log", "a"), stderr=subprocess.STDOUT)
except Exception:
    pass
