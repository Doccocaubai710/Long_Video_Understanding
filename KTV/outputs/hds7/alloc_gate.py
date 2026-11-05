"""alloc_gate.py <ten> -- phan bo thich ung (BX10 / SG10) co dang chay tiep khong? Thoat 0 = DAT, 1 = KHONG DAT.
Tieu chi (dat ra TRUOC khi co ket qua, giong Q1): TB VideoMME+LVBench >= chi KL (3 seed) + 1.0 VA MLVU dev >= chi KL + 0.5.
Ghi ly do vao outputs/hds7/ALLOC_gate.txt (moi ten 1 dong).
"""
import os, sys, statistics as st
sys.path.insert(0, os.getcwd())
from tools.hds_report import acc, load  # noqa: E402
name = sys.argv[1]
def sc(b, n):
    ps = [f"outputs/mlvu_dev/pred/{n}.json"] if b == "mlvu" else [f"outputs/{d}/{b}/{n}.json" for d in ["hds7", "hds2", "hds", "hds4"]]
    for p in ps:
        if os.path.exists(p): return acc(load(p, False))
    raise SystemExit(f"thieu {b} {n}")
kl = ["H3_r010", "H3s1_r010", "H3s2_r010"]
klm = st.mean(sc("mlvu", n) for n in kl); klv = st.mean((sc("videomme", n) + sc("lvbench", n)) / 2 for n in kl)
v = f"{name}_r010"
qm = sc("mlvu", v); qv = (sc("videomme", v) + sc("lvbench", v)) / 2
ok = qv >= klv + 1.0 and qm >= klm + 0.5
msg = (f"{name}: MLVU dev {qm:.2f} (chi KL {klm:.2f}), TB VM+LVB {qv:.2f} (chi KL {klv:.2f}) -> "
       + ("DAT: chay seed 1 + muc 5%" if ok else "KHONG DAT: chuyen sang seed bang chinh + thi nghiem luong data"))
with open("outputs/hds7/ALLOC_gate.txt", "a") as f:
    f.write(msg + "\n")
print(msg)
sys.exit(0 if ok else 1)
