"""q1_gate.py -- Q1 (KL tren mau hoi-dap) co dang chay tiep khong? Thoat 0 = co, 1 = khong. Ghi ly do ra Q1_gate.txt.
Tieu chi (dat ra truoc khi co ket qua): TB VideoMME+LVBench >= chi KL (3 seed) + 1.0 VA MLVU dev >= chi KL + 0.5.
"""
import os, sys, statistics as st
sys.path.insert(0, os.getcwd())
from tools.hds_report import acc, load  # noqa: E402
F = {"mlvu": "outputs/mlvu_dev/pred/{}.json", "videomme": "outputs/{}/videomme/{}.json", "lvbench": "outputs/{}/lvbench/{}.json"}
def sc(b, n):
    for d in (["x"] if b == "mlvu" else ["hds7", "hds2", "hds", "hds4"]):
        p = F[b].format(n) if b == "mlvu" else F[b].format(d, n)
        if os.path.exists(p): return acc(load(p, False))
kl = ["H3_r010", "H3s1_r010", "H3s2_r010"]
klm = st.mean(sc("mlvu", n) for n in kl); klv = st.mean((sc("videomme", n) + sc("lvbench", n)) / 2 for n in kl)
qm = sc("mlvu", "Q1_r010"); qv = (sc("videomme", "Q1_r010") + sc("lvbench", "Q1_r010")) / 2
ok = qv >= klv + 1.0 and qm >= klm + 0.5
msg = f"Q1: MLVU dev {qm:.2f} (chi KL {klm:.2f}), TB VM+LVB {qv:.2f} (chi KL {klv:.2f}) -> {'DAT, chay them Q1s1' if ok else 'KHONG DAT, bo huong nay'}"
open("outputs/hds7/Q1_gate.txt", "w").write(msg + "\n"); print(msg)
sys.exit(0 if ok else 1)
