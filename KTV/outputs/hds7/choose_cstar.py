"""choose_cstar.py -- chon cau hinh C* sau giai doan 1 dot hds7, CHI dua tren MLVU dev (khong nhin VideoMME/LVBench).

Quy tac: diem = trung binh do chinh xac overall MLVU dev cua cac seed trong nhom. Mac dinh H3ce (KL + CE-gate 0.2, cua so +-3).
Chi doi sang cau hinh khac neu hon mac dinh >= 1,0 diem (nhieu giua cac seed o 10% ~1,2 diem tren VideoMME).
Thoat ma 1 neu con ung vien chua cham xong (queue se thu lai sau). Ghi outputs/hds7/CSTAR.txt (dong 1 ten, dong 2 tham so).
Chay tu thu muc KTV: python outputs/hds7/choose_cstar.py
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from tools.hds_report import acc, load  # noqa: E402

O, MD, N = "outputs/hds7", "outputs/mlvu_dev/pred", 2174
CANDS = [  # ten C*, cac checkpoint (seed) trong nhom, tham so train
    ("H3ce", ["H3ce", "L4"], "--w_ceg 0.2"),
    ("H3", ["H3", "H3s1"], "--w_ceg 0"),
    ("L1", ["L1"], "--w_ceg 0.5"),
    ("L2", ["L2"], "--w_ceg 1.0"),
    ("L3", ["L3"], "--w_ce 0.2 --w_ceg 0"),
    ("W1", ["W1"], "--w_ceg 0.2 --window 1"),
    ("W2", ["W2"], "--w_ceg 0.2 --window 5"),
]
DEFAULT, MARGIN = "H3ce", 1.0

status = "".join(open(p).read() for p in glob.glob(f"{O}/STATUS_gpu*.txt"))
skipped = lambda n: re.search(rf"(FAIL|SKIP) train {n}\b", status) is not None  # noqa: E731

scores, missing = {}, []
for name, members, args in CANDS:
    vals = []
    for m in members:
        p = f"{MD}/{m}_r010.json"
        if skipped(m):
            continue
        if not os.path.exists(p) or sum(1 for _ in open(p)) < N:
            missing.append(m)
            continue
        vals.append(acc(load(p, False)))
    if vals:
        scores[name] = (sum(vals) / len(vals), vals, args)
if missing:
    print(f"chua du: {missing}")
    sys.exit(1)

best = max(scores, key=lambda k: scores[k][0])
d = scores.get(DEFAULT, (float("-inf"),))[0]
pick = best if best == DEFAULT or scores[best][0] >= d + MARGIN or DEFAULT not in scores else DEFAULT
lines = [f"{k:<5} MLVU dev overall = {v[0]:.2f}  (seed: {', '.join(f'{x:.2f}' for x in v[1])})  {v[2]}"
         for k, v in sorted(scores.items(), key=lambda kv: -kv[1][0])]
why = (f"C* = {pick}. Cao nhat: {best} ({scores[best][0]:.2f}); mac dinh {DEFAULT} = {d:.2f}; "
       f"nguong doi = +{MARGIN} diem.")
open(f"{O}/CSTAR_ly_do.txt", "w").write(why + "\n\n" + "\n".join(lines) + "\n")
open(f"{O}/CSTAR.txt", "w").write(f"{pick}\n{scores[pick][2]}\n")
print(why)
print("\n".join(lines))
