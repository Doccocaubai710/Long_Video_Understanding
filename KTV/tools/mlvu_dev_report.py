"""mlvu_dev_report.py -- diem MLVU dev: M-Avg (trung binh 7 tac vu, nhu paper) + overall + so cap McNemar.
  python -m tools.mlvu_dev_report  ->  in ra va ghi outputs/mlvu_dev/KET_QUA.txt
"""
import glob
import os

from tools.hds_report import acc, load, paired

D = "outputs/mlvu_dev/pred"
REFS = {"005": ["dense", "uniform_r005", "holitom_r005"], "002": ["dense", "uniform_r002", "holitom_r002"], "040": ["dense", "uniform_r040", "holitom_r040"], "025": ["dense", "uniform_r025", "holitom_r025"],
        "010": ["dense", "uniform_r010", "holitom_r010"]}
res = {os.path.splitext(os.path.basename(p))[0]: load(p, False) for p in sorted(glob.glob(f"{D}/*.json"))}
tasks = sorted({b for d in res.values() for _, b in d.values() if b})
L = ["KET QUA MLVU dev (2.174 cau trac nghiem, 4 lua chon), Qwen2.5-VL-7B, 128 frame.",
     "M-Avg = trung binh do chinh xac 7 tac vu (cach paper bao cao). z = McNemar so cap, |z| > 1,96 la co y nghia.", "",
     f"{'cau hinh':<16} {'n':>5} " + " ".join(f"{t[:7]:>7}" for t in tasks) + f" {'M-Avg':>7} {'overall':>8}   so voi moc"]
for n, d in res.items():
    mavg = sum(acc(d, t) for t in tasks) / len(tasks)
    cmp = []
    for r in REFS.get(n[-3:], []) if n.startswith("H") else []:
        if r in res:
            dd, z, _ = paired(d, res[r])
            cmp.append(f"{r} {dd:+.2f} (z={z:+.2f})")
    L.append(f"{n:<16} {len(d):>5} " + " ".join(f"{acc(d, t):>7.2f}" for t in tasks) + f" {mavg:>7.2f} {acc(d):>8.2f}   " + " | ".join(cmp))
open("outputs/mlvu_dev/KET_QUA.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
