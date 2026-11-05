"""pick_lambda.py <x25|x40> -- buoc 2 cua do luoi phat lech tam (nhom C): co MLVU dev cua 4 moc lambda 0 / 0,5 / 1 / 2
-> chon moc tot nhat, in ra 2 moc lan can de do min (ten run + gia tri), them hang vao bang Ablation (public + local).
Chay tu thu muc KTV: python outputs/hds7/pick_lambda.py   (in moi dong "TEN GIA_TRI"; ghi ly do vao PEN_pick.txt)
Moc 0,5 = mac dinh (lambda de train duoc nhung dung yen ~0,5), trung binh 3 seed H3 / H3s1 / H3s2.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "outputs/hds7")
from push_ablation import LOCAL, PUBLIC, score  # noqa: E402

LV = sys.argv[1] if len(sys.argv) > 1 else "x25"          # muc giu: x25 | x40
BASE = {"x25": ["K25", "K25s1", "K25s2"], "x40": ["K40", "K40s1", "K40s2"]}[LV]
GRID = {0.0: ["PN0" + LV], 0.5: BASE, 1.0: ["PNf1" + LV], 2.0: ["PNf2" + LV]}
# moc lan can theo moc tot nhat (giua 2 moc luoi; o bien thi mo rong ra ngoai)
NEIGH = {0.0: [0.25], 0.5: [0.25, 0.75], 1.0: [0.75, 1.5], 2.0: [1.5, 3.0]}


def name(v):
    return "PNf" + ("%g" % v).replace(".", "") + LV   # 0.25 -> PNf025x25, 1.5 -> PNf15x40


def mlvu(names):
    vals = []
    for n in names:
        v, k = score("mlvu", n)
        if v is None or k < 2174:
            return None
        vals.append(v)
    return sum(vals) / len(vals)


def add_rows(state, new):
    b = next(x for x in state["benchmarks"] if x["id"] == "bb_ablation")
    cat = next(c for c in b["categories"] if c["name"].startswith("C%s." % LV[1:]))
    have = {r["cells"].get("c_code", {}).get("v") for r in b["rows"]}
    rows = [r for r in b["rows"] if r.get("categoryId") == cat["id"]]
    pos = b["rows"].index(rows[-1]) + 1
    for v in new:
        if name(v) in have:
            continue
        cells = {"c_exp": "Dò mịn: phạt cố định λ = %g (1 λ chung)" % v, "c_s1": "Phạt cố định %g" % v, "c_s2": "Có (đầy đủ)",
                 "c_ld": "Chỉ KL · 2.569 video", "c_ret": LV[1:] + "%", "c_seed": "1", "c_st": "🕒 Trong hàng đợi",
                 "c_mlvu": "", "c_vm": "", "c_lvb": "", "c_avg": "", "c_d": "", "c_code": name(v),
                 "c_note": "Thêm tự động quanh mốc tốt nhất của lưới 0 / 0,5 / 1 / 2"}
        ref = [r for r in rows if r["cells"].get("c_exp", {}).get("v", "").startswith("★")][0].get("ref")
        b["rows"].insert(pos, {"id": "ab_" + name(v), "categoryId": cat["id"], "ref": ref,
                               "cells": {k: {"v": x} for k, x in cells.items()}})
        pos += 1


def main():
    res = {v: mlvu(n) for v, n in GRID.items()}
    if any(x is None for x in res.values()):
        print("chua du MLVU dev cho 4 moc", res, file=sys.stderr)
        sys.exit(1)
    best = max(res, key=lambda v: res[v])
    new = NEIGH[best]
    why = "MLVU dev theo lambda: " + ", ".join("%g: %.2f" % (v, res[v]) for v in sorted(res)) + \
          " -> tot nhat lambda = %g -> do min them: %s" % (best, ", ".join("%g" % v for v in new))
    open("outputs/hds7/PEN_pick_%s.txt" % LV, "w").write(why + "\n")
    try:
        with urllib.request.urlopen(PUBLIC, timeout=30) as f:
            st = json.load(f)
        add_rows(st, new)
        req = urllib.request.Request(PUBLIC, data=json.dumps(st, ensure_ascii=False).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print("them hang public loi:", e, file=sys.stderr)
    st = json.load(open(LOCAL, encoding="utf-8"))
    add_rows(st, new)
    json.dump(st, open(LOCAL + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(LOCAL + ".tmp", LOCAL)
    for v in new:
        print(name(v), v)


if __name__ == "__main__":
    main()
