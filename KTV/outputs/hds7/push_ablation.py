"""push_ablation.py -- dien so lieu moi vao bang "Ablation study" tren web public (va ban local).
Chay tu thu muc KTV: python outputs/hds7/push_ablation.py   (report.py goi o cuoi moi lan chay)

Chi cap nhat cac hang co cot "Ma run" = dung 1 ten run (vd. PN0, BOm1): MLVU dev / VideoMME / LVBench khi cham DU,
TB VM+LVB, cot Chenh (theo row["ref"]), Trang thai; roi to vang lai o cao nhat moi nhom. Khong dung vao hang khac.
"""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.getcwd())
from tools.hds_report import acc, load  # noqa: E402

PUBLIC = "https://longvideo.taile09185.ts.net/api/state"
LOCAL = "/home/hoang/Long_Video_Understanding/vlm-paper-site/state.json"
O = "outputs/hds7"
N = {"mlvu": 2174, "videomme": 2700, "lvbench": 1549}
DIRS = {"mlvu": ["outputs/mlvu_dev/pred"],
        "videomme": [f"outputs/{d}/videomme" for d in ["hds7", "hds6", "hds5", "hds4", "hds3", "hds2", "hds"]],
        "lvbench": [f"outputs/{d}/lvbench" for d in ["hds7", "hds6", "hds5", "hds4", "hds3", "hds2", "hds"]]}
COL = {"mlvu": "c_mlvu", "videomme": "c_vm", "lvbench": "c_lvb"}
NUMC = ("c_mlvu", "c_vm", "c_lvb", "c_avg")
DONE, RUN = "✅ Đã có", "⏳ Đang chạy"
RC = {"005": re.compile(r"^(K5|K5s\d+|SM5|SMg5|BX5|SG5|X5)$"), "025": re.compile(r"^(K25|K25s\d+|SM25|X25)$"),
      "040": re.compile(r"^(K40|K40s\d+|X40|X40s1)$"), "002": re.compile(r"^(K2s\d+|X2|H6)$")}


def var(name):
    if re.search(r"x(25|40)$", name):
        return f"{name}_r0{name[-2:]}"
    return name + "_r" + next((rc for rc, p in RC.items() if p.match(name)), "010")


def score(b, name):
    """(diem, so cau) -- diem = None neu chua co file."""
    for d in DIRS[b]:
        p = f"{d}/{var(name)}.json"
        if os.path.exists(p):
            r = load(p, False)
            return (acc(r) if r else None), len(r)
    return None, 0


def training(name):
    log = f"{O}/logs/{name}.train.log"
    return os.path.exists(log) or os.path.exists(f"{O}/train/{name}.pt")


def update(bench):
    changed = False
    for row in bench["rows"]:
        c = row["cells"]
        code = (c.get("c_code") or {}).get("v", "")
        if not re.fullmatch(r"[A-Za-z0-9]+", code or "") or code in ("dense", "uniform", "holitom"):
            continue
        got, full = {}, True
        for b, col in COL.items():
            v, n = score(b, code)
            if v is not None and n >= N[b]:
                got[col] = "%.2f" % v
            else:
                full = False
        if row.get("only"):   # hang chi cham mot so bo (vd. chi MLVU dev)
            full = all(k in got for k in row["only"])
        if not got and not training(code):
            continue
        new = dict(got)
        vm, lvb = got.get("c_vm"), got.get("c_lvb")
        new["c_avg"] = "%.2f" % ((float(vm) + float(lvb)) / 2) if vm and lvb else ""
        ref = row.get("ref")
        if ref:
            parts = ["%+.2f" % (float(got[k]) - r) if k in got else "–" for k, r in zip(("c_mlvu", "c_vm"), ref)]
            new["c_d"] = " / ".join(parts) if got else ""
        cur = (c.get("c_st") or {}).get("v", "")
        new["c_st"] = DONE if full else (cur if cur == DONE else RUN)   # hang cu thieu 1 bo van giu "Da co"
        for k, v in new.items():
            if (c.get(k) or {}).get("v", "") != v:
                c.setdefault(k, {})["v"] = v
                changed = True
    # to vang o cao nhat moi nhom (giong build_ablation.py)
    for cat in bench["categories"]:
        rs = [r for r in bench["rows"] if r.get("categoryId") == cat["id"] and not r.get("noMax") and "c_mlvu" in r["cells"]]
        for col in NUMC:
            vals = []
            for r in rs:
                try:
                    vals.append((float(r["cells"][col]["v"]), r))
                except (KeyError, ValueError, TypeError):
                    pass
            best = max(v for v, _ in vals) if len(vals) >= 2 else None
            for r in rs:
                cell = r["cells"].get(col)
                if cell is None:
                    continue
                want = best is not None and any(r is rr and v == best for v, rr in vals)
                if bool(cell.get("marked")) != want:
                    cell["marked"] = want
                    changed = True
    return changed


def push_public():
    with urllib.request.urlopen(PUBLIC, timeout=30) as f:
        state = json.load(f)
    bench = next((b for b in state["benchmarks"] if b["id"] == "bb_ablation"), None)
    if bench is None or not update(bench):
        return "public: khong doi"
    req = urllib.request.Request(PUBLIC, data=json.dumps(state, ensure_ascii=False).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as f:
        return "public: " + f.read().decode()[:40]


def push_local():
    state = json.load(open(LOCAL, encoding="utf-8"))
    bench = next((b for b in state["benchmarks"] if b["id"] == "bb_ablation"), None)
    if bench is None or not update(bench):
        return "local: khong doi"
    json.dump(state, open(LOCAL + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(LOCAL + ".tmp", LOCAL)
    return "local: da ghi"


if __name__ == "__main__":
    for fn in (push_public, push_local):
        try:
            print(fn())
        except Exception as e:   # mang loi / web tat: lan sau thu lai
            print(f"{fn.__name__} loi: {type(e).__name__}: {e}")
