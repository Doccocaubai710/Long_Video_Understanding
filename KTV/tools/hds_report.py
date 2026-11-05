"""hds_report.py -- viet lai outputs/hds/KET_QUA.txt tu moi file du doan + log train cua dot HDS-MAP 3D.

Usage (tu thu muc KTV): python -m tools.hds_report
"""
import csv
import datetime
import glob
import json
import math
import os
import re

from eval.compute_accuracy import load_jsonl, prediction_to_index

O = os.environ.get("HDS_DIR", "outputs/hds")   # dot moi: HDS_DIR=outputs/hds2
OLD = "outputs/hds"
PREDS = "outputs/preds"
CSV = "playground/gt_qa_files/VideoMME/val_qa.csv"
DESC = {
    "P0_center": "Pha 0: tam hop 3D (khong train)",
    "P0_var": "Pha 0: token bien doi 3D lon nhat trong hop",
    "P0_norm": "Pha 0: token co chuan lon nhat trong hop",
    "P0_medoid": "Pha 0: medoid cua hop (tong cosine lon nhat)",
    "H1": "Chi giai doan 1: bo truot CB-STE co hoc, chon 100% token goc",
    "H2": "Chi giai doan 2 (M3D-BD+): neo Uniform + M3D-BD + che token neo + cong (1-r)",
    "H3": "HDS-MAP 3D day du: giai doan 1 + giai doan 2 (mo tu buoc 2000)",
    "H4": "Ban tot nhat H1-H3, train them tren video dai ghep (3 clip 1-3 phut)",
    "H5": "Ban tot nhat H1-H3, Elastic: 1 checkpoint cho 2/5/10/25%",
    "H6": "Ban tot nhat H1-H3, train rieng o muc 2%",
    "uniform": "Moc: Uniform (luoi linspace)", "holitom": "Moc: HoliTom w/o M", "dense": "Moc: 100% token",
    "R10S": "RAP R10E + video dai ghep (doi chung cua H4)", "H4s1": "H4 seed 1", "H3s1": "H3 seed 1",
    "H3noSM": "H3 bo self-mask", "H3noDF": "H3 bo deform (hop nhin tu dich)", "H3noGate": "H3 bo cong (1-r)",
    "H10ce": "H3 o 40%, KL + CE-gate (CE chi tren token thay doan dung)", "H9ce": "H3 o 25%, KL + CE-gate",
    "H9": "H3 o 25% (chi KL)", "H8ce": "H3 o 5%, KL + CE-gate", "H8": "H3 o 5% (chi KL)",
    "H6ce": "H3 o 2%, KL + CE-gate", "H6s1": "H6 (H3 o 2%) seed 1",
    "R10E": "Moc: RAP R10E (10%)", "R02E": "Moc: RAP R02E (2%)", "runE": "Moc: RAP Run E (25%)",
}
with open(CSV, newline="", encoding="utf-8") as f:
    VMME_BUCKET = {r["question_id"]: r["duration"] for r in csv.DictReader(f)}


def load(path, use_csv):
    out = {}
    for r in load_jsonl(path):
        idx = prediction_to_index(r.get("pred"), r.get("candidates", []))
        qid = str(r.get("question_id"))
        b = VMME_BUCKET.get(qid) if use_csv else r.get("bucket")
        out[qid] = (idx is not None and idx == int(r.get("answer_number")), b)
    return out


def acc(d, b=None):
    v = [c for c, bb in d.values() if b is None or bb == b]
    return 100.0 * sum(v) / len(v) if v else float("nan")


def paired(a, b):
    common = [q for q in a if q in b]
    if not common:
        return float("nan"), float("nan"), 0
    oa = sum(a[q][0] and not b[q][0] for q in common)
    ob = sum(b[q][0] and not a[q][0] for q in common)
    d = 100.0 * (oa - ob) / len(common)
    return d, ((oa - ob) / math.sqrt(oa + ob) if oa + ob else 0.0), len(common)


def ret_of(name):
    m = re.search(r"_r(\d{3})$", name)
    return m.group(1) if m else None


def desc(name):
    for k in sorted(DESC, key=len, reverse=True):
        if name.startswith(k) or f"_{k}_" in f"_{name}_":
            return DESC[k]
    return ""


def section(title, files, use_csv, refs_fn, buckets):
    L = [f"== {title} ==", ""]
    if not files:
        return L + ["(chua co ket qua)", ""]
    data = {n: load(p, use_csv) for n, p in files}
    hdr = f"{'cau hinh':<22} {'n':>5} " + " ".join(f"{b[:7]:>7}" for b in buckets) + f" {'overall':>8}   so voi moc (diem chenh, z)"
    L.append(hdr)
    for n, _ in files:
        d = data[n]
        row = f"{n:<22} {len(d):>5} " + " ".join(f"{acc(d, b):>7.2f}" for b in buckets) + f" {acc(d):>8.2f}  "
        cmp_ = []
        for rn, rp in refs_fn(n):
            if rn == n or not os.path.exists(rp):
                continue
            dd, z, nc = paired(d, data.get(rn) or load(rp, use_csv))
            cmp_.append(f"{rn} {dd:+.2f} (z={z:+.2f})")
        L.append(row + " | ".join(cmp_))
        if desc(n):
            L.append(f"{'':<22} -> {desc(n)}")
    return L + [""]


def vmme_refs(name):
    r = ret_of(name)
    if r is None:
        return []
    refs = [(f"uniform_r{r}", f"{PREDS}/pred_videomme_f128_uniform_r{r}.json"),
            (f"holitom_r{r}", f"{PREDS}/pred_videomme_f128_holitom_r{r}.json")]
    rap = {"010": "rap_R10E_r010", "002": "rap_R02E_r002", "005": "rap_R05E_r005", "025": "rap_runE_r025",
           "015": "rap_R15E_r015", "040": "rap_R40E_r040", "001": "R01E_r001"}.get(r)
    if rap:
        refs.append((f"RAP_r{r}", f"{PREDS}/pred_videomme_f128_{rap}.json"))
    refs += [(f"vidcom2_r{r}", f"{PREDS}/pred_videomme_f128_vidcom2_r{r}.json"),
             (f"ELA_r{r}", f"{PREDS}/pred_videomme_f128_ELA_r{r}.json"),
             ("dense", f"{PREDS}/pred_videomme_f128_dense.json")]
    if O != OLD:   # dot moi: so them voi H3 / H4 cua dot cu
        refs += [(f"{h}_r{r}", f"{OLD}/videomme/{h}_r{r}.json") for h in ("H3", "H4", "H5", "H6") if not name.startswith(h + "_")]
    return refs


def train_lines():
    L = ["== Train (dong val moi nhat trong *.metrics.jsonl) ==", ""]
    for p in sorted(glob.glob(f"{O}/train/*.pt.metrics.jsonl")):
        last, nstep = None, 0
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "val" in r:
                last = r
            elif "step" in r:
                nstep = r["step"]
        name = os.path.basename(p).replace(".pt.metrics.jsonl", "")
        if last is None:
            L.append(f"{name}: buoc {nstep}, chua co val")
            continue
        v = last["val"]
        keys = ["real/kl_rap", "real/kl_uniform", "real/gap_closed", "real/hop_rate", "real/move_t", "real/move_s", "real/raer",
                "real/ce_rap", "real/ce_dense", "real/ceg_rap", "real/ceg_dense"]
        L.append(f"{name}: buoc train {nstep}, val o buoc {last['step']}: "
                 + ", ".join(f"{k.split('/')[1]}={v[k]:.4f}" for k in keys if k in v))
    L.append("  (kl_rap < kl_uniform = ban nen gan ban xem du hon Uniform; hop_rate = ti le neo roi tam hop)")
    return L + [""]


def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    L = [f"KET QUA DOT HDS-MAP 3D ({O}) -- cap nhat {now}",
         "Video-MME 2.700 cau (short/medium/long), LVBench 1.549 cau (video ~1 gio), MLVU-Test 502 cau (3 phut - 2 gio).",
         "Qwen2.5-VL-7B, 128 frame. z = McNemar so cap tung cau, |z| > 1,96 la co y nghia. Muc chinh: giu 10% token.",
         f"Tien do tung buoc: {O}/STATUS*.txt. Ke hoach: {O}/queue.sh (dau file).", ""]
    vm = sorted((os.path.basename(p)[:-5], p) for p in glob.glob(f"{O}/videomme/*.json"))
    L += section("Video-MME", vm, True, vmme_refs, ["short", "medium", "long"])
    for ds, title in [("lvbench", "LVBench"), ("mlvu", "MLVU-Test")]:
        files = sorted((os.path.basename(p)[:-5], p) for p in glob.glob(f"{O}/{ds}/*.json"))
        buckets = []
        if files:
            bs = {}
            for r in load_jsonl(files[0][1]):
                bs[r.get("bucket")] = bs.get(r.get("bucket"), 0) + 1
            buckets = [b for b, _ in sorted(bs.items(), key=lambda x: -x[1]) if b][:6]

        def refs(n, ds=ds):
            r = ret_of(n)
            if not r:
                return []
            base = OLD if O != OLD else O
            out = [(f"uniform_r{r}", f"{base}/{ds}/uniform_r{r}.json"), (f"holitom_r{r}", f"{base}/{ds}/holitom_r{r}.json")]
            if O != OLD:
                out += [(f"R{r[1:]}E_r{r}", f"{OLD}/{ds}/R{r[1:]}E_r{r}.json")]   # RAP cung muc: R10E, R02E, ...
                out += [(f"{h}_r{r}", f"{OLD}/{ds}/{h}_r{r}.json") for h in ("H4", "H5", "H6")]
            return out
        L += section(title, files, False, refs, buckets)
    L += train_lines()
    for st in sorted(glob.glob(f"{O}/STATUS*.txt")):
        L += [f"== 20 dong cuoi {os.path.basename(st)} ==", ""] + [x.rstrip() for x in open(st).readlines()[-20:]] + [""]
    open(f"{O}/KET_QUA.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
