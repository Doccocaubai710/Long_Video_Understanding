"""Cong cu cho hang doi rap_elastic (chay tu thu muc KTV).

  python outputs/rap_elastic/tools.py delta A.json B.json
      -> in 1 so: (diem A - diem B) tren cac cau chung, don vi diem (%). Dung de quyet dinh nhanh re.
  python outputs/rap_elastic/tools.py report
      -> viet lai outputs/rap_elastic/KET_QUA.txt tu moi file du doan hien co.
"""
import csv
import datetime
import math
import os
import sys

sys.path.insert(0, ".")
from eval.compute_accuracy import load_jsonl, prediction_to_index  # noqa: E402

O = "outputs/rap_elastic"
B = "outputs/kitoke_baseline"
CSV = "playground/gt_qa_files/VideoMME/val_qa.csv"

with open(CSV, newline="", encoding="utf-8") as f:
    BUCKET = {r["question_id"]: r["duration"] for r in csv.DictReader(f)}


def load(path):
    out = {}
    for r in load_jsonl(path):
        idx = prediction_to_index(r.get("pred"), r.get("candidates", []))
        out[str(r.get("question_id"))] = (idx is not None and idx == r.get("answer_number"))
    return out


def acc(d, key="overall"):
    ids = [q for q in d if key == "overall" or BUCKET.get(q) == key]
    return 100.0 * sum(d[q] for q in ids) / len(ids) if ids else float("nan")


def paired(a, b):
    common = [q for q in a if q in b]
    if not common:
        return float("nan"), float("nan"), 0
    da = 100.0 * sum(a[q] for q in common) / len(common)
    db = 100.0 * sum(b[q] for q in common) / len(common)
    oa = sum(a[q] and not b[q] for q in common)
    ob = sum(b[q] and not a[q] for q in common)
    z = (oa - ob) / math.sqrt(oa + ob) if oa + ob else 0.0
    return da - db, z, len(common)


# (ten, mo ta, file du doan, [(ten moc, file moc), ...])
RUNS_FILE = lambda name: os.path.join(O, name)  # noqa: E731
P = "pred_videomme_f128_"
REF = {
    "Uniform10": f"{B}/{P}uniform_r010.json", "Uniform25": f"{B}/{P}uniform_r025.json",
    "Uniform02": f"{B}/{P}uniform_r002.json", "Uniform05": f"{B}/{P}uniform_r005.json",
    "HoliTom10": f"{B}/{P}holitom_r010.json", "HoliTom25": f"{B}/{P}holitom_r025.json",
    "HoliTom02": f"{B}/{P}holitom_r002.json", "HoliTom05": f"{B}/{P}holitom_r005.json",
    "Dense128": f"{B}/{P}dense.json",
    "RunE25": f"outputs/rap/{P}rap_runE_r025.json", "R10E": f"outputs/rap/{P}rap_R10E_r010.json",
    "V10E": f"outputs/rap_vjepa/{P}rap_V10E_r010.json", "V25E": f"outputs/rap_vjepa25/{P}rap_V25E_r025.json",
    "M3DB": f"{O}/{P}M3D-B_r025.json",
    "R02E": f"outputs/rap_next/{P}rap_R02E_r002.json", "R05E": f"outputs/rap_next/{P}rap_R05E_r005.json",
}
EXPS = [
    ("Q1a sub300", "Uniform 10% + chia lai neo theo canh (khong train)", "sub300_AS10U.json", ["Uniform10"]),
    ("Q1b sub300", "RAP R10E + chia lai neo theo canh (khong train lai)", "sub300_AS10R.json", ["R10E", "Uniform10"]),
    ("Q1a full", "Uniform 10% + chia lai neo theo canh", f"{P}AS10U_r010.json", ["Uniform10"]),
    ("Q1b full", "RAP R10E + chia lai neo theo canh", f"{P}AS10R_r010.json", ["R10E", "Uniform10", "HoliTom10"]),
    ("Q3 ELA@25", "RAP-Elastic (1 checkpoint), cham o 25%", f"{P}ELA_r025.json", ["RunE25", "Uniform25", "HoliTom25"]),
    ("Q3 ELA@10", "RAP-Elastic (1 checkpoint), cham o 10%", f"{P}ELA_r010.json", ["R10E", "Uniform10", "HoliTom10"]),
    ("Q4 R02W3", "RAP 2%, cua so +-3 unit", f"{P}R02W3_r002.json", ["R02E", "Uniform02", "HoliTom02"]),
    ("Q5b sub300", "Dense 256 frame (thay manh hon?)", "sub300_DENSE256.json", ["Dense128"]),
    ("F2 DENSE256", "Dense 256 frame, du 2.700 cau", f"{P}DENSE256_r100.json", ["Dense128"]),
    ("Q5c T256", "Tro nen 256 frame xuong 12,5% (~4.000 token), thay = dense 256", f"{P}T256_r0125.json",
     ["Dense128", "RunE25", "Uniform25"]),
    ("Q5a R05W3", "RAP 5%, cua so +-3 unit", f"{P}R05W3_r005.json", ["R05E", "Uniform05", "HoliTom05"]),
    ("Q6 R10S", "RAP 10% train voi neo chia theo canh", f"{P}R10S_r010.json", ["R10E", "Uniform10", "HoliTom10"]),
    ("H1 SU10", "Uniform 10% + token tom tat moi canh (HTP-LVU, khong train)", f"{P}SU10_r010.json", ["Uniform10", "HoliTom10"]),
    ("H2 RSS10", "RAP 10% + neo chi nhin cung canh + token tom tat (HTP-LVU)", f"{P}RSS10_r010.json", ["R10E", "Uniform10", "HoliTom10"]),
    ("C0", "Medoid cum k-means trong tung canh, khong train, 25% (HTP-LVU 1.2 + MAP-3D tube)", f"{P}C0_r025.json", ["Uniform25", "RunE25", "HoliTom25"]),
    # queue6 (20/10 toi): MAP-3D 3.3, 3.2 + HTP-LVU bypass + kiem chung V-JEPA
    ("Q6-1 M3D-B", "RAP 25% + cross-attn phat khoang cach 3D (MAP-3D 3.3), KV +-3", f"{P}M3D-B_r025.json", ["RunE25", "Uniform25", "HoliTom25"]),
    ("Q6-2 SU25", "Uniform 25% + token tom tat moi canh (HTP-LVU residual bypass, khong train)", f"{P}SU25_r025.json", ["Uniform25", "HoliTom25", "RunE25"]),
    ("Q6-3 M3D-BD", "M3D-B + Deformable 3D Tube Adapter (MAP-3D 3.2)", f"{P}M3D-BD_r025.json", ["RunE25", "M3DB", "Uniform25", "HoliTom25"]),
    ("Q6-4a V10E_s1", "V-JEPA pretrain + KL 10%, seed 1 (kiem chung V10E)", f"{P}V10E_s1_r010.json", ["R10E", "V10E", "Uniform10"]),
    ("Q6-4b R10E7k", "Doi chung: RAP 10% KL 7.000 buoc, khong V-JEPA", f"{P}R10E7k_r010.json", ["R10E", "V10E", "Uniform10"]),
    ("Q6-5 RSS25", "RAP 25% + neo chi nhin cung canh (HTP-LVU S1) + token tom tat", f"{P}RSS25_r025.json", ["RunE25", "Uniform25", "HoliTom25"]),
    ("Q6-5 RSW25", "RAP 25% + neo chi nhin cung canh (HTP-LVU S1), khong token tom tat", f"{P}RSW25_r025.json", ["RunE25", "Uniform25", "HoliTom25"]),
    ("Q6-6 M3DF@25", "MAP-3D day du: 3.3 + 3.2 + Elastic 4.1, cham 25%", f"{P}M3DF_r025.json", ["RunE25", "M3DB", "Uniform25", "HoliTom25"]),
    ("Q6-6 M3DF@10", "MAP-3D day du: 3.3 + 3.2 + Elastic 4.1, cham 10%", f"{P}M3DF_r010.json", ["R10E", "Uniform10", "HoliTom10"]),
    ("F1 ELA_s1@25", "RAP-Elastic seed 1, 25%", f"{P}ELA_s1_r025.json", ["RunE25", "Uniform25"]),
    ("F1 ELA_s1@10", "RAP-Elastic seed 1, 10%", f"{P}ELA_s1_r010.json", ["R10E", "Uniform10"]),
]


def report():
    lines = [f"KET QUA HANG DOI rap_elastic -- cap nhat {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC",
             "Tien do chi tiet: outputs/rap_elastic/STATUS.txt. Ke hoach: outputs/rap_elastic/queue.sh (dau file).",
             "So 'sub300' chi tren 300 cau (de quyet dinh noi bo, nhieu nhieu). So 'full' tren du 2.700 cau.",
             "z: McNemar so cap tung cau, |z| > 1,96 la co y nghia.", ""]
    for tag, desc, fn, refs in EXPS:
        path = RUNS_FILE(fn)
        if not os.path.exists(path):
            lines.append(f"[{tag}] {desc}: CHUA CO")
            continue
        a = load(path)
        lines.append(f"[{tag}] {desc}  ({len(a)} cau)")
        lines.append("    " + "  ".join(f"{k}={acc(a, k):.2f}" for k in ("short", "medium", "long", "overall")))
        for r in refs:
            if os.path.exists(REF[r]):
                d, z, n = paired(a, load(REF[r]))
                lines.append(f"    so voi {r:10s}: {d:+.2f} diem (z={z:+.2f}, {n} cau chung)")
        lines.append("")
    stops = os.path.join(O, "STOPS.txt")
    if os.path.exists(stops):
        lines += ["CAC RUN BI NGAT SOM (thap hon moc co y nghia, z <= -1,96 sau >= 900 cau):"]
        lines += ["    " + l.rstrip() for l in open(stops, encoding="utf-8") if l.strip()]
    tmp = os.path.join(O, "KET_QUA.txt.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(O, "KET_QUA.txt"))


if __name__ == "__main__":
    if sys.argv[1] == "delta":
        if not (os.path.exists(sys.argv[2]) and os.path.exists(sys.argv[3])):
            print("nan")
        else:
            print(f"{paired(load(sys.argv[2]), load(sys.argv[3]))[0]:.3f}")
    elif sys.argv[1] == "guard":   # guard A.json Uniform25,RunE25 [min_n] [z] -> "STOP ..." hoac "OK ..."
        a = load(sys.argv[2]) if os.path.exists(sys.argv[2]) else {}
        min_n = int(sys.argv[4]) if len(sys.argv) > 4 else 900
        zt = float(sys.argv[5]) if len(sys.argv) > 5 else -1.96
        msg = []
        for r in sys.argv[3].split(","):
            d, z, n = paired(a, load(REF[r]))
            msg.append(f"{r} {d:+.2f} (z={z:+.2f}, {n} cau)")
            if n >= min_n and z <= zt:
                print(f"STOP thap hon {r}: {d:+.2f} diem, z={z:+.2f}, {n} cau | " + "; ".join(msg))
                sys.exit(0)
        print("OK " + "; ".join(msg))
    elif sys.argv[1] == "report":
        report()
