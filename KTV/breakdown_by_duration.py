import csv, json, sys
sys.path.insert(0, ".")
from eval.compute_accuracy import prediction_to_index, load_jsonl

pred_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/pred_casqta_fixed1872_kmax28.json"
csv_path = "playground/gt_qa_files/VideoMME/val_qa.csv"

qid_to_duration = {r["question_id"]: r["duration"] for r in csv.DictReader(open(csv_path))}

records = load_jsonl(pred_path)
by_bucket = {}
for r in records:
    qid = r.get("question_id", r.get("id"))
    bucket = qid_to_duration.get(qid, "UNKNOWN")
    stats = by_bucket.setdefault(bucket, {"correct": 0, "total": 0, "unparsed": 0})
    stats["total"] += 1
    idx = prediction_to_index(r.get("pred"), r.get("candidates", []))
    if idx is None:
        stats["unparsed"] += 1
    elif idx == r.get("answer_number"):
        stats["correct"] += 1

for b in ("short", "medium", "long", "UNKNOWN"):
    if b in by_bucket:
        s = by_bucket[b]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"{b:8s} correct={s['correct']:4d} total={s['total']:4d} unparsed={s['unparsed']:3d} acc={acc:.4f}")
