"""
check_bare_vs_elaborated.py

Kiem tra xem prediction "tra loi cut" (chi 1 chu cai, vd "A)") co lien quan
den ty le sai cao hon prediction "co giai thich" hay khong. Dung duoc ngay
ca khi run_inference_multiple_choice_qa.py CHUA chay xong (file la JSONL,
moi dong ghi ngay sau khi xong 1 cau).

Usage:
  python check_bare_vs_elaborated.py outputs/pred_casqta_gamma90_kmax80_bbar80.json
"""
import re
import sys

sys.path.insert(0, ".")
from eval.compute_accuracy import prediction_to_index, load_jsonl

BARE_RE = re.compile(r"^[A-Fa-f][\).:\-]?\s*$")


def main():
    path = sys.argv[1]
    records = load_jsonl(path)

    groups = {"bare": {"correct": 0, "total": 0}, "elaborated": {"correct": 0, "total": 0}}
    for r in records:
        pred = (r.get("pred") or "").strip()
        idx = prediction_to_index(pred, r.get("candidates", []))
        if idx is None:
            continue
        g = "bare" if BARE_RE.match(pred) else "elaborated"
        groups[g]["total"] += 1
        if idx == r.get("answer_number"):
            groups[g]["correct"] += 1

    print(f"questions so far: {len(records)}")
    for g in ("bare", "elaborated"):
        c, t = groups[g]["correct"], groups[g]["total"]
        acc = c / t if t else 0
        print(f"{g:12s} n={t:5d} correct={c:5d} acc={acc:.4f}")


if __name__ == "__main__":
    main()
