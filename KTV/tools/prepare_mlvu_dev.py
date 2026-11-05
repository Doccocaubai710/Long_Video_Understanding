"""prepare_mlvu_dev.py -- file cau hoi dang Video-MME cho MLVU dev (7 tac vu trac nghiem, 4 lua chon).
  Datasets/MLVU_Dev/MLVU/json/{1..7}_*.json  ->  playground/gt_qa_files/MLVU_Dev/val_qa_mc.json
video_name = "<thu muc tac vu>/<file>", video_dir = Datasets/MLVU_Dev/MLVU/video. bucket = question_type (de tinh M-Avg).
"""
import glob
import json
import os

LETTERS = "ABCDEFGH"
rows = []
for f in sorted(glob.glob("Datasets/MLVU_Dev/MLVU/json/[1-7]_*.json")):
    task = os.path.splitext(os.path.basename(f))[0]
    for i, s in enumerate(json.load(open(f))):
        cands = [f"{LETTERS[j]}. {x}" for j, x in enumerate(s["candidates"])]
        a = s["candidates"].index(s["answer"])
        rows.append({"task_name": "MLVU_Dev", "video_name": f"{task}/{s['video']}", "question_id": f"{s['question_type']}_{i}",
                     "question": s["question"], "candidates": cands, "answer_number": a, "answer": cands[a],
                     "bucket": s["question_type"], "duration": s["duration"]})
os.makedirs("playground/gt_qa_files/MLVU_Dev", exist_ok=True)
json.dump(rows, open("playground/gt_qa_files/MLVU_Dev/val_qa_mc.json", "w"), ensure_ascii=False, indent=1)
miss = sum(not os.path.exists(os.path.join("Datasets/MLVU_Dev/MLVU/video", r["video_name"])) for r in rows)
print(f"MLVU_Dev: {len(rows)} cau, {len({r['video_name'] for r in rows})} video, thieu video {miss} cau")
