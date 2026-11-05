"""prepare_long_gt.py -- file cau hoi dang Video-MME cho LVBench va MLVU-Test (video da giai nen).

Usage (tu thu muc KTV): python -m tools.prepare_long_gt
  -> playground/gt_qa_files/LVBench/val_qa.json, playground/gt_qa_files/MLVU_Test/val_qa_mc.json
Truong "bucket" dung de chia diem: LVBench = loai cau hoi dau tien, MLVU = nhom tac vu (AO, TR, ...).
"""
import ast
import json
import os
import re

LETTERS = "ABCDEF"


def find_videos(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
                out.setdefault(os.path.splitext(f)[0], os.path.relpath(os.path.join(d, f), root))
    return out


def lvbench():
    root = "Datasets/LVBench/videos"
    vids = find_videos(root)
    rows, miss = [], 0
    for line in open("Datasets/LVBench/meta/video_info.meta.jsonl"):
        m = json.loads(line)
        if m["key"] not in vids:
            miss += 1
            continue
        for q in m["qa"]:
            parts = re.split(r"\n\(([A-F])\)\s*", q["question"])
            stem, opts = parts[0].strip(), parts[2::2]
            letters = parts[1::2]
            cands = [f"{l}. {o.strip()}" for l, o in zip(letters, opts)]
            rows.append({"task_name": "LVBench", "video_name": vids[m["key"]], "question_id": str(q["uid"]),
                         "question": stem, "candidates": cands, "answer_number": letters.index(q["answer"]),
                         "answer": cands[letters.index(q["answer"])], "bucket": q["question_type"][0]})
    os.makedirs("playground/gt_qa_files/LVBench", exist_ok=True)
    json.dump(rows, open("playground/gt_qa_files/LVBench/val_qa.json", "w"), ensure_ascii=False, indent=1)
    print(f"LVBench: {len(rows)} cau, {len(vids)} video, thieu {miss} video")


def mlvu():
    root = "Datasets/MLVU_Test/videos"
    vids = find_videos(root)
    rows, miss = [], 0
    for s in json.load(open("playground/gt_qa_files/MLVU_Test/val_qa.json")):
        key = os.path.splitext(s["video_name"])[0]
        if key not in vids:
            miss += 1
            continue
        c = s["candidates"]
        c = ast.literal_eval(c) if isinstance(c, str) else c
        cands = [f"{LETTERS[i]}. {x}" for i, x in enumerate(c)]
        a = int(s["answer_number"])
        rows.append({"task_name": "MLVU_Test", "video_name": vids[key], "question_id": s["question_id"],
                     "question": s["question"], "candidates": cands, "answer_number": a, "answer": cands[a],
                     "bucket": s["question_id"].split("_")[0]})
    json.dump(rows, open("playground/gt_qa_files/MLVU_Test/val_qa_mc.json", "w"), ensure_ascii=False, indent=1)
    print(f"MLVU_Test: {len(rows)} cau, {len(vids)} video, thieu {miss} cau")


if __name__ == "__main__":
    lvbench()
    mlvu()
