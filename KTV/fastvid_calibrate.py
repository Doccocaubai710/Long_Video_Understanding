"""
fastvid_calibrate.py -- do ti le giu THUC cua FastVID theo ti le danh nghia, tren 60 video (moi 15 video lay 1,
deu ca short/medium/long), cung cau hinh danh gia cua du an (128 frame, max_pixels_units=16384).
FastVID khong giu dung ti le danh nghia (phu thuoc so segment cua tung video) -> muon so cung ngan sach voi
Uniform thi phai chon ti le danh nghia sao cho trung binh ti le giu thuc ~ muc muc tieu.
Ghi: outputs/kitoke_baseline/fastvid_calib_levels.json  {"target": {"nominal":..., "actual_mean":...}}
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import fastvid_port as fv
import run_inference_qwen25vl_dyseg as dm
from qwen_vision_process_fastvid import process_vision_info

TARGETS = [0.40, 0.15, 0.10, 0.05, 0.02]
GRID = [0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.055, 0.06, 0.07, 0.085, 0.10, 0.105, 0.11, 0.12, 0.13, 0.15, 0.16,
        0.17, 0.18, 0.20, 0.25, 0.30, 0.36, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55]
O = "outputs/kitoke_baseline"
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 15

max_pixels = 16384 * 28 * 28
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)
llm = model.model if hasattr(model, "model") else model

gt = json.load(open("playground/gt_qa_files/VideoMME/val_qa.json"))
videos, seen = [], set()
for s in gt:
    if s["video_name"] not in seen:
        seen.add(s["video_name"]); videos.append(s)
rows = []
log = open(f"{O}/fastvid_calib_levels.log", "w")
for vi in range(STEP // 2, len(videos), STEP):
    s = videos[vi]
    path = os.path.join(dm.resolve_path("Datasets/Video-MME/data/data"), s["video_name"])
    try:
        messages = dm.build_messages(path, s["question"], s["candidates"], max_pixels)
        messages[1]["content"][0]["max_frames"] = 128
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=None, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            emb, attn = fv.fastvid_frame_attn(llm, inputs["pixel_values_videos"], inputs["video_grid_thw"])
            emb = emb.to(torch.bfloat16)
            row = {"vi": vi, "bucket": vi // 300}
            for r in GRID:
                torch.manual_seed(0)
                try:
                    kept, _, diag = fv.fastvid_select(emb, attn, inputs["video_grid_thw"],
                                                      fv.fastvid_params(SimpleNamespace(retention=r)))
                    row[f"{r}"] = kept.numel() / emb.shape[0]
                    row["segs"] = diag.get("num_segments")
                except Exception as e:  # muc nen qua manh co the lam ban goc loi
                    row[f"{r}"] = None
                    row[f"err{r}"] = f"{type(e).__name__}: {e}"[:160]
        rows.append(row)
        log.write(json.dumps(row) + "\n"); log.flush()
    except Exception as e:
        log.write(f"SKIP {vi}: {type(e).__name__}: {e}\n"); log.flush()
    torch.cuda.empty_cache()

mean, fail = {}, {}
for r in GRID:
    v = [x[f"{r}"] for x in rows]
    fail[r] = sum(y is None for y in v)
    ok = [y for y in v if y is not None]
    mean[r] = float(np.mean(ok)) if ok else None
out = {"n_videos": len(rows), "grid_mean": {str(r): mean[r] for r in GRID}, "grid_fail": {str(r): fail[r] for r in GRID},
       "by_bucket": {b: {str(r): float(np.mean([x[f"{r}"] for x in rows if x["bucket"] == b and x[f"{r}"] is not None] or [np.nan]))
                         for r in GRID} for b in (0, 1, 2)}, "levels": {}}
xs = [r for r in GRID if mean[r] is not None and fail[r] == 0]
ys = [mean[r] for r in xs]
for t in TARGETS:
    if xs and min(ys) <= t <= max(ys):
        nom = float(np.interp(t, ys, xs))
        out["levels"][f"{t}"] = {"nominal": round(nom, 4)}
    else:
        out["levels"][f"{t}"] = {"nominal": None, "note": f"ngoai khoang do duoc [{min(ys) if ys else None}, {max(ys) if ys else None}]"}
json.dump(out, open(f"{O}/fastvid_calib_levels.json", "w"), indent=1)
log.write("DONE\n"); log.close()
print(json.dumps(out["levels"], indent=1)); print(json.dumps(out["grid_mean"])); print(json.dumps(out["grid_fail"]))
