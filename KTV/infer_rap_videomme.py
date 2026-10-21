"""
infer_rap_videomme.py -- danh gia Perceiver co neo (rap_model.py) tren VideoMME,
cung harness voi bo moc (run_inference_qwen25vl_kitoke.py): 128 frame,
max_pixels_units=16384, time_mode=lmms, giu 25%.

--policy none  : khong co Perceiver, dau ra = token neo = Uniform 25%. Dung de
                 kiem tra harness: ket qua phai trung pred_videomme_f128_uniform_r025.
--ablate noise : thay delta cua Perceiver bang nhieu Gauss cung do lon (doi chung).
--placer P.pt  : vi tri neo do AnchorPlacer (place_model.py) chon greedy thay cho luoi Uniform.

Usage (chay tu thu muc KTV):
  python infer_rap_videomme.py --policy outputs/rap/runA.pt --output_name pred_videomme_f128_rap_runA_r025
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import torch
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from alloc_model import scene_alloc_indices
from scene_model import cluster_medoid_indices, merge_kept, scene_ids, summary_layout
from place_model import load_placer, place, uniform_layout
from local_data import local_mirror
from rap_model import load_resampler, uniform_indices
from run_inference_qwen25vl_2pass import build_pruned_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--output_name", required=True)
    ap.add_argument("--output_dir", default="outputs/rap")
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--video_dir", default="Datasets/Video-MME/data/data")
    ap.add_argument("--gt_file", default="playground/gt_qa_files/VideoMME/val_qa.json")
    ap.add_argument("--retention", type=float, default=0.25)
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--max_pixels_units", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ablate", default="")
    ap.add_argument("--placer", default="")
    ap.add_argument("--alloc", default="")             # "scene": chia lai so neo giua cac unit (alloc_model.py)
    ap.add_argument("--alloc_floor", type=float, default=0.5)
    ap.add_argument("--cluster_lam", type=float, default=1.0)  # --alloc cluster: trong so toa do (t, y, x)
    ap.add_argument("--scene_window", type=int, default=0)  # 1: neo chi nhin token cung canh (scene_model.py)
    ap.add_argument("--summary", type=int, default=0)       # 1: moi canh 1 token tom tat, lay tu ngan sach neo
    ap.add_argument("--scene_kappa", type=float, default=2.0)
    ap.add_argument("--scene_local", type=int, default=8)
    ap.add_argument("--scene_min_len", type=int, default=2)
    ap.add_argument("--scene_max_len", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=2)  # so video giai ma truoc (luong nen); 0 = tat
    args = ap.parse_args()
    args.video_dir = local_mirror(args.video_dir)

    device = "cuda"
    max_pixels = args.max_pixels_units * 28 * 28
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)
    resampler = None
    if args.policy != "none":
        resampler, ckpt = load_resampler(args.policy, device)
        resampler.eval()
        print(f"Loaded resampler {args.policy} step={ckpt.get('step')} config={ckpt['config']}", flush=True)
    placer = None
    if args.placer:
        placer, pckpt = load_placer(args.placer, device)
        placer.eval()
        print(f"Loaded placer {args.placer} step={pckpt.get('step')} config={pckpt['config']}", flush=True)
    torch.manual_seed(0)

    gt = json.load(open(args.gt_file))
    if args.limit > 0:
        gt = gt[: args.limit]
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    stats_path = os.path.join(args.output_dir, f"{args.output_name}.stats.jsonl")
    seen = dm.load_answered_ids(out_path)
    llm = model.model
    vid = model.config.video_token_id
    cache = {}

    # Giai ma truoc cac video sap toi trong luong nen de GPU khong phai doi CPU. Ket qua y het ban tuan tu.
    def decode(video_path):
        msgs = dm.build_messages(video_path, "", [], max_pixels)
        msgs[1]["content"][0]["max_frames"] = args.max_frames
        return process_vision_info(msgs, return_video_kwargs=True)[1]

    todo = []
    for s_ in gt:
        vp = os.path.join(args.video_dir, s_["video_name"])
        if str(s_.get("id", s_.get("question_id"))) not in seen and os.path.exists(vp) and (not todo or todo[-1] != vp):
            todo.append(vp)
    pool = ThreadPoolExecutor(max(1, args.prefetch)) if args.prefetch > 0 else None
    pending, nxt = {}, 0

    def fill(cur):
        nonlocal nxt
        while pool is not None and nxt < len(todo) and len(pending) < args.prefetch:
            if todo[nxt] not in pending and todo[nxt] != cur:
                pending[todo[nxt]] = pool.submit(decode, todo[nxt])
            nxt += 1

    with open(out_path, "a", encoding="utf-8") as fo, open(stats_path, "a", encoding="utf-8") as fs:
        for sample in tqdm(gt):
            aid = str(sample.get("id", sample.get("question_id")))
            if aid in seen:
                continue
            video_path = os.path.join(args.video_dir, sample["video_name"])
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}", flush=True)
                continue
            try:
                messages = dm.build_messages(video_path, sample["question"], sample["candidates"], max_pixels)
                messages[1]["content"][0]["max_frames"] = args.max_frames
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if cache.get("video_path") != video_path:
                    cache.clear()
                    fut = pending.pop(video_path, None)
                    fill(video_path)
                    if fut is not None:
                        video_inputs = fut.result()
                    else:
                        _, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
                    cache["video_path"], cache["video_inputs"] = video_path, video_inputs
                inputs = processor(text=[text], images=None, videos=cache["video_inputs"], padding=True, return_tensors="pt").to(device)
                inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], 1.0)
                with torch.inference_mode():
                    full_embeds = model.get_input_embeddings()(inputs["input_ids"])
                    thw = inputs["video_grid_thw"][0]
                    T, H, W = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
                    N = H * W
                    new_video = "kept_embeds" not in cache
                    if new_video:
                        feat = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]).pooler_output
                        if isinstance(feat, (list, tuple)):
                            feat = torch.cat(feat, dim=0)
                        feat = feat.to(full_embeds.dtype)
                        assert feat.shape[0] == T * N
                        kept_idx = uniform_indices(T * N, args.retention, device)
                        moved = 0.0
                        if args.alloc == "scene":
                            kept_idx, counts = scene_alloc_indices(feat.float().view(T, N, -1), H, W, args.retention,
                                                                   args.alloc_floor)
                            moved = float(counts.max() - counts.min())  # do chenh so neo giua cac unit
                        if args.alloc == "cluster":  # C0: medoid cua cum k-means trong tung canh (scene_model.py)
                            sid0 = scene_ids(feat.float().view(T, N, -1), H, W, args.scene_kappa, args.scene_local,
                                             args.scene_min_len, args.scene_max_len)
                            kept_idx = cluster_medoid_indices(feat.float().view(T, N, -1), H, W, args.retention, sid0,
                                                              lam=args.cluster_lam)
                            moved = 1.0 - float(torch.isin(kept_idx, uniform_indices(T * N, args.retention, device)).float().mean())
                        if placer is not None:
                            _, umask, counts = uniform_layout(T, N, args.retention, device)
                            kept_idx, _, _ = place(placer(feat.float().view(T, N, -1), W) + placer.prior * umask, counts, sample=False)
                            moved = 1.0 - float(umask.view(-1)[kept_idx].float().mean())
                        sid, sum_idx = None, None
                        if args.scene_window or args.summary:
                            sid = scene_ids(feat.float().view(T, N, -1), H, W, args.scene_kappa, args.scene_local,
                                            args.scene_min_len, args.scene_max_len)
                        if args.summary:
                            kept_idx, sum_idx, sum_emb = summary_layout(feat.float().view(T, N, -1), H, W, args.retention, sid)
                        anchors = feat[kept_idx]
                        delta_ratio = 0.0
                        if resampler is None:
                            kept = anchors
                        else:
                            T2, _ = resampler(feat.float().view(T, N, -1), kept_idx, H, W,
                                              scene=sid if args.scene_window else None)
                            delta = T2 - anchors.float()
                            delta_ratio = float(delta.norm(dim=-1).mean() / anchors.float().norm(dim=-1).mean())
                            if args.ablate == "noise":
                                delta = torch.randn_like(delta) * delta.std()
                            kept = (anchors.float() + delta).to(full_embeds.dtype)
                        if sum_idx is not None:
                            kept_idx, kept = merge_kept(kept_idx, kept, sum_idx, sum_emb)
                        cache["kept_idx"], cache["kept_embeds"] = kept_idx, kept
                        fs.write(json.dumps({"video_name": sample["video_name"], "T": T, "N": N, "n_total": T * N,
                                             "n_kept": int(kept_idx.numel()), "delta_ratio": delta_ratio,
                                             "moved": moved, "n_scene": 0 if sid is None else int(sid.max()) + 1}) + "\n")
                        fs.flush()
                    vpos = (inputs["input_ids"][0] == vid).nonzero(as_tuple=True)[0]
                    position_ids, _ = llm.get_rope_index(
                        input_ids=inputs["input_ids"], image_grid_thw=None, video_grid_thw=inputs["video_grid_thw"],
                        second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=inputs["attention_mask"],
                        mm_token_type_ids=inputs.get("mm_token_type_ids"),
                    )
                    fe, pid, am, _ = build_pruned_sequence(full_embeds, inputs, vid, vpos, position_ids,
                                                           cache["kept_idx"], cache["kept_embeds"])
                    gen = model.generate(inputs_embeds=fe, position_ids=pid, attention_mask=am,
                                         eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
                                         do_sample=False, num_beams=1, max_new_tokens=16)
                output = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            except Exception as e:
                print(f"SKIP {aid} ({video_path}): {type(e).__name__}: {e}", flush=True)
                cache.clear()
                torch.cuda.empty_cache()
                continue
            fo.write(json.dumps({
                "task_name": sample.get("task_name", "unknown"), "question": sample["question"],
                "question_id": str(sample.get("question_id", aid)), "id": aid,
                "answer_number": sample["answer_number"], "candidates": sample["candidates"],
                "answer": sample["answer"], "pred": output,
            }) + "\n")
            fo.flush()
            seen.add(aid)
    print(f"[infer_rap_videomme] done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
