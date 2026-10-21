"""
analyze_perceiver_phase2.py -- bo sung cho analyze_perceiver_tokens.py:
  A. (anh) Perceiver thuc chat chon token T1 nao: norm cua token duoc chu y,
     vi tri, so cum T2 phan biet; CE khi chi dua dung cac token T1 duoc chon;
     CE "native" dung mRoPE 2D (ban truoc quen mm_token_type_ids).
  B. (Video-MME subset300, 128 frame) tach dong gop: T2+T1* (day du) / chi T2 /
     chi T1* / nhieu thay T2 + T1* / uniform cung so token / khong video.
Chi doc checkpoint, ghi vao outputs/perceiver_analysis/.
"""

import argparse
import json
import os
import random

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vision_process_fastvid import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from analyze_perceiver_tokens import K, agg, ce_seq, load_perceiver, perceiver_forward
from run_inference_qwen25vl_dyseg import dyseg_segments
from train_perceiver_stage1 import CAPTION_PROMPT, MIN_PIXELS, SYSTEM_PROMPT


def greedy_clusters(X, thr):
    Xn = F.normalize(X.float(), dim=-1)
    reps = []
    for i in range(X.shape[0]):
        if not reps or float((Xn[i] @ Xn[reps].t()).max()) < thr:
            reps.append(i)
    return reps


def part_a(model, processor, p1, p2, args, device, out_dir):
    d = json.load(open(args.coco_captions))
    img_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    cap_by_img = {}
    for ann in d["annotations"]:
        cap_by_img.setdefault(img_by_id[ann["image_id"]], ann["caption"].strip())
    names = sorted(cap_by_img)
    random.Random(0).shuffle(names)
    names = names[: args.n_images]
    llm = model.model if hasattr(model, "model") else model
    tok_id = model.config.image_token_id
    tk = processor.tokenizer
    max_pixels = args.image_units * 28 * 28
    rows, pos_y, pos_x = [], [], []
    for si, name in enumerate(names):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": os.path.join(args.coco_dir, name), "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
                    {"type": "text", "text": CAPTION_PROMPT},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ii, vi, _ = process_vision_info(messages, return_video_kwargs=True)
            inp = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(device)
            feat = llm.get_image_features(inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            T1 = (feat if not isinstance(feat, (list, tuple)) else torch.cat(feat, dim=0)).float()
            thw = inp["image_grid_thw"][0].tolist()
            h, w = int(thw[1]) // 2, int(thw[2]) // 2
            n = T1.shape[0]
            if n != h * w or n < 100:
                continue
            row = {"name": name, "n": n}
            nrm = T1.norm(dim=-1)
            rank = nrm.argsort().argsort().float() / (n - 1)  # percentile norm cua tung token
            top5 = nrm >= nrm.quantile(0.95)
            for tag, p in (("s1", p1), ("s2", p2)):
                T2, maps, _ = perceiver_forward(p, T1)
                for l, A in enumerate(maps):
                    am = A.argmax(-1).flatten()  # [heads*Q] token moi head chon
                    row[f"{tag}/L{l}/attended_unique_tokens"] = int(am.unique().numel())
                    row[f"{tag}/L{l}/attended_norm_percentile"] = float(rank[am].mean())
                    row[f"{tag}/L{l}/attended_norm_over_mean"] = float(nrm[am].mean() / nrm.mean())
                    row[f"{tag}/L{l}/mass_on_top5pct_norm"] = float(A.mean(0)[:, top5].sum(-1).mean())
                    row[f"{tag}/L{l}/mass_on_border"] = float(
                        A.mean((0, 1)).view(h, w)[[0, -1], :].sum() + A.mean((0, 1)).view(h, w)[1:-1, [0, -1]].sum())
                    row[f"{tag}/L{l}/border_area_frac"] = (2 * w + 2 * (h - 2)) / n
                    if tag == "s1" and l == len(maps) - 1:
                        u = am.unique()
                        pos_y += ((u // w).float() / max(h - 1, 1)).tolist()
                        pos_x += ((u % w).float() / max(w - 1, 1)).tolist()
                for thr in (0.9, 0.95, 0.99):
                    row[f"{tag}/T2_clusters_cos{thr}"] = len(greedy_clusters(T2, thr))
                if tag == "s1":
                    T2_s1, maps_s1 = T2, maps
            row["T1/top5pct_norm_over_mean"] = float(nrm[top5].mean() / nrm.mean())
            row["T1/norm_max_over_median"] = float(nrm.max() / nrm.median())

            # ---- CE
            full_ids = inp["input_ids"][0]
            nz = (full_ids == tok_id).nonzero(as_tuple=True)[0]
            pre_ids, post_ids = full_ids[: nz[0]], full_ids[nz[-1] + 1:]
            cap_ids = torch.tensor(tk(cap_by_img[name], add_special_tokens=False).input_ids + [tk.eos_token_id],
                                   dtype=full_ids.dtype, device=device)
            C = cap_ids.shape[0]
            ce1 = lambda X: float(ce_seq(model, pre_ids, post_ids, cap_ids, X.unsqueeze(0).float(), tok_id)[0])
            ids_nat = torch.cat([full_ids, cap_ids]).unsqueeze(0)
            kw = {}
            if "mm_token_type_ids" in inp:
                kw["mm_token_type_ids"] = torch.cat(
                    [inp["mm_token_type_ids"][0], torch.zeros(C, dtype=inp["mm_token_type_ids"].dtype, device=device)]).unsqueeze(0)
            out = model(input_ids=ids_nat, attention_mask=torch.ones_like(ids_nat), pixel_values=inp["pixel_values"],
                        image_grid_thw=inp["image_grid_thw"], return_dict=True, **kw)
            Ln = ids_nat.shape[1]
            row["ce/T1_full_native_mrope"] = float(F.cross_entropy(out.logits[0, Ln - C - 1: Ln - 1].float(), cap_ids))
            row["has_mm_token_type_ids"] = int("mm_token_type_ids" in inp)
            row["ce/T1_full_seqpos"] = ce1(T1)
            row["ce/T2"] = ce1(T2_s1)
            reps = greedy_clusters(T2_s1, 0.95)
            row["ce/T2_dedup_cos0.95"] = ce1(T2_s1[reps])
            row["n/T2_dedup_cos0.95"] = len(reps)
            sel = maps_s1[-1].mean(0).argmax(-1).unique()  # token T1 ma cac query chu y nhat (trung binh head)
            row["n/attended_T1_headavg"] = int(sel.numel())
            row["ce/attended_T1_only"] = ce1(T1[sel])
            selh = maps_s1[-1].argmax(-1).flatten().unique()  # tinh theo tung head
            row["n/attended_T1_perhead"] = int(selh.numel())
            row["ce/attended_T1_perhead_only"] = ce1(T1[selh])
            kk = int(selh.numel())
            row["ce/uniform_same_count_as_perhead"] = ce1(T1[torch.linspace(0, n - 1, kk).round().long()])
            row["ce/topnorm_same_count_as_perhead"] = ce1(T1[nrm.topk(kk).indices.sort().values])
            row["ce/topnorm64"] = ce1(T1[nrm.topk(K).indices.sort().values])
            row["ce/lownorm64"] = ce1(T1[(-nrm).topk(K).indices.sort().values])
            row["ce/uniform4"] = ce1(T1[torch.linspace(0, n - 1, 4).round().long()])
            row["ce/topnorm4"] = ce1(T1[nrm.topk(4).indices.sort().values])
            rows.append(row)
        except Exception as e:
            print(f"SKIP image {name}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        if len(rows) % 20 == 0:
            print(f"[A] {len(rows)} done", flush=True)
    res = {"agg": agg(rows), "rows": rows,
           "attended_pos_hist_y": np.histogram(pos_y, bins=6, range=(0, 1))[0].tolist(),
           "attended_pos_hist_x": np.histogram(pos_x, bins=6, range=(0, 1))[0].tolist()}
    json.dump(res, open(os.path.join(out_dir, "phase2_image_results.json"), "w"))
    print("[A] DONE", flush=True)


def part_b(model, processor, p2, args, device, out_dir):
    qa = json.load(open(args.videomme_gt))
    llm = model.model if hasattr(model, "model") else model
    tok_id = model.config.video_token_id
    max_pixels = args.video_units * 28 * 28
    out_path = os.path.join(out_dir, "phase2_videomme_decomp.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    cache = {}
    f = open(out_path, "a")
    for qi, s in enumerate(qa):
        qid = str(s.get("id", s.get("question_id")))
        if qid in done:
            continue
        try:
            vname = s["video_name"]
            messages = dm.build_messages(os.path.join(args.videomme_dir, vname), s["question"], s["candidates"], max_pixels)
            messages[1]["content"][0]["max_frames"] = args.max_frames
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ii, vv, _ = process_vision_info(messages, return_video_kwargs=True)
            inp = processor(text=[text], images=ii, videos=vv, padding=True, return_tensors="pt").to(device)
            if cache.get("name") != vname:
                feat = llm.get_video_features(inp["pixel_values_videos"], video_grid_thw=inp["video_grid_thw"])
                if hasattr(feat, "pooler_output"):
                    feat = feat.pooler_output
                T1f = (feat if not isinstance(feat, (list, tuple)) else torch.cat(feat, dim=0)).float()
                thw = inp["video_grid_thw"][0].tolist()
                T, N = int(thw[0]), (int(thw[1]) // 2) * (int(thw[2]) // 2)
                T1 = T1f.view(T, N, -1)
                segs = dyseg_segments(T1.mean(1), c=args.dyseg_c, tau=args.dyseg_tau)
                full, t2only, staronly, noisefull = [], [], [], []
                g = torch.Generator(device="cpu").manual_seed(qi)
                for (a, b) in segs:
                    X = T1[a:b].reshape(-1, T1.shape[-1])
                    T2, _, _ = perceiver_forward(p2, X)
                    sim = F.normalize(T2, dim=-1) @ F.normalize(X, dim=-1).t()
                    star = X[sim.max(dim=0).values >= args.pointing_tau]
                    noise = (torch.randn(T2.shape, generator=g) * float(T2.std()) + float(T2.mean())).to(device)
                    full.append(torch.cat([T2, star]))
                    t2only.append(T2)
                    staronly.append(star)
                    noisefull.append(torch.cat([noise, star]))
                cat = lambda L: torch.cat(L, dim=0)
                nf, nt = cat(full).shape[0], cat(t2only).shape[0]
                uni = lambda k: T1f[torch.linspace(0, T1f.shape[0] - 1, k).round().long()] if k > 0 else T1f[:0]
                cache = {"name": vname, "n_total": T1f.shape[0], "n_segments": len(segs), "variants": {
                    "full_T2_plus_T1star": cat(full), "T2_only": cat(t2only), "T1star_only": cat(staronly),
                    "noiseT2_plus_T1star": cat(noisefull), "uniform_same_as_full": uni(nf),
                    "uniform_same_as_T2only": uni(nt), "blind_no_video": T1f[:0],
                }}
            full_ids = inp["input_ids"][0]
            nz = (full_ids == tok_id).nonzero(as_tuple=True)[0]
            pre_ids, post_ids = full_ids[: nz[0]], full_ids[nz[-1] + 1:]
            rec = {"id": qid, "video": vname, "answer": s["answer"], "task_name": s.get("task_name", ""),
                   "n_total": cache["n_total"], "n_segments": cache["n_segments"], "pred": {}, "ntok": {}}
            for nm, X in cache["variants"].items():
                k = X.shape[0]
                ids = torch.cat([pre_ids, torch.full((k,), tok_id, dtype=pre_ids.dtype, device=device), post_ids]).unsqueeze(0)
                emb = model.get_input_embeddings()(ids).clone()
                if k > 0:
                    emb[ids == tok_id] = X.to(emb.dtype)
                L = ids.shape[1]
                pos = torch.arange(L, device=device).view(1, 1, L).expand(3, 1, L)
                gen = model.generate(inputs_embeds=emb, position_ids=pos, attention_mask=torch.ones_like(ids),
                                     eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
                                     do_sample=False, num_beams=1, max_new_tokens=8)
                rec["pred"][nm] = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
                rec["ntok"][nm] = k
            f.write(json.dumps(rec) + "\n")
            f.flush()
        except Exception as e:
            print(f"SKIP q {qid}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        if qi % 10 == 0:
            print(f"[B] {qi+1}/{len(qa)}", flush=True)
    print("[B] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--ckpt_s1", default="outputs/perceiver_stage1_selfattn.pt")
    ap.add_argument("--ckpt_s2", default="outputs/perceiver_stage2_v3.pt")
    ap.add_argument("--coco_dir", default="Datasets/COCO/val2017")
    ap.add_argument("--coco_captions", default="Datasets/COCO/annotations/captions_val2017.json")
    ap.add_argument("--videomme_dir", default="Datasets/Video-MME/data/data")
    ap.add_argument("--videomme_gt", default="playground/gt_qa_files/VideoMME/val_qa_subset300.json")
    ap.add_argument("--out_dir", default="outputs/perceiver_analysis")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--image_units", type=int, default=1024)
    ap.add_argument("--video_units", type=int, default=16384)
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--dyseg_c", type=int, default=8)
    ap.add_argument("--dyseg_tau", type=float, default=0.9)
    ap.add_argument("--pointing_tau", type=float, default=0.5)
    ap.add_argument("--skip_a", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    p1, _ = load_perceiver(args.ckpt_s1, device)
    p2, _ = load_perceiver(args.ckpt_s2, device)
    with torch.no_grad():
        if not args.skip_a:
            processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.image_units * 28 * 28, min_pixels=MIN_PIXELS)
            part_a(model, processor, p1, p2, args, device, args.out_dir)
        processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.video_units * 28 * 28, min_pixels=dm.MIN_PIXELS)
        part_b(model, processor, p2, args, device, args.out_dir)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
