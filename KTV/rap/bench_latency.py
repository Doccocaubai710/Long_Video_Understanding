"""
bench_latency.py -- do thoi gian / bo nho cua tung cau hinh nen tren cung harness voi eval_multi.py
(Qwen2.5-VL-7B, 128 frame, max_pixels_units=16384, time_mode=lmms, 16 token sinh, greedy, batch 1).

Moi video: giai ma + vision encoder 1 lan (do rieng), roi voi moi cau hinh do
  - nen:    thoi gian chon/gop token (HDS: ca 2 giai doan; HoliTom: chi buoc chon, attention lay tu lan vision rieng)
  - prefill: 1 lan forward LLM tren chuoi da nen (= thoi gian den token dau tien, khong tinh vision)
  - sinh:   generate 16 token (prefill + giai ma)
  - KV-cache: dung luong that cua past_key_values sau prefill; dinh bo nho GPU tang them trong prefill
Lap --reps lan / video sau 1 lan khoi dong, lay trung vi.

Usage (tu thu muc KTV):
  python -m rap.bench_latency --out outputs/hds7/LATENCY.json \\
      --variant dense=dense --variant uniform_r010=uniform@0.10 --variant hds_r010=ckpt:outputs/hds/train/H3.pt@0.10
"""

import argparse
import json
import os
import statistics
import time
from types import SimpleNamespace

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import run_inference_qwen25vl_dyseg as dm
from common.local_data import local_mirror
from common.qwen_vision_process_fastvid import process_vision_info
from common.run_inference_qwen25vl_2pass import build_pruned_sequence
from rap.eval_multi import norm_sample, parse_variant
from rap.hds_model import HDSPolicy, load_policy
from rap.rap_model import uniform_indices


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def cache_bytes(pkv):
    n = 0
    layers = getattr(pkv, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values"):
                x = getattr(layer, name, None)
                if torch.is_tensor(x):
                    n += x.numel() * x.element_size()
        return n
    for kv in pkv:
        for x in kv:
            n += x.numel() * x.element_size()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt_file", default="playground/gt_qa_files/VideoMME/val_qa.json")
    ap.add_argument("--video_dir", default="Datasets/Video-MME/data/data")
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--max_pixels_units", type=int, default=16384)
    ap.add_argument("--n_videos", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--question_ids", default="", help="vd 001-1,301-1,601-1 (VideoMME: ngan / vua / dai); trong = cach deu")
    args = ap.parse_args()
    args.video_dir = local_mirror(args.video_dir)
    args.model_path = local_mirror(args.model_path)
    device = "cuda"
    max_pixels = args.max_pixels_units * 28 * 28
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)
    llm, vid = model.model, model.config.video_token_id
    variants = [parse_variant(v) for v in args.variant]
    for v in variants:
        if v["kind"] == "ckpt":
            v["policy"], _ = load_policy(v["arg"], device)
            v["policy"].eval()
    need_attn = any(v["kind"] == "holitom" for v in variants)
    if need_attn:
        from baselines import holitom_port as ht

    # cac video cach deu trong bo cau hoi (VideoMME: cuoi file la video dai), 1 cau hoi / video
    gt = [norm_sample(s) for s in json.load(open(args.gt_file))]
    seen, pool = set(), []
    for s in gt:
        if s["video_name"] not in seen and os.path.exists(os.path.join(args.video_dir, s["video_name"])):
            seen.add(s["video_name"]); pool.append(s)
    if args.question_ids:
        by = {str(s["question_id"]): s for s in gt}
        samples = [by[q] for q in args.question_ids.split(",")]
    else:
        step = max(1, len(pool) // args.n_videos)
        samples = pool[step // 2:: step][: args.n_videos]

    R = {v["name"]: {"total": [], "total_gpu": [], "compress": [], "prefill": [], "generate": [], "kv_mb": [], "peak_mb": [], "n_visual": [], "seq": []} for v in variants}
    R["_vision"] = {"vision": [], "vision_holitom": [], "decode": []}
    for si, sample in enumerate(samples):
        vp = os.path.join(args.video_dir, sample["video_name"])
        messages = dm.build_messages(vp, sample["question"], sample["candidates"], max_pixels)
        messages[1]["content"][0]["max_frames"] = args.max_frames
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t = time.perf_counter()
        video_inputs = process_vision_info(messages, return_video_kwargs=True)[1]
        R["_vision"]["decode"].append(time.perf_counter() - t)
        inputs = processor(text=[text], images=None, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], 1.0)
        with torch.inference_mode():
            full_embeds = model.get_input_embeddings()(inputs["input_ids"])
            thw = inputs["video_grid_thw"][0]
            T, H, W = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
            N = H * W

            def vis():
                f = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]).pooler_output
                return torch.cat(f, 0) if isinstance(f, (list, tuple)) else f
            feat, tv = timed(vis)
            feat, tv = timed(vis) if si == 0 else (feat, tv)   # lan dau co khoi dong
            R["_vision"]["vision"].append(tv)
            feat = feat.to(full_embeds.dtype)
            X = feat.float().view(T, N, -1)
            attn = None
            if need_attn:
                (hf, attn), th = timed(lambda: ht.holitom_frame_attn(llm, inputs["pixel_values_videos"], inputs["video_grid_thw"]))
                R["_vision"]["vision_holitom"].append(th)
            vpos = (inputs["input_ids"][0] == vid).nonzero(as_tuple=True)[0]
            position_ids, _ = llm.get_rope_index(
                input_ids=inputs["input_ids"], image_grid_thw=None, video_grid_thw=inputs["video_grid_thw"],
                second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=inputs["attention_mask"],
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
            for v in variants:
                def compress():
                    if v["kind"] == "dense":
                        return torch.arange(T * N, device=device), feat
                    if v["kind"] == "uniform":
                        ki = uniform_indices(T * N, v["ret"], device)
                        return ki, feat[ki]
                    if v["kind"] == "holitom":
                        ki, ke, _ = ht.holitom_select(feat, attn.to(feat.device), inputs["video_grid_thw"],
                                                      ht.holitom_params(SimpleNamespace(retention=v["ret"])))
                        return ki, ke
                    if v["kind"] == "ckpt" and isinstance(v["policy"], HDSPolicy):
                        T2, lay, _ = v["policy"](X, H, W, v["ret"])
                        return lay["kept_idx"], T2.to(feat.dtype)
                    raise ValueError(v["kind"])
                cs, ps, gs, kv, pk = [], [], [], [], []
                for rep in range(args.reps + 1):
                    (ki, ke), tc = timed(compress)
                    fe, pid, am, _ = build_pruned_sequence(full_embeds, inputs, vid, vpos, position_ids, ki, ke)
                    torch.cuda.empty_cache()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    out, tp = timed(lambda: model(inputs_embeds=fe, position_ids=pid, attention_mask=am, use_cache=True))
                    peak = torch.cuda.max_memory_allocated() - base
                    kvb = cache_bytes(out.past_key_values)
                    del out
                    _, tg = timed(lambda: model.generate(inputs_embeds=fe, position_ids=pid, attention_mask=am,
                                                         eos_token_id=None, pad_token_id=processor.tokenizer.pad_token_id,
                                                         do_sample=False, num_beams=1, max_new_tokens=16, min_new_tokens=16))
                    if rep == 0:
                        continue   # khoi dong
                    cs.append(tc); ps.append(tp); gs.append(tg); kv.append(kvb / 2**20); pk.append(peak / 2**20)
                r = R[v["name"]]
                r["compress"].append(statistics.median(cs)); r["prefill"].append(statistics.median(ps))
                r["generate"].append(statistics.median(gs)); r["kv_mb"].append(statistics.median(kv))
                r["peak_mb"].append(statistics.median(pk)); r["n_visual"].append(int(ki.numel())); r["seq"].append(int(fe.shape[1]))
                tvis = R["_vision"]["vision_holitom"][-1] if v["kind"] == "holitom" else tv
                r.setdefault("total", []).append(R["_vision"]["decode"][-1] + tvis + statistics.median(cs) + statistics.median(gs))
                r.setdefault("total_gpu", []).append(tvis + statistics.median(cs) + statistics.median(gs))
                print(f"[bench] video {si} {v['name']}: n_visual={ki.numel()} seq={fe.shape[1]} compress={statistics.median(cs)*1e3:.1f}ms "
                      f"prefill={statistics.median(ps)*1e3:.1f}ms gen16={statistics.median(gs)*1e3:.1f}ms kv={statistics.median(kv):.1f}MB "
                      f"peak={statistics.median(pk):.0f}MB vision={tvis*1e3:.0f}ms decode={R['_vision']['decode'][-1]:.2f}s "
                      f"TOTAL={r['total'][-1]:.2f}s (khong tinh giai ma: {r['total_gpu'][-1]:.2f}s)", flush=True)
        del inputs, full_embeds, feat, X, attn
        torch.cuda.empty_cache()

    summ = {"gpu": torch.cuda.get_device_name(0), "videos": [s["video_name"] for s in samples], "reps": args.reps,
            "vision_s": statistics.mean(R["_vision"]["vision"]), "decode_s": statistics.mean(R["_vision"]["decode"]),
            "vision_holitom_s": statistics.mean(R["_vision"]["vision_holitom"]) if R["_vision"]["vision_holitom"] else None,
            "variants": {}}
    for v in variants:
        r = R[v["name"]]
        summ["variants"][v["name"]] = {k: statistics.mean(x) for k, x in r.items() if x}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summ, open(args.out, "w"), indent=1)
    d = summ["variants"].get("dense")
    print(f"\nGPU {summ['gpu']}, {len(samples)} video, giai ma {summ['decode_s']:.2f}s, vision {summ['vision_s']*1e3:.0f}ms")
    print("TOAN TRINH theo video (giai ma + vision + nen + prefill + sinh 16 token), giay:")
    for i, sm in enumerate(samples):
        print(f"  {sm['question_id']:8s} " + "  ".join(f"{v['name']}={R[v['name']]['total'][i]:.2f}" for v in variants)
              + (f"   (giai ma {R['_vision']['decode'][i]:.2f}s)"))
    print(f"{'cau hinh':16s} {'token':>6s} {'nen ms':>7s} {'prefill ms':>10s} {'x':>5s} {'gen16 ms':>9s} {'x':>5s} {'KV MB':>7s} {'x':>5s} {'peak MB':>8s}")
    for name, s in summ["variants"].items():
        f = (lambda a: f"{d[a] / s[a]:5.1f}") if d else (lambda a: "    -")
        print(f"{name:16s} {s['n_visual']:6.0f} {s['compress']*1e3:7.1f} {s['prefill']*1e3:10.1f} {f('prefill')} {s['generate']*1e3:9.1f} {f('generate')} "
              f"{s['kv_mb']:7.1f} {f('kv_mb')} {s['peak_mb']:8.0f}")


if __name__ == "__main__":
    main()
