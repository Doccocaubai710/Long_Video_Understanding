"""
bench_speed_rap.py -- do toc do / bo nho / FLOPs kieu FastVID cho Dense, Uniform va RAP.

Cung harness voi infer_rap_videomme.py (128 frame, max_pixels_units=16384,
time_mode=lmms, attn mac dinh cua transformers). Moi video lay cau hoi dau tien.
Doc video (CPU) khong tinh vi giong nhau cho moi phuong phap.

Cac moc do (cuda.synchronize truoc/sau, ms):
  vision  : bo ma hoa anh cua Qwen (get_video_features), dung chung cho moi cau hinh
  compress: chon Uniform + Perceiver RAP (0 voi Dense)
  prefill : 1 forward LLM tren toan bo chuoi (co tao KV cache)
  ttft    : vision + compress + prefill
  e2e     : vision + compress + generate 16 token (min_new_tokens=16 de cong bang)
Moi moc chay --repeats lan (mac dinh 3), lay trung vi.
  peak_gb : bo nho dinh trong luc prefill (gom ca trong so mo hinh)
FLOPs dem bang torch.utils.flop_counter tren --flop_videos video dau.

Usage (tu thu muc KTV):
  python bench_speed_rap.py --out outputs/speed/bench_f128.jsonl
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import torch
from qwen_vision_process_fastvid import process_vision_info
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import fastvid_port as fv
import holitom_port as ht
import run_inference_qwen25vl_dyseg as dm
from rap_model import load_resampler, uniform_indices
from run_inference_qwen25vl_2pass import build_pruned_sequence

CKPTS = {0.40: "outputs/rap_next/R40E.pt", 0.25: "outputs/rap/runE.pt", 0.15: "outputs/rap_next/R15E.pt",
         0.10: "outputs/rap/R10E.pt", 0.05: "outputs/rap_next/R05E.pt", 0.02: "outputs/rap_next/R02E.pt"}


def pick_samples(gt, per_bucket):
    """Cau hoi dau tien cua moi video; VideoMME: 001-300 short, 301-600 medium, 601-900 long."""
    first = {}
    for s in gt:
        vnum = int(str(s["question_id"]).split("-")[0])
        first.setdefault(vnum, s)
    out = []
    for name, lo in (("short", 1), ("medium", 301), ("long", 601)):
        nums = [n for n in sorted(first) if lo <= n < lo + 300]
        step = max(1, len(nums) // per_bucket)
        out += [(name, first[n]) for n in nums[::step][:per_bucket]]
    return out


REPEATS = 3


def timed(fn, setup=None):
    """Chay REPEATS lan, tra ve ket qua lan cuoi va trung vi thoi gian (ms).
    setup() chay truoc moi lan, ngoai vung bam gio."""
    ts, r = [], None
    for _ in range(REPEATS):
        r = None  # bo ket qua lan truoc (vd KV cache) truoc khi do lan moi
        if setup is not None:
            setup()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return r, sorted(ts)[len(ts) // 2]


def _gqa_sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs):
    """Cong thuc sdpa cua torch nhung chap nhan GQA (so head cua K/V it hon Q).
    Nhu torch, khong chia doi cho causal."""
    b, h, s_q, d = query_shape
    s_k, d_v = key_shape[-2], value_shape[-1]
    return 2 * b * h * s_q * s_k * d + 2 * b * h * s_q * s_k * d_v


_SDPA_OPS = {op: _gqa_sdpa_flop for op in (torch.ops.aten._scaled_dot_product_efficient_attention,
                                           torch.ops.aten._scaled_dot_product_flash_attention,
                                           torch.ops.aten._scaled_dot_product_cudnn_attention)}


def count_flops(fn):
    with FlopCounterMode(display=False, custom_mapping=_SDPA_OPS) as fc:
        fn()
    return fc.get_total_flops()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/speed/bench_f128.jsonl")
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--video_dir", default="Datasets/Video-MME/data/data")
    ap.add_argument("--gt_file", default="playground/gt_qa_files/VideoMME/val_qa.json")
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--max_pixels_units", type=int, default=16384)
    ap.add_argument("--per_bucket", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--flop_videos", type=int, default=3)
    ap.add_argument("--retentions", default="0.40,0.25,0.15,0.10,0.05,0.02")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--methods", default="uniform,rap")  # them fastvid,holitom neu can
    args = ap.parse_args()
    global REPEATS
    REPEATS = args.repeats

    device = "cuda"
    max_pixels = args.max_pixels_units * 28 * 28
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)
    llm = model.model
    vid = model.config.video_token_id
    tok = processor.tokenizer
    retentions = [float(r) for r in args.retentions.split(",")]
    resamplers = {}
    methods = args.methods.split(",")
    for r in retentions if "rap" in methods else []:
        rs, _ = load_resampler(CKPTS[r], device)
        resamplers[r] = rs.eval()
    n_params = {r: sum(p.numel() for p in rs.parameters()) for r, rs in resamplers.items()}
    print(f"attn_implementation={model.config._attn_implementation} rap_params={n_params}", flush=True)

    configs = [("dense", 1.0)] + [(m, r) for r in retentions for m in methods]
    samples = pick_samples(json.load(open(args.gt_file)), args.per_bucket)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fo = open(args.out, "w")
    # vong khoi dong tren video dau tien, khong ghi lai
    for i, (bucket, s) in enumerate([samples[0]] * args.warmup + samples):
        is_warm = i < args.warmup
        vi = i - args.warmup
        video_path = os.path.join(args.video_dir, s["video_name"])
        messages = dm.build_messages(video_path, s["question"], s["candidates"], max_pixels)
        messages[1]["content"][0]["max_frames"] = args.max_frames
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=None, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], 1.0)
        with torch.inference_mode():
            full_embeds = model.get_input_embeddings()(inputs["input_ids"])
            thw = inputs["video_grid_thw"][0]
            T, H, W = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
            N = H * W

            def vision():
                f = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]).pooler_output
                return (torch.cat(f, dim=0) if isinstance(f, (list, tuple)) else f).to(full_embeds.dtype)

            feat, t_vision = timed(vision)
            vision_flops = count_flops(vision) if 0 <= vi < args.flop_videos else None
            vpos = (inputs["input_ids"][0] == vid).nonzero(as_tuple=True)[0]
            position_ids, _ = llm.get_rope_index(
                input_ids=inputs["input_ids"], image_grid_thw=None, video_grid_thw=inputs["video_grid_thw"],
                second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=inputs["attention_mask"],
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
            for method, r in configs:
                def compress():
                    if method == "dense":
                        return torch.arange(T * N, device=device), feat
                    idx = uniform_indices(T * N, r, device)
                    if method == "uniform":
                        return idx, feat[idx]
                    if method in ("fastvid", "holitom"):
                        # ban port chay lai ca vision tower de lay attention block cuoi; phan vision duoc tru ra ben duoi
                        torch.manual_seed(0)
                        p = SimpleNamespace(retention=r)
                        if method == "fastvid":
                            emb, attn = fv.fastvid_frame_attn(llm, inputs["pixel_values_videos"], inputs["video_grid_thw"])
                            ki, ke, _ = fv.fastvid_select(emb.to(full_embeds.dtype), attn, inputs["video_grid_thw"], fv.fastvid_params(p))
                        else:
                            emb, attn = ht.holitom_frame_attn(llm, inputs["pixel_values_videos"], inputs["video_grid_thw"])
                            ki, ke, _ = ht.holitom_select(emb.to(full_embeds.dtype), attn, inputs["video_grid_thw"], ht.holitom_params(p))
                        return ki, ke.to(full_embeds.dtype)
                    T2, _ = resamplers[r](feat.float().view(T, N, -1), idx, H, W)
                    return idx, T2.to(full_embeds.dtype)

                (kept_idx, kept), t_comp = timed(compress)
                if method == "dense":
                    t_comp = 0.0
                elif method in ("fastvid", "holitom"):
                    t_comp = max(0.0, t_comp - t_vision)  # chi tinh phan them so voi 1 lan vision
                fe, pid, am, _ = build_pruned_sequence(full_embeds, inputs, vid, vpos, position_ids, kept_idx, kept)
                prefill = lambda: model(inputs_embeds=fe, position_ids=pid, attention_mask=am, use_cache=True)
                def reset_mem():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                _, t_prefill = timed(prefill, setup=reset_mem)
                peak = torch.cuda.max_memory_allocated() / 2**30
                _, t_gen = timed(lambda: model.generate(inputs_embeds=fe, position_ids=pid, attention_mask=am,
                                                        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                                                        do_sample=False, num_beams=1, max_new_tokens=16, min_new_tokens=16))
                rec = {"video_name": s["video_name"], "bucket": bucket, "method": method, "retention": r,
                       "T": T, "N": N, "n_video": int(kept_idx.numel()), "seq_len": int(fe.shape[1]),
                       "vision_ms": t_vision, "compress_ms": t_comp, "prefill_ms": t_prefill,
                       "ttft_ms": t_vision + t_comp + t_prefill, "e2e_ms": t_vision + t_comp + t_gen, "peak_gb": peak}
                if vision_flops is not None:
                    rec["vision_flops"] = vision_flops
                    rec["prefill_flops"] = count_flops(prefill)
                    rec["compress_flops"] = (count_flops(compress) if method == "rap" else
                                             count_flops(compress) - vision_flops if method in ("fastvid", "holitom") else 0)
                if not is_warm:
                    fo.write(json.dumps(rec) + "\n")
                    fo.flush()
                del fe, pid, am, kept
        print(f"{'warmup' if is_warm else vi} {bucket} {s['video_name']} T={T} N={N} vision={t_vision:.0f}ms", flush=True)
        del feat, inputs, full_embeds
        torch.cuda.empty_cache()
    print(f"[bench_speed_rap] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
