"""
run_inference_qwen25vl_loqo.py

Thu nghiem TRAN (khong train gi): "do quan trong cua token KHONG phu thuoc
cau hoi" co ton tai o muc token hay khong?

VideoMME co 3 cau hoi / video. Voi moi video:
  1. Encode 1 lan (dung y setting cua run_inference_qwen25vl_kitoke.py:
     max_frames=128, max_pixels_units=16384, time_mode=lmms).
  2. Voi TUNG cau hoi, lay attention (lop cuoi, vi tri cuoi prompt -> token
     video, trung binh theo head) tren TOAN BO token video. Chuoi day du ~16K
     token khong chay eager duoc (1 lop ~15GB), nen chia token thanh
     attn_chunks tap xen ke (stride), moi tap 1 lan forward ngan -- dung lai
     nguyen forward_pass1 cua run_inference_qwen25vl_2pass.py.
  3. Chon K = floor(retention * T*N) token roi generate, 2 cach cham diem:
     loqo : diem = trung binh attention cua CAC CAU HOI KHAC cung video
            (leave-one-question-out) -> bo chon KHONG nhin thay cau dang hoi.
            Day la tran cua moi scorer query-agnostic hoc tu attention.
     self : diem = attention cua CHINH cau dang hoi (query-aware) -> tran
            cua tin hieu attention. Neu self cung khong hon KiToke thi loi
            nam o tin hieu attention, khong phai o tinh query-agnostic.
  uniform_frac phan ngan sach duoc rai deu truoc (san do phu), phan con lai
  lay top theo diem.

Doc ket qua (so voi KiToke / Uniform CUNG retention, cung 2700 cau):
  loqo >  KiToke ro rang : dang dau tu chung cat attention -> scorer.
  loqo ~= KiToke, self >> : do quan trong phu thuoc cau hoi, query-agnostic
                            o muc token khong co du dia.
  self ~= KiToke          : attention lop cuoi khong phai tin hieu tot.

Ghi 2 file pred (<output_name>_loqo.json, <output_name>_self.json, dinh dang
giong cac script khac) + <output_name>.stats.jsonl moi cau 1 dong (Jaccard
giua 2 tap chon, so unit thoi gian co token).

Usage (chay tu thu muc KTV, env qwen25vl):
  python run_inference_qwen25vl_loqo.py \
    +video_dir=Datasets/Video-MME/data/data \
    +gt_file=playground/gt_qa_files/VideoMME/val_qa.json \
    +model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    +retention=0.25 +uniform_frac=0.25 \
    +output_dir=outputs/kitoke_baseline +output_name=pred_videomme_f128_attn_r025
  # smoke: them +limit=2 (2 video = 6 cau)
"""

import json
import os
from itertools import groupby

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "131072")

import hydra
import torch
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from run_inference_qwen25vl_2pass import build_pruned_sequence, forward_pass1
from utils import get_chunk

MODES = ("loqo", "self")


def prepare_question(model, processor, video_path, sample, args, max_pixels, cache):
    """Moi thu phu thuoc cau hoi (text embeds, position_ids); feature video lay tu cache."""
    messages = dm.build_messages(video_path, sample["question"], sample["candidates"], max_pixels)
    messages[1]["content"][0]["max_frames"] = int(getattr(args, "max_frames", 128))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if "video_inputs" not in cache:
        _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        cache["video_inputs"] = video_inputs
        cache["sample_fps"] = float(video_kwargs["fps"][0])
    inputs = processor(
        text=[text], images=None, videos=cache["video_inputs"], padding=True, return_tensors="pt",
    ).to(model.device)

    # Xem run_inference_qwen25vl_kitoke.py: gia tri mac dinh cua processor lam
    # moi unit trung vi tri thoi gian.
    time_mode = getattr(args, "time_mode", "lmms")
    if time_mode == "lmms":
        second_per_grid = 1.0
    elif time_mode == "true":
        second_per_grid = max(1.0, round(2.0 / cache["sample_fps"]))
    else:
        raise ValueError(f"unknown time_mode={time_mode!r}, expected 'lmms' or 'true'")
    inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], second_per_grid)

    llm = model.model if hasattr(model, "model") else model
    with torch.inference_mode():
        full_embeds = model.get_input_embeddings()(inputs["input_ids"])
        if "video_embeds_full" not in cache:
            video_feat = llm.get_video_features(
                inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]
            )
            video_embeds_full = video_feat.pooler_output
            if isinstance(video_embeds_full, (list, tuple)):
                video_embeds_full = torch.cat(video_embeds_full, dim=0)
            cache["video_embeds_full"] = video_embeds_full.to(full_embeds.dtype).to(full_embeds.device)
            grid = inputs["video_grid_thw"][0]
            cache["T"] = int(grid[0])
            cache["N"] = (int(grid[1]) // 2) * (int(grid[2]) // 2)
            assert cache["video_embeds_full"].shape[0] == cache["T"] * cache["N"]

        video_token_id = model.config.video_token_id
        video_positions_in_seq = (inputs["input_ids"][0] == video_token_id).nonzero(as_tuple=True)[0]
        position_ids, _ = llm.get_rope_index(
            input_ids=inputs["input_ids"], image_grid_thw=None, video_grid_thw=inputs["video_grid_thw"],
            second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )
    # pixel_values_videos (~GB) khong can nua sau khi da co feature
    inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    return {
        "inputs": inputs, "full_embeds": full_embeds, "position_ids": position_ids,
        "video_token_id": video_token_id, "video_positions_in_seq": video_positions_in_seq,
    }


def question_attention(model, q, video_embeds_full, attn_chunks):
    """Attention cau hoi -> TUNG token video, chuan hoa tong = 1."""
    n_total = video_embeds_full.shape[0]
    scores = torch.zeros(n_total, device=video_embeds_full.device, dtype=torch.float32)
    for offset in range(attn_chunks):
        idx = torch.arange(offset, n_total, attn_chunks, device=video_embeds_full.device)
        fe, pid, am, vpos = build_pruned_sequence(
            q["full_embeds"], q["inputs"], q["video_token_id"], q["video_positions_in_seq"],
            q["position_ids"], idx, video_embeds_full[idx],
        )
        _, attn = forward_pass1(model, fe, pid, am, vpos)
        scores[idx] = attn.to(scores.dtype)
        del fe, pid, am, attn
    return scores / scores.sum().clamp_min(1e-12)


def select_by_score(scores, retention, uniform_frac):
    n_total = scores.shape[0]
    K = max(1, int(retention * n_total))
    scores = scores.clone()
    n_uniform = int(uniform_frac * K)
    if n_uniform > 0:
        floor_idx = torch.linspace(0, n_total - 1, n_uniform, device=scores.device).round().long()
        scores[floor_idx] = float("inf")
    return torch.topk(scores, K).indices.sort().values


def generate_answer(model, processor, q, video_embeds_full, kept_indices):
    fe, pid, am, _ = build_pruned_sequence(
        q["full_embeds"], q["inputs"], q["video_token_id"], q["video_positions_in_seq"],
        q["position_ids"], kept_indices, video_embeds_full[kept_indices],
    )
    with torch.inference_mode():
        generated_ids = model.generate(
            inputs_embeds=fe, position_ids=pid, attention_mask=am,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False, num_beams=1, max_new_tokens=16,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def run_one_video(model, processor, video_path, samples, todo, args, max_pixels):
    """todo[i] = tap mode con thieu cua cau i. Tra ve list (i, mode, pred, diag)."""
    retention = float(getattr(args, "retention", 0.25))
    uniform_frac = float(getattr(args, "uniform_frac", 0.25))
    attn_chunks = int(getattr(args, "attn_chunks", 4))

    cache = {}
    questions = [prepare_question(model, processor, video_path, s, args, max_pixels, cache) for s in samples]
    video_embeds_full, T, N = cache["video_embeds_full"], cache["T"], cache["N"]
    del cache["video_inputs"]
    attn = [question_attention(model, q, video_embeds_full, attn_chunks) for q in questions]

    results = []
    for i, q in enumerate(questions):
        kept = {"self": select_by_score(attn[i], retention, uniform_frac)}
        if len(questions) > 1:
            others = torch.stack([attn[j] for j in range(len(questions)) if j != i]).mean(dim=0)
            kept["loqo"] = select_by_score(others, retention, uniform_frac)
        diag = {"T": T, "N": N, "n_total": T * N, "n_kept": int(kept["self"].numel())}
        if "loqo" in kept:
            both = torch.isin(kept["loqo"], kept["self"]).sum().item()
            diag["jaccard_loqo_self"] = both / (kept["loqo"].numel() + kept["self"].numel() - both)
        for mode, idx in kept.items():
            diag[f"units_covered_{mode}"] = int(torch.unique(idx // N).numel())
        for mode in MODES:
            if mode in kept and mode in todo[i]:
                results.append((i, mode, generate_answer(model, processor, q, video_embeds_full, kept[mode]), diag))
    return results


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels = int(getattr(args, "max_pixels_units", 16384)) * 28 * 28
    print(f"retention={getattr(args, 'retention', 0.25)} uniform_frac={getattr(args, 'uniform_frac', 0.25)} "
          f"attn_chunks={getattr(args, 'attn_chunks', 4)} max_frames={getattr(args, 'max_frames', 128)} "
          f"time_mode={getattr(args, 'time_mode', 'lmms')}")

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)

    video_dir = dm.resolve_path(args.video_dir)
    output_dir = dm.resolve_path(args.output_dir)
    with open(dm.resolve_path(args.gt_file), "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, int(getattr(args, "num_chunks", 1)), int(getattr(args, "chunk_idx", 0)))
    # Gom theo video; cac cau cung video nam lien nhau trong val_qa.json.
    videos = [(name, list(group)) for name, group in groupby(gt_qa_pairs, key=lambda s: s["video_name"])]
    limit = int(getattr(args, "limit", 0))  # >0: chi chay N VIDEO dau (smoke test)
    if limit > 0:
        videos = videos[:limit]

    os.makedirs(output_dir, exist_ok=True)
    pred_paths = {m: os.path.join(output_dir, f"{args.output_name}_{m}.json") for m in MODES}
    stats_path = os.path.join(output_dir, f"{args.output_name}.stats.jsonl")
    seen = {m: dm.load_answered_ids(p) for m, p in pred_paths.items()}
    files = {m: open(p, "a", encoding="utf-8") for m, p in pred_paths.items()}

    with open(stats_path, "a", encoding="utf-8") as stats_file:
        for video_name, samples in tqdm(videos):
            ids = [str(s.get("id", s.get("question_id"))) for s in samples]
            todo = [{m for m in MODES if qid not in seen[m]} for qid in ids]
            if not any(todo):
                continue
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}")
                continue
            try:
                results = run_one_video(model, processor, video_path, samples, todo, args, max_pixels)
            except Exception as e:
                print(f"SKIP {video_name}: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue

            for i, mode, output, diag in results:
                sample, answer_id = samples[i], ids[i]
                files[mode].write(json.dumps({
                    "task_name": sample.get("task_name", "unknown"), "question": sample["question"],
                    "question_id": str(sample.get("question_id", answer_id)), "id": answer_id,
                    "answer_number": sample["answer_number"], "candidates": sample["candidates"],
                    "answer": sample["answer"], "pred": output,
                }) + "\n")
                files[mode].flush()
                seen[mode].add(answer_id)
                if mode == MODES[-1]:
                    stats_file.write(json.dumps({"video_name": video_name, "id": answer_id, **diag}) + "\n")
                    stats_file.flush()
            print(f"{video_name}: " + " ".join(f"{ids[i]}/{mode}={output!r}" for i, mode, output, _ in results))

    for f in files.values():
        f.close()
    print(f"[run_inference_qwen25vl_loqo] done -> {', '.join(pred_paths.values())} (stats -> {stats_path})")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
