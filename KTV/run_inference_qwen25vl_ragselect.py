"""
run_inference_qwen25vl_ragselect.py

Tang 2+3 cua huong "RAG cho video + LLM chon doan bang text" (cam hung tu
FocusGraph, bien the cua "Huong 1" -- xem idea.md thao luan). Doi voi
DySeg+STPrune (hoan toan query-agnostic) va Huong 2 2-pass (dung attention
cua chinh Qwen), huong nay CHU DONG doc cau hoi som hon -- nhung chi o dang
TEXT (khong xu ly lai video), nen re hon nhieu so voi mot lan forward video
day du.

Yeu cau da chay precompute_scene_captions.py truoc (Tang 1, cache captions
+ segments_units + T_llm moi video). Script nay chi doc cache do.

Tang 2 (moi cau hoi, RE -- chi la 1 lan generate() TEXT, khong co video):
  Danh sach caption cac doan (danh so) + cau hoi + dap an -> Qwen (text-only)
  -> chon ra toi da `max_selected_segments` doan lien quan nhat (tra loi
  bang so thu tu doan).

Tang 3 (tra loi that):
  Doan duoc chon -> giu gan nhu toan bo token (trong pham vi
  `hot_budget_frac` cua tokens_num, tu dong bot neu vuot -- xem
  cap_to_budget). Doan con lai -> Uniform tren phan ngan sach con lai (san
  do phu toi thieu). Ghep lai -> generate() cau tra loi that.

CANH BAO: ban thu nghiem dau tien, chua kiem chung tren du lieu that -- rui
ro lon nhat la o buoc parse output cua selector (dinh dang tra loi co the
khong nhu mong doi, can doc log/dieu chinh SELECTOR_PROMPT neu vay).

Usage:
  python run_inference_qwen25vl_ragselect.py experiment=videomme \
    max_pixels_units=49152 tokens_num=16384 \
    +caption_cache_dir=outputs/scene_caption_cache \
    +max_selected_segments=3 +hot_budget_frac=0.5 \
    output_name=pred_videomme_qwen25vl_ragselect
"""

import json
import os
import re

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_inference_qwen25vl_2pass import build_pruned_sequence, generate_final
from run_inference_qwen25vl_dyseg import (
    build_messages,
    load_answered_ids,
    resolve_path,
    uniform_prune_video_tokens,
)
from utils import get_chunk

SYSTEM_PROMPT = "You are a helpful assistant."
SELECTOR_PROMPT_HEADER = (
    "Duoi day la mo ta tung doan cua 1 video, theo thu tu thoi gian, va 1 cau "
    "hoi trac nghiem ve video do. Hay chon toi da {k} doan CO KHA NANG chua "
    "thong tin de tra loi cau hoi nhat. CHI tra loi bang cac so thu tu doan, "
    "cach nhau boi dau phay (vi du: 2, 5, 7). Khong giai thich gi them.\n\n"
)


def build_selector_prompt(captions, question, candidates, max_selected):
    lines = [SELECTOR_PROMPT_HEADER.format(k=max_selected)]
    for i, cap in enumerate(captions):
        lines.append(f"Doan {i}: {cap}")
    lines.append("\nCau hoi: " + question)
    lines.append("\n".join(candidates))
    return "\n".join(lines)


def select_segments(model, processor, captions, question, candidates, max_selected):
    prompt = build_selector_prompt(captions, question, candidates, max_selected)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=32,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
        )
    trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()
    nums = [int(x) for x in re.findall(r"\d+", raw)]
    valid = [n for n in nums if 0 <= n < len(captions)]
    return sorted(set(valid))[:max_selected], raw


def cap_to_budget(indices, budget):
    """If a selected-segment token pool exceeds its budget, uniformly
    subsample it down (reuses uniform_prune_video_tokens on a plain index
    list by faking a [-1,1]-shaped embeds tensor -- only the COUNT/spacing
    matters here, not the actual embeddings)."""
    if len(indices) <= budget:
        return indices
    idx_t = torch.linspace(0, len(indices) - 1, budget).round().long().tolist()
    idx_t = sorted(set(idx_t))
    return [indices[i] for i in idx_t]


def run_one_question(model, processor, video_path, question, candidates, args, max_pixels, cache):
    captions = cache["captions"]
    seg_units = cache["segments_units"]
    cached_T_llm = cache["T_llm"]

    max_selected = int(getattr(args, "max_selected_segments", 3))
    selected, raw_selector_output = select_segments(
        model, processor, captions, question, candidates, max_selected
    )

    messages = build_messages(video_path, question, candidates, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(model.device)

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model

    with torch.inference_mode():
        full_embeds = model.get_input_embeddings()(inputs["input_ids"])
        video_feat = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"])
        video_embeds_full = video_feat.pooler_output
        if isinstance(video_embeds_full, (list, tuple)):
            video_embeds_full = torch.cat(video_embeds_full, dim=0)
        video_embeds_full = video_embeds_full.to(full_embeds.dtype).to(full_embeds.device)

        video_token_id = model.config.video_token_id
        video_mask = (inputs["input_ids"][0] == video_token_id)
        video_positions_in_seq = video_mask.nonzero(as_tuple=True)[0]

        position_ids, _ = llm.get_rope_index(
            input_ids=inputs["input_ids"], image_grid_thw=None, video_grid_thw=inputs["video_grid_thw"],
            second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

    T_llm = int(video_grid_thw[0])
    H_llm = int(video_grid_thw[1]) // 2
    W_llm = int(video_grid_thw[2]) // 2
    N = H_llm * W_llm
    n_total = video_embeds_full.shape[0]

    if T_llm != cached_T_llm:
        # Sampling drifted vs. precompute time (e.g. different max_pixels_units
        # between the two scripts) -- unit ranges from the cache no longer line
        # up. Fall back to plain Uniform for this question rather than using a
        # wrong mapping silently.
        kept, embeds, _ = uniform_prune_video_tokens(video_embeds_full, int(args.tokens_num))
        fe, pid, am, _ = build_pruned_sequence(
            full_embeds, inputs, video_token_id, video_positions_in_seq, position_ids, kept, embeds
        )
        output = generate_final(model, processor, fe, pid, am)
        return output, dict(mode="ragselect", fallback="T_llm_mismatch", selected=[], raw=raw_selector_output)

    hot_unit_set = set()
    for i in selected:
        s, e = seg_units[i]
        hot_unit_set.update(range(s, e))

    tokens_num = int(args.tokens_num)
    hot_budget_frac = float(getattr(args, "hot_budget_frac", 0.5))
    hot_budget = int(tokens_num * hot_budget_frac)

    hot_indices = [i for i in range(n_total) if (i // N) in hot_unit_set]
    hot_indices = cap_to_budget(hot_indices, hot_budget)
    cold_pool_indices = [i for i in range(n_total) if (i // N) not in hot_unit_set]
    remaining_budget = max(0, tokens_num - len(hot_indices))
    if cold_pool_indices and remaining_budget > 0:
        cold_embeds_pool = video_embeds_full[cold_pool_indices]
        idx_local, _, _ = uniform_prune_video_tokens(cold_embeds_pool, remaining_budget)
        cold_kept = [cold_pool_indices[j] for j in idx_local]
    else:
        cold_kept = []
    kept = sorted(hot_indices + cold_kept)
    embeds = video_embeds_full[kept]

    fe, pid, am, _ = build_pruned_sequence(
        full_embeds, inputs, video_token_id, video_positions_in_seq, position_ids, kept, embeds
    )
    output = generate_final(model, processor, fe, pid, am)
    diag = dict(mode="ragselect", selected=selected, n_selected_units=len(hot_unit_set),
                actual_total=len(kept), raw=raw_selector_output)
    return output, diag


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384 * 3))
    max_pixels = max_pixels_units * 28 * 28
    caption_cache_dir = resolve_path(getattr(args, "caption_cache_dir", "outputs/scene_caption_cache"))

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    output_dir = resolve_path(args.output_dir)

    print(f"Loading {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=16 * 28 * 28)

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    generated_ids_seen = load_answered_ids(output_path)
    output_mode = "a" if os.path.exists(output_path) else "w"

    with open(output_path, output_mode, encoding="utf-8") as ans_file:
        for sample in tqdm(gt_qa_pairs):
            video_name = sample["video_name"]
            answer_id = str(sample.get("id", sample.get("question_id")))
            if answer_id in generated_ids_seen:
                continue

            question = sample["question"]
            candidates = sample["candidates"]
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}")
                continue

            cache_path = os.path.join(caption_cache_dir, os.path.basename(video_name) + ".json")
            if not os.path.exists(cache_path):
                print(f"SKIP {answer_id}: no caption cache at {cache_path} -- run precompute_scene_captions.py first")
                continue
            with open(cache_path, "r") as cf:
                cache = json.load(cf)

            try:
                output, diag = run_one_question(
                    model, processor, video_path, question, candidates, args, max_pixels, cache
                )
            except Exception as e:
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue
            print(f"pred={output!r} diag={diag}")

            sample_set = {
                "task_name": sample["task_name"], "question": question,
                "question_id": str(sample.get("question_id", answer_id)), "id": answer_id,
                "answer_number": sample["answer_number"], "candidates": candidates,
                "answer": sample["answer"], "pred": output, "diag": diag,
            }
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()
            generated_ids_seen.add(answer_id)

    print(f"[ragselect] done -> {output_path}")


@hydra.main(config_path="configs/inference_qwen25vl", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
