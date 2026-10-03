"""
infer_dsta_scorer_videomme.py

Danh gia DSTA-Scorer (train_dsta_scorer_2a.py) tren VideoMME, cau hoi trac
nghiem that, dinh dang JSON giong het cac script khac trong repo (dung
duoc thang voi eval/compute_accuracy_by_bucket.py).

QUAN TRONG -- phep kiem chung quyet dinh (dung nguyen tac da ap dung cho
RL-Gating truoc day): script nay ho tro 2 mode, CHON TOP-K UNIT THEO DUNG
1 MUC GIU CO DINH (khong phai xac suat rieng tung video) de so sanh cong
bang:
  mode=dsta    : chon K unit co diem so (tu DSTAScorer da train) CAO NHAT.
  mode=uniform : chon K unit CACH DEU (khong hoc gi, quy tac cung -- doi
                 chieu true cua RL-Gating/Uniform 15% da dung truoc day).
K = round(target_retention * T_llm), CO DINH cho ca 2 mode -- neu 2 mode
cho ra ket qua giong het nhau (nhu da xay ra voi RL-Gating), ket luan
DSTA-Scorer khong hoc duoc gi hon quy tac ngau nhien.

Luu y: chon TOP-K theo diem (deterministic) o day khac Bernoulli sampling
luc train (stochastic, chi de tao G mask da dang cho GRPO) -- dung deterministic
o infer la chuan cua ca DySeg/RL-Gating/2-Pass trong repo nay.

Usage:
  python infer_dsta_scorer_videomme.py \
    video_dir=Datasets/Video-MME/data/data \
    gt_file=playground/gt_qa_files/VideoMME/val_qa_subset300.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    scorer_path=outputs/dsta_scorer_2a.pt \
    mode=dsta target_retention=0.35 \
    output_dir=outputs output_name=pred_videomme_dsta_2a_subset300 \
    num_chunks=1 chunk_idx=0

  # doi chieu true, DOI DUY NHAT mode=uniform, giu nguyen target_retention:
  python infer_dsta_scorer_videomme.py \
    video_dir=Datasets/Video-MME/data/data \
    gt_file=playground/gt_qa_files/VideoMME/val_qa_subset300.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    scorer_path=outputs/dsta_scorer_2a.pt \
    mode=uniform target_retention=0.35 \
    output_dir=outputs output_name=pred_videomme_uniform_matched_subset300 \
    num_chunks=1 chunk_idx=0
"""

import json
import os

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from run_inference_qwen25vl_2pass import build_pruned_sequence
from train_dsta_scorer_2a import DSTAScorer, units_to_token_indices
from utils import get_chunk


def select_units_dsta(scorer, X_units, delta_units, K):
    """Deterministic top-K unit theo diem hoc duoc (khong sample Bernoulli
    -- dung deterministic o infer, giong het DySeg/2-Pass/RL-Gating)."""
    with torch.no_grad():
        logits = scorer(X_units, delta_units)  # [T]
    T = logits.shape[0]
    K = max(1, min(K, T))
    top_idx = torch.topk(logits, K).indices
    mask = torch.zeros(T, dtype=torch.bool, device=logits.device)
    mask[top_idx] = True
    return mask


def select_units_uniform(T, K, device):
    """K unit cach deu -- doi chieu true, khong hoc gi."""
    K = max(1, min(K, T))
    idx = torch.linspace(0, T - 1, K, device=device).round().long()
    idx = torch.unique(idx, sorted=True)
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    mask[idx] = True
    return mask


def run_one_question(model, processor, scorer, video_path, question, candidates, args, max_pixels, device, debug=False):
    messages = dm.build_messages(video_path, question, candidates, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model

    with torch.inference_mode():
        full_embeds = model.get_input_embeddings()(inputs["input_ids"])
        video_feat = llm.get_video_features(
            inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]
        )
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
        assert n_total == T_llm * N, f"unit-major layout assumption broken: {n_total} != {T_llm}*{N}"

        target_retention = float(getattr(args, "target_retention", 0.35))
        K = max(1, round(target_retention * T_llm))
        mode = getattr(args, "mode", "dsta")

        if mode == "dsta":
            X_units = video_embeds_full.detach().view(T_llm, N, -1).mean(dim=1).float()
            delta_units = torch.zeros_like(X_units)
            delta_units[1:] = X_units[1:] - X_units[:-1]
            unit_mask = select_units_dsta(scorer, X_units, delta_units, K)
        elif mode == "uniform":
            unit_mask = select_units_uniform(T_llm, K, device)
        else:
            raise ValueError(f"unknown mode={mode!r}, expected 'dsta' or 'uniform'")

        if debug:
            kept_units = unit_mask.nonzero(as_tuple=True)[0].tolist()
            gaps = [kept_units[i + 1] - kept_units[i] for i in range(len(kept_units) - 1)]
            max_gap = max(gaps) if gaps else 0
            span = (kept_units[-1] - kept_units[0]) if kept_units else 0
            print(f"  [debug] mode={mode} T_llm={T_llm} K={K} kept_units={kept_units}")
            print(f"  [debug] span_covered={span}/{T_llm - 1} max_gap_between_kept={max_gap}")

        kept_indices = units_to_token_indices(unit_mask, N)
        kept_embeds = video_embeds_full[kept_indices]

        final_embeds, final_position_ids, final_attention_mask, _ = build_pruned_sequence(
            full_embeds, inputs, video_token_id, video_positions_in_seq, position_ids, kept_indices, kept_embeds,
        )

        generated_ids = model.generate(
            inputs_embeds=final_embeds, position_ids=final_position_ids, attention_mask=final_attention_mask,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False, num_beams=1, max_new_tokens=16,
        )
    output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    retention = kept_indices.numel() / n_total
    return output, retention, kept_indices.numel(), n_total


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    max_pixels = max_pixels_units * 28 * 28

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)

    mode = getattr(args, "mode", "dsta")
    scorer = None
    if mode == "dsta":
        scorer_path = dm.resolve_path(args.scorer_path)
        ckpt = torch.load(scorer_path, map_location=device)
        scorer = DSTAScorer(ckpt["video_dim"]).to(device).float()
        scorer.load_state_dict(ckpt["state_dict"])
        scorer.eval()
        print(f"Loaded DSTAScorer from {scorer_path} (step={ckpt.get('step', '?')})")
    else:
        print("mode=uniform -- khong can scorer, chi chon unit cach deu.")

    video_dir = dm.resolve_path(args.video_dir)
    gt_file = dm.resolve_path(args.gt_file)
    output_dir = dm.resolve_path(args.output_dir)

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, int(getattr(args, "num_chunks", 1)), int(getattr(args, "chunk_idx", 0)))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    generated_ids_seen = dm.load_answered_ids(output_path)
    output_mode = "a" if os.path.exists(output_path) else "w"

    debug_n = int(getattr(args, "debug_n", 0))  # in ra phan bo unit duoc giu cho N cau dau (chan doan dong cuc)
    n_seen = 0
    retentions = []
    with open(output_path, output_mode, encoding="utf-8") as ans_file:
        for sample in tqdm(gt_qa_pairs):
            video_name = sample["video_name"]
            answer_id = str(sample.get("id", sample.get("question_id")))
            if answer_id in generated_ids_seen:
                continue
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}")
                continue

            try:
                output, retention, n_kept, n_total = run_one_question(
                    model, processor, scorer, video_path, sample["question"], sample["candidates"],
                    args, max_pixels, device, debug=(n_seen < debug_n),
                )
                n_seen += 1
            except Exception as e:
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue

            retentions.append(retention)
            print(f"pred={output!r} retention={retention:.4f} ({n_kept}/{n_total})")

            sample_set = {
                "task_name": sample.get("task_name", "unknown"), "question": sample["question"],
                "question_id": str(sample.get("question_id", answer_id)), "id": answer_id,
                "answer_number": sample["answer_number"], "candidates": sample["candidates"],
                "answer": sample["answer"], "pred": output,
            }
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()
            generated_ids_seen.add(answer_id)

    if retentions:
        print(f"[infer_dsta_scorer_videomme] done -> {output_path} (mean retention={sum(retentions)/len(retentions):.4f})")
    else:
        print(f"[infer_dsta_scorer_videomme] done -> {output_path} (khong co cau hoi moi)")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
