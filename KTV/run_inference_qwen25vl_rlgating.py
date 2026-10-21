"""
run_inference_qwen25vl_rlgating.py

Real inference + accuracy measurement for a trained RL-Gating policy
(train_rl_gating.py checkpoint) -- unlike visualize_rl_gating.py (which only
renders which units get kept, no generate()), this actually compresses the
video with the policy's decision and calls generate() for a real answer,
writing output in the same JSONL format as every other pred_*.json in this
repo so it drops straight into eval/compute_accuracy_by_bucket.py.

Query-agnostic at inference: the policy only sees video features (same
forward as visualize_rl_gating.py), the question is used ONLY to build the
final prompt for generate() -- never fed to the policy.

Deterministic at inference (no Bernoulli sampling): keep unit t if
probs[t] >= threshold, then top up to min_retention_frac via the same
evenly-spaced floor used during training (so inference-time retention
behavior matches what the policy was actually trained/rewarded under).

Usage (VideoMME, to compare directly against the other rows on the
benchmark site -- the policy was trained on NExT-QA, so this is a genuine
out-of-domain generalization test, no train/eval leakage):
  python run_inference_qwen25vl_rlgating.py \
    video_dir=Datasets/Video-MME/data/data \
    gt_file=playground/gt_qa_files/VideoMME/val_qa.json \
    policy_path=outputs/rl_gating_policy_exp3.pt \
    output_dir=outputs output_name=pred_videomme_qwen25vl_rlgating_exp3_step1000 \
    max_pixels_units=16384 threshold=0.5 min_retention_frac=0.15 \
    num_chunks=1 chunk_idx=0

Recommend a SMALL subset first (num_chunks=... or a hand-built subset gt
file) before committing to the full 2700 questions, given how early-stage
this checkpoint is (entropy still near-max at step 1000, see training log).
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
from run_inference_qwen25vl_2pass import build_pruned_sequence, generate_final
from train_rl_gating import GatingPolicy
from utils import get_chunk


def build_video_index(video_dir):
    index = {}
    for root, _dirs, files in os.walk(video_dir):
        for fname in files:
            index.setdefault(fname, os.path.join(root, fname))
    return index


def resolve_video_path(video_dir, video_name, video_index):
    direct = os.path.join(video_dir, video_name)
    if os.path.exists(direct):
        return direct
    return video_index.get(os.path.basename(video_name))


def run_one_question(model, processor, policy, video_path, question, candidates, args, max_pixels, device):
    messages = dm.build_messages(video_path, question, candidates, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model

    with torch.no_grad():
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
            input_ids=inputs["input_ids"], image_grid_thw=None,
            video_grid_thw=inputs["video_grid_thw"], second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=inputs["attention_mask"], mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        T_llm = int(video_grid_thw[0])
        H_llm = int(video_grid_thw[1]) // 2
        W_llm = int(video_grid_thw[2]) // 2
        N = H_llm * W_llm

        # ---- policy decision (query-agnostic: never sees `question`/`candidates`) ----
        X_units = video_embeds_full.view(T_llm, N, -1).mean(dim=1).float()
        delta_units = torch.zeros_like(X_units)
        delta_units[1:] = X_units[1:] - X_units[:-1]
        probs = torch.sigmoid(policy(X_units, delta_units))

        threshold = float(getattr(args, "threshold", 0.5))
        min_retention_frac = float(getattr(args, "min_retention_frac", 0.15))
        mask = probs >= threshold
        min_units = max(1, int(round(min_retention_frac * T_llm)))
        if int(mask.sum().item()) < min_units:
            floor_idx = torch.linspace(0, T_llm - 1, steps=min_units, device=mask.device).round().long()
            mask[floor_idx] = True

        kept_units = mask.nonzero(as_tuple=True)[0]
        offsets = torch.arange(N, device=mask.device)
        kept_indices = torch.sort((kept_units.unsqueeze(1) * N + offsets.unsqueeze(0)).reshape(-1)).values
        kept_embeds = video_embeds_full[kept_indices]
        retention = kept_indices.numel() / video_embeds_full.shape[0]

        fe, pid, am, _ = build_pruned_sequence(
            full_embeds, inputs, video_token_id, video_positions_in_seq, position_ids, kept_indices, kept_embeds,
        )

    output = generate_final(model, processor, fe, pid, am)
    return output, retention, T_llm


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

    policy_path = dm.resolve_path(args.policy_path)
    ckpt = torch.load(policy_path, map_location=device)
    policy = GatingPolicy(ckpt["video_dim"]).to(device).float()
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    print(f"Loaded policy from {policy_path} (trained {ckpt.get('step', '?')} steps)")

    video_dir = dm.resolve_path(args.video_dir)
    gt_file = dm.resolve_path(args.gt_file)
    output_dir = dm.resolve_path(args.output_dir)

    print(f"Indexing videos under {video_dir} ...")
    video_index = build_video_index(video_dir)
    print(f"Indexed {len(video_index)} video files")

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, int(getattr(args, "num_chunks", 1)), int(getattr(args, "chunk_idx", 0)))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    generated_ids_seen = dm.load_answered_ids(output_path)
    output_mode = "a" if os.path.exists(output_path) else "w"

    retentions = []
    with open(output_path, output_mode, encoding="utf-8") as ans_file:
        for sample in tqdm(gt_qa_pairs):
            video_name = sample["video_name"]
            answer_id = str(sample.get("id", sample.get("question_id")))
            if answer_id in generated_ids_seen:
                continue
            video_path = resolve_video_path(video_dir, video_name, video_index)
            if video_path is None:
                print(f"Missing video: {video_name}")
                continue

            try:
                output, retention, T_llm = run_one_question(
                    model, processor, policy, video_path,
                    sample["question"], sample["candidates"], args, max_pixels, device,
                )
            except Exception as e:
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue

            retentions.append(retention)
            print(f"pred={output!r} retention={retention:.3f} T_units={T_llm}")

            sample_set = {
                "task_name": sample.get("task_name", "unknown"),
                "question": sample["question"],
                "question_id": str(sample.get("question_id", answer_id)),
                "id": answer_id,
                "answer_number": sample["answer_number"],
                "candidates": sample["candidates"],
                "answer": sample["answer"],
                "pred": output,
            }
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()
            generated_ids_seen.add(answer_id)

    if retentions:
        print(f"[rlgating inference] done -> {output_path} (mean retention={sum(retentions)/len(retentions):.3f})")
    else:
        print(f"[rlgating inference] done -> {output_path} (no new questions answered)")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
