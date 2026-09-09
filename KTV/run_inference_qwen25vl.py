"""
run_inference_qwen25vl.py

Standalone inference script for Qwen2.5-VL-7B-Instruct on the multiple-choice
video QA format used throughout this repo. Independent from ktv/llava --
Qwen2.5-VL has its own vision encoder (native dynamic resolution, not
CLIP-ViT) and its own chat template/processor, so none of the custom
encode_images/token-pruning code applies here.

v1 scope: uniform frame sampling (key_frame_path=null) at a fixed total
visual-token budget, split evenly across frames via min_pixels=max_pixels
on the Qwen processor (1 merged visual token ~= 28x28 px, from
patch_size=14 * spatial_merge_size=2). This mirrors the exact experimental
design of run_phase0_resolution_sweep.sh, so results are directly
comparable to the existing num_frames/tokens_num table on the old backbone.
Per-question variable keyframe files (key_frame_path set) are supported for
frame selection, but then tokens_num/num_frames no longer determines a
single valid min_pixels/max_pixels -- pick num_frames to match what the
keyframe file actually yields per question if you use this path.

Output format matches run_inference_multiple_choice_qa.py exactly (JSONL,
one record per question, resumable), so eval/compute_accuracy.py and
eval/compute_accuracy_by_bucket.py work unchanged.
"""

import json
import os

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from dataset import load_video
from utils import get_chunk


def get_option_prompt(candidates):
    """Inlined from prompt.py to avoid importing ktv.llava (pins an old,
    incompatible transformers version) into this separate environment."""
    option_prompt = ""
    for idx, candidate in enumerate(candidates):
        choice = chr(ord("A") + idx)
        option_prompt += f"({choice}) {candidate}\n"
    return option_prompt


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_answered_ids(output_path):
    answered_ids = set()
    if not os.path.exists(output_path):
        return answered_ids
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            answer_id = data.get("id", data.get("question_id"))
            if answer_id is not None:
                answered_ids.add(str(answer_id))
    return answered_ids


def build_messages(num_frames, question, candidates):
    option_prompt = get_option_prompt(candidates)
    text = (
        "The input consists of a sequence of key frames from a video.\n"
        f"Question: {question}\nOptions:\n{option_prompt}"
        'Answer with ONLY the option letter (e.g. "C"), nothing else.'
    )
    content = [{"type": "image"} for _ in range(num_frames)]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    output_dir = resolve_path(args.output_dir)
    key_frame_path = resolve_path(args.key_frame_path)

    tokens_per_frame = max(1, args.tokens_num // args.num_frames)
    pixel_budget = tokens_per_frame * 28 * 28
    print(
        f"num_frames={args.num_frames} tokens_num={args.tokens_num} "
        f"-> ~{tokens_per_frame} tokens/frame -> min_pixels=max_pixels={pixel_budget}"
    )

    print(f"Loading {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=pixel_budget, max_pixels=pixel_budget
    )

    keyframes_by_question = {}
    if key_frame_path:
        with open(key_frame_path, "r") as f:
            keyframes_by_question = {
                str(qid): kf for qid, kf in json.load(f).items()
            }

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

            keyframe = keyframes_by_question.get(answer_id)
            frames, _ = load_video(video_path, keyframe, num_frms=args.num_frames)
            if not frames:
                print(f"No frames loaded for {answer_id}")
                continue

            messages = build_messages(len(frames), question, candidates)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=frames, videos=None,
                padding=True, return_tensors="pt",
            ).to(model.device)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
            )
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_kwargs)
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            output = processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            print(output)

            sample_set = {
                "task_name": sample["task_name"],
                "question": question,
                "question_id": str(sample.get("question_id", answer_id)),
                "id": answer_id,
                "answer_number": sample["answer_number"],
                "candidates": candidates,
                "answer": sample["answer"],
                "pred": output,
            }
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()
            generated_ids_seen.add(answer_id)

    print(f"[Qwen2.5-VL] done -> {output_path}")


@hydra.main(
    config_path="configs/inference_qwen25vl", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
