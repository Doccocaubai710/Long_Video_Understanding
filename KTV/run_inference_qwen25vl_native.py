"""
run_inference_qwen25vl_native.py

G1 gate test (idea.md, Buoc 0): does matching FastVID/lmms-eval's own
Qwen2.5-VL harness exactly -- native video path (not extracted frames), their
generation config, their prompt template -- close the gap between our
measured Vanilla number and the literature's reported one (63.8%, VideoMME
w/o subtitles)?

Rewritten after a controlled experiment: running FastVID's own code through
lmms-eval (retention_ratio=0.27, --model qwen2_5_vl) landed at ~62.9%
overall on Qwen2.5-VL-7B-Instruct -- above anything this codebase's own
scripts had produced even at 100% token budget (dense). Diffing
lmms_eval/models/qwen2_5_vl.py and lmms_eval/tasks/videomme/{utils.py,
videomme.yaml} (cloned at ../FastVID/fastvid_qwen25vl/lmms-eval) against the
previous version of this script turned up four concrete mismatches, all
fixed here:

  1. Frame/resolution sampling. FastVID's harness never fixes a frame count
     -- it hands the raw video path to qwen_vl_utils with only min_pixels/
     max_pixels bounds (16*28*28 / 16384*28*28) and lets Qwen's own
     fps-based smart_nframes()/smart_resize() choose frame count (up to 768)
     and per-frame resolution. The previous version forced a fixed 128-frame
     cv2 extraction with uniform per-frame resolution -- not the sampling
     regime the model/harness was actually benchmarked with.

     IMPORTANT SUB-FIX (found after the first harness-matched attempt OOM'd
     on video #8): the PyPI qwen_vl_utils package treats "max_pixels" as a
     near-fixed PER-FRAME ceiling (clamped by its own hardcoded
     VIDEO_MAX_PIXELS=768*28*28 global), so total sequence length grows
     ~linearly with video duration/frame count, unbounded -- this is what
     caused the OOM on longer videos. FastVID's own vendored
     vision_process.py (lmms_eval/models/qwenvlutils/vision_process.py in
     the FastVID repo) patches fetch_video() to instead divide the supplied
     max_pixels by nframes (fetch_video():
     `max_pixels = max(int(max_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))`),
     which is what actually keeps the total video-token budget bounded near
     ~16384 regardless of video length, matching the FastVID paper's stated
     "max 16384 video tokens" setting. This is a FastVID-specific patch, not
     upstream qwen_vl_utils behavior. Fixed here by importing
     process_vision_info from a local copy of that exact patched file
     (qwen_vision_process_fastvid.py, copied verbatim from
     ../FastVID/fastvid_qwen25vl/lmms-eval/lmms_eval/models/qwenvlutils/vision_process.py)
     instead of the PyPI package.
  2. Generation config. videomme.yaml's task-level generation_kwargs
     (max_new_tokens=16, do_sample=False, i.e. pure greedy) overrides the
     model class's own defaults. The previous version used temperature=0.2
     (sampling -> run-to-run noise) and max_new_tokens=32.
  3. System prompt. lmms-eval's Qwen2_5_VL model class always prepends
     {"role": "system", "content": "You are a helpful assistant."}. The
     previous version sent no system message at all.
  4. post_prompt text. Verified in lmms_eval/api/task.py (Task.__init__):
     the preset is selected by matching --model's registered name against
     lmms_eval_specific_kwargs' keys, falling back to "default" on no match.
     "qwen2_5_vl" matches nothing in videomme.yaml (no such preset; "qwen_vl"
     is commented out) -> falls back to "default", whose post_prompt is
     "\nAnswer with the option's letter from the given choices directly."
     The previous version hardcoded "The best answer is:", which is actually
     the "llava_vid" preset -- never used for Qwen in this harness.

IMPORTANT: pass a fresh `output_name=` override when running this -- the old
cached output file (from the pre-fix version) has different predictions for
the same question ids, and the resume/skip logic will silently treat them as
already-answered if the filename collides.

Requires: pip install qwen-vl-utils   (in the qwen25vl env)
"""

import json
import os

# A handful of VideoMME videos trip decord's EOF-retry limit when seeking
# near the end of file (intermittent). Must be set before decord is used;
# doubling the default per decord's own suggested fix in the error message.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils import get_chunk

# Matches lmms_eval.models.qwen2_5_vl.Qwen2_5_VL's own defaults exactly --
# both the processor-level bounds and the per-content-item bounds it sets.
# max_pixels_units is overridable via hydra (max_pixels_units=<N>) purely to
# let a verify_mode run of run_inference_qwen25vl_dyseg.py at a non-default
# MAX_PIXELS be diffed against a same-resolution native pass -- default 16384
# reproduces the validated 63.33%-vs-Vanilla-63.8% Gate G1 result unchanged.
MIN_PIXELS = 16 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."


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


# Verbatim lmms-eval videomme_doc_to_text, "default" preset -- the one
# actually selected for --model qwen2_5_vl (see docstring point 4).
VIDEOMME_OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question "
    "based on the video and the subtitles. Respond with only the letter "
    "(A, B, C, or D) of the correct option."
)
VIDEOMME_POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def build_messages(video_path, question, candidates, max_pixels, min_pixels=MIN_PIXELS, force_nframes=None):
    option_block = "\n".join(candidates)
    text = (
        f"{VIDEOMME_OPTION_PROMPT}\n{question}\n{option_block}\n{VIDEOMME_POST_PROMPT}"
    )
    video_item = {"type": "video", "video": video_path, "max_pixels": max_pixels, "min_pixels": min_pixels}
    if force_nframes:
        # qwen_vision_process_fastvid's smart_nframes() honors "nframes"
        # directly instead of deriving frame count from fps=2.0 default --
        # and fetch_video()'s max_pixels/nframes*FRAME_FACTOR division
        # already keeps total tokens ~constant regardless of nframes, so
        # this alone tests "same ~16384 token budget, more temporal
        # coverage" without any other code change.
        video_item["nframes"] = force_nframes
    content = [
        video_item,
        {"type": "text", "text": text},
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    force_nframes = getattr(args, "force_nframes", None)
    force_nframes = int(force_nframes) if force_nframes else None
    max_pixels = max_pixels_units * 28 * 28

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    output_dir = resolve_path(args.output_dir)

    print(f"Loading {args.model_path} ... (max_pixels_units={max_pixels_units})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto"
    )
    # Same processor-level pixel bounds lmms-eval's Qwen2_5_VL model class
    # uses (independent of the per-content-item bounds set in build_messages).
    processor = AutoProcessor.from_pretrained(
        args.model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS
    )

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

            try:
                messages = build_messages(video_path, question, candidates, max_pixels, force_nframes=force_nframes)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Deliberately NOT forwarding the fps metadata (return_video_kwargs)
                # into processor() -- lmms_eval's own Qwen2_5_VL.generate_until
                # computes video_kwargs but never passes them to self.processor()
                # either. Matching that exactly, not "fixing" it, is the point of
                # this gate test.
                image_inputs, video_inputs, _ = process_vision_info(
                    messages, return_video_kwargs=True
                )
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                if "video_grid_thw" in inputs:
                    thw = inputs["video_grid_thw"][0].tolist()
                    n_vis_tok = int(thw[0] * thw[1] * thw[2] / 4)  # /spatial_merge_size**2
                    approx_frames = thw[0] * 2  # temporal_patch_size=2
                    print(
                        f"video_grid_thw={thw} visual_tokens={n_vis_tok} "
                        f"~n_frames={approx_frames} tok/frame~={n_vis_tok / max(1, approx_frames):.1f} "
                        f"total_input_ids_len={inputs['input_ids'].shape[1]}"
                    )
                elif "image_grid_thw" in inputs:
                    print(
                        "WARNING: went down the IMAGE path, not video. "
                        f"image_grid_thw={inputs['image_grid_thw'].tolist()}"
                    )
                else:
                    print("WARNING: neither video_grid_thw nor image_grid_thw in inputs -- unexpected.")

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=16,
                    )
                trimmed = [
                    out[len(inp):]
                    for inp, out in zip(inputs.input_ids, generated_ids)
                ]
                output = processor.batch_decode(
                    trimmed, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
            except Exception as e:
                # A small number of VideoMME videos fail to decode (decord
                # EOF-retry timeout, corrupt container, etc.) -- skip this
                # question rather than crash the whole run. Not marked as
                # answered, so a later rerun (same output_name) will retry it.
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue
            print(f"pred={output!r}")

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

    print(f"[Qwen2.5-VL native, harness-matched] done -> {output_path}")


@hydra.main(
    config_path="configs/inference_qwen25vl", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
