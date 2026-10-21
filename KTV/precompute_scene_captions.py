"""
precompute_scene_captions.py

Tang 1 cua huong "RAG cho video + LLM chon doan bang text" (cam hung tu
FocusGraph -- Scene-Caption Selector, xem idea.md thao luan). Query-agnostic,
chay 1 lan/video, cache lai:

  1. Encode ca video 1 lan (X_pooler), dung DySeg (dyseg_segments, y het co
     che dang dung o Tang 1 cua pipeline chinh) chia thanh N doan theo don
     vi thoi gian cua chinh Qwen.
  2. Quy doi moi doan unit-range -> khoang giay xap xi (unit t ~ t/T_llm *
     tong_thoi_luong -- gan dung, vi lay mau 2fps/tran 768 khong hoan toan
     tuyen tinh voi video rat dai, chap nhan sai so cho ban thu nghiem nay).
  3. Voi MOI doan: goi lai pipeline Qwen chuan (video_start/video_end, giong
     het native script) tren dung khoang giay do, sinh 1 caption ngan gon --
     khong tai su dung X_pooler cho buoc nay (don gian/an toan hon thao tac
     embedding, chi tra them chi phi decode lai tung doan, chap nhan duoc vi
     day la tien xu ly 1 lan/video).
  4. Luu {segments: [[start_sec,end_sec], ...], captions: [...]} ra 1 file
     JSON/video, dung lai cho MOI cau hoi cua video do (run_inference_
     qwen25vl_ragselect.py doc file nay, khong xu ly lai video).

Usage:
  python precompute_scene_captions.py \
    video_dir=Datasets/Video-MME/data/data \
    gt_file=playground/gt_qa_files/VideoMME/val_qa.json \
    cache_dir=outputs/scene_caption_cache \
    max_pixels_units=16384
"""

import json
import os

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import decord
import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_inference_qwen25vl_dyseg import dyseg_segments

MIN_PIXELS = 16 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."
CAPTION_PROMPT = (
    "Mo ta ngan gon (1-2 cau) noi dung chinh cua doan video nay: co nhung "
    "doi tuong/nhan vat nao, dang lam gi, boi canh la gi. Chi tra loi phan mo ta."
)


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def caption_segment(model, processor, video_path, start_sec, end_sec, max_pixels):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": max_pixels,
             "min_pixels": MIN_PIXELS, "video_start": max(0.0, start_sec), "video_end": end_sec},
            {"type": "text", "text": CAPTION_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=64,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def get_segments_and_duration(model, processor, video_path, max_pixels):
    """Encode the whole video once, run DySeg to get unit-range segments, and
    return (segments_in_units, T_llm, duration_sec) -- unit t maps to roughly
    t/T_llm * duration_sec seconds (linear approximation)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
            {"type": "text", "text": "placeholder"},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(model.device)
    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model
    with torch.inference_mode():
        video_feat = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"])
        X_pooler = video_feat.pooler_output
        if isinstance(X_pooler, (list, tuple)):
            X_pooler = torch.cat(X_pooler, dim=0)
    T_llm = int(video_grid_thw[0])
    H_llm = int(video_grid_thw[1]) // 2
    W_llm = int(video_grid_thw[2]) // 2
    N = H_llm * W_llm
    X = X_pooler.reshape(T_llm, N, -1)
    segments = dyseg_segments(X.mean(dim=1), c=8, tau=0.9)

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    duration_sec = len(vr) / vr.get_avg_fps()
    del vr
    return segments, T_llm, duration_sec


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    max_pixels = max_pixels_units * 28 * 28

    print(f"Loading {args.model_path if hasattr(args, 'model_path') else 'Qwen/Qwen2.5-VL-7B-Instruct'} ...")
    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    cache_dir = resolve_path(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    video_names = sorted({sample["video_name"] for sample in gt_qa_pairs})
    print(f"{len(video_names)} unique videos")

    for video_name in tqdm(video_names):
        cache_path = os.path.join(cache_dir, os.path.basename(video_name) + ".json")
        if os.path.exists(cache_path):
            continue
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue
        try:
            segments, T_llm, duration_sec = get_segments_and_duration(model, processor, video_path, max_pixels)
            seg_ranges_sec = []
            captions = []
            for (s, e) in segments:
                start_sec = s / T_llm * duration_sec
                end_sec = e / T_llm * duration_sec
                cap = caption_segment(model, processor, video_path, start_sec, end_sec, max_pixels)
                seg_ranges_sec.append([round(start_sec, 2), round(end_sec, 2)])
                captions.append(cap)
        except Exception as e:
            print(f"SKIP {video_name}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue
        with open(cache_path, "w") as f:
            json.dump({"segments_sec": seg_ranges_sec, "segments_units": segments,
                       "T_llm": T_llm, "captions": captions,
                       "duration_sec": duration_sec}, f, ensure_ascii=False)

    print(f"[scene caption precompute] done -> {cache_dir}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
