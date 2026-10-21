"""
infer_perceiver_stage2.py

Infer that (khong phai train) cho checkpoint Perceiver Giai doan 2 (video,
train_perceiver_stage2.py) -- lay 1 video, dung Perceiver (da train, dong
bang) nen T1 (video) -> T2, ghep vao LLM, GOI generate() THAT de xem
Perceiver sinh ra caption gi, so sanh bang mat voi caption that.

Y het infer_perceiver_stage1.py ve co che, chi doi anh -> video
(get_video_features/video_token_id thay vi get_image_features/image_token_id).
Cung ho tro xuat montage PNG (20 ca/1 file), thumbnail lay khung hinh GIUA
video (qua fetch_video, giong cach visualize_pruning.py da dung).

Usage:
  python infer_perceiver_stage2.py \
    video_dir=Datasets/MSRVTT/video \
    captions_file=Datasets/MSRVTT/msrvtt_train_9k.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    policy_path=outputs/perceiver_stage2_smoke.pt \
    num_samples=10 out_image=outputs/infer_perceiver_stage2_montage.png
"""

import os
import random

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from qwen_vision_process_fastvid import fetch_video, process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from infer_perceiver_stage1 import build_montage
from train_perceiver_stage1 import MIN_PIXELS, SYSTEM_PROMPT, PerceiverResampler
from run_inference_qwen25vl_dyseg import dyseg_segments
from train_perceiver_stage2 import CAPTION_PROMPT, load_msrvtt_captions


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def middle_frame(video_path, max_pixels):
    """Lay 1 khung hinh GIUA video de lam thumbnail cho montage -- khong
    anh huong gi den infer (chi de nhin bang mat), infer thuc su van dung
    toan bo video qua process_vision_info nhu binh thuong."""
    raw_video = fetch_video({"type": "video", "video": video_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS})
    frame = raw_video[raw_video.shape[0] // 2]
    arr = frame.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr)


def infer_one_video(model, processor, perceiver, video_path, dyseg_c, dyseg_tau, pointing_tau, max_pixels, device):
    """Kien truc v3 -- chia T_llm unit thanh cac DOAN do dai khac nhau theo
    noi dung (dyseg_segments), moi doan qua Perceiver 1 lan + Pointing,
    giong het run_one_video() trong train_perceiver_stage2.py (khong co
    backward/loss)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
            {"type": "text", "text": CAPTION_PROMPT},
        ]},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    llm = model.model if hasattr(model, "model") else model
    video_token_id = model.config.video_token_id

    with torch.no_grad():
        video_feat = llm.get_video_features(
            prompt_inputs["pixel_values_videos"], video_grid_thw=prompt_inputs["video_grid_thw"]
        )
        if hasattr(video_feat, "pooler_output"):
            video_feat = video_feat.pooler_output
        T1_flat = video_feat if not isinstance(video_feat, (list, tuple)) else torch.cat(video_feat, dim=0)
        T1_flat = T1_flat.float()

        thw = prompt_inputs["video_grid_thw"][0].tolist()
        T_llm, H_llm, W_llm = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
        N = H_llm * W_llm
        T1 = T1_flat.view(T_llm, N, -1)

        frame_feats = T1.mean(dim=1)  # [T_llm, D]
        segments = dyseg_segments(frame_feats, c=dyseg_c, tau=dyseg_tau)
        per_seg_tokens = []
        total_pointed = 0
        for start, end in segments:
            T1_seg = T1[start:end].reshape(-1, T1.shape[-1])
            T2_seg = perceiver(T1_seg)
            T2n = torch.nn.functional.normalize(T2_seg, dim=-1)
            T1n = torch.nn.functional.normalize(T1_seg, dim=-1)
            sim = T2n @ T1n.t()
            max_sim_per_tok, _ = sim.max(dim=0)
            point_mask = max_sim_per_tok >= pointing_tau
            T1_star = T1_seg[point_mask]
            total_pointed += T1_star.shape[0]
            per_seg_tokens.append(torch.cat([T2_seg, T1_star], dim=0))
        T2 = torch.cat(per_seg_tokens, dim=0)
        n_queries = T2.shape[0]
        n_segments_used = len(segments)

        full_input_ids = prompt_inputs["input_ids"][0]
        video_mask_full = (full_input_ids == video_token_id)
        pre_ids = full_input_ids[: video_mask_full.nonzero(as_tuple=True)[0][0]]
        post_ids = full_input_ids[video_mask_full.nonzero(as_tuple=True)[0][-1] + 1:]
        video_placeholder = torch.full((n_queries,), video_token_id, dtype=full_input_ids.dtype, device=device)
        input_ids = torch.cat([pre_ids, video_placeholder, post_ids]).unsqueeze(0)

        text_embeds = model.get_input_embeddings()(input_ids)
        video_mask = (input_ids[0] == video_token_id).unsqueeze(0).unsqueeze(-1)
        final_embeds = text_embeds.masked_scatter(video_mask, T2.to(text_embeds.dtype))
        attention_mask = torch.ones_like(input_ids)

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).view(1, 1, seq_len).expand(3, 1, seq_len)

        generated_ids = model.generate(
            inputs_embeds=final_embeds, position_ids=position_ids, attention_mask=attention_mask,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False, num_beams=1, max_new_tokens=40,
        )
    pred = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return pred, n_segments_used, n_queries, total_pointed


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 1024))
    max_pixels = max_pixels_units * 28 * 28

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)

    policy_path = resolve_path(args.policy_path)
    ckpt = torch.load(policy_path, map_location=device)
    perceiver = PerceiverResampler(ckpt["dim"], num_queries=ckpt["num_queries"], num_layers=ckpt["num_layers"]).to(device).float()
    perceiver.load_state_dict(ckpt["state_dict"])
    perceiver.eval()
    dyseg_c = int(getattr(args, "dyseg_c", ckpt.get("dyseg_c", 8)))
    dyseg_tau = float(getattr(args, "dyseg_tau", ckpt.get("dyseg_tau", 0.9)))
    pointing_tau = float(getattr(args, "pointing_tau", ckpt.get("pointing_tau", 0.5)))
    print(f"Loaded Perceiver from {policy_path} (step={ckpt.get('step', '?')}, num_queries={ckpt['num_queries']}, "
          f"dyseg_c={dyseg_c}, dyseg_tau={dyseg_tau}, pointing_tau={pointing_tau})")

    video_dir = resolve_path(args.video_dir)
    captions_file = resolve_path(args.captions_file)
    rows = load_msrvtt_captions(captions_file)
    random.seed(123)  # khac seed train (0)
    random.shuffle(rows)

    seen = set()
    unique_rows = []
    for r in rows:
        if r["file_name"] not in seen:
            seen.add(r["file_name"])
            unique_rows.append(r)

    num_samples = int(getattr(args, "num_samples", 10))
    out_image = getattr(args, "out_image", None)
    montage_items = []
    for row in unique_rows[:num_samples]:
        video_path = os.path.join(video_dir, row["file_name"])
        if not os.path.exists(video_path):
            continue
        pred, n_segments_used, n_queries, total_pointed = infer_one_video(
            model, processor, perceiver, video_path, dyseg_c, dyseg_tau, pointing_tau, max_pixels, device,
        )
        print(f"\n=== {row['file_name']} ===")
        print(f"  Caption THAT : {row['caption']}")
        print(f"  Caption Perceiver sinh ra: {pred!r}")
        print(f"  n_segments_used={n_segments_used} n_final_tokens={n_queries} n_pointed_total={total_pointed}")
        if out_image:
            montage_items.append((middle_frame(video_path, max_pixels), row["caption"], pred, row["file_name"]))

    if out_image and montage_items:
        build_montage(montage_items, resolve_path(out_image))


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
