"""
infer_perceiver_stage1.py

Infer that (khong phai train) cho checkpoint Perceiver da train o
train_perceiver_stage1.py -- lay 1 anh, dung Perceiver (da train, dong bang)
nen T1 -> T2, ghep vao LLM, GOI generate() THAT (khong teacher-force) de xem
Perceiver sinh ra caption gi, so sanh bang mat voi caption that.

Usage:
  python infer_perceiver_stage1.py \
    image_dir=Datasets/COCO/val2017 \
    captions_file=Datasets/COCO/annotations/captions_val2017.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    policy_path=outputs/perceiver_stage1_smoke.pt \
    num_samples=10
"""

import json
import os
import random
import textwrap

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from qwen_vision_process_fastvid import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from train_perceiver_stage1 import (
    CAPTION_PROMPT,
    MIN_PIXELS,
    SYSTEM_PROMPT,
    PerceiverResampler,
    load_captions_auto,
)


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def build_montage(items, out_path, thumb_w=300, thumb_h=225, cols=4, text_h=150, label_h=16):
    """items: list[(PIL.Image, real_caption, gen_caption)] hoac
    list[(PIL.Image, real_caption, gen_caption, file_name)] -- file_name la
    tuy chon, neu co se in ra 1 dong ten file (den, in dam) o DAU moi o, de
    tien tra lai duong dan anh/video that (VD dua vao check_native_color.py)
    ma khong phai doi chieu qua log train/infer. Ve tat ca vao DUNG 1 file
    PNG -- moi o gom [ten file] + thumbnail + 2 dong caption (that/mau xanh,
    Perceiver sinh ra/mau do) da wrap chu."""
    n = len(items)
    rows = (n + cols - 1) // cols
    cell_w = thumb_w
    cell_h = label_h + thumb_h + text_h
    montage = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    for idx, item in enumerate(items):
        if len(item) == 4:
            img, real_cap, gen_cap, file_name = item
        else:
            img, real_cap, gen_cap = item
            file_name = None
        r, c = idx // cols, idx % cols
        cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
        draw = ImageDraw.Draw(cell)
        if file_name:
            label = file_name if len(file_name) * 6 <= thumb_w else file_name[-(thumb_w // 6):]
            draw.text((2, 1), label, fill=(0, 0, 0))
        thumb = img.convert("RGB").resize((thumb_w, thumb_h))
        cell.paste(thumb, (0, label_h))
        draw.rectangle([0, label_h, cell_w - 1, label_h + thumb_h - 1], outline=(180, 180, 180), width=1)
        y = label_h + thumb_h + 4
        for line in wrap_caption(f"THAT: {real_cap}", thumb_w):
            draw.text((4, y), line, fill=(0,110, 0))
            y += 14
        y += 4
        for line in wrap_caption(f"GEN : {gen_cap}", thumb_w):
            draw.text((4, y), line, fill=(180, 0, 0))
            y += 14
        montage.paste(cell, (c * cell_w, r * cell_h))
    montage.save(out_path, quality=92)
    print(f"Saved montage -> {out_path}")


def wrap_caption(text, thumb_w, chars_per_line=None):
    # ~6px/ky tu voi font PIL mac dinh -- uoc luong so ky tu vua khop thumb_w
    width = chars_per_line or max(20, thumb_w // 6)
    return textwrap.wrap(text, width=width) or [""]


def infer_one_image(model, processor, perceiver, image_path, max_pixels, device):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
            {"type": "text", "text": CAPTION_PROMPT},
        ]},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    llm = model.model if hasattr(model, "model") else model
    image_token_id = model.config.image_token_id

    with torch.no_grad():
        image_feat = llm.get_image_features(
            prompt_inputs["pixel_values"], image_grid_thw=prompt_inputs["image_grid_thw"]
        )
        if hasattr(image_feat, "pooler_output"):
            image_feat = image_feat.pooler_output
        T1 = image_feat if not isinstance(image_feat, (list, tuple)) else torch.cat(image_feat, dim=0)
        T1 = T1.float()

        T2 = perceiver(T1)
        n_queries = T2.shape[0]

        full_input_ids = prompt_inputs["input_ids"][0]
        image_mask_full = (full_input_ids == image_token_id)
        pre_ids = full_input_ids[: image_mask_full.nonzero(as_tuple=True)[0][0]]
        post_ids = full_input_ids[image_mask_full.nonzero(as_tuple=True)[0][-1] + 1:]
        image_placeholder = torch.full((n_queries,), image_token_id, dtype=full_input_ids.dtype, device=device)
        input_ids = torch.cat([pre_ids, image_placeholder, post_ids]).unsqueeze(0)

        text_embeds = model.get_input_embeddings()(input_ids)
        image_mask = (input_ids[0] == image_token_id).unsqueeze(0).unsqueeze(-1)
        final_embeds = text_embeds.masked_scatter(image_mask, T2.to(text_embeds.dtype))
        attention_mask = torch.ones_like(input_ids)

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).view(1, 1, seq_len).expand(3, 1, seq_len)

        generated_ids = model.generate(
            inputs_embeds=final_embeds, position_ids=position_ids, attention_mask=attention_mask,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False, num_beams=1, max_new_tokens=40,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


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
    print(f"Loaded Perceiver from {policy_path} (step={ckpt.get('step', '?')}, num_queries={ckpt['num_queries']})")

    image_dir = resolve_path(args.image_dir)
    captions_file = resolve_path(args.captions_file)
    rows = load_captions_auto(captions_file)
    random.seed(123)  # khac seed train (0) -- xem anh CHUA thay trong luc train
    random.shuffle(rows)

    # 1 caption/anh (bo trung anh, vi COCO co nhieu caption/anh) de xem da dang
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
        image_path = os.path.join(image_dir, row["file_name"])
        if not os.path.exists(image_path):
            continue
        pred = infer_one_image(model, processor, perceiver, image_path, max_pixels, device)
        print(f"\n=== {row['file_name']} ===")
        print(f"  Caption THAT : {row['caption']}")
        print(f"  Caption Perceiver sinh ra: {pred!r}")
        if out_image:
            montage_items.append((Image.open(image_path), row["caption"], pred, row["file_name"]))

    if out_image and montage_items:
        build_montage(montage_items, resolve_path(out_image))


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
