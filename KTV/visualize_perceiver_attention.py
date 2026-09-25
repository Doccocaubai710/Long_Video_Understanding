"""
visualize_perceiver_attention.py

Ve ban do attention cua Perceiver (lop cuoi, forward_with_attn) tren 1 anh
that -- moi query trong so 64 query dang "nhin" vao vung nao cua anh nhieu
nhat. Query-agnostic (khong lien quan cau hoi/text), chi de xem BEN TRONG
Perceiver dang hoc chu y vao dau, khong goi generate().

Ra 1 file PNG duy nhat: anh goc + heatmap trung binh CA 64 query, cong voi
mot vai query rieng le lam vi du (mac dinh 8 query dau tien).

Usage:
  python visualize_perceiver_attention.py \
    image_path=Datasets/COCO/val2017/000000129062.jpg \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    policy_path=outputs/perceiver_stage1_smoke.pt \
    out_path=outputs/viz_perceiver_attn.png \
    num_example_queries=8
"""

import os

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from qwen_vision_process_fastvid import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from train_perceiver_stage1 import CAPTION_PROMPT, MIN_PIXELS, SYSTEM_PROMPT, PerceiverResampler


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def attn_to_heatmap_overlay(base_img, attn_row, H_llm, W_llm, alpha=0.55):
    """attn_row: [n_tokens] (= H_llm*W_llm), gia tri cang cao = query cang
    chu y vung do. Chuan hoa 0-1 roi phu mau (do=chu y nhieu) len anh goc."""
    grid = attn_row.reshape(H_llm, W_llm).cpu().numpy()
    grid = grid - grid.min()
    if grid.max() > 1e-8:
        grid = grid / grid.max()
    heat = Image.fromarray((grid * 255).astype(np.uint8)).resize(base_img.size, Image.BILINEAR)
    heat_rgb = np.zeros((*heat.size[::-1], 4), dtype=np.uint8)
    heat_arr = np.array(heat)
    heat_rgb[..., 0] = 255  # do
    heat_rgb[..., 1] = 255 - heat_arr  # vang o vung thap, do dam o vung cao
    heat_rgb[..., 2] = 0
    heat_rgb[..., 3] = (heat_arr.astype(np.float32) * alpha).astype(np.uint8)
    overlay = Image.fromarray(heat_rgb, mode="RGBA")
    return Image.alpha_composite(base_img.convert("RGBA"), overlay).convert("RGB")


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
    print(f"Loaded Perceiver from {policy_path} (step={ckpt.get('step', '?')})")

    image_path = resolve_path(args.image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
            {"type": "text", "text": CAPTION_PROMPT},
        ]},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    llm = model.model if hasattr(model, "model") else model
    with torch.no_grad():
        image_feat = llm.get_image_features(inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
        if hasattr(image_feat, "pooler_output"):
            image_feat = image_feat.pooler_output
        T1 = image_feat if not isinstance(image_feat, (list, tuple)) else torch.cat(image_feat, dim=0)
        T1 = T1.float()

        _, attn_maps = perceiver.forward_with_attn(T1)

    # image_grid_thw la luoi RAW patch (truoc merge 2x2) -- T1/attn dang o
    # muc DA merge (giong X_units/N o RL-Gating), nen H_llm/W_llm phai chia 2.
    thw = inputs["image_grid_thw"][0].tolist()
    H_llm, W_llm = thw[1] // 2, thw[2] // 2
    n_tokens = H_llm * W_llm
    print(f"image_grid_thw={thw} -> H_llm={H_llm} W_llm={W_llm} n_tokens={n_tokens} (T1.shape={tuple(T1.shape)})")

    last_layer_attn = attn_maps[-1]  # [num_queries, n_tokens]
    base_img = Image.open(image_path).convert("RGB")

    num_example_queries = int(getattr(args, "num_example_queries", 8))
    out_path = resolve_path(getattr(args, "out_path", "outputs/viz_perceiver_attn.png"))

    # 1) Heatmap trung binh CA so query -- "nhin chung Perceiver dang chu y vung nao"
    avg_attn = last_layer_attn.mean(dim=0)
    avg_overlay = attn_to_heatmap_overlay(base_img, avg_attn, H_llm, W_llm)

    # 2) Vai query rieng le lam vi du
    example_overlays = []
    step = max(1, last_layer_attn.shape[0] // num_example_queries)
    query_indices = list(range(0, last_layer_attn.shape[0], step))[:num_example_queries]
    for qi in query_indices:
        example_overlays.append((qi, attn_to_heatmap_overlay(base_img, last_layer_attn[qi], H_llm, W_llm)))

    # Ghep tat ca vao 1 anh: hang 1 = goc + trung binh, hang 2+ = tung query vi du
    thumb_w, thumb_h = 260, 195
    cols = min(4, len(example_overlays) + 2)
    all_imgs = [("GOC", base_img), ("TRUNG BINH (64 query)", avg_overlay)] + \
               [(f"query #{qi}", img) for qi, img in example_overlays]
    rows = (len(all_imgs) + cols - 1) // cols
    montage = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 22)), (255, 255, 255))
    from PIL import ImageDraw
    for idx, (label, img) in enumerate(all_imgs):
        r, c = idx // cols, idx % cols
        thumb = img.resize((thumb_w, thumb_h))
        montage.paste(thumb, (c * thumb_w, r * (thumb_h + 22)))
        draw = ImageDraw.Draw(montage)
        draw.text((c * thumb_w + 4, r * (thumb_h + 22) + thumb_h + 4), label, fill=(0, 0, 0))
    montage.save(out_path, quality=92)
    print(f"Saved -> {out_path}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
