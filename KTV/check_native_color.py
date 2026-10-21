"""
check_native_color.py

Kiem tra xem loi doan sai mau sac la do Perceiver nen mat thong tin, hay do
mot loi tien xu ly anh (VD lat kenh mau) anh huong ca pipeline -- hoi
THANG Qwen2.5-VL, dung TOAN BO token anh goc (KHONG qua Perceiver, khong
nen gi ca), tren nhung anh da thay sai mau trong montage.

Neu Qwen (khong qua Perceiver) VAN doan sai mau y het -> co the la loi
tien xu ly, can kiem tra tiep pixel that.
Neu Qwen (khong qua Perceiver) doan DUNG mau -> loi nam o Perceiver
(nen mat thong tin mau), khong phai loi doc anh.

Usage:
  python check_native_color.py \
    image_paths=Datasets/COCO/val2017/000000123456.jpg,Datasets/COCO/val2017/000000654321.jpg \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct
"""

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MIN_PIXELS = 16 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant."
COLOR_PROMPT = "Describe this image in detail, paying special attention to the exact color of each object you see."


def resolve_path(path):
    path = path.strip()
    if path.startswith("http") or "/" not in path:
        return path
    import os
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)

    image_paths = [resolve_path(p) for p in str(args.image_paths).split(",")]

    for image_path in image_paths:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
                {"type": "text", "text": COLOR_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=150,
                eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            )
        trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        print(f"\n=== {image_path} ===")
        print(f"Qwen (token GOC, khong qua Perceiver): {output}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
