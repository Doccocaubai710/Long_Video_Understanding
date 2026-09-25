"""
recaption_coco_detailed.py

Gop 5 caption ngan (do nhieu nguoi viet doc lap) cua MOI anh COCO thanh
DUNG 1 caption chi tiet, on dinh (khong doi qua tung buoc train nua) --
dung Qwen2.5-VL-7B (dong bang, KHONG train o day) o CHE DO THUAN TEXT
(KHONG doc lai anh, chi doc 5 caption goc) de gop nhanh.

Vi CHI dua tren 5 caption co san (khong nhin lai anh that), yeu cau ve so
luong/vi tri tuong quan CHI duoc dam bao trong pham vi nhung gi 5 caption
gop lai co nhac toi -- KHONG bia them chi tiet moi khong co trong 5 caption
goc.

Output: 1 dong JSON/anh (khong phai 5 dong/anh nhu file goc) trong file
.jsonl, dinh dang {"file_name":..., "caption":...}.

Ho tro chia chunk (num_chunks/chunk_idx) de chay song song tren nhieu
server, va tu resume (bo qua anh da xong) neu bi ngat giua chung.

Nhanh hon nhieu ban doc lai anh: khong co buoc ma hoa thi giac, moi anh
chi la 1 lan sinh van ban ngan (~1-3 cau).

XU LY THEO LO (batch_size, mac dinh 16): goi generate() 1 LAN cho ca lo
anh thay vi tung anh mot -- GPU 7B o batch_size=1 chi dung ~1 phan nho
suc tinh, phan lon thoi gian la do cho tuan tu tung request khong lien
quan gi den do phuc tap that. Gop lo giup dung het suc GPU hon, nhanh hon
nhieu lan so voi ban tung-anh-mot truoc do (tung do that ~40 tieng cho
118K anh o batch_size=1).

Usage:
  python recaption_coco_detailed.py \
    captions_file=Datasets/COCO/annotations/captions_train2017.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    output_path=outputs/coco_train2017_recaptioned.jsonl \
    batch_size=16 num_chunks=1 chunk_idx=0
"""

import json
import os
from collections import defaultdict

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYSTEM_PROMPT = "You are a helpful assistant."

RECAPTION_PROMPT_TEMPLATE = """Here are {n} short captions written independently by different people, all describing the SAME image:
{captions_block}

Combine them into ONE single detailed caption (2-4 sentences) that:
1. Includes every distinct object mentioned across the {n} captions (do not drop any).
2. States the COUNT of an object if any of the captions mentions how many there are (e.g. "three people", "two dogs").
3. States the RELATIVE POSITION between objects if any of the captions mentions it (e.g. "to the left of", "behind", "on top of").
4. Resolves minor disagreements between the captions by keeping the most specific/detailed version.
5. Does NOT invent any object, count, or position that is not stated in at least one of the {n} captions.

Write only the final combined caption, no preamble, no list format."""


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_grouped_captions(captions_file):
    """Tra ve list[{file_name, image_id, captions: [5 caption goc]}] -- 1
    dong/ANH (khac load_coco_captions() cua train_perceiver_stage1.py, ham
    do 1 dong/CAPTION)."""
    d = json.load(open(captions_file, "r", encoding="utf-8"))
    img_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    caps_by_img = defaultdict(list)
    for ann in d["annotations"]:
        caps_by_img[ann["image_id"]].append(ann["caption"].strip())
    rows = []
    for image_id, caps in caps_by_img.items():
        fname = img_by_id.get(image_id)
        if fname is None:
            continue
        rows.append({"file_name": fname, "image_id": image_id, "captions": caps})
    return rows


def get_chunk(items, num_chunks, chunk_idx):
    n = len(items)
    per = (n + num_chunks - 1) // num_chunks
    return items[chunk_idx * per: (chunk_idx + 1) * per]


def load_done_file_names(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["file_name"])
                except json.JSONDecodeError:
                    continue
    return done


def build_prompt_text(processor, captions):
    captions_block = "\n".join(f"- {c}" for c in captions)
    prompt = RECAPTION_PROMPT_TEMPLATE.format(n=len(captions), captions_block=captions_block)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def recaption_batch(model, processor, rows_batch, device):
    """Gop nhieu anh vao 1 lan goi generate() -- can padding_side='left' de
    cac cau tra loi sinh ra dung vi tri du prompt dai ngan khac nhau (chuan
    khi batch-generate voi mo hinh causal)."""
    texts = [build_prompt_text(processor, r["captions"]) for r in rows_batch]
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=150,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
        )
    trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    outputs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [o.strip() for o in outputs]


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ... (che do thuan text, khong nhin anh)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    captions_file = resolve_path(args.captions_file)
    rows = load_grouped_captions(captions_file)
    print(f"Loaded {len(rows)} anh (moi anh trung binh {sum(len(r['captions']) for r in rows) / len(rows):.2f} caption goc)")

    rows = get_chunk(rows, int(getattr(args, "num_chunks", 1)), int(getattr(args, "chunk_idx", 0)))

    output_path = resolve_path(getattr(args, "output_path", "outputs/coco_recaptioned.jsonl"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    done = load_done_file_names(output_path)
    todo = [r for r in rows if r["file_name"] not in done]
    print(f"Da xong truoc do: {len(done)} anh. Con lai trong chunk nay: {len(todo)}")

    batch_size = int(getattr(args, "batch_size", 16))
    with open(output_path, "a", encoding="utf-8") as f:
        pbar = tqdm(total=len(todo))
        i = 0
        while i < len(todo):
            batch = todo[i:i + batch_size]
            try:
                merged_list = recaption_batch(model, processor, batch, device)
                for row, merged in zip(batch, merged_list):
                    f.write(json.dumps({"file_name": row["file_name"], "caption": merged}) + "\n")
                f.flush()
            except Exception as e:
                # Ca lo loi (VD OOM) -- rot lai xu ly tung anh mot cho DUNG
                # lo nay, khong mat het ca lo vi 1 anh loi.
                print(f"[canh bao] lo {i}-{i+len(batch)} loi ({type(e).__name__}: {e}), rot ve tung anh mot")
                torch.cuda.empty_cache()
                for row in batch:
                    try:
                        merged = recaption_batch(model, processor, [row], device)[0]
                        f.write(json.dumps({"file_name": row["file_name"], "caption": merged}) + "\n")
                        f.flush()
                    except Exception as e2:
                        print(f"SKIP {row['file_name']}: {type(e2).__name__}: {e2}")
                        torch.cuda.empty_cache()
            i += len(batch)
            pbar.update(len(batch))
        pbar.close()

    print(f"[recaption_coco_detailed] done -> {output_path}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
