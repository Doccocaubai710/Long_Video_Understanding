"""
show_coco_caption_example.py

Script demo don gian -- doc file annotation COCO that, in ra vi du 1 anh
kem toan bo cac caption cua dung anh do (de thay ro 1 anh co nhieu caption
khac nhau do nhieu nguoi viet). Chi de xem/trinh bay, khong lien quan gi
den viec train.

Usage:
  python show_coco_caption_example.py \
    --captions_file Datasets/COCO/annotations/captions_train2017.json \
    --num_examples 3
"""

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_file", default="Datasets/COCO/annotations/captions_train2017.json")
    parser.add_argument("--num_examples", type=int, default=3, help="So anh vi du muon xem")
    args = parser.parse_args()

    with open(args.captions_file, "r", encoding="utf-8") as f:
        d = json.load(f)

    img_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    captions_by_img = defaultdict(list)
    for ann in d["annotations"]:
        captions_by_img[ann["image_id"]].append(ann["caption"].strip())

    print(f"Tong so anh: {len(d['images'])}")
    print(f"Tong so caption: {len(d['annotations'])}")
    print(f"Trung binh caption/anh: {len(d['annotations']) / len(d['images']):.2f}")
    print()

    shown = 0
    for image_id, caps in captions_by_img.items():
        if len(caps) < 4:  # chon anh co nhieu caption de vi du ro nhat
            continue
        print(f"--- Anh: {img_by_id[image_id]} (image_id={image_id}, {len(caps)} caption) ---")
        for i, c in enumerate(caps, start=1):
            print(f"  Caption {i}: {c}")
        print()
        shown += 1
        if shown >= args.num_examples:
            break


if __name__ == "__main__":
    main()
