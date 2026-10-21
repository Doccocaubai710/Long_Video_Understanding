"""
infer_perceiver_videomme_grouped.py

THI NGHIEM kiem chung gia thuyet "Perceiver kem vi chi nhin 30/384 unit":
thay vi chon 30 unit dai dien roi bo phan con lai, script nay cho Perceiver
DOC THEO NHOM group_size unit lien nhau (mac dinh 12), moi nhom -> 64 token.
Toan bo unit cua video deu duoc phu, khong bo unit nao. KHONG can train lai:
cross-attention cua Perceiver nhan chuoi token do dai bat ky.

Ngan sach token cuoi = so_nhom x 64 = ceil(T / group_size) x 64
  VD video 384 unit, group_size=12 -> 32 nhom -> 2.048 token
  (~16,7% cua 12.288 token tho o max_pixels_units=16384, gan bang retention
  15% cua RL-Gating -> so sanh duoc cong bang).

Pointing mac dinh TAT (pointing_tau=-1) de ngan sach dung bang so_nhom x 64;
bat bang +pointing_tau=0.5 neu muon them token goc nhu ban chinh.

Usage:
  python infer_perceiver_videomme_grouped.py \
    +video_dir=Datasets/Video-MME/data/data \
    +gt_file=playground/gt_qa_files/VideoMME/val_qa_subset300.json \
    +model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    +policy_path=outputs/perceiver_stage2_v2.pt \
    +max_pixels_units=16384 +group_size=12 \
    +output_dir=outputs +output_name=pred_videomme_perceiver_grouped12_16384_subset300 \
    +num_chunks=1 +chunk_idx=0
"""

import json
import math
import os

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from train_perceiver_stage1 import PerceiverResampler
from utils import get_chunk


def run_one_question(model, processor, perceiver, video_path, question, candidates,
                     group_size, pointing_tau, max_pixels, device):
    messages = dm.build_messages(video_path, question, candidates, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    llm = model.model if hasattr(model, "model") else model
    video_token_id = model.config.video_token_id

    with torch.no_grad():
        video_feat = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"])
        if hasattr(video_feat, "pooler_output"):
            video_feat = video_feat.pooler_output
        T1_flat = video_feat if not isinstance(video_feat, (list, tuple)) else torch.cat(video_feat, dim=0)
        T1_flat = T1_flat.float()
        n_total_real = T1_flat.shape[0]

        thw = inputs["video_grid_thw"][0].tolist()
        T_llm, H_llm, W_llm = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
        N = H_llm * W_llm
        T1 = T1_flat.view(T_llm, N, -1)  # [T_llm, N, D]

        # Phu TOAN BO unit: chia thanh ceil(T/group_size) nhom lien tiep.
        num_groups = math.ceil(T_llm / group_size)
        group_tokens = []
        for g in range(num_groups):
            chunk = T1[g * group_size:(g + 1) * group_size].reshape(-1, T1.shape[-1])  # [k*N, D]
            T2_g = perceiver(chunk)  # [64, D] -- khong nhin question/candidates
            if pointing_tau >= 0:
                T2n = torch.nn.functional.normalize(T2_g, dim=-1)
                T1n = torch.nn.functional.normalize(chunk, dim=-1)
                max_sim_per_tok, _ = (T2n @ T1n.t()).max(dim=0)
                T2_g = torch.cat([T2_g, chunk[max_sim_per_tok >= pointing_tau]], dim=0)
            group_tokens.append(T2_g)
        T2 = torch.cat(group_tokens, dim=0)
        n_queries = T2.shape[0]

        full_input_ids = inputs["input_ids"][0]
        video_mask_full = (full_input_ids == video_token_id)
        pre_ids = full_input_ids[: video_mask_full.nonzero(as_tuple=True)[0][0]]
        post_ids = full_input_ids[video_mask_full.nonzero(as_tuple=True)[0][-1] + 1:]
        video_placeholder = torch.full((n_queries,), video_token_id, dtype=full_input_ids.dtype, device=device)
        new_input_ids = torch.cat([pre_ids, video_placeholder, post_ids]).unsqueeze(0)

        text_embeds = model.get_input_embeddings()(new_input_ids)
        video_mask = (new_input_ids[0] == video_token_id).unsqueeze(0).unsqueeze(-1)
        final_embeds = text_embeds.masked_scatter(video_mask, T2.to(text_embeds.dtype))
        attention_mask = torch.ones_like(new_input_ids)

        seq_len = new_input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).view(1, 1, seq_len).expand(3, 1, seq_len)

        generated_ids = model.generate(
            inputs_embeds=final_embeds, position_ids=position_ids, attention_mask=attention_mask,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False, num_beams=1, max_new_tokens=16,
        )
    output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return output, n_queries / n_total_real, n_total_real, n_queries, num_groups


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    max_pixels = max_pixels_units * 28 * 28
    group_size = int(getattr(args, "group_size", 12))
    pointing_tau = float(getattr(args, "pointing_tau", -1))

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)

    policy_path = dm.resolve_path(args.policy_path)
    ckpt = torch.load(policy_path, map_location=device)
    perceiver = PerceiverResampler(ckpt["dim"], num_queries=ckpt["num_queries"], num_layers=ckpt["num_layers"]).to(device).float()
    perceiver.load_state_dict(ckpt["state_dict"])
    perceiver.eval()
    print(f"Loaded Perceiver from {policy_path} (step={ckpt.get('step', '?')}, num_queries={ckpt['num_queries']}), "
          f"max_pixels_units={max_pixels_units}, group_size={group_size}, pointing_tau={pointing_tau} "
          f"({'tat' if pointing_tau < 0 else 'bat'})")

    video_dir = dm.resolve_path(args.video_dir)
    gt_file = dm.resolve_path(args.gt_file)
    output_dir = dm.resolve_path(args.output_dir)

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
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}")
                continue

            try:
                output, retention, n_total_real, n_queries, num_groups = run_one_question(
                    model, processor, perceiver, video_path, sample["question"], sample["candidates"],
                    group_size, pointing_tau, max_pixels, device,
                )
            except Exception as e:
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue

            retentions.append(retention)
            print(f"pred={output!r} retention={retention:.4f} ({n_queries}/{n_total_real}) groups={num_groups}")

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
        print(f"[perceiver grouped infer] done -> {output_path} (mean retention={sum(retentions)/len(retentions):.4f})")
    else:
        print(f"[perceiver grouped infer] done -> {output_path} (khong co cau hoi moi)")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
