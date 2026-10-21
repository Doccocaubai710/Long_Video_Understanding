"""
infer_perceiver_videomme.py

Do THAT xem Perceiver (Giai doan 2, video) giu duoc bao nhieu thong tin de
tra loi cau hoi trac nghiem VideoMME -- thay TOAN BO token video goc bang
DUNG 64 token T2 (Perceiver tao ra), hoi cau hoi that (dinh dang A/B/C/D
giong het cac script RL-Gating/DySeg), generate() that, ghi ra cung dinh
dang JSON de dung truc tiep eval/compute_accuracy_by_bucket.py.

Query-agnostic o dung 1 cho: Perceiver KHONG nhin cau hoi khi nen T1->T2
(giong RL-Gating) -- cau hoi chi xuat hien luc build prompt cuoi cung cho
LLM, sau khi T2 da co san.

Usage (nen thu tren 1 tap con truoc, xem run_inference_qwen25vl_rlgating.py
vi du cach tao subset):
  python infer_perceiver_videomme.py \
    video_dir=Datasets/Video-MME/data/data \
    gt_file=playground/gt_qa_files/VideoMME/val_qa_subset300.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    policy_path=outputs/perceiver_stage2_smoke.pt \
    output_dir=outputs output_name=pred_videomme_perceiver_stage2_subset300 \
    num_chunks=1 chunk_idx=0
"""

import json
import os

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from run_inference_qwen25vl_dyseg import dyseg_segments
from train_perceiver_stage1 import PerceiverResampler
from utils import get_chunk


def run_one_question(model, processor, perceiver, video_path, question, candidates, dyseg_c, dyseg_tau, pointing_tau, max_pixels, device, ablate=None):
    """Kien truc v3 -- chia T_llm unit thanh cac DOAN do dai khac nhau theo
    noi dung (dyseg_segments), voi TUNG doan: Perceiver rieng (doc CA doan
    trong 1 lan) + Pointing rieng (nguong cosine tau), noi tat ca lai --
    giong het run_one_video() trong train_perceiver_stage2.py, chi khac o
    cho khong tinh caption_loss/backward ma generate() that de tra loi cau
    hoi."""
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

        frame_feats = T1.mean(dim=1)  # [T_llm, D]
        segments = dyseg_segments(frame_feats, c=dyseg_c, tau=dyseg_tau)  # [(start,end), ...] phu het T_llm

        per_seg_tokens = []
        for start, end in segments:
            T1_seg = T1[start:end].reshape(-1, T1.shape[-1])  # [(end-start)*N, D]
            T2_seg = perceiver(T1_seg)  # [num_queries, D] -- KHONG nhin question/candidates
            T2n = torch.nn.functional.normalize(T2_seg, dim=-1)
            T1n = torch.nn.functional.normalize(T1_seg, dim=-1)
            sim = T2n @ T1n.t()
            max_sim_per_tok, _ = sim.max(dim=0)
            point_mask = max_sim_per_tok >= pointing_tau
            T1_star = T1_seg[point_mask]
            per_seg_tokens.append(torch.cat([T2_seg, T1_star], dim=0))
        T2 = torch.cat(per_seg_tokens, dim=0)  # [n_queries, D] -- BIEN DOI theo video VA do phuc tap noi dung
        # Kiem chung: LLM co that su dung noi dung T2 khong? Giu nguyen so token va
        # thong ke (trung binh, do lech chuan), chi bo noi dung video.
        if ablate == "noise":
            T2 = torch.randn_like(T2) * T2.std() + T2.mean()
        elif ablate == "zeros":
            T2 = torch.zeros_like(T2)
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
    retention = n_queries / n_total_real
    return output, retention, n_total_real, n_queries


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    max_pixels = max_pixels_units * 28 * 28

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
    # dyseg_c/dyseg_tau/pointing_tau la sieu tham so cua kien truc v3, luon
    # lay TU checkpoint (train_perceiver_stage2.py da luu san) de dam bao
    # infer dung khop voi luc train -- CHO PHEP override qua args de thu
    # nghiem tham so khac luc infer.
    dyseg_c = int(getattr(args, "dyseg_c", ckpt.get("dyseg_c", 8)))
    dyseg_tau = float(getattr(args, "dyseg_tau", ckpt.get("dyseg_tau", 0.9)))
    pointing_tau = float(getattr(args, "pointing_tau", ckpt.get("pointing_tau", 0.5)))
    ablate = getattr(args, "ablate", None)
    if ablate:
        torch.manual_seed(0)
    print(f"Loaded Perceiver from {policy_path} (step={ckpt.get('step', '?')}, num_queries={ckpt['num_queries']}, "
          f"dyseg_c={dyseg_c}, dyseg_tau={dyseg_tau}, pointing_tau={pointing_tau}, ablate={ablate})")

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
                output, retention, n_total_real, n_queries = run_one_question(
                    model, processor, perceiver, video_path, sample["question"], sample["candidates"],
                    dyseg_c, dyseg_tau, pointing_tau, max_pixels, device, ablate=ablate,
                )
            except Exception as e:
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue

            retentions.append(retention)
            print(f"pred={output!r} retention={retention:.4f} ({n_queries}/{n_total_real})")

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
        print(f"[perceiver videomme infer] done -> {output_path} (mean retention={sum(retentions)/len(retentions):.4f})")
    else:
        print(f"[perceiver videomme infer] done -> {output_path} (khong co cau hoi moi)")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
