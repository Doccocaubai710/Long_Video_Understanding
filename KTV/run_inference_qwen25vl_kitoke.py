"""
run_inference_qwen25vl_kitoke.py

Chay Qwen2.5-VL-7B tren VideoMME voi bo nen KiToke GOC (import nguyen ven tu
kitoke/src/kitoke_repro, KHONG sua gi trong repo do), dung chung 1 duong
gather + position_ids voi cac script khac trong repo (build_pruned_sequence)
de moi mode so sanh duoc truc tiep voi nhau.

3 mode, CHI khac nhau o buoc chon token:
  mode=dense   : giu toan bo token (moc 100%). Van di qua DUNG duong gather
                 nhu 2 mode con lai -- neu duong nay sai thi moc 100% cung sai
                 theo, khong bi che di nhu khi goi thang model.generate().
  mode=kitoke  : KiTokeCompressor.build_plan + apply_compression_plan
                 (KDE toan cuc -> pivotal sampling -> interval -> merge).
  mode=uniform : lay deu K chi so tren chuoi phang T*N (doi chung).
K = floor(retention * T*N) cho ca kitoke lan uniform (dung cong thuc cua
KiToke), tinh RIENG tung video.

time_mode (vi tri thoi gian trong RoPE cua token video):
  lmms (mac dinh) : second_per_grid_ts=1.0, tuc fps=2.0 co dinh nhu lmms-eval.
  true            : giay that cua video (2/sample_fps).
  hf_default      : de nguyen gia tri processor tra ve. Voi transformers 5.x +
                    frame da decode san, gia tri do la 2/24 va bi int() ve 0
                    -> moi unit trung vi tri thoi gian (stats: t_pos_unique=1).

So frame: max_frames (mac dinh 128) duoc dua vao phan tu video ->
smart_nframes lay 2fps roi chan tran max_frames, rai deu. max_pixels_units
van la TONG ngan sach (fetch_video tu chia cho so frame), nen 128 frame @
16384 unit ~ 64 unit thoi gian x ~256 token ~ 16K token.

Vi nen la query-agnostic, ket qua (frame da decode, feature, tap token giu)
duoc cache theo video va dung lai cho cac cau hoi lien tiep cua cung video.

Ngoai file pred (dinh dang giong cac script khac, dung thang voi
eval/compute_accuracy_by_bucket.py), script ghi them <output_name>.stats.jsonl
moi video 1 dong: so token, so interval, thong ke mat do KDE, thoi gian nen
-- dung de kiem tra bo nguong alpha/tau co "song" tren feature Qwen2.5-VL
khong (density ~1 cho moi token = kernel chet; num_intervals luon = 1 hoac
luon = T = nguong interval khong hop thang do).

Usage (chay tu thu muc KTV):
  # smoke 12 cau
  python run_inference_qwen25vl_kitoke.py \
    +video_dir=Datasets/Video-MME/data/data \
    +gt_file=playground/gt_qa_files/VideoMME/val_qa.json \
    +model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    +mode=dense +max_frames=128 +limit=12 \
    +output_dir=outputs +output_name=pred_videomme_f128_dense_smoke

  # dense 100%, full 2700 cau
  python run_inference_qwen25vl_kitoke.py \
    +video_dir=Datasets/Video-MME/data/data \
    +gt_file=playground/gt_qa_files/VideoMME/val_qa.json \
    +model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    +mode=dense +max_frames=128 \
    +output_dir=outputs +output_name=pred_videomme_f128_dense

  # KiToke 25% (doi DUY NHAT mode/retention/output_name)
  ... +mode=kitoke +retention=0.25 +output_name=pred_videomme_f128_kitoke_r025
"""

import json
import os
import sys
import time

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from run_inference_qwen25vl_2pass import build_pruned_sequence
from utils import get_chunk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kitoke", "src"))
from kitoke_repro.compressor import KiTokeCompressor, apply_compression_plan  # noqa: E402
from kitoke_repro.config import KiTokeConfig  # noqa: E402


def select_dense(video_embeds_full):
    n_total = video_embeds_full.shape[0]
    kept_indices = torch.arange(n_total, device=video_embeds_full.device)
    return kept_indices, video_embeds_full, {}


def select_uniform(video_embeds_full, retention):
    n_total = video_embeds_full.shape[0]
    K = max(1, int(retention * n_total))
    idx = torch.linspace(0, n_total - 1, K, device=video_embeds_full.device).round().long()
    idx = torch.unique(idx, sorted=True)
    return idx, video_embeds_full[idx], {}


def select_kitoke(compressor, video_embeds_full, T, N):
    """KiToke goc: plan tren [T, N, D], roi gop token bi loai vao token giu.
    retained_indices da sap xep tang dan va hang cua embeds da gop khop dung
    thu tu do -- dung dieu build_pruned_sequence can."""
    tokens = video_embeds_full.float().view(T, N, -1)
    plan = compressor.build_plan(tokens)
    merged = apply_compression_plan(tokens.reshape(T * N, -1), plan, eps=compressor.config.eps)
    density = plan.diversity_scores.float().reciprocal()
    diag = {
        "num_intervals": len(plan.intervals),
        "rescued_intervals": plan.rescued_intervals,
        "density_min": float(density.min().item()),
        "density_median": float(density.median().item()),
        "density_max": float(density.max().item()),
    }
    return plan.retained_indices, merged, diag


def run_one_question(model, processor, compressor, video_path, question, candidates, args, max_pixels, device, cache):
    max_frames = int(getattr(args, "max_frames", 128))
    messages = dm.build_messages(video_path, question, candidates, max_pixels)
    messages[1]["content"][0]["max_frames"] = max_frames
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if cache.get("video_path") != video_path:
        cache.clear()
        _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        cache["video_path"] = video_path
        cache["video_inputs"] = video_inputs
        cache["sample_fps"] = float(video_kwargs["fps"][0])
    inputs = processor(
        text=[text], images=None, videos=cache["video_inputs"], padding=True, return_tensors="pt",
    ).to(device)

    # Frame da decode san nen processor khong co metadata -> sampled_fps mac dinh
    # 24 -> second_per_grid_ts = 2/24 = 0.083, va get_rope_index lam
    # tokens_per_second * int(0.083) = 0: MOI unit trung vi tri thoi gian.
    time_mode = getattr(args, "time_mode", "lmms")
    if time_mode == "lmms":      # fps=2.0 co dinh, nhu harness lmms-eval (transformers 4.x)
        second_per_grid = 1.0
    elif time_mode == "true":    # giay that cua video, nhu code eval chinh thuc cua Qwen
        second_per_grid = max(1.0, round(2.0 / cache["sample_fps"]))
    elif time_mode == "hf_default":  # giu nguyen hanh vi cu de doi chieu voi cac run truoc
        second_per_grid = None
    else:
        raise ValueError(f"unknown time_mode={time_mode!r}, expected 'lmms', 'true' or 'hf_default'")
    if second_per_grid is not None:
        inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], second_per_grid)

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model

    with torch.inference_mode():
        full_embeds = model.get_input_embeddings()(inputs["input_ids"])

        if "video_embeds_full" not in cache:
            video_feat = llm.get_video_features(
                inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]
            )
            video_embeds_full = video_feat.pooler_output
            if isinstance(video_embeds_full, (list, tuple)):
                video_embeds_full = torch.cat(video_embeds_full, dim=0)
            cache["video_embeds_full"] = video_embeds_full.to(full_embeds.dtype).to(full_embeds.device)
        video_embeds_full = cache["video_embeds_full"]

        video_token_id = model.config.video_token_id
        video_mask = (inputs["input_ids"][0] == video_token_id)
        video_positions_in_seq = video_mask.nonzero(as_tuple=True)[0]

        position_ids, _ = llm.get_rope_index(
            input_ids=inputs["input_ids"], image_grid_thw=None, video_grid_thw=inputs["video_grid_thw"],
            second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        T_llm = int(video_grid_thw[0])
        H_llm = int(video_grid_thw[1]) // 2
        W_llm = int(video_grid_thw[2]) // 2
        N = H_llm * W_llm
        n_total = video_embeds_full.shape[0]
        assert n_total == T_llm * N, f"unit-major layout assumption broken: {n_total} != {T_llm}*{N}"

        new_video = "kept_indices" not in cache
        if new_video:
            mode = getattr(args, "mode", "dense")
            retention = float(getattr(args, "retention", 1.0))
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if mode == "dense":
                kept_indices, kept_embeds, diag = select_dense(video_embeds_full)
            elif mode == "uniform":
                kept_indices, kept_embeds, diag = select_uniform(video_embeds_full, retention)
            elif mode == "kitoke":
                kept_indices, kept_embeds, diag = select_kitoke(compressor, video_embeds_full, T_llm, N)
            elif mode == "fastvid":
                # FastVID goc (fastvid_port.py): can them attention cua block cuoi vision tower -> chay lai vision 1 lan.
                import fastvid_port as fv
                torch.manual_seed(int(getattr(args, "seed", 0)))  # ban goc cong torch.rand*1e-6 vao density
                fv_embeds, fv_attn = fv.fastvid_frame_attn(llm, inputs["pixel_values_videos"], inputs["video_grid_thw"])
                fv_embeds = fv_embeds.to(full_embeds.dtype).to(full_embeds.device)
                kept_indices, kept_embeds, diag = fv.fastvid_select(fv_embeds, fv_attn, inputs["video_grid_thw"], fv.fastvid_params(args))
            elif mode == "holitom":
                # HoliTom ban w/o M (holitom_port.py): nen truoc LLM, can attention block cuoi vision tower.
                import holitom_port as ht
                torch.manual_seed(int(getattr(args, "seed", 0)))  # ban goc cong torch.rand*1e-6 vao density
                ht_embeds, ht_attn = ht.holitom_frame_attn(llm, inputs["pixel_values_videos"], inputs["video_grid_thw"])
                ht_embeds = ht_embeds.to(full_embeds.dtype).to(full_embeds.device)
                kept_indices, kept_embeds, diag = ht.holitom_select(ht_embeds, ht_attn, inputs["video_grid_thw"], ht.holitom_params(args))
            elif mode == "vidcom2":
                # VidCom^2 (vidcom2_port.py): chi chon token, truoc LLM; retention = R_RATIO cua ban goc.
                import vidcom2_port as vc
                kept_indices, kept_embeds, diag = vc.vidcom2_select(video_embeds_full, inputs["video_grid_thw"], retention)
            elif mode == "dycoke":
                # DyCoke chi giai doan TTM (dycoke_port.py); dycoke_k suy tu retention neu khong truyen.
                import dycoke_port as dk
                dk_k = float(getattr(args, "dycoke_k", dk.k_for_retention(retention)))
                kept_indices, kept_embeds, diag = dk.dycoke_ttm_select(video_embeds_full, inputs["video_grid_thw"], dk_k)
            else:
                raise ValueError(f"unknown mode={mode!r}, expected 'dense', 'kitoke', 'uniform', 'fastvid', 'holitom', 'vidcom2' or 'dycoke'")
            if device == "cuda":
                torch.cuda.synchronize()
            diag.update({
                "T": T_llm, "N": N, "n_total": n_total, "n_kept": int(kept_indices.numel()),
                "t_pos_unique": int(position_ids[0, 0, video_positions_in_seq].unique().numel()),
                "second_per_grid": float(inputs["second_per_grid_ts"][0]),
                "select_ms": (time.perf_counter() - t0) * 1000.0,
            })
            cache["kept_indices"], cache["kept_embeds"], cache["diag"] = kept_indices, kept_embeds, diag
        kept_indices, kept_embeds = cache["kept_indices"], cache["kept_embeds"]

        final_embeds, final_position_ids, final_attention_mask, _ = build_pruned_sequence(
            full_embeds, inputs, video_token_id, video_positions_in_seq, position_ids, kept_indices, kept_embeds,
        )

        generated_ids = model.generate(
            inputs_embeds=final_embeds, position_ids=final_position_ids, attention_mask=final_attention_mask,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False, num_beams=1, max_new_tokens=16,
        )
    output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return output, cache["diag"], new_video


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    max_pixels = max_pixels_units * 28 * 28
    mode = getattr(args, "mode", "dense")
    retention = float(getattr(args, "retention", 1.0))

    compressor = None
    if mode == "kitoke":
        # Mac dinh = sieu tham so cua paper (dung chung cho ca 3 backbone trong paper).
        compressor = KiTokeCompressor(KiTokeConfig(
            retention_ratio=retention,
            alpha=float(getattr(args, "alpha", 800.0)),
            tau_diff=float(getattr(args, "tau_diff", 110.0)),
            tau_delta=float(getattr(args, "tau_delta", 70.0)),
            tau_relative=float(getattr(args, "tau_relative", 0.40)),
            seed=int(getattr(args, "seed", 0)),
            sampler=str(getattr(args, "sampler", "pivotal")),
        ))
    print(f"mode={mode} retention={retention} max_frames={getattr(args, 'max_frames', 128)} "
          f"max_pixels_units={max_pixels_units} time_mode={getattr(args, 'time_mode', 'lmms')}")

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)

    video_dir = dm.resolve_path(args.video_dir)
    gt_file = dm.resolve_path(args.gt_file)
    output_dir = dm.resolve_path(args.output_dir)

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, int(getattr(args, "num_chunks", 1)), int(getattr(args, "chunk_idx", 0)))
    limit = int(getattr(args, "limit", 0))  # >0: chi chay N cau dau (smoke test)
    if limit > 0:
        gt_qa_pairs = gt_qa_pairs[:limit]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    stats_path = os.path.join(output_dir, f"{args.output_name}.stats.jsonl")
    generated_ids_seen = dm.load_answered_ids(output_path)
    output_mode = "a" if os.path.exists(output_path) else "w"

    cache = {}
    retentions = []
    with open(output_path, output_mode, encoding="utf-8") as ans_file, \
            open(stats_path, "a", encoding="utf-8") as stats_file:
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
                output, diag, new_video = run_one_question(
                    model, processor, compressor, video_path, sample["question"], sample["candidates"],
                    args, max_pixels, device, cache,
                )
            except Exception as e:
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                cache.clear()
                torch.cuda.empty_cache()
                continue

            retention_actual = diag["n_kept"] / diag["n_total"]
            retentions.append(retention_actual)
            print(f"pred={output!r} retention={retention_actual:.4f} ({diag['n_kept']}/{diag['n_total']}) "
                  f"T={diag['T']} N={diag['N']}")
            if new_video:
                stats_file.write(json.dumps({"video_name": video_name, "mode": mode, **diag}) + "\n")
                stats_file.flush()

            sample_set = {
                "task_name": sample.get("task_name", "unknown"), "question": sample["question"],
                "question_id": str(sample.get("question_id", answer_id)), "id": answer_id,
                "answer_number": sample["answer_number"], "candidates": sample["candidates"],
                "answer": sample["answer"], "pred": output,
            }
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()
            generated_ids_seen.add(answer_id)

    if retentions:
        print(f"[run_inference_qwen25vl_kitoke] done -> {output_path} "
              f"(mean retention={sum(retentions)/len(retentions):.4f}, stats -> {stats_path})")
    else:
        print(f"[run_inference_qwen25vl_kitoke] done -> {output_path} (khong co cau hoi moi)")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
