"""
run_inference_qwen25vl_2pass.py

"Huong 2" (idea.md discussion): thay vi doan ho "cho nao quan trong" bang mot
model phu (DINOv2/V-JEPA/motion-delta -- tat ca da khong cai thien gi, xem
README_DySeg_STPrune.md), de chinh Qwen2.5-VL tu lo ra no dang thieu thong
tin o dau, qua 1 lan forward re o ngan sach thap, roi cap them ngan sach
DUNG CHO DO cho lan sinh cau tra loi that.

Pass 1 (re): Uniform-prune ve ngan sach thap (`pass1_tokens`, vd 3000), MOT
lan forward (khong generate nhieu buoc -- VideoMME chi can 1 token dap an)
voi output_attentions=True. Lay 2 thu MIEN PHI tu cung 1 lan forward:
  - entropy cua logits vi tri cuoi -- proxy do tu tin
  - attention tu vi tri cuoi (vi tri sinh dap an) toi tung token thi giac
    con trong context pass-1

Vision encoder CHI CHAY 1 LAN cho ca 2 pass (video_embeds_full dung chung) --
"2 pass" thuc chat la 2 lan forward LLM re tren cung 1 pool da encode san,
khong phai encode lai video.

Pass 2 (chi chay neu pass 1 KHONG du tu tin -- entropy > nguong):
  - Gop diem attention theo UNIT (vi tri thoi gian trong luoi token cua
    chinh Qwen -- dung don vi voi DySeg's "unit").
  - Xep hang unit theo diem, chon top units "nong" duoc GIU TOAN BO token
    (full retention) trong pham vi `hot_budget_frac` cua tokens_num.
  - Phan con lai cua tokens_num duoc rai DEU (Uniform) tren cac unit "lanh"
    con lai -- dam bao van co do phu toi thieu toan video, khong bo trong
    hoan toan cho nao.
  - Forward/generate that voi tap token da phan bo lai nay.

CANH BAO KY THUAT: output_attentions=True KHONG hoat dong voi
attn_implementation mac dinh (sdpa/flash) trong nhieu ban transformers --
script nay ep attn_implementation="eager", co the cham hon dang ke so voi
cac script khac trong repo. Day la ban thu nghiem dau tien, nhieu kha nang
can debug them khi chay tren du lieu that (shape/indexing cua attention,
nguong entropy hop ly...) -- chua duoc kiem chung nhu cac script da on
dinh khac.

Usage:
  python run_inference_qwen25vl_2pass.py experiment=videomme \
    max_pixels_units=49152 tokens_num=16384 pass1_tokens=3000 \
    hot_budget_frac=0.4 confidence_entropy_max=2.0 \
    output_name=pred_videomme_qwen25vl_2pass
"""

import json
import os

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_inference_qwen25vl_dyseg import (
    build_messages,
    load_answered_ids,
    resolve_path,
    uniform_prune_video_tokens,
)
from utils import get_chunk


def build_pruned_sequence(full_embeds, inputs, video_token_id, video_positions_in_seq,
                           position_ids, kept_indices, kept_embeds):
    """Same gather/scatter pattern as run_inference_qwen25vl_dyseg.run_one_question,
    factored out so both passes can reuse it. Also returns the positions of the
    kept video tokens WITHIN the pruned sequence, in kept_indices order -- needed
    to read attention scores back out per kept token after pass 1."""
    video_mask = (inputs["input_ids"][0] == video_token_id)
    keep_seq_pos = video_positions_in_seq[kept_indices]
    drop_mask = video_mask.clone()
    drop_mask[keep_seq_pos] = False
    final_seq_mask = ~drop_mask

    final_embeds = full_embeds[0][final_seq_mask].unsqueeze(0)
    final_embeds = final_embeds.masked_scatter(
        (inputs["input_ids"][0][final_seq_mask] == video_token_id).unsqueeze(0).unsqueeze(-1),
        kept_embeds.to(final_embeds.dtype),
    )
    final_position_ids = position_ids[:, :, final_seq_mask]
    final_attention_mask = inputs["attention_mask"][0][final_seq_mask].unsqueeze(0)
    video_positions_in_pruned = (
        inputs["input_ids"][0][final_seq_mask] == video_token_id
    ).nonzero(as_tuple=True)[0]
    return final_embeds, final_position_ids, final_attention_mask, video_positions_in_pruned


import re

_ATTN_HOOK_STATE = {"module": None, "captured": None}


def _find_last_self_attn(model):
    """Locate the '...layers.N.self_attn' module with the highest N -- avoids
    hardcoding an exact attribute path, which can differ between transformers
    versions/wrappers for Qwen2.5-VL's inner language model."""
    best_idx, best_name, best_module = -1, None, None
    for name, module in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.self_attn$", name)
        if m:
            idx = int(m.group(1))
            if idx > best_idx:
                best_idx, best_name, best_module = idx, name, module
    if best_module is None:
        raise RuntimeError(
            "Could not find a '...layers.N.self_attn' module to hook -- "
            "inspect model.named_modules() and adjust the regex above."
        )
    return best_name, best_module


def _install_attention_hook(model):
    """Force output_attentions=True on ONLY the last self-attention layer via a
    pre-hook (instead of passing output_attentions=True to the whole model,
    which makes HF materialize the attention matrix for ALL layers at once --
    the actual cause of the CUDA OOM seen in the first run: ~28 layers x
    [n_heads, seq, seq] simultaneously). Capture that one layer's weights via
    a forward hook. Installed once, reused across all questions."""
    if _ATTN_HOOK_STATE["module"] is not None:
        return
    name, module = _find_last_self_attn(model)
    print(f"[2pass] hooking attention capture on module: {name}")

    def _pre_hook(mod, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["output_attentions"] = True
        return args, kwargs

    def _fwd_hook(mod, args, output):
        # HF attention module forward typically returns
        # (attn_output, attn_weights, past_key_value) when output_attentions=True.
        _ATTN_HOOK_STATE["captured"] = output[1] if isinstance(output, tuple) and len(output) > 1 else None

    module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    module.register_forward_hook(_fwd_hook)
    _ATTN_HOOK_STATE["module"] = module


def forward_pass1(model, final_embeds, final_position_ids, final_attention_mask,
                   video_positions_in_pruned):
    """Single forward (no generation loop) -- logits at the last position ARE the
    first-token distribution generate() would produce; attention FROM that same
    last position tells us where the model looked to produce it. Global
    output_attentions stays False (cheap); the hook forces just the last layer."""
    _install_attention_hook(model)
    _ATTN_HOOK_STATE["captured"] = None
    with torch.inference_mode():
        out = model(
            inputs_embeds=final_embeds,
            position_ids=final_position_ids,
            attention_mask=final_attention_mask,
            output_attentions=False,
            return_dict=True,
        )
    logits_last = out.logits[0, -1, :].float()
    attn_weights = _ATTN_HOOK_STATE["captured"]
    if attn_weights is None:
        raise RuntimeError(
            "Attention hook captured nothing -- the hooked module's forward "
            "signature/return shape likely differs from what _fwd_hook expects; "
            "needs live debugging (print(output) shape inside _fwd_hook)."
        )
    attn_last_layer = attn_weights[0].float()  # [n_heads, seq_len, seq_len], batch=1
    attn_to_visual = attn_last_layer[:, -1, video_positions_in_pruned].mean(dim=0)
    del out, attn_weights
    return logits_last, attn_to_visual


def generate_final(model, processor, final_embeds, final_position_ids, final_attention_mask):
    with torch.inference_mode():
        generated_ids = model.generate(
            inputs_embeds=final_embeds,
            position_ids=final_position_ids,
            attention_mask=final_attention_mask,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1,
            max_new_tokens=16,
        )
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def run_one_question_2pass(model, processor, video_path, question, candidates, args, max_pixels):
    messages = build_messages(video_path, question, candidates, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model

    with torch.inference_mode():
        full_embeds = model.get_input_embeddings()(inputs["input_ids"])
        video_feat = llm.get_video_features(
            inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]
        )
        video_embeds_full = video_feat.pooler_output
        if isinstance(video_embeds_full, (list, tuple)):
            video_embeds_full = torch.cat(video_embeds_full, dim=0)
        video_embeds_full = video_embeds_full.to(full_embeds.dtype).to(full_embeds.device)

        video_token_id = model.config.video_token_id
        video_mask = (inputs["input_ids"][0] == video_token_id)
        video_positions_in_seq = video_mask.nonzero(as_tuple=True)[0]

        position_ids, _ = llm.get_rope_index(
            input_ids=inputs["input_ids"],
            image_grid_thw=None,
            video_grid_thw=inputs["video_grid_thw"],
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

    T_llm = int(video_grid_thw[0])
    H_llm = int(video_grid_thw[1]) // 2
    W_llm = int(video_grid_thw[2]) // 2
    N = H_llm * W_llm
    n_total = video_embeds_full.shape[0]

    # ---------- pass 1 ----------
    pass1_budget = int(getattr(args, "pass1_tokens", 3000))
    kept1, embeds1, _ = uniform_prune_video_tokens(video_embeds_full, pass1_budget)
    fe1, pid1, am1, vpos1 = build_pruned_sequence(
        full_embeds, inputs, video_token_id, video_positions_in_seq, position_ids, kept1, embeds1
    )
    logits1, attn1 = forward_pass1(model, fe1, pid1, am1, vpos1)
    probs = torch.softmax(logits1, dim=-1)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())

    confidence_entropy_max = float(getattr(args, "confidence_entropy_max", 2.0))
    always_pass2 = bool(getattr(args, "always_pass2", False))
    if entropy <= confidence_entropy_max and not always_pass2:
        pred_id = int(logits1.argmax(-1).item())
        output = processor.tokenizer.decode([pred_id], skip_special_tokens=True).strip()
        diag = dict(mode="2pass", exited_at=1, entropy=entropy, n_hot_units=0, actual_total=len(kept1))
        return output, diag

    # ---------- pass 2: reallocate budget toward attention hotspots ----------
    unit_of_kept1 = torch.tensor([k // N for k in kept1], device=attn1.device, dtype=torch.long)
    unit_scores = torch.zeros(T_llm, device=attn1.device)
    unit_counts = torch.zeros(T_llm, device=attn1.device)
    unit_scores.scatter_add_(0, unit_of_kept1, attn1)
    unit_counts.scatter_add_(0, unit_of_kept1, torch.ones_like(attn1))
    unit_scores = unit_scores / unit_counts.clamp_min(1)  # units with 0 pass-1 tokens stay at 0

    tokens_num = int(args.tokens_num)
    hot_budget_frac = float(getattr(args, "hot_budget_frac", 0.4))
    hot_budget = int(tokens_num * hot_budget_frac)
    max_hot_units = max(1, hot_budget // N)
    order = torch.argsort(unit_scores, descending=True)
    hot_units = set(order[:max_hot_units].tolist())

    hot_indices = [i for i in range(n_total) if (i // N) in hot_units]
    cold_pool_indices = [i for i in range(n_total) if (i // N) not in hot_units]
    remaining_budget = max(0, tokens_num - len(hot_indices))
    if cold_pool_indices and remaining_budget > 0:
        cold_embeds_pool = video_embeds_full[cold_pool_indices]
        idx_local, _, _ = uniform_prune_video_tokens(cold_embeds_pool, remaining_budget)
        cold_kept = [cold_pool_indices[j] for j in idx_local]
    else:
        cold_kept = []
    kept2 = sorted(hot_indices + cold_kept)
    embeds2 = video_embeds_full[kept2]

    fe2, pid2, am2, _ = build_pruned_sequence(
        full_embeds, inputs, video_token_id, video_positions_in_seq, position_ids, kept2, embeds2
    )
    output = generate_final(model, processor, fe2, pid2, am2)
    diag = dict(mode="2pass", exited_at=2, entropy=entropy, n_hot_units=len(hot_units),
                actual_total=len(kept2))
    return output, diag


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384 * 3))
    max_pixels = max_pixels_units * 28 * 28

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    output_dir = resolve_path(args.output_dir)

    print(f"Loading {args.model_path} ... (attn_implementation=eager, required for output_attentions)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto", attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, max_pixels=max_pixels, min_pixels=16 * 28 * 28
    )

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.output_name}.json")
    generated_ids_seen = load_answered_ids(output_path)
    output_mode = "a" if os.path.exists(output_path) else "w"

    n_exit1 = 0
    n_exit2 = 0
    with open(output_path, output_mode, encoding="utf-8") as ans_file:
        for sample in tqdm(gt_qa_pairs):
            video_name = sample["video_name"]
            answer_id = str(sample.get("id", sample.get("question_id")))
            if answer_id in generated_ids_seen:
                continue

            question = sample["question"]
            candidates = sample["candidates"]
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Missing video: {video_path}")
                continue

            try:
                output, diag = run_one_question_2pass(
                    model, processor, video_path, question, candidates, args, max_pixels
                )
                if diag["exited_at"] == 1:
                    n_exit1 += 1
                else:
                    n_exit2 += 1
            except Exception as e:
                print(f"SKIP {answer_id} ({video_path}): {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue
            print(f"pred={output!r} diag={diag}")

            sample_set = {
                "task_name": sample["task_name"],
                "question": question,
                "question_id": str(sample.get("question_id", answer_id)),
                "id": answer_id,
                "answer_number": sample["answer_number"],
                "candidates": candidates,
                "answer": sample["answer"],
                "pred": output,
                "diag": diag,
            }
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()
            generated_ids_seen.add(answer_id)

    print(f"[2-pass attention-guided] done -> {output_path} "
          f"(exit@pass1={n_exit1}, ran pass2={n_exit2})")


@hydra.main(config_path="configs/inference_qwen25vl", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
