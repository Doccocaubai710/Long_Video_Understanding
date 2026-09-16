"""
train_rl_gating_2stage.py

2-tang RL-Gating (thiet ke sau khi thao luan sua bo ban "phang" 1-tang dau
tien): thay vi hoc xac suat doc lap cho toan bo T*N token cung luc (nhieu,
noisy, lang phi tinh cho nhung unit chac chan bi bo), tach thanh 2 quyet
dinh noi tiep:

  TANG 1 (tho, cap UNIT) -- CHINH LA policy cua train_rl_gating.py, dung lai
    nguyen kien truc GatingPolicy. Quyet dinh unit nao "dang xem xet ky
    hon", unit nao loai han. KHONG bat buoc giu bao nhieu unit -- tu hoc.

  TANG 2 (min, cap TOKEN) -- CHI chay tren nhung unit Tang 1 da giu. Voi
    MOI token trong unit do, 1 policy khac (GatingPolicyToken) quyet dinh
    giu/bo RIENG -- so token song sot moi unit la KET QUA cong don tu N
    quyet dinh doc lap, KHONG phai ngan sach dat truoc (vd khong co chuyen
    "moi unit giu dung 96 token" -- unit nhieu chuyen dong co the giu 150,
    unit tinh co the giu 15).

Cong thuc loss KHONG doi ve dang toan hoc so voi ban 1-tang (van la reward =
-CE_loss - lambda*retention, GRPO advantage, REINFORCE + entropy bonus) --
chi THEM 1 so hang: log_prob toan bo = log_prob(Tang 1) + log_prob(Tang 2 |
Tang 1 da giu), va entropy toan bo = entropy(Tang 1) + entropy(Tang 2 |
Tang 1 da giu). Tang 2 CHI duoc tinh/lay mau cho nhung unit Tang 1 giu --
khong ton compute cho unit da bi loai.

Sang tao gan voi DySeg+STPrune ve cau truc (Tang 1 ~ DySeg phan doan+ngan
sach, Tang 2 ~ STPrune chon trong-doan) -- khac o cho ca 2 tang deu HOC tu
du lieu thay vi cong thuc co dinh.

Usage (smoke test truoc, so cau it, so step it):
  python train_rl_gating_2stage.py \
    video_dir=Datasets/NExTQA/NExTVideo \
    gt_file=playground/gt_qa_files/NExTQA/train_qa.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    max_pixels_units=16384 group_size=4 \
    lambda_sparsity=0.3 min_retention_frac1=0.15 min_retention_frac2=0.1 \
    entropy_coef=0.01 lr=1e-3 num_steps=200 log_every=10 ckpt_every=50 \
    policy_save_path=outputs/rl_gating_2stage_exp1.pt
"""

import json
import os

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_inference_qwen25vl_2pass import build_pruned_sequence
from train_rl_gating import (
    LETTERS,
    GatingPolicy,
    answer_letter,
    build_messages_lettered,
    build_video_index,
    resolve_path,
    resolve_video_path,
)


class GatingPolicyToken(nn.Module):
    """Tang 2 -- nhin [X_i ; delta_i ; X_units[t]] cua TUNG TOKEN (khong
    phai trung binh ca unit nhu Tang 1) -> logit giu/bo rieng cho token do.
    Query-agnostic tuyet doi, giong Tang 1: khong bao gio nhan cau hoi.

    LayerNorm o dau: X_i/delta_i la gia tri THO cua Qwen (khong qua phep
    trung binh nhu X_units o Tang 1, nen bien do co the lon hon nhieu).
    Smoke test dau tien khong co LayerNorm cho thay entropy2 da bao hoa gan
    0 tu NHUNG STEP DAU (khong phai giam dan theo thoi gian train) -- dau
    hieu logit dau ra qua lon ngay tu luc khoi tao ngau nhien do input chua
    chuan hoa, chu khong phai RL "hoc ra" ket luan do."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim * 3)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_tok, delta_tok, unit_ctx_tok):
        feat = torch.cat([x_tok, delta_tok, unit_ctx_tok], dim=-1)
        feat = self.input_norm(feat)
        return self.net(feat).squeeze(-1)


def run_one_question(model, processor, policy1, policy2, optimizer, video_path, sample, args,
                      max_pixels, device):
    question = sample["question"]
    candidates = sample["candidates"]
    messages = build_messages_lettered(video_path, question, candidates, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model

    with torch.no_grad():
        full_embeds = model.get_input_embeddings()(inputs["input_ids"])
        video_feat = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"])
        video_embeds_full = video_feat.pooler_output
        if isinstance(video_embeds_full, (list, tuple)):
            video_embeds_full = torch.cat(video_embeds_full, dim=0)
        video_embeds_full = video_embeds_full.to(full_embeds.dtype).to(full_embeds.device)

        video_token_id = model.config.video_token_id
        video_mask = (inputs["input_ids"][0] == video_token_id)
        video_positions_in_seq = video_mask.nonzero(as_tuple=True)[0]

        position_ids, _ = llm.get_rope_index(
            input_ids=inputs["input_ids"], image_grid_thw=None,
            video_grid_thw=inputs["video_grid_thw"], second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=inputs["attention_mask"], mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

    T_llm = int(video_grid_thw[0])
    H_llm = int(video_grid_thw[1]) // 2
    W_llm = int(video_grid_thw[2]) // 2
    N = H_llm * W_llm
    n_total = video_embeds_full.shape[0]
    assert n_total == T_llm * N

    # ---------- Tang 1: unit-level (giong het train_rl_gating.py) ----------
    X_full = video_embeds_full.detach().view(T_llm, N, -1).float()  # [T, N, D]
    X_units = X_full.mean(dim=1)  # [T, D]
    delta_units = torch.zeros_like(X_units)
    delta_units[1:] = X_units[1:] - X_units[:-1]
    probs1 = torch.sigmoid(policy1(X_units, delta_units)).clamp(1e-4, 1 - 1e-4)  # [T]

    # per-token delta (khac delta_units -- day la delta CUA DUNG VI TRI
    # KHONG GIAN do so voi unit truoc, chi tiet hon delta trung binh ca unit)
    delta_full = torch.zeros_like(X_full)
    delta_full[1:] = X_full[1:] - X_full[:-1]  # [T, N, D]

    letters = LETTERS[: len(candidates)]
    letter_ids = torch.tensor(
        [processor.tokenizer(l, add_special_tokens=False).input_ids[0] for l in letters], device=device,
    )
    target_idx = letters.index(answer_letter(sample))

    K = int(getattr(args, "group_size", 4))
    lambda_sparsity = float(getattr(args, "lambda_sparsity", 0.3))
    min_retention_frac1 = float(getattr(args, "min_retention_frac1", 0.15))
    min_retention_frac2 = float(getattr(args, "min_retention_frac2", 0.1))
    entropy_coef = float(getattr(args, "entropy_coef", 0.01))
    # entropy2 cong qua n_kept_units*N token (hang nghin), trong khi entropy1
    # cong qua T unit (hang chuc) -- dung CHUNG entropy_coef cho ca 2 se lam
    # entropy2 ap dao hoan toan (phat hien tu smoke test: loss nhay len toi
    # -101, entropy_coef*entropy2 ~73 trong khi policy_loss chi ~1). Chia
    # entropy_coef cho N de entropy2's dong gop/unit khop dung thang do voi
    # entropy1's dong gop/unit -- tu suy ra tu N, khong can hyperparameter moi.
    entropy_coef2 = entropy_coef / N

    # Entropy cua Tang 1 khong doi qua K sample (probs1 co dinh trong buoc
    # nay) -- tinh 1 lan, cong .sum() dung thang do voi log_prob (da rut
    # kinh nghiem tu loi entropy .mean() o ban 1-tang).
    entropy1 = -(probs1 * torch.log(probs1) + (1 - probs1) * torch.log(1 - probs1))
    entropy1_sum = entropy1.sum()

    rewards, log_probs, entropy2_sums, retentions, entropy2_counts = [], [], [], [], []
    for _k in range(K):
        with torch.no_grad():
            mask1 = torch.bernoulli(probs1).bool()
            min_units = max(1, int(round(min_retention_frac1 * T_llm)))
            if int(mask1.sum().item()) < min_units:
                floor_idx = torch.linspace(0, T_llm - 1, steps=min_units, device=mask1.device).round().long()
                mask1[floor_idx] = True
            kept_units = mask1.nonzero(as_tuple=True)[0]  # [n_kept_units]

        # ---------- Tang 2: token-level, CHI tren cac unit Tang 1 da giu ----------
        n_kept_units = kept_units.numel()
        X_tok = X_full[kept_units].reshape(n_kept_units * N, -1)          # [n_kept_units*N, D]
        delta_tok = delta_full[kept_units].reshape(n_kept_units * N, -1)  # [n_kept_units*N, D]
        unit_ctx_tok = X_units[kept_units].unsqueeze(1).expand(-1, N, -1).reshape(n_kept_units * N, -1)
        logits2 = policy2(X_tok, delta_tok, unit_ctx_tok)  # [n_kept_units*N]
        probs2 = torch.sigmoid(logits2).clamp(1e-4, 1 - 1e-4)

        with torch.no_grad():
            mask2 = torch.bernoulli(probs2).bool()
            # san toi thieu MOI unit (khong de 1 unit da duoc Tang 1 giu
            # nhung Tang 2 lai bo trang hoan toan -- vo nghia viec Tang 1
            # da chon no). Ap rieng cho tung unit trong kept_units.
            min_tok_per_unit = max(1, int(round(min_retention_frac2 * N)))
            mask2_2d = mask2.view(n_kept_units, N)
            counts = mask2_2d.sum(dim=1)
            need_floor = counts < min_tok_per_unit
            if need_floor.any():
                floor_pos = torch.linspace(0, N - 1, steps=min_tok_per_unit, device=mask2.device).round().long()
                mask2_2d[need_floor.nonzero(as_tuple=True)[0].unsqueeze(1), floor_pos.unsqueeze(0)] = True
            mask2 = mask2_2d.reshape(-1)

            # anh xa lai ve chi so token trong toan bo n_total (unit-major,
            # giong units_to_token_indices o ban 1-tang)
            offsets = torch.arange(N, device=mask2.device)
            flat_idx_per_kept_unit = (kept_units.unsqueeze(1) * N + offsets.unsqueeze(0)).reshape(-1)
            kept_indices = torch.sort(flat_idx_per_kept_unit[mask2]).values
            kept_embeds = video_embeds_full[kept_indices]

            fe, pid, am, _ = build_pruned_sequence(
                full_embeds, inputs, video_token_id, video_positions_in_seq, position_ids,
                kept_indices, kept_embeds,
            )
            out = model(inputs_embeds=fe, position_ids=pid, attention_mask=am,
                        output_attentions=False, return_dict=True)
            logits_last = out.logits[0, -1, :].float()
            ce_loss = torch.nn.functional.cross_entropy(
                logits_last[letter_ids].unsqueeze(0), torch.tensor([target_idx], device=device)
            )
            retention = kept_indices.numel() / n_total
            reward = -float(ce_loss.item()) - lambda_sparsity * retention
            del out, logits_last

        mask1_f = mask1.float()
        log_prob1 = (mask1_f * torch.log(probs1) + (1 - mask1_f) * torch.log(1 - probs1)).sum()
        mask2_f = mask2.float()
        log_prob2 = (mask2_f * torch.log(probs2) + (1 - mask2_f) * torch.log(1 - probs2)).sum()

        entropy2 = -(probs2 * torch.log(probs2) + (1 - probs2) * torch.log(1 - probs2))

        rewards.append(reward)
        log_probs.append(log_prob1 + log_prob2)
        entropy2_sums.append(entropy2.sum())
        entropy2_counts.append(probs2.numel())
        retentions.append(retention)

    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
    advantage = (rewards_t - rewards_t.mean()).detach()
    log_probs_t = torch.stack(log_probs)
    entropy2_mean = torch.stack(entropy2_sums).mean()

    policy_loss = -(advantage * log_probs_t).mean()
    loss = policy_loss - entropy_coef * entropy1_sum - entropy_coef2 * entropy2_mean

    optimizer.zero_grad()
    loss.backward()
    # REINFORCE o action space lon (token-level co the len toi hang nghin
    # quyet dinh dong thoi) tu nhien co variance cao -- thay trong smoke
    # test policy_loss nhay tu -9.8 den +17.9 giua cac step. Clip grad norm
    # de tranh 1 buoc cap nhat qua lon lam hong policy dot ngot.
    torch.nn.utils.clip_grad_norm_(
        list(policy1.parameters()) + list(policy2.parameters()), max_norm=1.0
    )
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "entropy1_mean": float(entropy1.mean().item()),
        "entropy2_mean": float((entropy2_mean / max(1, sum(entropy2_counts) / len(entropy2_counts))).item()),
        "reward_mean": float(rewards_t.mean().item()),
        "retention_mean": float(sum(retentions) / len(retentions)),
        "n_kept_units_last": n_kept_units,
        "T_units": T_llm,
    }


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    max_pixels = max_pixels_units * 28 * 28

    video_dir = resolve_path(args.video_dir)
    gt_file = resolve_path(args.gt_file)
    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")

    print(f"Indexing videos under {video_dir} ...")
    video_index = build_video_index(video_dir)
    print(f"Indexed {len(video_index)} video files")

    print(f"Loading {model_path} ... (frozen, no grad)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=16 * 28 * 28)

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    import random
    random.seed(0)
    random.shuffle(gt_qa_pairs)

    resume_from = getattr(args, "resume_from", None)
    resume_ckpt = None
    if resume_from:
        resume_ckpt = torch.load(resolve_path(resume_from), map_location=device)
        video_dim = resume_ckpt["video_dim"]
        print(f"Resuming from {resume_from}: video_dim={video_dim}, previous step={resume_ckpt.get('step', 0)}")
    else:
        probe_sample = gt_qa_pairs[0]
        probe_path = resolve_video_path(video_dir, probe_sample["video_name"], video_index)
        assert probe_path is not None, f"Cannot resolve probe video: {probe_sample['video_name']}"
        with torch.no_grad():
            probe_messages = build_messages_lettered(probe_path, probe_sample["question"], probe_sample["candidates"], max_pixels)
            probe_text = processor.apply_chat_template(probe_messages, tokenize=False, add_generation_prompt=True)
            pi, pv, _ = process_vision_info(probe_messages, return_video_kwargs=True)
            probe_inputs = processor(text=[probe_text], images=pi, videos=pv, padding=True, return_tensors="pt").to(device)
            llm = model.model if hasattr(model, "model") else model
            probe_feat = llm.get_video_features(probe_inputs["pixel_values_videos"], video_grid_thw=probe_inputs["video_grid_thw"])
            probe_pooler = probe_feat.pooler_output
            if isinstance(probe_pooler, (list, tuple)):
                probe_pooler = torch.cat(probe_pooler, dim=0)
            video_dim = probe_pooler.shape[-1]
        print(f"Inferred video token dim: {video_dim}")

    policy1 = GatingPolicy(video_dim).to(device).float()
    policy2 = GatingPolicyToken(video_dim).to(device).float()
    lr = float(getattr(args, "lr", 1e-3))
    optimizer = torch.optim.Adam(list(policy1.parameters()) + list(policy2.parameters()), lr=lr)

    start_step = 0
    if resume_ckpt is not None:
        policy1.load_state_dict(resume_ckpt["state_dict1"])
        policy2.load_state_dict(resume_ckpt["state_dict2"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        else:
            print("WARNING: checkpoint has no optimizer_state -- resuming with fresh Adam state.")
        start_step = int(resume_ckpt.get("step", 0))

    num_steps = int(getattr(args, "num_steps", 20))
    log_every = int(getattr(args, "log_every", 1))
    ckpt_every = int(getattr(args, "ckpt_every", 200))
    policy_save_path = resolve_path(getattr(args, "policy_save_path", "outputs/rl_gating_2stage.pt"))
    os.makedirs(os.path.dirname(policy_save_path), exist_ok=True)
    metrics_log_path = policy_save_path + ".metrics.jsonl"
    metrics_f = open(metrics_log_path, "a", encoding="utf-8")

    def save_ckpt(path, step):
        torch.save({
            "state_dict1": policy1.state_dict(), "state_dict2": policy2.state_dict(),
            "video_dim": video_dim, "step": step, "optimizer_state": optimizer.state_dict(),
        }, path)

    step = start_step
    running_reward, running_retention = [], []
    pbar = tqdm(total=num_steps, initial=start_step)
    idx = start_step
    while step < num_steps:
        sample = gt_qa_pairs[idx % len(gt_qa_pairs)]
        idx += 1
        video_path = resolve_video_path(video_dir, sample["video_name"], video_index)
        if video_path is None:
            continue
        try:
            stats = run_one_question(model, processor, policy1, policy2, optimizer, video_path, sample, args,
                                      max_pixels, device)
        except Exception as e:
            print(f"SKIP step {step} ({sample['video_name']}): {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue

        running_reward.append(stats["reward_mean"])
        running_retention.append(stats["retention_mean"])
        step += 1
        pbar.update(1)
        metrics_f.write(json.dumps({"step": step, **stats}) + "\n")
        metrics_f.flush()
        if step % log_every == 0:
            print(
                f"step={step} loss={stats['loss']:.4f} policy_loss={stats['policy_loss']:.4f} "
                f"entropy1={stats['entropy1_mean']:.4f} entropy2={stats['entropy2_mean']:.4f} "
                f"reward={stats['reward_mean']:.4f} retention={stats['retention_mean']:.3f} "
                f"n_kept_units={stats['n_kept_units_last']}/{stats['T_units']} "
                f"(running reward={sum(running_reward)/len(running_reward):.4f}, "
                f"running retention={sum(running_retention)/len(running_retention):.3f})"
            )
        if ckpt_every > 0 and step % ckpt_every == 0:
            ckpt_path = policy_save_path + f".step{step}"
            save_ckpt(ckpt_path, step)
            save_ckpt(policy_save_path, step)
            print(f"[checkpoint] step={step} -> {ckpt_path} (and {policy_save_path})")
    pbar.close()
    metrics_f.close()

    save_ckpt(policy_save_path, step)
    print(f"[train_rl_gating_2stage] done -> saved policy to {policy_save_path}")
    print(f"Per-step metrics logged to {metrics_log_path}")
    print(f"Final running reward={sum(running_reward)/len(running_reward):.4f}, "
          f"running retention={sum(running_retention)/len(running_retention):.3f}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
