"""
train_dsta_scorer_2a.py

Huong 2A cua tai lieu 2 giai doan (Query-Agnostic Downstream Alignment),
BO HAN V-JEPA khoi vong lap -- da kiem chung bang test_vjepa_reward_correlation.py
(Spearman ~0.06-0.07 tren 2 nguon caption doc lap, duoi han nguong 0.3 --
xem outputs/vjepa_reward_correlation*.csv) rang reward vat ly tho cua
V-JEPA (L_predict) KHONG tuong quan voi muc do huu ich ngu nghia cho
Qwen2.5-VL. Thay vao do, reward = R_Stage2A that (log-likelihood caption
that qua chinh Qwen2.5-VL) -- da chinh la thu da do dat trong pilot, nen
biet chac no phan anh dung thu can toi uu.

KIEN TRUC: DSTAScorer (xem class ben duoi) la BAN RUT GON cua DSTA-Scorer
trong tai lieu, khop voi khong gian hanh dong UNIT (khong phai tung PATCH
rieng le) da dung san trong train_rl_gating.py -- giu nguyen "Temporal
Attention Block" (phan quan trong nhat, sua dung loi "mu thoi gian" cua
MLP don le ma tai lieu chi ra o muc 2.1), bo "Spatial Attention Block" vi
X_units da duoc mean-pool truoc do (khong con gi de attend trong 1 unit).

VONG LAP GRPO: TAI SU DUNG nguyen co che da kiem chung on dinh cua
train_rl_gating.py (Bernoulli mask + san toi thieu min_retention_frac +
entropy bonus SUM khong phai mean + build_pruned_sequence giu dung
position_ids goc) -- chi doi 2 cho: (1) DSTAScorer thay GatingPolicy,
(2) reward = -CE(caption that) thay vi -CE(4 dap an trac nghiem).

DU LIEU: Charades qua LLaVA-Video-178K/0_30_s_academic_v0_1 (caption GPT-4o
chi tiet, DA TAI 1/8 shard = 484 video -- xem test_vjepa_reward_correlation.py
de biet nguon/cach tai them shard neu can nhieu video hon).

Requires: giong het train_rl_gating.py (khong them dependency moi).

Usage (SMOKE TEST truoc, vai chuc step, vai phut):
  python train_dsta_scorer_2a.py \
    video_dir=Datasets/LLaVA-Video-178K-probe/academic_source/Charades \
    captions_file=Datasets/LLaVA-Video-178K-probe/0_30_s_academic_v0_1_cap_processed.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    max_pixels_units=16384 group_size=4 lambda_sparsity=0.01 \
    min_retention_frac=0.15 lr=1e-3 num_steps=20 log_every=1 \
    scorer_save_path=outputs/dsta_scorer_2a_smoke.pt

Sau khi smoke test qua (khong loi, reward/retention hop ly), chay that voi
num_steps lon hon.
"""

import json
import os
import random

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from qwen_vision_process_fastvid import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_inference_qwen25vl_2pass import build_pruned_sequence
from run_inference_qwen25vl_dyseg import MIN_PIXELS, SYSTEM_PROMPT


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_academic_caption_rows(captions_file):
    """LLaVA-Video-178K, split 0_30_s_academic_v0_1 (nguon Charades, caption
    GPT-4o sinh chi tiet). Moi dong = 1 video, khong can dedup (khac
    MSR-VTT). Field 'video': 'academic_source/Charades/XXXXX.mp4' -> chi
    lay basename. 'conversations': [0]=cau hoi goc (bo tien to '<image>\\n'),
    [1]=caption GPT-4o that."""
    d = json.load(open(captions_file, "r", encoding="utf-8"))
    rows = []
    for item in d:
        file_name = os.path.basename(item["video"])
        prompt = item["conversations"][0]["value"].replace("<image>\n", "").strip()
        caption = item["conversations"][1]["value"].strip()
        rows.append({"file_name": file_name, "caption": caption, "prompt": prompt})
    return rows


def load_caption_rows_multi(captions_files):
    """Ban mo rong (2026-10-07): nhieu file *_cap_processed.json cua
    LLaVA-Video-178K (phan cach bang dau phay), giu NGUYEN duong dan tuong doi
    'academic_source/<nguon>/<file>' trong field 'video' -- dung voi
    video_root=<thu muc giai nen tar>, de train tren moi nguon (Charades,
    activitynet, NextQA, youcook2, ego4d) va moi split do dai, khong chi
    Charades 0-30s."""
    rows = []
    for cf in captions_files:
        d = json.load(open(cf, "r", encoding="utf-8"))
        for item in d:
            conv = item["conversations"]
            rows.append({
                "file_name": item["video"],
                "caption": conv[1]["value"].strip(),
                "prompt": conv[0]["value"].replace("<image>\n", "").strip(),
                "split": item.get("data_source", os.path.basename(cf)),
            })
    return rows


def build_messages_caption(video_path, prompt, max_pixels):
    content = [
        {"type": "video", "video": video_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
        {"type": "text", "text": prompt},
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


class DSTAScorer(nn.Module):
    """Ban rut gon cua DSTA-Scorer (tai lieu VJEPA_REWARD_proposal muc 2.2),
    hoat dong tren khong gian UNIT (T vector, moi vector = N token khong
    gian/unit da mean-pool -- giong het GatingPolicy cu) thay vi tung PATCH
    (T*N) rieng le nhu ban goc trong tai lieu. Ly do rut gon, khong phai
    thieu sot:
      1. X_units da mean-pool N token/unit tu truoc (dung chung code voi
         GatingPolicy/DySeg) -- "Spatial Attention Block" (chu y giua N
         token NOI 1 khung hinh) khong con gi de attend.
      2. Giu action space o muc unit de TAI DUNG NGUYEN VEN co che GRPO +
         min-retention-floor + build_pruned_sequence da on dinh, thay vi
         viet lai 1 GRPO estimator moi cho top-k-khong-hoan-lai tren T*N
         chieu (log-prob cua kieu sampling do khong don gian, de sai).

    PHAN QUAN TRONG NHAT van giu nguyen: Temporal Attention Block -- moi
    unit gio tu-chu-y toan bo cac unit KHAC truoc khi cho diem (khac han
    GatingPolicy cu chi nhin dung vector cua chinh no + delta 1-buoc-lui,
    dung "mu thoi gian" ma tai lieu chi trich o muc 2.1)."""

    def __init__(self, dim, hidden=256, n_heads=4):
        super().__init__()
        self.w_in = nn.Linear(dim * 2, hidden)
        self.ln_in = nn.LayerNorm(hidden)
        self.temporal_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln_attn = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        # CHAN DOAN: reward_std do duoc gan 0 (0.006-0.026 tren reward
        # ~1.0-1.4) o moi muc khoi tao mac dinh (retention ngau nhien 45-80%,
        # vi sigmoid(logit~0) = 0.5) -- clip Charades ngan + caption cong
        # thuc nen giu bat ky ~50-80% deu du doan dung caption, CE gan nhu
        # khong doi giua cac mask. Khoi tao bias lop cuoi am de sigmoid ~
        # init_keep_prob ngay tu dau (khong doi qua W=0 cua nn.Linear, chi
        # dich bias) -- o muc giu thap hon, cac mask ngau nhien khac nhau
        # NHIEU hon ve noi dung thuc su giu lai, ky vong reward_std tang.
        init_keep_prob = 0.15
        bias_init = torch.log(torch.tensor(init_keep_prob / (1 - init_keep_prob)))
        with torch.no_grad():
            self.head[-1].bias.fill_(bias_init.item())

    def forward(self, x_units, delta_units):
        """x_units, delta_units: [T, D]. Tra ve logits [T]."""
        feat = torch.cat([x_units, delta_units], dim=-1)  # [T, 2D]
        z = self.ln_in(self.w_in(feat)).unsqueeze(0)  # [1, T, hidden]
        attn_out, _ = self.temporal_attn(z, z, z)  # self-attn xuyen suot T unit
        z = self.ln_attn(z + attn_out)
        logits = self.head(z).squeeze(0).squeeze(-1)  # [T]
        return logits


def units_to_token_indices(unit_mask, N):
    """unit_mask: bool tensor [T] (True = keep whole unit). Tra ve flat
    token index (da sap xep) trong tong T*N token, unit-major order."""
    kept_units = unit_mask.nonzero(as_tuple=True)[0]
    if kept_units.numel() == 0:
        return kept_units.new_zeros(0, dtype=torch.long)
    offsets = torch.arange(N, device=unit_mask.device)
    idx = (kept_units.unsqueeze(1) * N + offsets.unsqueeze(0)).reshape(-1)
    return torch.sort(idx).values


def run_one_video(model, processor, scorer, optimizer, video_path, caption, prompt, args, max_pixels, device, debug=False):
    messages = build_messages_caption(video_path, prompt, max_pixels)
    # max_frames (mac dinh 128) + time_mode=lmms (mac dinh): khop harness
    # baseline co dinh cua run_inference_qwen25vl_kitoke.py. time_mode=
    # hf_default = hanh vi cua moi lan train truoc 2026-10-07 (transformers
    # 5.x lam moi unit trung 1 vi tri thoi gian).
    max_frames = int(getattr(args, "max_frames", 128))
    if max_frames > 0:
        messages[1]["content"][0]["max_frames"] = max_frames
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)
    time_mode = getattr(args, "time_mode", "lmms")
    if time_mode == "lmms":
        inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], 1.0)
    elif time_mode != "hf_default":
        raise ValueError(f"unknown time_mode={time_mode!r}, expected 'lmms' or 'hf_default'")

    video_grid_thw = inputs["video_grid_thw"][0]
    llm = model.model if hasattr(model, "model") else model

    with torch.no_grad():
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

        # Caption that -- tokenize + embed 1 lan, dung chung cho ca G mask
        # (chi phan video thay doi giua cac mask, caption luon giu nguyen).
        caption_ids = processor.tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        max_caption_tokens = int(getattr(args, "max_caption_tokens", 0))
        if max_caption_tokens > 0:
            caption_ids = caption_ids[:, :max_caption_tokens]
        caption_embeds = model.get_input_embeddings()(caption_ids)
        L_cap = caption_ids.shape[1]

    T_llm = int(video_grid_thw[0])
    H_llm = int(video_grid_thw[1]) // 2
    W_llm = int(video_grid_thw[2]) // 2
    N = H_llm * W_llm
    n_total = video_embeds_full.shape[0]
    assert n_total == T_llm * N, f"unit-major layout assumption broken: {n_total} != {T_llm}*{N}"

    # ---------- DSTAScorer forward (grad ON, Qwen da @torch.no_grad o tren) ----------
    X_units = video_embeds_full.detach().view(T_llm, N, -1).mean(dim=1).float()
    delta_units = torch.zeros_like(X_units)
    delta_units[1:] = X_units[1:] - X_units[:-1]
    logits = scorer(X_units, delta_units)  # [T]
    probs = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)

    if debug:
        print(f"  [debug] logits: mean={logits.mean().item():.3f} std={logits.std().item():.3f} "
              f"min={logits.min().item():.3f} max={logits.max().item():.3f}")
        print(f"  [debug] probs : mean={probs.mean().item():.3f} min={probs.min().item():.3f} max={probs.max().item():.3f}")
        print(f"  [debug] head[-1].bias = {scorer.head[-1].bias.item():.4f}")

    K = int(getattr(args, "group_size", 4))
    lambda_sparsity = float(getattr(args, "lambda_sparsity", 0.01))
    # San toi thieu -- giong het train_rl_gating.py, tranh sup do ve mask
    # rong/deterministic khi probs bao hoa ve 0 (da la bai hoc that tu
    # Experiment 1 cua RL-Gating, xem comment goc trong file do).
    min_retention_frac = float(getattr(args, "min_retention_frac", 0.15))
    entropy_coef = float(getattr(args, "entropy_coef", 0.01))

    rewards, log_probs, retentions = [], [], []
    for _k in range(K):
        with torch.no_grad():
            mask = torch.bernoulli(probs).bool()
            min_units = max(1, int(round(min_retention_frac * T_llm)))
            if int(mask.sum().item()) < min_units:
                floor_idx = torch.linspace(0, T_llm - 1, steps=min_units, device=mask.device).round().long()
                mask[floor_idx] = True
            kept_indices = units_to_token_indices(mask, N)
            kept_embeds = video_embeds_full[kept_indices]

            fe, pid, am, _ = build_pruned_sequence(
                full_embeds, inputs, video_token_id, video_positions_in_seq,
                position_ids, kept_indices, kept_embeds,
            )

            # Noi caption that vao cuoi chuoi da rut gon, gan labels=-100
            # cho toan bo phan prompt+video (chi tinh loss tren token
            # caption) -- dung R_Stage2A(S): log-likelihood caption dieu
            # kien tren dung tap token video da giu.
            fe_full = torch.cat([fe, caption_embeds], dim=1)
            last_pos = pid[:, :, -1:]  # [3,1,1]
            cap_offsets = torch.arange(1, L_cap + 1, device=device).view(1, 1, L_cap)
            cap_pos = last_pos + cap_offsets  # [3,1,L_cap], mRoPE: text tiep tuc tuan tu ca 3 kenh
            pid_full = torch.cat([pid, cap_pos], dim=-1)
            am_full = torch.cat([am, am.new_ones((1, L_cap))], dim=1)
            labels_full = torch.cat(
                [torch.full((1, fe.shape[1]), -100, dtype=torch.long, device=device), caption_ids], dim=1,
            )

            out = model(
                inputs_embeds=fe_full, position_ids=pid_full, attention_mask=am_full,
                labels=labels_full, output_attentions=False, return_dict=True,
            )
            ce_loss = out.loss
            retention = kept_indices.numel() / n_total
            reward = -float(ce_loss.item()) - lambda_sparsity * retention
            del out

        mask_f = mask.float()
        lp = (mask_f * torch.log(probs) + (1 - mask_f) * torch.log(1 - probs)).sum()
        rewards.append(reward)
        log_probs.append(lp)
        retentions.append(retention)

    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
    baseline = rewards_t.mean()
    advantage = (rewards_t - baseline).detach()
    log_probs_t = torch.stack(log_probs)
    # Entropy SUM (khong phai mean) -- bai hoc da vap o train_rl_gating.py:
    # log_probs_t la tong tren T unit nen quy mo ty le voi T, entropy bonus
    # phai cong tuong tu de khong bi nuot mat boi policy_loss.
    entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
    policy_loss = -(advantage * log_probs_t).mean()
    loss = policy_loss - entropy_coef * entropy.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "entropy_mean": float(entropy.mean().item()),
        "reward_mean": float(rewards_t.mean().item()),
        # CHAN DOAN: do lech chuan reward TRONG 1 nhom G -- neu gan 0 he
        # thong, nghia la G mask ngau nhien cho CE gan nhu giong het nhau
        # (caption khong nhay voi viec giu dung unit nao) -> advantage luon
        # ~0 -> GRPO khong co gi de hoc, KHONG phai loi kien truc scorer.
        "reward_std": float(rewards_t.std().item()) if len(rewards) > 1 else 0.0,
        "retention_mean": float(sum(retentions) / len(retentions)),
        "T_units": T_llm,
    }


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    max_pixels_units = int(getattr(args, "max_pixels_units", 16384))
    max_pixels = max_pixels_units * 28 * 28

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    video_root = getattr(args, "video_root", None)
    if video_root:
        # Che do moi: video_root + captions_files (phan cach dau phay), moi nguon/moi split.
        video_dir = resolve_path(video_root)
        cap_files = [resolve_path(c.strip()) for c in str(args.captions_files).split(",") if c.strip()]
        print(f"Loading captions tu {cap_files} ...")
        rows = load_caption_rows_multi(cap_files)
    else:
        video_dir = resolve_path(args.video_dir)
        captions_file = resolve_path(args.captions_file)
        print(f"Loading captions tu {captions_file} ...")
        rows = load_academic_caption_rows(captions_file)
    random.seed(int(getattr(args, "seed", 0)))
    random.shuffle(rows)
    # Chi giu nhung video that su co tren dia (chi tai mot phan shard).
    rows = [r for r in rows if os.path.exists(os.path.join(video_dir, r["file_name"]))]
    print(f"{len(rows)} video co san (khop caption + file that su ton tai)")
    if rows and "split" in rows[0]:
        from collections import Counter
        print("  theo split:", dict(Counter(r["split"] for r in rows)))

    print(f"Loading {model_path} ... (frozen, no grad)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)

    resume_from = getattr(args, "resume_from", None)
    resume_ckpt = None
    if resume_from:
        resume_ckpt = torch.load(resolve_path(resume_from), map_location=device)
        video_dim = resume_ckpt["video_dim"]
        print(f"Resuming from {resume_from}: video_dim={video_dim}, previous step={resume_ckpt.get('step', 0)}")
    else:
        probe = rows[0]
        probe_path = os.path.join(video_dir, probe["file_name"])
        with torch.no_grad():
            probe_messages = build_messages_caption(probe_path, probe["prompt"], max_pixels)
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

    scorer = DSTAScorer(video_dim).to(device).float()
    lr = float(getattr(args, "lr", 1e-3))
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)
    start_step = 0
    if resume_ckpt is not None:
        scorer.load_state_dict(resume_ckpt["state_dict"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        else:
            print("WARNING: checkpoint khong co optimizer_state -- resume voi Adam state moi.")
        start_step = int(resume_ckpt.get("step", 0))

    num_steps = int(getattr(args, "num_steps", 20))
    log_every = int(getattr(args, "log_every", 1))
    ckpt_every = int(getattr(args, "ckpt_every", 200))
    scorer_save_path = resolve_path(getattr(args, "scorer_save_path", "outputs/dsta_scorer_2a.pt"))
    os.makedirs(os.path.dirname(scorer_save_path), exist_ok=True)
    metrics_log_path = scorer_save_path + ".metrics.jsonl"
    metrics_f = open(metrics_log_path, "a", encoding="utf-8")

    step = start_step
    running_reward, running_retention = [], []
    pbar = tqdm(total=num_steps, initial=start_step)
    idx = start_step
    while step < num_steps:
        row = rows[idx % len(rows)]
        idx += 1
        video_path = os.path.join(video_dir, row["file_name"])
        try:
            stats = run_one_video(
                model, processor, scorer, optimizer, video_path, row["caption"], row["prompt"],
                args, max_pixels, device, debug=(step == start_step),
            )
        except Exception as e:
            print(f"SKIP step {step} ({row['file_name']}): {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue

        running_reward.append(stats["reward_mean"])
        running_retention.append(stats["retention_mean"])
        step += 1
        pbar.update(1)
        metrics_f.write(json.dumps({
            "step": step, "loss": stats["loss"], "policy_loss": stats["policy_loss"],
            "entropy": stats["entropy_mean"], "reward": stats["reward_mean"],
            "reward_std": stats["reward_std"],
            "retention": stats["retention_mean"], "T_units": stats["T_units"],
            "video_name": row["file_name"],
        }) + "\n")
        metrics_f.flush()
        if step % log_every == 0:
            print(
                f"step={step} loss={stats['loss']:.4f} policy_loss={stats['policy_loss']:.4f} "
                f"entropy={stats['entropy_mean']:.4f} reward={stats['reward_mean']:.4f} "
                f"reward_std={stats['reward_std']:.4f} "
                f"retention={stats['retention_mean']:.3f} T_units={stats['T_units']} "
                f"(running reward={sum(running_reward)/len(running_reward):.4f}, "
                f"running retention={sum(running_retention)/len(running_retention):.3f})"
            )
        if ckpt_every > 0 and step % ckpt_every == 0:
            ckpt_blob = {"state_dict": scorer.state_dict(), "video_dim": video_dim, "step": step,
                         "optimizer_state": optimizer.state_dict()}
            torch.save(ckpt_blob, scorer_save_path + f".step{step}")
            torch.save(ckpt_blob, scorer_save_path)
            print(f"[checkpoint] step={step} -> {scorer_save_path}.step{step} (va {scorer_save_path})")
    pbar.close()
    metrics_f.close()

    torch.save({"state_dict": scorer.state_dict(), "video_dim": video_dim, "step": step,
                "optimizer_state": optimizer.state_dict()}, scorer_save_path)
    print(f"[train_dsta_scorer_2a] done -> saved scorer to {scorer_save_path}")
    print(f"Per-step metrics logged to {metrics_log_path}")
    print(f"Final running reward={sum(running_reward)/len(running_reward):.4f}, "
          f"running retention={sum(running_retention)/len(running_retention):.3f}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
