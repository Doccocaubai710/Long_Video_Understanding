"""
train_rl_gating.py

SCORE-style query-agnostic RL-Gating: mot policy nho (MLP) hoc CHON THEM
token video nao dang giu, thay vi doan ho bang similarity (DPC-frame) hay
doc attention noi bo cua chinh Qwen luc tra loi (2-pass -- QUERY-AWARE, khong
dung cho huong nay). Qwen2.5-VL-7B luon DONG BANG; chi update policy.

Do MINH BACH: day la ban PHAC THAO dau tien, chua chay tren du lieu that.
Muc tieu truoc mat la SMOKE TEST (--num_steps nho, vd 20-50) de xac nhan
code chay dung -- shape khop, reward tinh hop ly, khong OOM -- TRUOC KHI
dau tu vao 1 lan train dai (vai tram/nghin step) de do lambda that su.

Do minh, query-agnostic:
  - Policy KHONG BAO GIO nhin cau hoi/text -- chi nhin dac trung thi giac
    cua tung UNIT (mot moc thoi gian trong luoi token cua Qwen, gom N token
    khong gian) va delta so voi unit truoc do.
  - Quyet dinh giu/bo o muc UNIT (khong phai tung token rieng le) -- giam
    action space, tan dung lai don vi da dung o DySeg.

Vong lap (GRPO-style, moi cau hoi la 1 "episode"):
  1. Encode video 1 lan (video_embeds_full, video_grid_thw) -- dung chung
     cho toan bo K sample cua cau hoi nay.
  2. Gom token theo unit -> X_units [T,D] (mean-pool N token khong gian/unit).
  3. policy(X_units, delta_units) -> keep_prob per unit (Bernoulli).
  4. Lay mau K mat na doc lap (K units-mask/cau hoi).
  5. Voi MOI mat na: build lai inputs_embeds+position_ids (gather/scatter,
     tai su dung build_pruned_sequence tu run_inference_qwen25vl_2pass.py),
     1 lan forward DUY NHAT (KHONG generate) duoi @torch.no_grad, lay logits
     vi tri cuoi -> CE loss so voi chu cai dap an that (giong het cach tinh
     "do tu tin" o pass 1 cua 2pass, chi khac la co nhan that de tinh CE).
     reward = -CE_loss - lambda_sparsity * (ty le token giu)
  6. baseline = trung binh reward trong nhom K (GRPO) -> advantage.
  7. loss_policy = -(advantage.detach() * log_prob(mat_na | policy)).mean()
     CHI backward qua policy (Qwen luon @torch.no_grad).

Usage (SMOKE TEST truoc, K nho, so cau it, so step it -- vai phut):
  python train_rl_gating.py \
    video_dir=Datasets/NExTQA/NExTVideo \
    gt_file=playground/gt_qa_files/NExTQA/train_qa_smoke5.json \
    model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    max_pixels_units=16384 group_size=4 lambda_sparsity=0.01 \
    lr=1e-3 num_steps=20 log_every=1 \
    policy_save_path=outputs/rl_gating_policy_smoke.pt

Sau khi smoke test qua (khong loi, reward/retention hien thi hop ly), moi
chay that voi train_qa.json day du va num_steps lon hon de do lambda.
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

LETTERS = ["A", "B", "C", "D", "E", "F"]


def build_messages_lettered(video_path, question, candidates, max_pixels):
    """dyseg's build_messages assumes candidates already carry a letter
    prefix ("A. xxx") and hardcodes "A, B, C, or D" in the prompt -- true
    for VideoMME's gt files but NOT for NExT-QA's (raw text, no prefix, and
    NExT-QA MC questions have 5 options, not 4). Build the letter prefixes
    here instead of relying on the source JSON's formatting."""
    letters = LETTERS[: len(candidates)]
    option_block = "\n".join(f"{l}. {c}" for l, c in zip(letters, candidates))
    letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    option_prompt = (
        "Select the best answer to the following multiple-choice question "
        f"based on the video. Respond with only the letter ({letter_list}) "
        "of the correct option."
    )
    post_prompt = "\nAnswer with the option's letter from the given choices directly."
    text = f"{option_prompt}\n{question}\n{option_block}\n{post_prompt}"
    content = [
        {"type": "video", "video": video_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
        {"type": "text", "text": text},
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def build_video_index(video_dir):
    """Same nested-subfolder fix as run_inference_qwen25vl_native.py --
    NExT-QA stores videos under per-video subfolders, not flat."""
    index = {}
    for root, _dirs, files in os.walk(video_dir):
        for fname in files:
            index.setdefault(fname, os.path.join(root, fname))
    return index


def resolve_video_path(video_dir, video_name, video_index):
    direct = os.path.join(video_dir, video_name)
    if os.path.exists(direct):
        return direct
    return video_index.get(os.path.basename(video_name))


class GatingPolicy(nn.Module):
    """Nhin [X_i ; delta_X_i] cua tung UNIT -> logit giu/bo. Query-agnostic
    tuyet doi: khong bao gio nhan text/cau hoi lam input."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_units, delta_units):
        feat = torch.cat([x_units, delta_units], dim=-1)
        return self.net(feat).squeeze(-1)  # [T] logits


def answer_letter(sample):
    """Letter is determined by POSITION of answer_number in candidates, not
    by any prefix in the candidate text -- NExT-QA's candidates are raw
    text ("one", "three", ...) with no "A. "-style prefix at all, unlike
    VideoMME's gt files."""
    idx = int(sample["answer_number"])
    return LETTERS[idx]


def units_to_token_indices(unit_mask, N):
    """unit_mask: bool tensor [T] (True = keep whole unit). Returns sorted
    flat token indices among the n_total=T*N video tokens (unit-major order,
    matching video_embeds_full's own layout)."""
    kept_units = unit_mask.nonzero(as_tuple=True)[0]
    if kept_units.numel() == 0:
        return kept_units.new_zeros(0, dtype=torch.long)
    offsets = torch.arange(N, device=unit_mask.device)
    idx = (kept_units.unsqueeze(1) * N + offsets.unsqueeze(0)).reshape(-1)
    return torch.sort(idx).values


def run_one_question(model, processor, policy, optimizer, video_path, sample, args, max_pixels, device):
    question = sample["question"]
    candidates = sample["candidates"]
    messages = build_messages_lettered(video_path, question, candidates, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

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

    T_llm = int(video_grid_thw[0])
    H_llm = int(video_grid_thw[1]) // 2
    W_llm = int(video_grid_thw[2]) // 2
    N = H_llm * W_llm
    n_total = video_embeds_full.shape[0]
    assert n_total == T_llm * N, f"unit-major layout assumption broken: {n_total} != {T_llm}*{N}"

    # ---------- policy forward (grad ON, Qwen already detached above) ----------
    X_units = video_embeds_full.detach().view(T_llm, N, -1).mean(dim=1).float()
    delta_units = torch.zeros_like(X_units)
    delta_units[1:] = X_units[1:] - X_units[:-1]
    logits = policy(X_units, delta_units)  # [T]
    probs = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)

    # Reward must reflect "did it pick the right MC option", not "did it
    # predict the right token out of the whole ~150k vocab" -- restrict CE
    # to the 4 candidate-letter logits (softmax renormalized over just
    # {A,B,C,D}) instead of full-vocab cross_entropy. Full-vocab CE landed
    # at ~9-13 in the first smoke test (close to ln(vocab_size)~12, i.e.
    # near-random over the WHOLE vocab) which swamps the actual MC signal
    # and makes lambda_sparsity's contribution negligible by comparison.
    letters = LETTERS[: len(candidates)]
    letter_ids = torch.tensor(
        [processor.tokenizer(l, add_special_tokens=False).input_ids[0] for l in letters],
        device=device,
    )
    target_letter = answer_letter(sample)
    target_idx = letters.index(target_letter)

    K = int(getattr(args, "group_size", 4))
    lambda_sparsity = float(getattr(args, "lambda_sparsity", 0.01))
    # Minimum-coverage floor, evenly spaced across time -- replaces a first
    # version that force-kept only the single argmax unit when a Bernoulli
    # sample came back empty. That single-unit fallback was DETERMINISTIC
    # (same unit every time within one question), so once probs saturated
    # near 0 for every unit, all K samples of a group ended up identical --
    # advantage=reward-mean(group)=0 for all of them, zero gradient, policy
    # frozen exactly at the degenerate point (this is what Experiment 1
    # actually hit: retention collapsed to ~1-10% and loss pinned at -0.0).
    # An evenly-spaced floor keeps minimum temporal coverage (same idea as
    # Uniform sampling / 2-pass's cold pool) while leaving the REST of the
    # mask free to vary between the K samples, so advantage stays non-zero
    # even when probs are saturated.
    min_retention_frac = float(getattr(args, "min_retention_frac", 0.15))
    # Entropy bonus -- standard PPO/A2C-style regularizer that penalizes
    # probs collapsing to 0/1 too fast, so lambda_sparsity's downward
    # pressure settles at an equilibrium instead of running away to zero.
    entropy_coef = float(getattr(args, "entropy_coef", 0.01))

    rewards = []
    log_probs = []
    retentions = []
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
            out = model(inputs_embeds=fe, position_ids=pid, attention_mask=am,
                        output_attentions=False, return_dict=True)
            logits_last = out.logits[0, -1, :].float()
            option_logits = logits_last[letter_ids]  # [4] (or fewer)
            ce_loss = torch.nn.functional.cross_entropy(
                option_logits.unsqueeze(0), torch.tensor([target_idx], device=device)
            )
            retention = kept_indices.numel() / n_total
            reward = -float(ce_loss.item()) - lambda_sparsity * retention
            del out, logits_last

        # log-prob of THIS sampled mask under the (grad-enabled) policy dist
        mask_f = mask.float()
        lp = (mask_f * torch.log(probs) + (1 - mask_f) * torch.log(1 - probs)).sum()
        rewards.append(reward)
        log_probs.append(lp)
        retentions.append(retention)

    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
    baseline = rewards_t.mean()
    advantage = (rewards_t - baseline).detach()
    log_probs_t = torch.stack(log_probs)
    # log_probs_t is a SUM over T units (standard REINFORCE: log-prob of a
    # whole trajectory = sum of log-probs of each action in it), so its
    # magnitude scales with T (tens for a long video). The entropy bonus
    # must be summed the same way, not averaged -- an averaged entropy term
    # (0-0.69 regardless of T) is negligible next to a summed policy_loss
    # (which can be in the tens), so with .mean() the entropy bonus barely
    # moves the gradient at all. This was silently broken in the first
    # version of this fix and is why entropy_coef=0.01 did nothing to stop
    # probs saturating to 1.0 in Experiment 2 -- caught only after that run.
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
        "retention_mean": float(sum(retentions) / len(retentions)),
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=16 * 28 * 28)

    with open(gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    random.seed(0)
    random.shuffle(gt_qa_pairs)

    resume_from = getattr(args, "resume_from", None)
    resume_ckpt = None
    if resume_from:
        resume_ckpt = torch.load(resolve_path(resume_from), map_location=device)
        video_dim = resume_ckpt["video_dim"]
        print(f"Resuming from {resume_from}: video_dim={video_dim}, previous step={resume_ckpt.get('step', 0)}")
    else:
        # Infer video hidden dim from one real sample before building the
        # policy (avoids hardcoding a dim that may differ by model variant).
        # Skipped when resuming -- video_dim is already known from the
        # checkpoint, no need to spend a forward pass re-deriving it.
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

    policy = GatingPolicy(video_dim).to(device).float()
    lr = float(getattr(args, "lr", 1e-3))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    start_step = 0
    if resume_ckpt is not None:
        policy.load_state_dict(resume_ckpt["state_dict"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        else:
            print("WARNING: checkpoint has no optimizer_state (saved by an older version of this "
                  "script) -- resuming with a fresh Adam state (momentum/variance reset, policy "
                  "weights still resumed correctly).")
        start_step = int(resume_ckpt.get("step", 0))

    # num_steps is the ABSOLUTE target step count, not "additional steps" --
    # e.g. resuming from step=200 with num_steps=700 runs 500 more steps.
    num_steps = int(getattr(args, "num_steps", 20))
    log_every = int(getattr(args, "log_every", 1))
    # Checkpoint periodically -- a run of a few thousand steps can take
    # hours, and losing all progress to an SSH drop / a stray OOM on one
    # video / a GPU-sharing preemption with only an end-of-run save would
    # be wasteful. ckpt_every=0 disables (falls back to save-once-at-end).
    ckpt_every = int(getattr(args, "ckpt_every", 200))
    policy_save_path = resolve_path(getattr(args, "policy_save_path", "outputs/rl_gating_policy.pt"))
    os.makedirs(os.path.dirname(policy_save_path), exist_ok=True)
    metrics_log_path = policy_save_path + ".metrics.jsonl"
    metrics_f = open(metrics_log_path, "a", encoding="utf-8")

    step = start_step
    running_reward = []
    running_retention = []
    pbar = tqdm(total=num_steps, initial=start_step)
    # Advance the data cursor past what a previous run already consumed, so
    # a resume doesn't immediately re-train on the exact same first N
    # samples in the exact same order (shuffle order is fixed by the seed
    # above, so idx=start_step lines up with "the sample after wherever the
    # previous run left off", assuming the same gt_file/seed).
    idx = start_step
    while step < num_steps:
        sample = gt_qa_pairs[idx % len(gt_qa_pairs)]
        idx += 1
        video_path = resolve_video_path(video_dir, sample["video_name"], video_index)
        if video_path is None:
            continue
        try:
            stats = run_one_question(model, processor, policy, optimizer, video_path, sample, args, max_pixels, device)
        except Exception as e:
            print(f"SKIP step {step} ({sample['video_name']}): {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue

        running_reward.append(stats["reward_mean"])
        running_retention.append(stats["retention_mean"])
        step += 1
        pbar.update(1)
        metrics_f.write(json.dumps({
            "step": step, "loss": stats["loss"], "policy_loss": stats["policy_loss"],
            "entropy": stats["entropy_mean"], "reward": stats["reward_mean"],
            "retention": stats["retention_mean"], "T_units": stats["T_units"],
            "video_name": sample["video_name"],
        }) + "\n")
        metrics_f.flush()
        if step % log_every == 0:
            print(
                f"step={step} loss={stats['loss']:.4f} policy_loss={stats['policy_loss']:.4f} "
                f"entropy={stats['entropy_mean']:.4f} reward={stats['reward_mean']:.4f} "
                f"retention={stats['retention_mean']:.3f} T_units={stats['T_units']} "
                f"(running reward={sum(running_reward)/len(running_reward):.4f}, "
                f"running retention={sum(running_retention)/len(running_retention):.3f})"
            )
        if ckpt_every > 0 and step % ckpt_every == 0:
            ckpt_blob = {"state_dict": policy.state_dict(), "video_dim": video_dim, "step": step,
                         "optimizer_state": optimizer.state_dict()}
            ckpt_path = policy_save_path + f".step{step}"
            torch.save(ckpt_blob, ckpt_path)
            # Also overwrite the "latest" path so a resume/eval script can
            # always point at policy_save_path without knowing the step.
            torch.save(ckpt_blob, policy_save_path)
            print(f"[checkpoint] step={step} -> {ckpt_path} (and {policy_save_path})")
    pbar.close()
    metrics_f.close()

    torch.save({"state_dict": policy.state_dict(), "video_dim": video_dim, "step": step,
                "optimizer_state": optimizer.state_dict()}, policy_save_path)
    print(f"[train_rl_gating] done -> saved policy to {policy_save_path}")
    print(f"Per-step metrics logged to {metrics_log_path}")
    print(f"Final running reward={sum(running_reward)/len(running_reward):.4f}, "
          f"running retention={sum(running_retention)/len(running_retention):.3f}")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
