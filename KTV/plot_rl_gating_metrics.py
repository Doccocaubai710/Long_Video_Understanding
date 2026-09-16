"""
plot_rl_gating_metrics.py

Ve do thi tu file metrics.jsonl da log san trong luc train (train_rl_gating.py
/ train_rl_gating_2stage.py) -- KHONG can chay lai training, chi doc file da
co. Nhin so tren terminal tung step rat nhieu (chi K=4 mau/step), do thi co
lam min (rolling mean) moi thay ro xu huong that.

Chay duoc tren server khong man hinh (dung backend "Agg", chi luu PNG, khong
can hien thi).

Usage (1 file, VD kiem tra bo frame-level):
  python plot_rl_gating_metrics.py \
    metrics_path=outputs/rl_gating_policy_exp3.pt.metrics.jsonl \
    out_path=outputs/plot_exp3.png

So sanh nhieu file cung luc (VD frame-level vs 2-stage), cach nhau bang dau
phay, ten hien thi tuong ung cach nhau bang dau phay o "labels":
  python plot_rl_gating_metrics.py \
    metrics_path=outputs/rl_gating_policy_exp3.pt.metrics.jsonl,outputs/rl_gating_2stage_exp1.pt.metrics.jsonl \
    labels=frame-level,2-stage \
    out_path=outputs/plot_compare.png

Optional: window=30 (do rong rolling mean, mac dinh 30 step)

Neu bao loi "No module named matplotlib": pip install matplotlib
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling_mean(values, window):
    if len(values) < 2:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def load_metrics(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    # Goi kieu hydra "+key=value" (dong bo voi cac script train/infer khac
    # trong repo nay) -- argparse KHONG cho dang ky option bat dau bang "+",
    # nen tu parse sys.argv truc tiep thay vi dung argparse.
    kv = {}
    for tok in sys.argv[1:]:
        tok = tok.lstrip("+")
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    if "metrics_path" not in kv:
        raise SystemExit("Thieu +metrics_path=... (1 hoac nhieu file .jsonl, cach nhau bang dau phay)")
    metrics_path = kv["metrics_path"]
    labels_raw = kv.get("labels")
    out_path = kv.get("out_path", "outputs/rl_gating_plot.png")
    window = int(kv.get("window", 30))

    paths = [p.strip() for p in metrics_path.split(",")]
    labels = [l.strip() for l in labels_raw.split(",")] if labels_raw else [p.split("/")[-1] for p in paths]

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    ax_reward, ax_retention, ax_entropy, ax_loss = axes

    for path, label in zip(paths, labels):
        rows = load_metrics(path)
        if not rows:
            print(f"WARNING: {path} rong, bo qua")
            continue
        steps = [r["step"] for r in rows]

        reward = [r.get("reward", r.get("reward_mean")) for r in rows]
        retention = [r.get("retention", r.get("retention_mean")) for r in rows]
        loss = [r["loss"] for r in rows]
        policy_loss = [r.get("policy_loss", r["loss"]) for r in rows]

        ax_reward.plot(steps, reward, alpha=0.15, linewidth=0.8)
        ax_reward.plot(steps, rolling_mean(reward, window), label=label, linewidth=2)

        ax_retention.plot(steps, retention, alpha=0.15, linewidth=0.8)
        ax_retention.plot(steps, rolling_mean(retention, window), label=label, linewidth=2)

        ax_loss.plot(steps, policy_loss, alpha=0.15, linewidth=0.8)
        ax_loss.plot(steps, rolling_mean(policy_loss, window), label=f"{label} policy_loss", linewidth=2)

        # entropy: ban 1-tang co "entropy_mean", ban 2-tang co "entropy1_mean"/"entropy2_mean"
        if "entropy1_mean" in rows[0]:
            e1 = [r["entropy1_mean"] for r in rows]
            e2 = [r["entropy2_mean"] for r in rows]
            ax_entropy.plot(steps, rolling_mean(e1, window), label=f"{label} entropy (Tang 1, unit)", linewidth=2)
            ax_entropy.plot(steps, rolling_mean(e2, window), label=f"{label} entropy (Tang 2, token)",
                             linewidth=2, linestyle="--")
        elif "entropy" in rows[0]:  # ban 1-tang (train_rl_gating.py) ghi key "entropy", khong phai "entropy_mean"
            e = [r["entropy"] for r in rows]
            ax_entropy.plot(steps, rolling_mean(e, window), label=f"{label} entropy", linewidth=2)

    ax_reward.set_ylabel("reward")
    ax_reward.set_title(f"Reward (mo nhat = raw moi step, dam = rolling mean w={window})")
    ax_reward.legend()
    ax_reward.grid(alpha=0.3)

    ax_retention.set_ylabel("retention")
    ax_retention.set_title("Ty le token giu (0=bo het, 1=giu het)")
    ax_retention.axhline(0.15, color="red", linestyle=":", alpha=0.5, label="san toi thieu (vd)")
    ax_retention.legend()
    ax_retention.grid(alpha=0.3)
    ax_retention.set_ylim(0, 1)

    ax_entropy.set_ylabel("entropy")
    ax_entropy.set_title("Entropy chinh sach (0.693=ln2=chua hoc gi/gan random, cang thap = cang tu tin phan hoa)")
    ax_entropy.axhline(0.693, color="gray", linestyle=":", alpha=0.5, label="tran ly thuyet ln(2)")
    ax_entropy.legend()
    ax_entropy.grid(alpha=0.3)

    ax_loss.set_ylabel("policy_loss")
    ax_loss.set_xlabel("step")
    ax_loss.set_title("Policy loss (nhieu lon la binh thuong voi REINFORCE, nhin rolling mean la chinh)")
    ax_loss.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
