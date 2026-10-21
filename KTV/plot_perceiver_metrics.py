"""
plot_perceiver_metrics.py

Ve do thi tu file metrics.jsonl cua train_perceiver_stage1.py/stage2.py (loss,
caption_loss, anchor_loss theo step) -- rieng cho Perceiver, khac
plot_rl_gating_metrics.py (danh cho RL-Gating, field khac hoan toan:
reward/retention/entropy).

QUAN TRONG -- file .metrics.jsonl duoc MO O CHE DO APPEND ("a"), nen moi lan
resume train (ke ca cac lan bi crash het o dia / bi bug step reset ve 0
truoc khi sua) deu CONG DON them dong log moi vao CUNG 1 file, khong xoa log
cu. Ket qua: nhieu "lan chay" khac nhau co the trung lap khoang step tren
truc X (VD lan 1 chay step 0->2900 roi crash, lan 2 resume lai tu dau vi bug
cu lai chay step 0->300...), ve chung len 1 truc se chong cheo rat kho doc.

Script nay TU DONG tach cac lan chay (phat hien khi step[i] < step[i-1] =
bat dau 1 lan chay moi), chi ve CHI TIET (duong day, mau) cho LAN CHAY CUOI
CUNG (thuong la ban da sua bug, dang tin cay nhat / lien mach nhat), cac lan
chay cu hon ve MO + NET DUT, co chu thich rieng ghi ro step range de khong
bi hieu nham la "so sanh nhieu experiment".

Usage:
  python plot_perceiver_metrics.py \
    metrics_path=outputs/perceiver_stage1_smoke.pt.metrics.jsonl \
    out_path=outputs/plot_perceiver_smoke.png window=20
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


def split_runs(rows):
    """Tach thanh danh sach cac 'lan chay' -- moi khi step[i] < step[i-1]
    (bi/duoc reset lai, dau hieu 1 tien trinh train MOI da bat dau ghi vao
    CUNG file metrics), coi la ranh gioi 1 lan chay moi."""
    if not rows:
        return []
    runs = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        if cur["step"] < prev["step"]:
            runs.append([])
        runs[-1].append(cur)
    return runs


def plot_key(ax, runs, key, color, window, label_main=None, old_alpha=0.35):
    """Ve TAT CA cac lan chay cu (mo, net dut, xam) + lan chay CUOI (net
    lien, mau that, day du raw+rolling mean). Tra ve so lan chay cu bi mo
    de in canh bao ra console."""
    n_old = len(runs) - 1
    for ri, run in enumerate(runs[:-1]):
        steps = [r["step"] for r in run]
        vals = [r[key] for r in run]
        lo, hi = steps[0], steps[-1]
        ax.plot(steps, rolling_mean(vals, window), color="gray", alpha=old_alpha,
                 linewidth=1.2, linestyle="--",
                 label=f"lần chạy cũ #{ri+1} (step {lo}-{hi}, đã bỏ/ghi đè)")
    last = runs[-1]
    steps = [r["step"] for r in last]
    vals = [r[key] for r in last]
    ax.plot(steps, vals, alpha=0.2, linewidth=0.8, color=color)
    ax.plot(steps, rolling_mean(vals, window), color=color, linewidth=2,
             label=(label_main or key) + f" (lần chạy CUỐI, step {steps[0]}-{steps[-1]})")
    return n_old


def main():
    kv = {}
    for tok in sys.argv[1:]:
        tok = tok.lstrip("+")
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    if "metrics_path" not in kv:
        raise SystemExit("Thieu +metrics_path=...")
    metrics_path = kv["metrics_path"]
    out_path = kv.get("out_path", "outputs/perceiver_plot.png")
    window = int(kv.get("window", 20))

    rows = load_metrics(metrics_path)
    runs = split_runs(rows)
    if len(runs) > 1:
        print(f"[canh bao] file metrics co {len(runs)} lan chay khac nhau chong len nhau tren truc step:")
        for ri, run in enumerate(runs):
            tag = "CUOI (dung lam duong chinh)" if ri == len(runs) - 1 else "cu (chi ve mo)"
            print(f"  - lan {ri+1}: step {run[0]['step']}-{run[-1]['step']} ({len(run)} dong) [{tag}]")

    has_tokens = "n_final_tokens" in rows[0]

    nrows = 4 if has_tokens else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 11 if not has_tokens else 14), sharex=True)
    if has_tokens:
        ax_loss, ax_caption, ax_anchor, ax_tokens = axes
    else:
        ax_loss, ax_caption, ax_anchor = axes

    plot_key(ax_loss, runs, "loss", "tab:blue", window, label_main="loss (tong)")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"Tong loss = caption_loss + anchor_weight*anchor_loss (dam=rolling mean w={window})")
    ax_loss.legend(fontsize=7, loc="upper right")
    ax_loss.grid(alpha=0.3)

    plot_key(ax_caption, runs, "caption_loss", "tab:orange", window, label_main="caption_loss")
    ax_caption.set_ylabel("caption_loss")
    ax_caption.set_title("Cross-entropy sinh caption that (Perceiver hoc tom tat co ich -- cang thap cang tot)")
    ax_caption.legend(fontsize=7, loc="upper right")
    ax_caption.grid(alpha=0.3)

    plot_key(ax_anchor, runs, "anchor_loss", "tab:green", window, label_main="anchor_loss")
    ax_anchor.set_ylabel("anchor_loss")
    ax_anchor.set_xlabel("step")
    ax_anchor.set_title("1 - mean_j(max_i cos(T2_j,T1_i)) -- cang thap = T2 cang 'gan' token anh goc")
    ax_anchor.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax_anchor.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax_anchor.legend(fontsize=7, loc="upper right")
    ax_anchor.grid(alpha=0.3)

    if has_tokens:
        plot_key(ax_tokens, runs, "n_final_tokens", "tab:red", window, label_main="n_final_tokens")
        ax_tokens.set_ylabel("so luong token")
        ax_tokens.set_xlabel("step")
        ax_tokens.set_title("Tong token cuoi/video -- CHI lan chay cuoi ve day du, lan cu chi ve mo (xem chu thich)")
        ax_tokens.legend(fontsize=6.5, loc="upper left")
        ax_tokens.grid(alpha=0.3)

        last = runs[-1]
        steps_last = [r["step"] for r in last]
        n_pointed_last = [r["n_pointed_avg"] for r in last]
        ax_pointed = ax_tokens.twinx()
        ax_pointed.plot(steps_last, rolling_mean(n_pointed_last, window), color="tab:cyan", linewidth=1.5,
                         label="n_pointed_avg/unit (lần cuối)")
        ax_pointed.set_ylabel("n_pointed_avg (token/unit, lần chạy cuối)", color="tab:cyan")
        ax_pointed.tick_params(axis="y", labelcolor="tab:cyan")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
