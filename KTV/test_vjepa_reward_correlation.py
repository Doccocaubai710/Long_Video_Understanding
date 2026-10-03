"""
test_vjepa_reward_correlation.py

Buoc 0b (theo de xuat phan bien ve tai lieu VJEPA_REWARD_proposal): TRUOC KHI
dau tu viet DSTA-Scorer + vong lap GRPO day du, kiem tra thang gia dinh rui
ro nhat cua toan bo huong "V-JEPA lam reward": sai so tai tao latent cua
V-JEPA co tuong quan gi voi muc do huu ich NGU NGHIA cho model dich
(Qwen2.5-VL) hay khong. Neu tuong quan thap, Giai doan 1 (V-JEPA reward)
gan nhu chac chan vo dung, tiet kiem duoc toan bo cong crawl data +
GPU-gio cho GRPO.

TAI SU DUNG ha tang V-JEPA DA CO SAN va DA KIEM CHUNG trong repo nay
(precompute_vjepa_salience.py) -- facebook/vjepa2-vitl-fpc64-256 qua
transformers chuan (AutoModel/AutoVideoProcessor), KHONG phai V-JEPA 2.1
(2.1 chua co trong transformers, chi tai duoc qua torch.hub cua Meta, can
env Python 3.12 rieng -- ngoai pham vi cua 1 pilot). context_mask/
target_mask + so sanh predictor_output.last_hidden_state (du doan) voi
predictor_output.target_hidden_state (ground truth cua Target Encoder)
CHINH LA cong thuc L_predict(m) trong tai lieu VJEPA_REWARD_proposal.

Thiet ke phep do (khong can DSTA-Scorer, khong can GRPO -- day la chan doan
thuan tuy, khong phai training):
  Voi moi video:
    1. Doc 64 frame (dung frames_per_clip cua V-JEPA) -- chia thanh
       grid_depth=32 "tubelet" (moi tubelet = 2 frame lien tiep, dung
       tubelet_size cua V-JEPA).
    2. Voi 20 mask ngau nhien (giu ~40% tubelet, tha random):
       - reward_vjepa: AM sai so du doan cua V-JEPA tren cac tubelet BI
         CHE, khi chi cho context la cac tubelet duoc giu (patch cua tubelet
         giu = context_mask, patch cua tubelet bi che = target_mask).
       - reward_semantic: log-likelihood trung binh cua CAPTION THAT, tinh
         qua Qwen2.5-VL-7B (dung labels=..., HF loss chuan) khi chi dua
         DUNG cac frame thuoc tubelet duoc giu vao (ghi ra 1 clip tam ngan,
         dua qua dung pipeline process_vision_info/get_video_features da
         kiem chung o cac script khac trong repo).
    3. Voi moi video, tinh he so tuong quan Spearman giua 20 cap
       (reward_vjepa, reward_semantic).
  Tong hop: phan phoi he so tuong quan tren toan bo video (mean/median,
  % video > 0.3 -- muc phan bien de xuat lam nguong "dang lam tiep").

QUAN TRONG: day la CHAN DOAN, khong phai huan luyen -- khong co
DSTA-Scorer, khong co policy nao duoc cap nhat. Muc dich duy nhat: co nen
di tiep vao Giai doan 1 (V-JEPA + GRPO) hay khong.

DATA (v2 -- doi tu MSR-VTT sang LLaVA-Video-178K/0_30_s_academic_v0_1):
lan chay dau tren MSR-VTT ra Spearman rat thap (trung binh 0.064) --
nhung MSR-VTT caption ngan (1 cau), chung chung, 20 nguoi gan nhan doc
lap nen chat luong rat khong deu (vi du that: 1/20 caption cua 1 video la
"a group is dancing" trong khi video la canh lai xe). Nghi ngo tuong quan
thap mot phan do NHIEU o phia reward_semantic (ground truth te), khong
chi do V-JEPA. Doi sang bo caption GPT-4o sinh, chi tiet 5-8 cau, nguon
Charades (academic_source/Charades/*.mp4) tu LLaVA-Video-178K de loai bo
nghi ngo nay -- neu tuong quan van thap voi ground truth tot, ket luan
"V-JEPA reward khong an voi ngu nghia" moi thuc su vung.

Requires (them ngoai cac dependency da co trong repo):
  pip install scipy imageio imageio-ffmpeg

Usage (smoke test truoc, vai video vai mask, roi moi chay full):
  python test_vjepa_reward_correlation.py \
    video_dir=Datasets/LLaVA-Video-178K-probe/academic_source/Charades \
    captions_file=Datasets/LLaVA-Video-178K-probe/0_30_s_academic_v0_1_cap_processed.json \
    num_videos=3 num_masks=5 \
    output_csv=outputs/vjepa_reward_correlation_charades_smoke.csv

  python test_vjepa_reward_correlation.py \
    video_dir=Datasets/LLaVA-Video-178K-probe/academic_source/Charades \
    captions_file=Datasets/LLaVA-Video-178K-probe/0_30_s_academic_v0_1_cap_processed.json \
    num_videos=200 num_masks=20 keep_ratio=0.4 \
    output_csv=outputs/vjepa_reward_correlation_charades.csv
"""

import csv
import json
import math
import os
import random

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import hydra
import imageio
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoVideoProcessor, Qwen2_5_VLForConditionalGeneration

from precompute_vjepa_salience import load_video_frames
from qwen_vision_process_fastvid import process_vision_info
from train_perceiver_stage1 import MIN_PIXELS, SYSTEM_PROMPT

DEFAULT_CAPTION_PROMPT = "Describe this video in detail."


def resolve_path(path):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return to_absolute_path(path)


def load_academic_caption_rows(captions_file):
    """LLaVA-Video-178K, split 0_30_s_academic_v0_1 (nguon Charades, caption
    GPT-4o sinh chi tiet -- khac MSR-VTT: moi dong = 1 video, khong co
    nhieu annotator/video nen KHONG can dedup nhu load_msrvtt_captions.
    Field 'video': 'academic_source/Charades/XXXXX.mp4' -> chi lay
    basename, khop voi cach giai nen tar. Field 'conversations': [0]=cau
    hoi goc (co tien to '<image>\\n' can bo), [1]=caption GPT-4o that."""
    d = json.load(open(captions_file, "r", encoding="utf-8"))
    rows = []
    for item in d:
        file_name = os.path.basename(item["video"])
        prompt = item["conversations"][0]["value"].replace("<image>\n", "").strip()
        caption = item["conversations"][1]["value"].strip()
        rows.append({"file_name": file_name, "caption": caption, "prompt": prompt})
    return rows


def sample_keep_mask(n_tubelets, keep_ratio, rng):
    """Chon ngau nhien n_keep tubelet (trong tong so n_tubelets) de GIU lam
    context -- phan con lai la target (bi che). Dam bao luon co it nhat 1
    tubelet o ca 2 phia."""
    n_keep = min(n_tubelets - 1, max(1, round(n_tubelets * keep_ratio)))
    return sorted(rng.sample(range(n_tubelets), n_keep))


def tubelets_to_patch_idx(t_list, n_hw):
    idx = []
    for t in t_list:
        base = t * n_hw
        idx.extend(range(base, base + n_hw))
    return torch.tensor(idx, dtype=torch.long)


@torch.inference_mode()
def vjepa_reward(vjepa_model, pixel_values_videos, grid_depth, n_hw, keep_tubelets, device):
    """AM sai so du doan L1 cua V-JEPA tren cac tubelet BI CHE, khi context
    la cac tubelet duoc giu. Reward cao = phan bi che de doan duoc tu phan
    giu -> phan giu "du thong tin" theo nghia V-JEPA (dung cong thuc
    L_predict trong tai lieu VJEPA_REWARD_proposal, khong lay log vi
    Spearman bat bien voi bien doi don dieu)."""
    drop_tubelets = sorted(set(range(grid_depth)) - set(keep_tubelets))
    if not drop_tubelets:
        return None
    context_idx = tubelets_to_patch_idx(keep_tubelets, n_hw)
    target_idx = tubelets_to_patch_idx(drop_tubelets, n_hw)
    context_mask = [context_idx.unsqueeze(0).to(device)]
    target_mask = [target_idx.unsqueeze(0).to(device)]

    out = vjepa_model(
        pixel_values_videos=pixel_values_videos, context_mask=context_mask, target_mask=target_mask,
    )
    pred = out.predictor_output.last_hidden_state[0].float()
    target = out.predictor_output.target_hidden_state[0].float()
    err = (pred - target).norm(dim=-1).mean().item()
    return -err


@torch.no_grad()
def semantic_reward(qwen_model, qwen_processor, clip_path, caption, prompt, max_pixels, device, debug=False):
    """AM CE (= log-likelihood trung binh) cua caption THAT, tinh qua
    Qwen2.5-VL that (model dich cuoi cung cua du an), khi video dau vao CHI
    gom cac frame thuoc tubelet duoc giu. Dung labels=... (HF loss chuan)
    thay vi generate() -- nhanh hon va dung dung cong thuc R_Stage2A trong
    tai lieu 2 giai doan. prompt: cau hoi khop phong cach caption (vd
    "Elaborate on the visual and narrative elements..." cho caption dai,
    chi tiet -- khac voi 1 cau ngan CAPTION_PROMPT cu cua MSR-VTT)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": clip_path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
            {"type": "text", "text": prompt},
        ]},
    ]
    prompt_text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    prompt_inputs = qwen_processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    if debug:
        thw = prompt_inputs["video_grid_thw"][0].tolist()
        print(f"  [debug] Qwen thay video_grid_thw={thw} (T_llm={thw[0]})")

    caption_ids = qwen_processor.tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_input_ids = torch.cat([prompt_inputs["input_ids"], caption_ids], dim=1)
    labels = full_input_ids.clone()
    labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
    attention_mask = torch.ones_like(full_input_ids)

    out = qwen_model(
        input_ids=full_input_ids,
        attention_mask=attention_mask,
        pixel_values_videos=prompt_inputs["pixel_values_videos"],
        video_grid_thw=prompt_inputs["video_grid_thw"],
        labels=labels,
    )
    return -out.loss.item()


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    vjepa_checkpoint = getattr(args, "vjepa_checkpoint", "facebook/vjepa2-vitl-fpc64-256")
    print(f"Loading V-JEPA: {vjepa_checkpoint} ...")
    vjepa_processor = AutoVideoProcessor.from_pretrained(vjepa_checkpoint)
    vjepa_model = AutoModel.from_pretrained(vjepa_checkpoint, torch_dtype=dtype, device_map="auto").eval()
    grid_size = vjepa_model.config.crop_size // vjepa_model.config.patch_size
    grid_depth = vjepa_model.config.frames_per_clip // vjepa_model.config.tubelet_size
    n_hw = grid_size * grid_size
    print(f"V-JEPA grid: T={grid_depth} tubelets, H=W={grid_size} -> N={n_hw} patch/tubelet")

    model_path = getattr(args, "model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    max_pixels_units = int(getattr(args, "max_pixels_units", 1024))
    max_pixels = max_pixels_units * 28 * 28
    print(f"Loading Qwen2.5-VL: {model_path} ...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto").eval()
    qwen_processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=MIN_PIXELS)

    video_dir = resolve_path(args.video_dir)
    captions_file = resolve_path(args.captions_file)
    unique_rows = load_academic_caption_rows(captions_file)  # da la 1 dong/video, khong can dedup
    random.seed(0)
    random.shuffle(unique_rows)

    num_videos = int(getattr(args, "num_videos", 200))
    num_masks = int(getattr(args, "num_masks", 20))
    keep_ratio = float(getattr(args, "keep_ratio", 0.4))
    # Ma hoa clip tam o dung fps=2.0 -- khop target FPS mac dinh cua
    # qwen_vision_process_fastvid.py de Qwen KHONG tu sample-bot them frame
    # cua chinh chung ta da chon giu.
    clip_fps = float(getattr(args, "clip_fps", 2.0))

    tmp_dir = resolve_path(getattr(args, "tmp_dir", "outputs/tmp_vjepa_corr"))
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_clip_path = os.path.join(tmp_dir, "current_clip.mp4")

    output_csv = resolve_path(args.output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rng = random.Random(42)
    fieldnames = ["video_name", "mask_idx", "n_keep_tubelets", "reward_vjepa", "reward_semantic"]
    per_video_corr = []
    n_done = 0
    first_mask_debug = True

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(unique_rows, desc="videos"):
            if n_done >= num_videos:
                break
            video_path = os.path.join(video_dir, row["file_name"])
            if not os.path.exists(video_path):
                continue
            try:
                frames = load_video_frames(video_path, num_frames=vjepa_model.config.frames_per_clip)
            except Exception as e:
                print(f"SKIP {row['file_name']} (doc frame loi): {type(e).__name__}: {e}")
                continue

            vjepa_inputs = vjepa_processor(frames, return_tensors="pt").to(device)
            pixel_values_videos = vjepa_inputs["pixel_values_videos"].to(dtype)

            vjepa_rewards, semantic_rewards = [], []
            for m in range(num_masks):
                keep_tubelets = sample_keep_mask(grid_depth, keep_ratio, rng)
                r_v = vjepa_reward(vjepa_model, pixel_values_videos, grid_depth, n_hw, keep_tubelets, device)
                if r_v is None:
                    continue

                kept_frames = []
                for t in keep_tubelets:
                    kept_frames.append(frames[2 * t])
                    kept_frames.append(frames[2 * t + 1])

                try:
                    imageio.mimwrite(tmp_clip_path, kept_frames, fps=clip_fps)
                    r_s = semantic_reward(
                        qwen_model, qwen_processor, tmp_clip_path, row["caption"],
                        row.get("prompt") or DEFAULT_CAPTION_PROMPT, max_pixels, device,
                        debug=first_mask_debug,
                    )
                    first_mask_debug = False
                except Exception as e:
                    print(f"SKIP mask {m} cua {row['file_name']}: {type(e).__name__}: {e}")
                    continue

                vjepa_rewards.append(r_v)
                semantic_rewards.append(r_s)
                writer.writerow({
                    "video_name": row["file_name"], "mask_idx": m,
                    "n_keep_tubelets": len(keep_tubelets),
                    "reward_vjepa": r_v, "reward_semantic": r_s,
                })
            f.flush()

            min_masks_for_corr = min(int(getattr(args, "min_masks_for_corr", 5)), num_masks)
            if len(vjepa_rewards) >= min_masks_for_corr:
                rho, _ = spearmanr(vjepa_rewards, semantic_rewards)
                if not math.isnan(rho):
                    per_video_corr.append(rho)
                    print(f"{row['file_name']}: spearman={rho:.3f} (n={len(vjepa_rewards)})")
            n_done += 1

    if os.path.exists(tmp_clip_path):
        os.remove(tmp_clip_path)

    if per_video_corr:
        arr = np.array(per_video_corr)
        print(f"\n=== KET QUA TONG HOP ({len(arr)} video co du lieu) ===")
        print(f"Spearman trung binh : {arr.mean():.3f}")
        print(f"Spearman trung vi   : {np.median(arr):.3f}")
        print(f"% video Spearman > 0.3 : {(arr > 0.3).mean() * 100:.1f}%")
        print(f"% video Spearman > 0   : {(arr > 0).mean() * 100:.1f}%")
        print(f"Chi tiet tung mask -> {output_csv}")
    else:
        print("Khong co video nao du du lieu de tinh Spearman -- kiem tra log SKIP o tren.")


@hydra.main(config_path=None, version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
