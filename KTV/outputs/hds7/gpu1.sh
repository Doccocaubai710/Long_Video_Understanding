#!/bin/bash
# Dot hds7, GPU1. Plan: outputs/hds7/PLAN.txt. Chay: bash outputs/hds7/start.sh
GPU=1; source "$(dirname "$(readlink -f "$0")")/common.sh"
log "QUEUE START gpu1 (pid $$)"
while pgrep -f "eval_multi --gt_file playground/gt_qa_files/LVBench/val_qa.json .*--output_dir outputs/hds/lvbench" > /dev/null; do sleep 60; done  # eval RAP con thieu (hds6)

# --- Giai doan 1: ablation loss + cua so (10%) ---
train L2 --w_ceg 1.0
train L4 --w_ceg 0.2 --seed 1            # H3ce seed 1
train W2 --w_ceg 0.2 --window 5
evalb mlvu H3 H3s1 H3cv L1 L2 L3 L4 W2
train H3noGate --gate 0                  # train tiep tu buoc ~2000 (hds2)
evalb mlvu H3noGate

# --- Giai doan 2: data + seed, theo C* ---
wait_cstar; log "C* = $CNAME ($CARGS)"
if grep -q "DOWNLOAD DONE" Datasets/LLaVA-Video-178K-x2/download.log 2>/dev/null; then
  log "SKIP train Dx8 (da co data 2x)"; train D2x $CARGS --extra_video_root Datasets/LLaVA-Video-178K-x2 --max_extra 2569
else
  log "SKIP train D2x (chua tai xong data 2x)"; train Dx8 $CARGS --max_videos 320
fi
case $CNAME in H3|H3ce) s=2;; *) s=1;; esac
train S2 $CARGS --seed $s
if [ "$CNAME" = H3ce ]; then log "SKIP train V1 (C* = H3ce, da co H3cv)"; train CEo --w_kd 0 --w_ce 1.0 --w_ceg 0
else log "SKIP train CEo"; train V1 $CARGS $VRIPT; fi

# --- Giai doan 3: cham nhom B ---
evalb mlvu Dq Dh D2x Dx8 S2 V1 CEo
evalb videomme L1 L2 L3 L4 W1 W2 H3noGate V1 CEo
evalb lvbench L1 L2 L3 L4 W1 W2 H3noGate V1 CEo

# --- Du phong: giu card chay ---
if [ "$CNAME" = H3 ]; then train L4s2 --w_ceg 0.2 --seed 2; e=L4s2; else train H3s2 --w_ceg 0 --seed 2; e=H3s2; fi
train X2 $CARGS
evalb mlvu $e X2; evalb videomme $e X2; evalb lvbench $e X2
log "QUEUE DONE gpu1"
