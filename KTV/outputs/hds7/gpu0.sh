#!/bin/bash
# Dot hds7, GPU0. Plan: outputs/hds7/PLAN.txt. Chay: bash outputs/hds7/start.sh
GPU=0; source "$(dirname "$(readlink -f "$0")")/common.sh"
log "QUEUE START gpu0 (pid $$)"

# --- Giai doan 1: ablation loss + cua so (10%) ---
train H3cv --w_ceg 0.2 $VRIPT            # dang chay tu hds6: cho xong (hoac tu resume neu bi ngat)
train L1 --w_ceg 0.5
train L3 --w_ce 0.2 --w_ceg 0            # CE khong gate
train W1 --w_ceg 0.2 --window 1
evalb mlvu W1

# --- Giai doan 2: data + chuyen muc nen, theo C* ---
wait_cstar; log "C* = $CNAME ($CARGS)"
train Dq $CARGS --max_videos 640         # 1/4 data
train Dh $CARGS --max_videos 1280        # 1/2 data
train X25 $CARGS
train X5 $CARGS

# --- Giai doan 3: cham cuoi nhom A ---
evalb mlvu X25 X5
evalb videomme S2 Dq Dh D2x Dx8 H3cv X25 X5
evalb lvbench S2 Dq Dh D2x Dx8 H3cv X25 X5

# --- Du phong: giu card chay ---
case $CNAME in H3|H3ce) s=3;; *) s=2;; esac
train S3 $CARGS --seed $s
train X40 $CARGS
evalb mlvu S3 X40; evalb videomme S3 X40; evalb lvbench S3 X40
log "QUEUE DONE gpu0"
