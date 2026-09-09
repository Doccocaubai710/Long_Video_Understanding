#!/bin/bash
# Verify RECON-Select + QTA (CAS-QTA) for real, before trusting the earlier
# ~0.40-0.41 number. Run in the (ktv) env, NOT qwen25vl.
#
# Order: (1) confirm the order>=12 patch is actually present in this
# server's llava_arch.py, (2) regenerate Stage1/Stage2 from a clean cache
# (stale cache from before the patch would silently serve old, possibly
# broken results), (3) verify_token_budget.py sanity check (no GPU), (4) a
# 30-question smoke test. Full 2700-question run is a SEPARATE manual step
# after you review the smoke test -- not automatic here.
set -euo pipefail

PY="${PY:-python}"

echo "=== 1) Confirm order>=12 patch is present ==="
if grep -q "order >= 12" ktv/llava/model/llava_arch.py; then
  echo "OK: patch found."
else
  echo "MISSING: 'order >= 12' not found in ktv/llava/model/llava_arch.py."
  echo "CAS-QTA token budgets will be silently flattened to a single value."
  echo "Fix this before continuing -- do not proceed."
  exit 1
fi

echo "=== 2) Fresh Stage1 (RECON) + Stage2 (QTA) run, clearing stale cache ==="
rm -rf outputs/casqta_stage1 outputs/casqta_stage1_stats.txt \
       outputs/vmme_casqta_kf outputs/vmme_casqta.json
${PY} recon_select.py experiment=vmme_casqta
${PY} qta_allocated.py experiment=vmme_casqta

echo "=== 3) Token budget sanity check (no GPU) ==="
${PY} verify_token_budget.py outputs/vmme_casqta.json \
  --num_frames 6 --tokens_num 1872 --rope_scaling_factor 2

echo "=== 4) Smoke test: 30 questions ==="
${PY} run_inference_multiple_choice_qa.py \
  experiment=vmme_casqta \
  output_name=smoke_casqta_verified_30q \
  key_frame_path=outputs/vmme_casqta.json \
  prune_mode=cls_new_token_sim rate=0.2 tokens_num=1872 \
  num_chunks=90 chunk_idx=0
${PY} eval/compute_accuracy.py outputs/smoke_casqta_verified_30q.json

echo "=== Done. Review the output above before running the full 2700 pass. ==="
echo "If it looks sane (accuracy well above ~25% chance, unparsed=0, no OVERFLOW/FLATTENED in step 3):"
echo "  nohup python run_inference_multiple_choice_qa.py \\"
echo "    experiment=vmme_casqta \\"
echo "    output_name=pred_casqta_verified \\"
echo "    key_frame_path=outputs/vmme_casqta.json \\"
echo "    prune_mode=cls_new_token_sim rate=0.2 tokens_num=1872 \\"
echo "    > casqta_verify_run.log 2>&1 &"
echo "  disown"
