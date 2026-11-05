# KTV / RAP – nén token video dài cho Qwen2.5-VL-7B

Thư mục làm việc của RAP (Anchored Resampler) và các mốc so sánh trên Video-MME.
Chuyển server: xem `docs/prompts/PROMPT_AGENT_SERVER_MOI.md` và `docs/prompts/SETUP_SERVER_MOI.txt`
(đường dẫn script trong hai file này là theo cấu trúc cũ, trước 23/10).

## Cấu trúc

```
common/      dùng chung: đọc video (qwen_vision_process_fastvid), dựng chuỗi token (run_inference_qwen25vl_2pass, _dyseg),
             local_data, utils
rap/         RAP: rap_model, alloc_model, scene_model, place_model,
             train_rap (bản gốc), train_rap_elastic (bản chính), train_rap_vjepa, train_place, precompute_vjepa_salience,
             infer_rap_videomme (chấm Video-MME), hds_model (HDS-MAP 3D: trượt hộp 3D + M3D-BD+),
             eval_multi (chấm nhiều cấu hình / 1 lần đọc video, mọi bộ: Video-MME, LVBench, MLVU),
             bench_latency (đo prefill / KV-cache / thời gian toàn trình)
baselines/   mốc so sánh: run_inference_qwen25vl_kitoke (+mode=uniform/holitom/vidcom2/fastvid/dycoke/kitoke),
             holitom_port, vidcom2_port, fastvid_port, dycoke_port, run_qwen25vl_kitoke_baseline.sh
tools/       compare (so cặp z), hds_report (outputs/hds/KET_QUA.txt), mlvu_dev_report, prep_vript, bench_speed_rap
eval/        compute_accuracy, compute_accuracy_by_bucket
configs/     config Hydra (inference_qwen25vl)
kitoke/      repo KiToke (kitoke_repro, baselines/run_inference_qwen25vl_kitoke import)
outputs/     hds/ … hds6/ (các đợt HDS-MAP 3D: queue, KET_QUA.txt), KET_QUA_TONG_HOP.md (mọi kết quả), preds/ (dự đoán Video-MME 2.700 câu),
             checkpoints/ (RAP, không đưa lên git)
             hds7/ (đợt hiện tại): PLAN.txt, KET_QUA.txt, common.sh + gpu{0,1}{,_extra}.sh + keepalive.sh + start.sh (hàng đợi 2 GPU),
                   report.py (viết KET_QUA.txt), build_ablation.py / push_ablation.py / pick_lambda.py (tab Ablation study trên web),
                   alloc_gate.py, q1_gate.py, choose_cstar.py (tiêu chí đặt trước), LATENCY.json (đo tốc độ)
docs/        báo cáo, đề xuất, idea.md; docs/prompts/ prompt agent + setup server
playground/gt_qa_files/  câu hỏi; Datasets/ video (Video-MME, LVBench, MLVU_Test, LLaVA-Video-178K-probe); Qwen/, facebook/ trọng số Qwen2.5-VL, V-JEPA 2
scripts/data/            chuẩn bị dataset
```

## Chạy (luôn đứng ở thư mục KTV, dùng `python -m`)

```bash
PY=/root/local_env/qwen25vl/bin/python
# train RAP
$PY -m rap.train_rap_elastic --out outputs/R10X.pt --retention 0.10 --num_steps 4000 \
    --pseudo_steps 0 --pseudo_prob_after 0 --w_ce 0 --w_kd 1.0
# chấm RAP
$PY -m rap.infer_rap_videomme --policy outputs/checkpoints/R10E.pt --retention 0.10 \
    --output_dir outputs/preds --output_name pred_videomme_f128_rap_R10X_r010
# mốc so sánh (Hydra)
$PY -m baselines.run_inference_qwen25vl_kitoke +mode=holitom +retention=0.10 ...
# điểm theo short/medium/long, so cặp z
$PY eval/compute_accuracy_by_bucket.py outputs/preds/<file>.json --csv playground/gt_qa_files/VideoMME/val_qa.csv
$PY -m tools.compare outputs/preds/<A>.json outputs/preds/<B>.json
```
