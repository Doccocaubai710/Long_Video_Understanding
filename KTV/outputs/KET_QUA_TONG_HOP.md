# KET QUA TONG HOP – outputs/ (gom ngay 2026-10-23)

Video-MME, Qwen2.5-VL-7B, 128 frame. Ten file `pred_videomme_f128_<run>_r<muc giu>`: r025 = giu 25% token.
n = so cau (2700 = du bo, 300 = tap con sub300). z = McNemar so cap tung cau, |z| > 1,96 la co y nghia.
Log, jsonl, file du doan tung cau cua cac huong cu da xoa; file du doan day du (2700 cau) giu o `preds/` de so cap z.
Duong dan trong cac bao cao ben duoi (outputs/rap/..., outputs/rap_elastic/...) la vi tri cu, nay da xoa.

So cap 2 file: `python -m tools.compare outputs/preds/<A>.json outputs/preds/<B>.json`

Checkpoint giu lai (`checkpoints/`, nap bang `rap.rap_model.load_resampler`):

| file | la gi | muc giu khi cham |
|---|---|---|
| runE.pt | RAP Run E (KL, khong warm-up, 4.000 buoc) – ban chinh | 25% |
| R40E.pt / R15E.pt / R10E.pt / R05E.pt / R02E.pt / R01E.pt | cong thuc Run E train rieng tung muc | 40 / 15 / 10 / 5 / 2 / 1% |
| ELA.pt | RAP-Elastic: 1 checkpoint cham duoc moi muc | 1–40% |

## 1. Bang do chinh xac (tu cac file *_accuracy_by_bucket.txt)

### kitoke_baseline/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| f128_attn_r010_loqo | 2700 | 71.56 | 58.89 | 50.89 | **60.44** |
| f128_attn_r010_self | 2700 | 75.44 | 62.11 | 53.00 | **63.52** |
| f128_attn_r025_loqo | 2700 | 74.22 | 62.22 | 52.78 | **63.07** |
| f128_attn_r025_self | 2700 | 76.22 | 64.00 | 53.33 | **64.52** |
| f128_dense | 2700 | 76.89 | 64.44 | 54.11 | **65.15** |
| f128_dense_time-hf_default | 2700 | 73.22 | 64.89 | 54.67 | **64.26** |
| f128_dsta_v2_r025 | 2700 | 67.56 | 59.89 | 53.22 | **60.22** |
| f128_dycoke_r025 | 2700 | 71.33 | 58.44 | 51.78 | **60.52** |
| f128_dycoke_r040 | 2700 | 75.67 | 63.11 | 54.33 | **64.37** |
| f128_dyseg_grid_r025 | 2700 | 72.11 | 62.44 | 52.67 | **62.41** |
| f128_dyseg_r002 | 2700 | 59.22 | 50.56 | 45.78 | **51.85** |
| f128_dyseg_r005 | 2700 | 64.22 | 54.89 | 48.00 | **55.70** |
| f128_dyseg_r010 | 2700 | 66.11 | 58.22 | 50.89 | **58.41** |
| f128_dyseg_r025 | 2700 | 70.78 | 61.33 | 52.78 | **61.63** |
| f128_dyseg_r040 | 2700 | 73.22 | 61.78 | 54.56 | **63.19** |
| f128_fastvid_r002 | 2700 | 61.11 | 50.33 | 46.22 | **52.56** |
| f128_fastvid_r005 | 2700 | 66.00 | 53.89 | 50.00 | **56.63** |
| f128_fastvid_r010 | 2700 | 69.56 | 57.56 | 50.78 | **59.30** |
| f128_fastvid_r015 | 2700 | 72.78 | 59.22 | 51.56 | **61.19** |
| f128_fastvid_r025 | 2700 | 74.22 | 60.89 | 53.89 | **63.00** |
| f128_fastvid_r040 | 2700 | 74.89 | 64.56 | 55.89 | **65.11** |
| f128_holitom_notemporal_r010 | 2700 | 69.44 | 59.11 | 52.44 | **60.33** |
| f128_holitom_notemporal_r025 | 2700 | 75.22 | 62.67 | 54.44 | **64.11** |
| f128_holitom_r002 | 2700 | 64.67 | 53.11 | 48.33 | **55.37** |
| f128_holitom_r005 | 2700 | 68.33 | 57.22 | 50.56 | **58.70** |
| f128_holitom_r010 | 2700 | 71.00 | 59.56 | 53.44 | **61.33** |
| f128_holitom_r015 | 2700 | 72.22 | 61.56 | 53.00 | **62.26** |
| f128_holitom_r025 | 2700 | 74.33 | 62.00 | 55.22 | **63.85** |
| f128_holitom_r040 | 2700 | 76.00 | 64.67 | 55.11 | **65.26** |
| f128_holitom_tau080_r010 | 2700 | 70.67 | 59.11 | 52.44 | **60.74** |
| f128_kitoke_r002 | 2700 | 59.44 | 49.78 | 43.89 | **51.04** |
| f128_kitoke_r005 | 2700 | 61.33 | 51.44 | 47.22 | **53.33** |
| f128_kitoke_r025 | 2700 | 71.11 | 57.56 | 51.22 | **59.96** |
| f128_kitoke_r040 | 2700 | 73.11 | 61.67 | 52.11 | **62.30** |
| f128_uniform_r002 | 2700 | 61.56 | 51.78 | 46.89 | **53.41** |
| f128_uniform_r005 | 2700 | 67.89 | 54.78 | 50.11 | **57.59** |
| f128_uniform_r010 | 2700 | 70.56 | 56.89 | 51.33 | **59.59** |
| f128_uniform_r015 | 2700 | 71.33 | 60.00 | 51.89 | **61.07** |
| f128_uniform_r025 | 2700 | 72.67 | 63.22 | 53.33 | **63.07** |
| f128_uniform_r040 | 2700 | 76.11 | 61.78 | 54.33 | **64.07** |
| f128_unituniform_r025 | 2700 | 71.33 | 60.00 | 53.11 | **61.48** |
| f128_vidcom2_r002 | 2700 | 49.56 | 49.56 | 44.56 | **47.89** |
| f128_vidcom2_r005 | 2700 | 62.67 | 55.67 | 47.11 | **55.15** |
| f128_vidcom2_r010 | 2700 | 69.44 | 58.56 | 51.78 | **59.93** |
| f128_vidcom2_r015 | 2700 | 72.00 | 58.78 | 52.89 | **61.22** |
| f128_vidcom2_r025 | 2700 | 73.78 | 63.67 | 54.44 | **63.96** |
| f128_vidcom2_r040 | 2700 | 76.00 | 65.89 | 54.44 | **65.44** |
| f256_uniform_r050 | 2700 | 76.56 | 64.33 | 57.00 | **65.96** |

### kitoke_baseline/abl/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| abl_dyseg_delta_nomerge_r025 | 2700 | 70.78 | 60.89 | 52.00 | **61.22** |
| abl_dyseg_grid_dtm0_r025 | 2700 | 73.44 | 62.56 | 53.44 | **63.15** |
| abl_dyseg_grid_nomerge_r025 | 2700 | 71.78 | 62.11 | 52.44 | **62.11** |
| abl_len_delta_r025 | 2700 | 71.78 | 62.44 | 53.22 | **62.48** |
| abl_len_grid_dtm0_r025 | 2700 | 73.56 | 61.89 | 53.22 | **62.89** |
| abl_vcast_r025 | 2700 | 71.89 | 61.33 | 53.89 | **62.37** |

### rap/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| f128_rap_R10E_r010 | 2700 | 71.33 | 61.00 | 52.44 | **61.59** |
| f128_rap_runA_r025 | 2700 | 73.44 | 62.33 | 52.89 | **62.89** |
| f128_rap_runB_noise_r025 | 2700 | 72.78 | 63.56 | 53.33 | **63.22** |
| f128_rap_runB_r025 | 2700 | 73.67 | 65.11 | 53.22 | **64.00** |
| f128_rap_runC_r025 | 2700 | 74.00 | 64.33 | 53.56 | **63.96** |
| f128_rap_runD_r025 | 2700 | 73.78 | 65.11 | 54.33 | **64.41** |
| f128_rap_runE_r025 | 2700 | 73.33 | 65.11 | 54.11 | **64.19** |
| f128_rap_runG_r025 | 2700 | 72.67 | 65.22 | 53.22 | **63.70** |
| f128_rap_runH_r025 | 2700 | 73.67 | 64.89 | 53.11 | **63.89** |
| sub300_runA | 300 | 67.86 | 67.47 | 55.24 | **63.33** |
| sub300_runB | 300 | 72.32 | 67.47 | 54.29 | **64.67** |
| sub300_runC | 300 | 73.21 | 66.27 | 55.24 | **65.00** |

### rap_elastic/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| f128_AS10R_r010 | 2700 | 71.78 | 59.89 | 51.11 | **60.93** |
| f128_AS10U_r010 | 2700 | 70.00 | 57.22 | 52.56 | **59.93** |
| f128_C0_r025 | 2700 | 73.67 | 61.33 | 52.89 | **62.63** |
| f128_ELA_r010 | 2700 | 71.44 | 59.00 | 53.00 | **61.15** |
| f128_ELA_r025 | 2700 | 73.78 | 65.11 | 53.67 | **64.19** |
| f128_ELA_s1_r010 | 2700 | 71.44 | 58.89 | 52.78 | **61.04** |
| f128_ELA_s1_r025 | 2700 | 73.56 | 65.00 | 54.11 | **64.22** |
| f128_M3D-B_r025 | 2700 | 73.33 | 64.33 | 53.89 | **63.85** |
| f128_R02W3_r002 | 2700 | 65.89 | 54.44 | 49.11 | **56.48** |
| f128_SU25_r025 | 2700 | 75.22 | 62.22 | 53.33 | **63.59** |
| sub300_AS10R | 300 | 72.32 | 62.65 | 50.48 | **62.00** |
| sub300_AS10U | 300 | 67.86 | 57.83 | 53.33 | **60.00** |
| sub300_DENSE256 | 300 | 76.79 | 65.06 | 57.14 | **66.67** |

### rap_next/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| f128_rap_R02E_r002 | 2700 | 65.67 | 54.00 | 48.67 | **56.11** |
| f128_rap_R05E_r005 | 2700 | 68.00 | 59.56 | 52.44 | **60.00** |
| f128_rap_R10D_r010 | 2700 | 71.33 | 59.56 | 52.67 | **61.19** |
| f128_rap_R15E_r015 | 2700 | 72.33 | 63.11 | 52.56 | **62.67** |
| f128_rap_R40E_r040 | 2700 | 75.44 | 63.89 | 53.67 | **64.33** |

### rap_night/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| f128_ELA_r001 | 2700 | 62.33 | 54.11 | 49.78 | **55.41** |
| f128_ELA_r002 | 2700 | 64.67 | 54.44 | 49.44 | **56.19** |
| f128_ELA_r005 | 2700 | 67.44 | 58.78 | 51.44 | **59.22** |
| f128_ELA_r015 | 2700 | 72.67 | 62.22 | 53.00 | **62.63** |
| f128_ELA_r040 | 2700 | 75.78 | 63.44 | 54.56 | **64.59** |
| f128_R01E_r001 | 2700 | 61.89 | 52.56 | 49.33 | **54.59** |
| f128_R10KA_r010 | 2700 | 71.33 | 60.11 | 52.78 | **61.41** |
| f128_holitom_r001 | 2700 | 59.00 | 51.67 | 45.56 | **52.07** |
| f128_uniform_r001 | 2700 | 58.00 | 50.22 | 46.33 | **51.52** |

### rap_seeds/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| f128_rap_runD_s1_r025 | 2700 | 73.00 | 65.00 | 54.44 | **64.15** |
| f128_rap_runD_s2_r025 | 2700 | 74.44 | 65.00 | 53.33 | **64.26** |
| f128_rap_runE_s1_r025 | 2700 | 73.11 | 65.44 | 54.89 | **64.48** |
| f128_rap_runE_s2_r025 | 2700 | 73.67 | 64.44 | 54.11 | **64.07** |

### rap_vjepa/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| f128_rap_V10E_r010 | 2700 | 72.33 | 60.56 | 53.33 | **62.07** |

### rap_vjepa25/

| run | n | short | medium | long | overall |
|---|---:|---:|---:|---:|---:|
| f128_rap_V25E_r025 | 2700 | 73.78 | 65.33 | 53.67 | **64.26** |

## 2. Cac bao cao da viet trong tung dot

### rap/KET_QUA.md

```
# Perceiver có neo (RAP) — nhật ký thử nghiệm, mức giữ 25%

Mốc cần vượt: Uniform 25% = 63,07 (VideoMME 2.700 câu, 128 frame, cùng harness).
Số báo cáo chỉ lấy từ bản đủ 2.700 câu (`pred_videomme_f128_rap_<run>_r025*`); bản 300 câu chỉ để quyết định nội bộ.
Tiến độ hàng đợi: `STATUS.txt`. Kế hoạch: `plan.txt`. Chạy lại hàng đợi: xem đầu `queue.sh`.

## Tóm tắt (cập nhật 12/10 00:00 UTC)

| Run | Cách train | 2.700 câu | So với Uniform 63,07 | z |
|---|---|---|---|---|
| A | lỗi caption + đa dạng, có warm-up | 62,89 | -0,18 | -0,28 |
| B | bắt chước bản không nén + đa dạng, có warm-up | 64,00 | +0,93 | +1,72 |
| C | B + câu hỏi trắc nghiệm | 63,96 | +0,89 | +1,77 |
| E | B, không warm-up | 64,19 | +1,12 | +2,04 |
| D | B, mô hình lớn (4 lớp, ±2 unit) | 64,41 | +1,34 | +2,53 |
| G | E, bỏ loss đa dạng | 63,70 | +0,63 | +1,17 |
| H | mô hình lớn, không warm-up, 3.000 bước | 63,89 | +0,82 | +1,53 |

- Train bằng lỗi caption (A) không giúp: mô hình học kiểu viết caption. Train bằng cách bắt chước bản không nén (B, C, D, E, H) cho +0,8 đến +1,3, trung bình khoảng +1,0.
- Chênh giữa các biến thể bắt chước (warm-up, câu hỏi trắc nghiệm, cỡ mô hình) nằm trong mức dao động giữa các lần chạy (~±0,3); chưa biến thể nào hơn hẳn.
- Bỏ loss đa dạng (G) thấp hơn khoảng 0,5 điểm so với có (E), chưa đủ chắc về thống kê.
- Còn cách bản không nén (65,15) khoảng 1 điểm.

## Run A — caption CE + coverage + DPP, warm-up 1.500 bước video giả, tổng 4.000 bước
- Train xong 09/10 15:45 UTC, không lỗi.
- Sai số tả video thật (8 video giữ riêng): Uniform 1,312 → RAP 1,054, thấp hơn cả bản không nén (1,268).
  Tức là mô hình học "kiểu viết" caption của bộ dữ liệu, không phải giữ thêm thông tin hình.
- Độ đa dạng token giữ nguyên (erank ~50, bằng token gốc) → lỗi co cụm của Perceiver cũ không lặp lại.
- 300 câu: 63,33 so với Uniform 64,00 (16 câu chỉ RAP đúng, 18 câu chỉ Uniform đúng, z = -0,34) → không hơn.
- **2.700 câu: 62,89 so với Uniform 63,07** (157 câu chỉ RAP đúng, 162 câu chỉ Uniform đúng, z = -0,28) → ngang Uniform, không hơn.
  Short 73,44 / Medium 62,33 / Long 52,89 (Uniform: 72,67 / 63,22 / 53,33).

Kết luận tạm: lỗi caption giảm không kéo theo điểm hỏi–đáp. Đổi mục tiêu train.

## Run B — chỉ bắt chước bản không nén (KL, bỏ caption CE), cùng lịch với Run A
- Lý do: "thầy" là chính Qwen xem đủ 100% token nên không thể học tắt kiểu viết.
- Ghi chú: trên caption, độ lệch KL của Uniform 25% so với bản không nén vốn đã nhỏ (0,037 trên video thật) → dư địa của mục tiêu này hẹp.
- Train xong 10/10 01:19 UTC, không lỗi. Độ lệch KL so với bản không nén trên video thật giảm một nửa: 0,037 → 0,0175.
  Sai số caption nằm giữa Uniform và bản không nén (1,285; Uniform 1,312; không nén 1,268) → không còn học tắt kiểu viết.
- 300 câu: 64,67 so với Uniform 64,00 (13 câu chỉ RAP đúng, 11 câu chỉ Uniform đúng, z = +0,41) → chưa phân biệt được với Uniform.
- **2.700 câu: 64,00 so với Uniform 63,07 (+0,93)**; 118 câu chỉ RAP đúng, 93 câu chỉ Uniform đúng, z = +1,72 (sát ngưỡng 1,96, chưa đủ để khẳng định chắc).
  Short 73,67 / Medium 65,11 / Long 53,22 (Uniform: 72,67 / 63,22 / 53,33) → lợi ở video ngắn và vừa, video dài không đổi.

## Run C — thêm câu hỏi trắc nghiệm (LLaVA-Video-178K mc QA, 535 video train, 9.058 câu)
- 70% bước sau warm-up là mẫu hỏi–đáp nhiều lượt (CE đáp án + KL với bản không nén), còn lại là caption (chỉ KL).
- Dữ liệu kéo thêm: 3 file `*_mc_v0_1_qa_processed.json` (~26 MB) trong `Datasets/LLaVA-Video-178K-probe/`.
- Chạy thử 8 bước (smokeQA) không lỗi.
- Train xong 10/10 11:37 UTC, không lỗi.
- Lại thấy học tắt, lần này trên hỏi–đáp: sai số đáp án 0,44 → 0,10, thấp hơn nhiều so với bản không nén (0,48),
  trong khi độ lệch KL với bản không nén trên hỏi–đáp lại TĂNG (0,014 → 0,055). Mô hình học thiên lệch đáp án của bộ câu hỏi sinh tự động.
- 300 câu: 65,00 so với Uniform 64,00 (12 so với 9 câu, z = +0,65).
- **2.700 câu: 63,96 so với Uniform 63,07 (+0,89)**; 104 so với 80 câu, z = +1,77. Short 74,00 / Medium 64,33 / Long 53,56.
- So thẳng với Run B: 63,96 và 64,00, 60 so với 61 câu → như nhau. Thêm câu hỏi trắc nghiệm không đem lại gì thêm.

## Run E — như Run B nhưng KHÔNG warm-up video giả (train thẳng trên video thật)
- Mục đích: đo xem warm-up kiểu SCORE đóng góp bao nhiêu.
- Train xong 10/10 20:54 UTC, không lỗi. Độ lệch KL trên video thật 0,0168 (Run B: 0,0175).
- **2.700 câu: 64,19 so với Uniform 63,07 (+1,12)**; 123 so với 93 câu, z = +2,04 → run đầu tiên qua ngưỡng 1,96.
  Short 73,33 / Medium 65,11 / Long 54,11 (Uniform: 72,67 / 63,22 / 53,33).
- So thẳng với Run B (có warm-up): 64,19 và 64,00, 43 so với 38 câu, z = +0,56 → như nhau.
  Warm-up video giả không đem lại lợi thêm với cách train bắt chước; bỏ đi còn nhỉnh hơn ở video dài (54,11 so với 53,22).

## Run D — như Run B nhưng mô hình lớn hơn (4 lớp, mỗi token nhìn ±2 unit; 78 triệu tham số thay vì 45 triệu)
- Train xong 11/10, không lỗi (27 GB bộ nhớ GPU). Độ lệch KL trên video thật 0,0177, bằng Run B.
- **2.700 câu: 64,41 so với Uniform 63,07 (+1,34)**; 119 so với 83 câu, z = +2,53 → kết quả tốt nhất đến giờ, qua ngưỡng 1,96.
  Short 73,78 / Medium 65,11 / Long 54,33 (Uniform: 72,67 / 63,22 / 53,33).
- So thẳng với Run B: 41 so với 30 câu (z = +1,31); phần hơn nằm ở video dài (54,33 so với 53,22, 14 so với 4 câu, z = +2,36).

## Run G — như Run E nhưng bỏ hai thành phần đa dạng (coverage + DPP), chỉ còn bắt chước
- Mục đích: đo đóng góp của phần loss đa dạng theo gợi ý của sếp.
- Train xong 11/10, không lỗi. Độ lệch KL trên video thật 0,0156 (thấp hơn Run E 0,0168), nhưng token đầu ra giống nhau hơn một chút (cos 0,336 so với 0,302 của token gốc).
- **2.700 câu: 63,70 so với Uniform 63,07 (+0,63)**; 114 so với 97 câu, z = +1,17 → không qua ngưỡng.
  Short 72,67 / Medium 65,22 / Long 53,22.
- So thẳng với Run E (có loss đa dạng): 63,70 và 64,19, 39 so với 52 câu, z = -1,36 → có loss đa dạng nhỉnh hơn khoảng 0,5 điểm, nhưng chênh này chưa đủ chắc về thống kê.

## Run H — gộp hai điểm tốt nhất: mô hình lớn (như Run D) + không warm-up (như Run E), 3.000 bước
- Thay cho Run F (train tiếp Run E) vì không đủ thời gian chạy cả hai.
- Train xong 11/10, không lỗi.
- **2.700 câu: 63,89 so với Uniform 63,07 (+0,82)**; 115 so với 93 câu, z = +1,53. Short 73,67 / Medium 64,89 / Long 53,11.
- Không hơn Run D (64,41) hay Run E (64,19): gộp hai điểm tốt không cộng dồn. Phần hơn của Run D ở video dài không lặp lại ở đây.

## Đối chứng nhiễu trên Run B (`runB_noise`)
- Thay phần chỉnh của Perceiver bằng nhiễu ngẫu nhiên cùng độ lớn; nếu điểm tụt về hoặc dưới Uniform thì phần +0,9 là do nội dung học được.
- **2.700 câu: 63,22 so với Uniform 63,07** (27 so với 23 câu, z = +0,57) → nhiễu không giúp, điểm về lại mức Uniform. So với Run B thật: 63,22 và 64,00 (96 so với 117 câu, z = -1,44).
- Vậy phần hơn của Run B đến từ nội dung phần chỉnh học được.

## Bản trình bày chi tiết
- Tab Idea trên https://longvideo.taile09185.ts.net/ : một mục duy nhất "13. Perceiver có neo (RAP)", gồm Phần I tổng quan và Phần II chi tiết từng run, đăng 12/10.
- Script sinh hình và nội dung được lưu ở `outputs/rap/web/`.

## Dọn dẹp 12/10
- Đã xoá checkpoint `runA.pt`, `runC.pt`, `runG.pt`, `runH.pt` (theo yêu cầu). Còn giữ `runB.pt`, `runD.pt`, `runE.pt`.
- File dự đoán, log train và metrics của cả 7 run vẫn còn nguyên.

## Run R10E — cấu hình Run E nhưng ở mức giữ 10% token (4.000 bước, không warm-up, chỉ loss bắt chước + đa dạng)
- Mục đích: xem RAP còn giúp không khi nén mạnh hơn (25% → 10%), nơi Uniform mất nhiều điểm hơn.
- Train xong 12/10, không lỗi. Độ lệch KL trên video thật giảm từ 0,094 (Uniform 10%) xuống 0,038.
- **2.700 câu: 61,59 so với Uniform 10% 59,59 (+2,00)**; 179 so với 125 câu, z = +3,10 → qua ngưỡng 1,96.
  Short 71,33 / Medium 61,00 / Long 52,44 (Uniform: 70,56 / 56,89 / 51,33). Phần hơn chủ yếu ở video trung bình (z = +3,45).
- So với các phương pháp không cần train ở 10%:
  - HoliTom 61,33: ngang nhau (204 so với 197 câu, z = +0,35); video dài HoliTom nhỉnh hơn (53,44 so với 52,44, z = -0,80).
  - VidCom² 59,93: hơn 1,66 điểm (z = +2,09).
  - FastVID 59,30: hơn 2,29 điểm (z = +2,98).
- File: `pred_videomme_f128_rap_R10E_r010.json`, checkpoint `R10E.pt`.
- Run R10D (mô hình lớn ở 10%) đã gỡ khỏi `plan.txt` để nhường GPU, chuyển sang card kia chạy.
```

### rap_seeds/KET_QUA.md

```
# Lặp lại Run E và Run D của RAP với seed khác

Qwen2.5-VL-7B, VideoMME không phụ đề, 128 frame, time_mode=lmms, mức giữ 25%. Chỉ báo run đã đủ 2.700 câu.
Mốc: Uniform 25% = 63,07; Dense = 65,15. z là kiểm định theo cặp (McNemar); |z| > 1,96 là có ý nghĩa.

## Run E

| Seed | Short | Medium | Long | Overall | Chênh so với Uniform 25% | z so với Uniform | Chênh so với seed 0 | z so với seed 0 |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 73,33 | 65,11 | 54,11 | 64,19 | +1,11 | +2,04 | – | – |
| seed 1 | 73,11 | 65,44 | 54,89 | 64,48 | +1,41 | +2,52 | +0,30 | +0,88 |
| seed 2 | 73,67 | 64,44 | 54,11 | 64,07 | +1,00 | +1,87 | -0,11 | -0,33 |
| **Trung bình (3 seed)** | 73,37 | 65,00 | 54,37 | 64,25 | | | | |
| **Độ lệch chuẩn (n−1) (3 seed)** | 0,28 | 0,51 | 0,45 | 0,21 | | | | |

## Run D

| Seed | Short | Medium | Long | Overall | Chênh so với Uniform 25% | z so với Uniform | Chênh so với seed 0 | z so với seed 0 |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 73,78 | 65,11 | 54,33 | 64,41 | +1,33 | +2,53 | – | – |
| seed 1 | 73,00 | 65,00 | 54,44 | 64,15 | +1,07 | +2,03 | -0,26 | -0,86 |
| seed 2 | 74,44 | 65,00 | 53,33 | 64,26 | +1,19 | +2,24 | -0,15 | -0,45 |
| **Trung bình (3 seed)** | 73,74 | 65,04 | 54,04 | 64,27 | | | | |
| **Độ lệch chuẩn (n−1) (3 seed)** | 0,72 | 0,06 | 0,61 | 0,13 | | | | |

## Kết luận

Mức hơn Uniform 25% lặp lại ở cả 6 run: Run E trung bình 64,25 (lệch chuẩn 0,21), Run D trung bình 64,27 (lệch chuẩn 0,13), so với Uniform 63,07; 5/6 run hơn Uniform có ý nghĩa thống kê theo cặp (z > 1,96). Run D hơn Run E 0,02 điểm, trong khi dao động giữa các seed khoảng 0,18 điểm, nên chênh lệch D–E nằm TRONG mức dao động giữa các seed (ngưỡng dùng: 2 lần độ lệch chuẩn gộp; mỗi công thức chỉ có 3 seed).
```

### rap_next/STATUS.txt

```
2026-10-12 12:57:39 QUEUE START (pid 369496), doi outputs/rap_seeds xong
2026-10-13 04:26:48 QUEUE2 START (pid 877243)
2026-10-13 06:04:35 tam dung hang doi seed de chay R40E truoc
2026-10-13 06:04:57 START train R40E (lan 1)
2026-10-13 12:42:51 OK   train R40E
2026-10-13 12:42:51 START eval R40E (lan 1, da co 0/2700)
2026-10-13 15:23:39 DONE eval R40E (2700/2700): overall      2700     1737         0  0.6433
2026-10-13 15:23:39 OK   eval R40E
2026-10-13 15:23:39 chay tiep hang doi seed (Run D seed 2)
2026-10-14 00:16:00 START train R10D (lan 1)
2026-10-14 03:55:58 QUEUE3 START (pid 1699020), doi queue2.sh xong
2026-10-14 03:58:06 QUEUE4 START (pid 1700293)
2026-10-14 04:16:07 tam dung R10D sau checkpoint (step=2500)
2026-10-14 04:16:38 START train R05E (lan 1)
2026-10-14 09:44:44 OK   train R05E
2026-10-14 09:44:44 START eval R05E (lan 1, da co 0/2700)
2026-10-14 10:23:30 QUEUE5 START (pid 1932146): R02E chuyen sang card con lai
2026-10-14 10:29:52 QUEUE6 START (pid 1939367), doi queue5.sh xong
2026-10-14 10:42:35 QUEUE7 START (pid 1956175): R05E -> R02E -> R10D -> R15E
2026-10-14 12:02:37 OK   train R05E
2026-10-14 12:02:39 DONE eval R05E (2700/2700): overall      2700     1620         0  0.6000
2026-10-14 12:02:40 OK   eval R05E
2026-10-14 12:02:40 START train R02E (lan 1)
2026-10-14 17:50:25 OK   train R02E
2026-10-14 17:50:25 START eval R02E (lan 1, da co 0/2700)
2026-10-14 19:57:36 DONE eval R02E (2700/2700): overall      2700     1515         0  0.5611
2026-10-14 19:57:36 OK   eval R02E
2026-10-14 19:57:36 START train R10D (lan 1)
2026-10-14 22:14:01 OK   train R10D
2026-10-14 22:14:01 START eval R10D (lan 1, da co 0/2700)
2026-10-15 00:23:47 DONE eval R10D (2700/2700): overall      2700     1652         0  0.6119
2026-10-15 00:23:47 OK   eval R10D
2026-10-15 00:23:47 START train R15E (lan 1)
2026-10-15 06:05:58 OK   train R15E
2026-10-15 06:05:58 START eval R15E (lan 1, da co 0/2700)
2026-10-15 08:19:52 START eval R15E (lan 2, da co 2699/2700)
2026-10-15 08:28:12 DONE eval R15E (2700/2700): overall      2700     1692         0  0.6267
2026-10-15 08:28:12 OK   eval R15E
2026-10-15 08:28:12 QUEUE7 DONE
```

### rap_elastic/KET_QUA.txt

```
KET QUA HANG DOI rap_elastic -- cap nhat 2026-10-22 07:34 UTC
Tien do chi tiet: outputs/rap_elastic/STATUS.txt. Ke hoach: outputs/rap_elastic/queue.sh (dau file).
So 'sub300' chi tren 300 cau (de quyet dinh noi bo, nhieu nhieu). So 'full' tren du 2.700 cau.
z: McNemar so cap tung cau, |z| > 1,96 la co y nghia.

[Q1a sub300] Uniform 10% + chia lai neo theo canh (khong train)  (300 cau)
    short=67.86  medium=57.83  long=53.33  overall=60.00
    so voi Uniform10 : +0.67 diem (z=+0.38, 300 cau chung)

[Q1b sub300] RAP R10E + chia lai neo theo canh (khong train lai)  (300 cau)
    short=72.32  medium=62.65  long=50.48  overall=62.00
    so voi R10E      : +0.67 diem (z=+0.32, 300 cau chung)
    so voi Uniform10 : +2.67 diem (z=+1.13, 300 cau chung)

[Q1a full] Uniform 10% + chia lai neo theo canh  (2700 cau)
    short=70.00  medium=57.22  long=52.56  overall=59.93
    so voi Uniform10 : +0.33 diem (z=+0.50, 2700 cau chung)

[Q1b full] RAP R10E + chia lai neo theo canh  (2700 cau)
    short=71.78  medium=59.89  long=51.11  overall=60.93
    so voi R10E      : -0.67 diem (z=-1.01, 2700 cau chung)
    so voi Uniform10 : +1.33 diem (z=+1.74, 2700 cau chung)
    so voi HoliTom10 : -0.41 diem (z=-0.51, 2700 cau chung)

[Q3 ELA@25] RAP-Elastic (1 checkpoint), cham o 25%  (2700 cau)
    short=73.78  medium=65.11  long=53.67  overall=64.19
    so voi RunE25    : +0.00 diem (z=+0.00, 2700 cau chung)
    so voi Uniform25 : +1.11 diem (z=+2.00, 2700 cau chung)
    so voi HoliTom25 : +0.33 diem (z=+0.51, 2700 cau chung)

[Q3 ELA@10] RAP-Elastic (1 checkpoint), cham o 10%  (2700 cau)
    short=71.44  medium=59.00  long=53.00  overall=61.15
    so voi R10E      : -0.44 diem (z=-1.13, 2700 cau chung)
    so voi Uniform10 : +1.56 diem (z=+2.48, 2700 cau chung)
    so voi HoliTom10 : -0.19 diem (z=-0.25, 2700 cau chung)

[Q4 R02W3] RAP 2%, cua so +-3 unit  (2700 cau)
    short=65.89  medium=54.44  long=49.11  overall=56.48
    so voi R02E      : +0.37 diem (z=+0.88, 2700 cau chung)
    so voi Uniform02 : +3.07 diem (z=+4.14, 2700 cau chung)
    so voi HoliTom02 : +1.11 diem (z=+1.30, 2700 cau chung)

[Q5b sub300] Dense 256 frame (thay manh hon?)  (300 cau)
    short=76.79  medium=65.06  long=57.14  overall=66.67
    so voi Dense128  : -1.00 diem (z=-0.51, 300 cau chung)

[F2 DENSE256] Dense 256 frame, du 2.700 cau: CHUA CO
[Q5c T256] Tro nen 256 frame xuong 12,5% (~4.000 token), thay = dense 256: CHUA CO
[Q5a R05W3] RAP 5%, cua so +-3 unit: CHUA CO
[Q6 R10S] RAP 10% train voi neo chia theo canh: CHUA CO
[H1 SU10] Uniform 10% + token tom tat moi canh (HTP-LVU, khong train)  (294 cau)
    short=68.71  medium=nan  long=nan  overall=68.71
    so voi Uniform10 : -3.40 diem (z=-1.67, 294 cau chung)
    so voi HoliTom10 : -1.02 diem (z=-0.45, 294 cau chung)

[H2 RSS10] RAP 10% + neo chi nhin cung canh + token tom tat (HTP-LVU): CHUA CO
[C0] Medoid cum k-means trong tung canh, khong train, 25% (HTP-LVU 1.2 + MAP-3D tube)  (2700 cau)
    short=73.67  medium=61.33  long=52.89  overall=62.63
    so voi Uniform25 : -0.44 diem (z=-0.79, 2700 cau chung)
    so voi RunE25    : -1.56 diem (z=-2.42, 2700 cau chung)
    so voi HoliTom25 : -1.22 diem (z=-2.03, 2700 cau chung)

[Q6-1 M3D-B] RAP 25% + cross-attn phat khoang cach 3D (MAP-3D 3.3), KV +-3  (2700 cau)
    short=73.33  medium=64.33  long=53.89  overall=63.85
    so voi RunE25    : -0.33 diem (z=-1.10, 2700 cau chung)
    so voi Uniform25 : +0.78 diem (z=+1.48, 2700 cau chung)
    so voi HoliTom25 : +0.00 diem (z=+0.00, 2700 cau chung)

[Q6-2 SU25] Uniform 25% + token tom tat moi canh (HTP-LVU residual bypass, khong train)  (2700 cau)
    short=75.22  medium=62.22  long=53.33  overall=63.59
    so voi Uniform25 : +0.52 diem (z=+0.94, 2700 cau chung)
    so voi HoliTom25 : -0.26 diem (z=-0.43, 2700 cau chung)
    so voi RunE25    : -0.59 diem (z=-0.91, 2700 cau chung)

[Q6-3 M3D-BD] M3D-B + Deformable 3D Tube Adapter (MAP-3D 3.2): CHUA CO
[Q6-4a V10E_s1] V-JEPA pretrain + KL 10%, seed 1 (kiem chung V10E): CHUA CO
[Q6-4b R10E7k] Doi chung: RAP 10% KL 7.000 buoc, khong V-JEPA: CHUA CO
[Q6-5 RSS25] RAP 25% + neo chi nhin cung canh (HTP-LVU S1) + token tom tat: CHUA CO
[Q6-5 RSW25] RAP 25% + neo chi nhin cung canh (HTP-LVU S1), khong token tom tat: CHUA CO
[Q6-6 M3DF@25] MAP-3D day du: 3.3 + 3.2 + Elastic 4.1, cham 25%: CHUA CO
[Q6-6 M3DF@10] MAP-3D day du: 3.3 + 3.2 + Elastic 4.1, cham 10%: CHUA CO
[F1 ELA_s1@25] RAP-Elastic seed 1, 25%  (2700 cau)
    short=73.56  medium=65.00  long=54.11  overall=64.22
    so voi RunE25    : +0.04 diem (z=+0.10, 2700 cau chung)
    so voi Uniform25 : +1.15 diem (z=+2.06, 2700 cau chung)

[F1 ELA_s1@10] RAP-Elastic seed 1, 10%  (2700 cau)
    short=71.44  medium=58.89  long=52.78  overall=61.04
    so voi R10E      : -0.56 diem (z=-1.30, 2700 cau chung)
    so voi Uniform10 : +1.44 diem (z=+2.24, 2700 cau chung)
```

### rap_night/KET_QUA.txt

```
KET QUA rap_night -- cap nhat 2026-10-23 08:41 UTC
Tien do: outputs/rap_night/STATUS.txt. z: McNemar so cap tung cau, |z| >= 1,96 la co y nghia.

[U01] Uniform 1% (khong train)  (2700 cau)
    short=58.00  medium=50.22  long=46.33  overall=51.52

[R01E] RAP 1%, cong thuc Run E  (2700 cau)
    short=61.89  medium=52.56  long=49.33  overall=54.59
    so voi Uniform01 : +3.07 diem (z=+4.03, 2700 cau chung)
    so voi HoliTom01 : +2.52 diem (z=+2.86, 2700 cau chung)

[H01] HoliTom 1% (w/o M)  (2700 cau)
    short=59.00  medium=51.67  long=45.56  overall=52.07
    so voi Uniform01 : +0.56 diem (z=+0.75, 2700 cau chung)

[R10KA] R10E nhung Perceiver chi nhin token neo (ablation)  (2700 cau)
    short=71.33  medium=60.11  long=52.78  overall=61.41
    so voi R10E      : -0.19 diem (z=-0.50, 2700 cau chung)
    so voi Uniform10 : +1.81 diem (z=+2.78, 2700 cau chung)
    so voi HoliTom10 : +0.07 diem (z=+0.10, 2700 cau chung)

[ELA@2] Elastic (1 checkpoint) cham o 2%  (2700 cau)
    short=64.67  medium=54.44  long=49.44  overall=56.19
    so voi R02E      : +0.07 diem (z=+0.15, 2700 cau chung)
    so voi Uniform02 : +2.78 diem (z=+3.78, 2700 cau chung)
    so voi HoliTom02 : +0.81 diem (z=+0.97, 2700 cau chung)

[ELA@5] Elastic cham o 5%  (2700 cau)
    short=67.44  medium=58.78  long=51.44  overall=59.22
    so voi R05E      : -0.78 diem (z=-1.62, 2700 cau chung)
    so voi Uniform05 : +1.63 diem (z=+2.28, 2700 cau chung)
    so voi HoliTom05 : +0.52 diem (z=+0.64, 2700 cau chung)

[ELA@15] Elastic cham o 15%  (2700 cau)
    short=72.67  medium=62.22  long=53.00  overall=62.63
    so voi R15E      : -0.04 diem (z=-0.09, 2700 cau chung)
    so voi Uniform15 : +1.56 diem (z=+2.58, 2700 cau chung)
    so voi HoliTom15 : +0.37 diem (z=+0.53, 2700 cau chung)

[ELA@40] Elastic cham o 40%  (2700 cau)
    short=75.78  medium=63.44  long=54.56  overall=64.59
    so voi R40E      : +0.26 diem (z=+0.80, 2700 cau chung)
    so voi Uniform40 : +0.52 diem (z=+1.03, 2700 cau chung)
    so voi HoliTom40 : -0.67 diem (z=-1.14, 2700 cau chung)

[ELA@1] Elastic cham o 1% (khong train o muc nay)  (2700 cau)
    short=62.33  medium=54.11  long=49.78  overall=55.41
    so voi R01E      : +0.81 diem (z=+1.51, 2700 cau chung)
    so voi Uniform01 : +3.89 diem (z=+5.27, 2700 cau chung)
    so voi HoliTom01 : +3.33 diem (z=+3.90, 2700 cau chung)

[R01W3] RAP 1%, cua so +-3: CHUA CO
```

### rap_night/TONG_HOP_NEN_1PCT.txt

```
TỔNG HỢP KẾT QUẢ: NÉN CỰC SÂU 1% TRÊN VIDEO-MME (Qwen2.5-VL-7B, 128 frame)
Cập nhật: 23/10/2026, 01:40 UTC

1. Thiết lập
- Mức giữ 1%: còn khoảng 161 / 16.128 token thị giác cho mỗi video.
- Chấm trên toàn bộ Video-MME (2.700 câu), chia theo độ dài video: Ngắn / Trung bình / Dài.
- So sánh từng câu theo cặp (McNemar); |z| ≥ 1,96 nghĩa là chênh lệch có ý nghĩa thống kê.
- RAP (R01E): train 4.000 bước bằng KL tới Qwen nhìn 100% token (công thức Run E), cửa sổ ±1.

2. Kết quả ở mức 1%

  Phương pháp        Overall   Ngắn    Trung bình   Dài
  Uniform            51,52     58,00   50,22        46,33
  HoliTom            52,07     59,00   51,67        45,56
  RAP (R01E)         54,59     61,89   52,56        49,33

  So sánh theo cặp:
  - RAP so với HoliTom:     +2,52 điểm (z = +2,86) → có ý nghĩa
  - RAP so với Uniform:     +3,07 điểm (z = +4,03) → có ý nghĩa
  - HoliTom so với Uniform: +0,56 điểm (z = +0,75) → không có ý nghĩa

3. Nhận xét chính
- Đây là lần đầu RAP hơn HoliTom có ý nghĩa thống kê. Ở các mức 2–25% trước đây,
  RAP chưa lần nào hơn HoliTom có ý nghĩa (tốt nhất là z = 1,63 ở mức 5%).
- Ở mức 1%, HoliTom gần như mất hết lợi thế so với Uniform; ở nhóm video dài,
  HoliTom còn thấp hơn Uniform (45,56 so với 46,33). Điều này khớp với nhận xét
  của KiToke: các phương pháp heuristic sụp khi nén cực sâu.
- Ở nhóm video dài, RAP hơn HoliTom 3,8 điểm (49,33 so với 45,56).
- Khoảng cách RAP – HoliTom tăng dần khi nén sâu hơn:
    10%: +0,26 | 5%: +1,30 | 2%: +0,74 | 1%: +2,52 (z = 2,86)
  → Đây là bằng chứng mạnh nhất cho hướng "nén cực sâu cho video dài".

4. Quá trình train R01E
- KL trên tập kiểm tra: 0,2917 (bằng Uniform, lúc chưa train) → 0,1611 (bước 4.000).
- Khoảng 4,5 giây/bước, dùng 21,4 GB GPU.

5. Đang chạy tiếp (server cũ, 1 GPU, hàng đợi tự động)
- R10KA (ablation): giống R10E nhưng Perceiver CHỈ được nhìn token neo, không nhìn
  token bị bỏ. Mục đích: xem phần lợi của RAP có đến từ việc gom thông tin của token bị bỏ hay không.
  Kết quả train sơ bộ (KL trên tập kiểm tra, mức 10%):
    Uniform 0,0940 | R10E 0,0376 (bước 4.000) | R10KA 0,0387 (bước 3.750)
  → KL gần như bằng R10E. Nếu điểm chấm cũng xấp xỉ thì phần lớn lợi ích đến từ việc
    tinh chỉnh chính các token neo, không phải từ việc gom token bị bỏ. Cần chờ điểm chấm.
  Dự kiến train xong khoảng 02:00 UTC, chấm xong khoảng 03:00–04:00 UTC.
- Sau đó: chấm checkpoint Elastic (một checkpoint cho mọi mức) ở 2%, 5%, 15%, 40%, 1%.
- Cuối cùng: R01W3 (RAP 1%, cửa sổ ±3).
- File kết quả tự cập nhật: outputs/rap_night/KET_QUA.txt
```

### kitoke_baseline/REPORT.md

```
# Bao cao thi nghiem qua dem (tu dong cap nhat)

Cap nhat luc: 2026-10-11 20:09:33 (gio may chu, UTC)

VideoMME khong subtitle, Qwen2.5-VL-7B, time_mode=lmms tru khi ghi khac. Run chua du 2700 cau la DANG CHAY: so cua no chua phai so cuoi (file chay theo thu tu short -> medium -> long nen overall tam thoi bi cao).

## Trang thai hang doi

```
2026-10-06 10:05:16 WAIT uniform_r025 (pid 328214)
2026-10-07 02:01:29 START attn_r025 (chay tay, tung buoc)
2026-10-07 06:55:37 DONE attn_r025 (rc=0)
2026-10-07 06:55:53 START uniform_r040 (chay tay, tung buoc)
2026-10-07 07:58:20 STOP uniform_r040 theo yeu cau nguoi dung (1274/2700, co the resume)
2026-10-07 09:58:05 START uniform r040(resume)/r015/r010, 128 frame (moc baseline co dinh)
2026-10-07 16:19:45 DONE uniform r040/r015/r010 (rc=0)
2026-10-07 16:20:26 START dsta_v2 train (128 frame, lmms, 1049 cap video 1-3 phut, 2000 step)
2026-10-07 20:31:19 DONE dsta_v2 train (rc=0)
2026-10-07 20:31:54 START pred_videomme_f128_dsta_v2_r025
2026-10-07 23:48:54 DONE pred_videomme_f128_dsta_v2_r025 (rc=0)
2026-10-07 23:48:54 START pred_videomme_f128_unituniform_r025
2026-10-08 03:07:53 DONE pred_videomme_f128_unituniform_r025 (rc=0)
2026-10-08 03:08:16 START pred_videomme_f128_dyseg_r025 (fill_budget=on)
2026-10-08 04:22:18 START pred_videomme_f128_dyseg_r040 (fill_budget=on, card container vlm-hoang03)
2026-10-08 05:32:54 DONE pred_videomme_f128_dyseg_r025 (rc=0)
2026-10-08 06:51:46 START pred_videomme_f128_dyseg_r010 (fill_budget=on, card nay)
2026-10-08 07:24:35 DONE pred_videomme_f128_dyseg_r040 (rc=0, 2700/2700, cau 613-1 chay bu)
2026-10-08 07:36:40 START pred_videomme_f128_dyseg_grid_r025 (salience_mode=grid, fill_budget=on, container vlm-hoang03)
2026-10-08 09:06:07 DONE pred_videomme_f128_dyseg_r010 (rc=0)
2026-10-08 09:29:26 START nen cuc manh: uniform r005/r002 roi kitoke r005/r002 (128 frame, lmms)
2026-10-08 10:07:05 DONE pred_videomme_f128_dyseg_grid_r025 (rc=0)
2026-10-08 10:50:25 night_queue: cho job nen cuc manh xong
2026-10-08 17:55:32 DONE nen cuc manh (uniform rc=0, kitoke rc=0)
2026-10-08 17:56:41 SKIP pred_videomme_f128_uniform_r005 (da du 2700 cau)
2026-10-08 17:56:41 SKIP pred_videomme_f128_uniform_r002 (da du 2700 cau)
2026-10-08 17:56:41 SKIP pred_videomme_f128_kitoke_r005 (da du 2700 cau)
2026-10-08 17:56:41 SKIP pred_videomme_f128_kitoke_r002 (da du 2700 cau)
2026-10-08 17:56:41 START pred_videomme_f128_dyseg_r005 (fill_budget=on)
2026-10-08 20:03:28 DONE pred_videomme_f128_dyseg_r005 (rc=0, 2700/2700)
2026-10-08 20:03:38 START pred_videomme_f128_dyseg_r002 (fill_budget=on)
2026-10-08 22:10:44 DONE pred_videomme_f128_dyseg_r002 (rc=0, 2700/2700)
2026-10-08 22:10:54 START pred_videomme_f128_attn_r010 (loqo + self)
2026-10-09 01:34:45 PAUSE night_queue (nguoi dung yeu cau); attn_r010 dung o 2049/2700
2026-10-09 04:38:51 START pred_videomme_f128_fastvid_r025 (FastVID ma chinh thuc, danh nghia 0.2625 ~ thuc giu 25%, 128 frame, lmms)
2026-10-09 07:24:44 DONE pred_videomme_f128_fastvid_r025 (rc=0, 2700/2700)
2026-10-09 08:06:13 RESUME pred_videomme_f128_attn_r010 (chi buoc nay, tu cau 2050)
2026-10-09 09:41:36 DONE pred_videomme_f128_attn_r010 (rc=0, loqo 2700/2700)
2026-10-09 10:02:07 START fastvid_calibrate (60 video, cac muc 40/15/10/5/2)
2026-10-09 10:17:20 weekend_queue: bat dau
2026-10-09 10:17:20 BO QUA fastvid 0.10: buoc hieu chinh khong tim duoc ti le danh nghia
2026-10-09 10:17:20 BO QUA fastvid 0.40: buoc hieu chinh khong tim duoc ti le danh nghia
2026-10-09 10:17:20 START pred_videomme_f128_holitom_r025 (HoliTom w/o M, retention 0.25)
2026-10-09 10:20:28 weekend_queue: dung de sua loi doc file hieu chinh (khoa 0.10/0.40); holitom_r025 dung o 52/2700, se tu chay tiep
2026-10-09 10:20:35 weekend_queue: bat dau
2026-10-09 10:20:35 START pred_videomme_f128_fastvid_r010 (FastVID, muc tieu 0.10, danh nghia 0.1015)
2026-10-09 12:53:59 DONE pred_videomme_f128_fastvid_r010 (rc=0, 2700/2700, SKIP=0)
2026-10-09 12:54:12 START pred_videomme_f128_fastvid_r040 (FastVID, muc tieu 0.40, danh nghia 0.4524)
2026-10-09 15:50:03 DONE pred_videomme_f128_fastvid_r040 (rc=0, 2700/2700, SKIP=0)
2026-10-09 15:50:16 START pred_videomme_f128_holitom_r025 (HoliTom w/o M, retention 0.25)
2026-10-09 18:32:10 DONE pred_videomme_f128_holitom_r025 (rc=0, 2700/2700, SKIP=0)
2026-10-09 18:32:22 START pred_videomme_f128_holitom_r010 (HoliTom w/o M, retention 0.10)
2026-10-09 21:14:58 DONE pred_videomme_f128_holitom_r010 (rc=0, 2700/2700, SKIP=0)
2026-10-09 21:15:10 START pred_videomme_f128_holitom_r040 (HoliTom w/o M, retention 0.40)
2026-10-10 00:10:41 DONE pred_videomme_f128_holitom_r040 (rc=0, 2700/2700, SKIP=0)
2026-10-10 00:10:54 START pred_videomme_f128_fastvid_r015 (FastVID, muc tieu 0.15, danh nghia 0.1535)
2026-10-10 02:52:02 DONE pred_videomme_f128_fastvid_r015 (rc=0, 2700/2700, SKIP=0)
2026-10-10 02:52:16 START pred_videomme_f128_holitom_r015 (HoliTom w/o M, retention 0.15)
2026-10-10 05:34:23 DONE pred_videomme_f128_holitom_r015 (rc=0, 2700/2700, SKIP=0)
2026-10-10 05:34:38 START pred_videomme_f128_fastvid_r005 (FastVID, muc tieu 0.05, danh nghia 0.0527)
2026-10-10 08:09:50 DONE pred_videomme_f128_fastvid_r005 (rc=0, 2700/2700, SKIP=0)
2026-10-10 08:10:04 START pred_videomme_f128_holitom_r005 (HoliTom w/o M, retention 0.05)
2026-10-10 10:44:51 DONE pred_videomme_f128_holitom_r005 (rc=0, 2700/2700, SKIP=0)
2026-10-10 10:45:05 START pred_videomme_f128_fastvid_r002 (FastVID, muc tieu 0.02, danh nghia 0.0206)
2026-10-10 13:21:01 DONE pred_videomme_f128_fastvid_r002 (rc=0, 2700/2700, SKIP=0)
2026-10-10 13:21:56 START pred_videomme_f128_holitom_r002 (HoliTom w/o M, retention 0.02)
2026-10-10 15:59:27 DONE pred_videomme_f128_holitom_r002 (rc=0, 2700/2700, SKIP=0)
2026-10-10 15:59:42 weekend_queue: XONG TOAN BO
2026-10-10 17:35:58 holitom_ablation_queue: bat dau
2026-10-10 17:35:58 START pred_videomme_f128_holitom_notemporal_r010 (HoliTom w/o M, retention 0.10, tau 1.01)
2026-10-10 17:49:39 holitom_ablation_queue: TAM DUNG o 141/2700 (uu tien VidCom2 + DyCoke theo yeu cau nguoi dung), se chay tiep sau
2026-10-10 17:55:35 score_table_queue: bat dau
2026-10-10 17:55:35 START pred_videomme_f128_vidcom2_r010 (mode=vidcom2 retention=0.10 )
2026-10-10 19:58:04 DONE pred_videomme_f128_vidcom2_r010 (rc=0, 2700/2700, SKIP=0)
2026-10-10 19:58:19 START pred_videomme_f128_vidcom2_r025 (mode=vidcom2 retention=0.25 )
2026-10-10 22:19:35 DONE pred_videomme_f128_vidcom2_r025 (rc=0, 2700/2700, SKIP=0)
2026-10-10 22:19:50 START pred_videomme_f128_vidcom2_r040 (mode=vidcom2 retention=0.40 )
2026-10-11 00:43:35 DONE pred_videomme_f128_vidcom2_r040 (rc=0, 2700/2700, SKIP=0)
2026-10-11 00:43:51 START pred_videomme_f128_dycoke_r040 (mode=dycoke retention=0.40 +dycoke_k=0.817)
2026-10-11 03:07:22 DONE pred_videomme_f128_dycoke_r040 (rc=0, 2700/2700, SKIP=0)
2026-10-11 03:07:45 START pred_videomme_f128_dycoke_r025 (mode=dycoke retention=0.25 +dycoke_k=1.0)
2026-10-11 05:36:16 DONE pred_videomme_f128_dycoke_r025 (rc=0, 2700/2700, SKIP=0)
2026-10-11 05:37:30 START pred_videomme_f128_vidcom2_r015 (mode=vidcom2 retention=0.15 )
2026-10-11 07:55:01 DONE pred_videomme_f128_vidcom2_r015 (rc=0, 2700/2700, SKIP=0)
2026-10-11 07:55:18 START pred_videomme_f128_vidcom2_r005 (mode=vidcom2 retention=0.05 )
2026-10-11 10:00:06 DONE pred_videomme_f128_vidcom2_r005 (rc=0, 2700/2700, SKIP=0)
2026-10-11 10:00:23 START pred_videomme_f128_vidcom2_r002 (mode=vidcom2 retention=0.02 )
2026-10-11 12:02:25 DONE pred_videomme_f128_vidcom2_r002 (rc=0, 2700/2700, SKIP=0)
2026-10-11 12:02:42 score_table_queue: XONG, chay tiep ablation HoliTom
2026-10-11 12:02:42 holitom_ablation_queue: bat dau
2026-10-11 12:02:42 START pred_videomme_f128_holitom_notemporal_r010 (HoliTom w/o M, retention 0.10, tau 1.01)
2026-10-11 14:35:42 DONE pred_videomme_f128_holitom_notemporal_r010 (rc=0, 2700/2700, SKIP=0)
2026-10-11 14:35:58 START pred_videomme_f128_holitom_notemporal_r025 (HoliTom w/o M, retention 0.25, tau 1.01)
2026-10-11 17:25:55 DONE pred_videomme_f128_holitom_notemporal_r025 (rc=0, 2700/2700, SKIP=0)
2026-10-11 17:26:12 START pred_videomme_f128_holitom_tau080_r010 (HoliTom w/o M, retention 0.10, tau 0.80)
2026-10-11 20:09:21 DONE pred_videomme_f128_holitom_tau080_r010 (rc=0, 2700/2700, SKIP=0)
```

## Ket qua

| Cau hinh | Da chay | Short | Medium | Long | Overall |
|---|---|---|---|---|---|
| Dense 100% (128 frame, ~16K token) | 2700/2700 XONG | 76.89 | 64.44 | 54.11 | 65.15 |
| KiToKe 40% | 2700/2700 XONG | 73.11 | 61.67 | 52.11 | 62.30 |
| KiToKe 25% | 2700/2700 XONG | 71.11 | 57.56 | 51.22 | 59.96 |
| Uniform 40% | 2700/2700 XONG | 76.11 | 61.78 | 54.33 | 64.07 |
| Uniform 25% | 2700/2700 XONG | 72.67 | 63.22 | 53.33 | 63.07 |
| Uniform 15% | 2700/2700 XONG | 71.33 | 60.00 | 51.89 | 61.07 |
| Uniform 10% | 2700/2700 XONG | 70.56 | 56.89 | 51.33 | 59.59 |
| DySeg+STPrune 40% | 2700/2700 XONG | 73.22 | 61.78 | 54.56 | 63.19 |
| DySeg+STPrune 25% | 2700/2700 XONG | 70.78 | 61.33 | 52.78 | 61.63 |
| DySeg+STPrune 10% | 2700/2700 XONG | 66.11 | 58.22 | 50.89 | 58.41 |
| DSTA v2 25% (train lai 07/10) | 2700/2700 XONG | 67.56 | 59.89 | 53.22 | 60.22 |
| Uniform theo unit 25% (doi chung DSTA) | 2700/2700 XONG | 71.33 | 60.00 | 53.11 | 61.48 |
| Attention 25%, KHONG thay cau hoi (loqo) | 2700/2700 XONG | 74.22 | 62.22 | 52.78 | 63.07 |
| Attention 25%, thay cau hoi (self) | 2700/2700 XONG | 76.22 | 64.00 | 53.33 | 64.52 |
| 256 frame, Uniform 50% (~16K token vao LLM) | 2700/2700 XONG | 76.56 | 64.33 | 57.00 | 65.96 |
| Dense 100%, vi tri thoi gian kieu cu (loi RoPE) | 2700/2700 XONG | 73.22 | 64.89 | 54.67 | 64.26 |
| Uniform 5% | 2700/2700 XONG | 67.89 | 54.78 | 50.11 | 57.59 |
| Uniform 2% | 2700/2700 XONG | 61.56 | 51.78 | 46.89 | 53.41 |
| KiToKe 5% | 2700/2700 XONG | 61.33 | 51.44 | 47.22 | 53.33 |
| KiToKe 2% | 2700/2700 XONG | 59.44 | 49.78 | 43.89 | 51.04 |
| DySeg+STPrune 5% | 2700/2700 XONG | 64.22 | 54.89 | 48.00 | 55.70 |
| DySeg+STPrune 2% | 2700/2700 XONG | 59.22 | 50.56 | 45.78 | 51.85 |
| Attention 10%, KHONG thay cau hoi (loqo) | 2700/2700 XONG | 71.56 | 58.89 | 50.89 | 60.44 |
| Attention 10%, thay cau hoi (self) | 2700/2700 XONG | 75.44 | 62.11 | 53.00 | 63.52 |
| DySeg+STPrune 25% (salience=grid) | 2700/2700 XONG | 72.11 | 62.44 | 52.67 | 62.41 |
| FastVID 25% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 74.22 | 60.89 | 53.89 | 63.00 |
| VidCom2 40% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 76.00 | 65.89 | 54.44 | 65.44 |
| VidCom2 25% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 73.78 | 63.67 | 54.44 | 63.96 |
| VidCom2 15% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 72.00 | 58.78 | 52.89 | 61.22 |
| VidCom2 10% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 69.44 | 58.56 | 51.78 | 59.93 |
| VidCom2 5% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 62.67 | 55.67 | 47.11 | 55.15 |
| VidCom2 2% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 49.56 | 49.56 | 44.56 | 47.89 |
| DyCoke chi TTM 40% (port sang Qwen2.5-VL) | 2700/2700 XONG | 75.67 | 63.11 | 54.33 | 64.37 |
| DyCoke chi TTM 25% (port sang Qwen2.5-VL) | 2700/2700 XONG | 71.33 | 58.44 | 51.78 | 60.52 |
| HoliTom w/o M 10% bo gop thoi gian | 2700/2700 XONG | 69.44 | 59.11 | 52.44 | 60.33 |
| HoliTom w/o M 25% bo gop thoi gian | 2700/2700 XONG | 75.22 | 62.67 | 54.44 | 64.11 |
| HoliTom w/o M 10% tau 0.80 | 2700/2700 XONG | 70.67 | 59.11 | 52.44 | 60.74 |
| FastVID 40% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 74.89 | 64.56 | 55.89 | 65.11 |
| FastVID 15% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 72.78 | 59.22 | 51.56 | 61.19 |
| FastVID 10% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 69.56 | 57.56 | 50.78 | 59.30 |
| FastVID 5% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 66.00 | 53.89 | 50.00 | 56.63 |
| FastVID 2% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700/2700 XONG | 61.11 | 50.33 | 46.22 | 52.56 |
| HoliTom w/o M 40% (port sang Qwen2.5-VL) | 2700/2700 XONG | 76.00 | 64.67 | 55.11 | 65.26 |
| HoliTom w/o M 25% (port sang Qwen2.5-VL) | 2700/2700 XONG | 74.33 | 62.00 | 55.22 | 63.85 |
| HoliTom w/o M 15% (port sang Qwen2.5-VL) | 2700/2700 XONG | 72.22 | 61.56 | 53.00 | 62.26 |
| HoliTom w/o M 10% (port sang Qwen2.5-VL) | 2700/2700 XONG | 71.00 | 59.56 | 53.44 | 61.33 |
| HoliTom w/o M 5% (port sang Qwen2.5-VL) | 2700/2700 XONG | 68.33 | 57.22 | 50.56 | 58.70 |
| HoliTom w/o M 2% (port sang Qwen2.5-VL) | 2700/2700 XONG | 64.67 | 53.11 | 48.33 | 55.37 |

## So cap tren cac cau chung (McNemar)

|z| > 1.96 thi khac biet co y nghia thong ke (p < 0.05). Dau duong = A tot hon B.

| A | B | So cau chung | Acc A | Acc B | Chi A dung | Chi B dung | z |
|---|---|---|---|---|---|---|---|
| Dense 100% (128 frame, ~16K token) | Uniform 40% | 2700 | 65.15 | 64.07 | 129 | 100 | +1.92 |
| Dense 100% (128 frame, ~16K token) | Uniform 25% | 2700 | 65.15 | 63.07 | 182 | 126 | +3.19 |
| Dense 100% (128 frame, ~16K token) | Uniform 15% | 2700 | 65.15 | 61.07 | 258 | 148 | +5.46 |
| Dense 100% (128 frame, ~16K token) | Uniform 10% | 2700 | 65.15 | 59.59 | 309 | 159 | +6.93 |
| DySeg+STPrune 40% | Uniform 40% | 2700 | 63.19 | 64.07 | 123 | 147 | -1.46 |
| DySeg+STPrune 25% | Uniform 25% | 2700 | 61.63 | 63.07 | 134 | 173 | -2.23 |
| DySeg+STPrune 10% | Uniform 10% | 2700 | 58.41 | 59.59 | 174 | 206 | -1.64 |
| DSTA v2 25% (train lai 07/10) | Uniform 25% | 2700 | 60.22 | 63.07 | 186 | 263 | -3.63 |
| DSTA v2 25% (train lai 07/10) | Uniform theo unit 25% (doi chung DSTA) | 2700 | 60.22 | 61.48 | 221 | 255 | -1.56 |
| Uniform 25% | KiToKe 25% | 2700 | 63.07 | 59.96 | 206 | 122 | +4.64 |
| Uniform 40% | KiToKe 40% | 2700 | 64.07 | 62.30 | 161 | 113 | +2.90 |
| Attention 25%, KHONG thay cau hoi (loqo) | Uniform 25% | 2700 | 63.07 | 63.07 | 158 | 158 | +0.00 |
| Attention 25%, KHONG thay cau hoi (loqo) | KiToKe 25% | 2700 | 63.07 | 59.96 | 257 | 173 | +4.05 |
| Attention 25%, thay cau hoi (self) | Attention 25%, KHONG thay cau hoi (loqo) | 2700 | 64.52 | 63.07 | 91 | 52 | +3.26 |
| Attention 25%, thay cau hoi (self) | Uniform 25% | 2700 | 64.52 | 63.07 | 185 | 146 | +2.14 |
| 256 frame, Uniform 50% (~16K token vao LLM) | Dense 100% (128 frame, ~16K token) | 2700 | 65.96 | 65.15 | 153 | 131 | +1.31 |
| Dense 100% (128 frame, ~16K token) | Dense 100%, vi tri thoi gian kieu cu (loi RoPE) | 2700 | 65.15 | 64.26 | 152 | 128 | +1.43 |
| Dense 100% (128 frame, ~16K token) | Uniform 5% | 2700 | 65.15 | 57.59 | 367 | 163 | +8.86 |
| KiToKe 5% | Uniform 5% | 2700 | 53.33 | 57.59 | 166 | 281 | -5.44 |
| DySeg+STPrune 5% | Uniform 5% | 2700 | 55.70 | 57.59 | 169 | 220 | -2.59 |
| Dense 100% (128 frame, ~16K token) | Uniform 2% | 2700 | 65.15 | 53.41 | 487 | 170 | +12.37 |
| KiToKe 2% | Uniform 2% | 2700 | 51.04 | 53.41 | 188 | 252 | -3.05 |
| DySeg+STPrune 2% | Uniform 2% | 2700 | 51.85 | 53.41 | 183 | 225 | -2.08 |
| Attention 10%, KHONG thay cau hoi (loqo) | Uniform 10% | 2700 | 60.44 | 59.59 | 223 | 200 | +1.12 |
| DySeg+STPrune 25% (salience=grid) | Uniform 25% | 2700 | 62.41 | 63.07 | 143 | 161 | -1.03 |
| DySeg+STPrune 25% (salience=grid) | DySeg+STPrune 25% | 2700 | 62.41 | 61.63 | 138 | 117 | +1.32 |
| FastVID 25% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 25% | 2700 | 63.00 | 63.07 | 156 | 158 | -0.11 |
| FastVID 25% (ma chinh thuc, ghep vao cau hinh cua du an) | DySeg+STPrune 25% | 2700 | 63.00 | 61.63 | 189 | 152 | +2.00 |
| Dense 100% (128 frame, ~16K token) | FastVID 25% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700 | 65.15 | 63.00 | 199 | 141 | +3.15 |
| VidCom2 40% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 40% | 2700 | 65.44 | 64.07 | 138 | 101 | +2.39 |
| VidCom2 40% (ma chinh thuc, ghep vao cau hinh cua du an) | HoliTom w/o M 40% (port sang Qwen2.5-VL) | 2700 | 65.44 | 65.26 | 109 | 104 | +0.34 |
| VidCom2 25% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 25% | 2700 | 63.96 | 63.07 | 197 | 173 | +1.25 |
| VidCom2 25% (ma chinh thuc, ghep vao cau hinh cua du an) | HoliTom w/o M 25% (port sang Qwen2.5-VL) | 2700 | 63.96 | 63.85 | 154 | 151 | +0.17 |
| VidCom2 15% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 15% | 2700 | 61.22 | 61.07 | 211 | 207 | +0.20 |
| VidCom2 15% (ma chinh thuc, ghep vao cau hinh cua du an) | HoliTom w/o M 15% (port sang Qwen2.5-VL) | 2700 | 61.22 | 62.26 | 172 | 200 | -1.45 |
| VidCom2 10% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 10% | 2700 | 59.93 | 59.59 | 244 | 235 | +0.41 |
| VidCom2 10% (ma chinh thuc, ghep vao cau hinh cua du an) | HoliTom w/o M 10% (port sang Qwen2.5-VL) | 2700 | 59.93 | 61.33 | 199 | 237 | -1.82 |
| VidCom2 5% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 5% | 2700 | 55.15 | 57.59 | 254 | 320 | -2.75 |
| VidCom2 5% (ma chinh thuc, ghep vao cau hinh cua du an) | HoliTom w/o M 5% (port sang Qwen2.5-VL) | 2700 | 55.15 | 58.70 | 215 | 311 | -4.19 |
| VidCom2 2% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 2% | 2700 | 47.89 | 53.41 | 231 | 380 | -6.03 |
| VidCom2 2% (ma chinh thuc, ghep vao cau hinh cua du an) | HoliTom w/o M 2% (port sang Qwen2.5-VL) | 2700 | 47.89 | 55.37 | 224 | 426 | -7.92 |
| DyCoke chi TTM 40% (port sang Qwen2.5-VL) | Uniform 40% | 2700 | 64.37 | 64.07 | 123 | 115 | +0.52 |
| DyCoke chi TTM 25% (port sang Qwen2.5-VL) | Uniform 25% | 2700 | 60.52 | 63.07 | 151 | 220 | -3.58 |
| HoliTom w/o M 10% bo gop thoi gian | Uniform 10% | 2700 | 60.33 | 59.59 | 201 | 181 | +1.02 |
| HoliTom w/o M 10% bo gop thoi gian | HoliTom w/o M 10% (port sang Qwen2.5-VL) | 2700 | 60.33 | 61.33 | 95 | 122 | -1.83 |
| HoliTom w/o M 25% bo gop thoi gian | Uniform 25% | 2700 | 64.11 | 63.07 | 154 | 126 | +1.67 |
| HoliTom w/o M 25% bo gop thoi gian | HoliTom w/o M 25% (port sang Qwen2.5-VL) | 2700 | 64.11 | 63.85 | 74 | 67 | +0.59 |
| HoliTom w/o M 10% tau 0.80 | Uniform 10% | 2700 | 60.74 | 59.59 | 200 | 169 | +1.61 |
| HoliTom w/o M 10% tau 0.80 | HoliTom w/o M 10% (port sang Qwen2.5-VL) | 2700 | 60.74 | 61.33 | 60 | 76 | -1.37 |
| FastVID 40% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 40% | 2700 | 65.11 | 64.07 | 129 | 101 | +1.85 |
| FastVID 15% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 15% | 2700 | 61.19 | 61.07 | 172 | 169 | +0.16 |
| FastVID 10% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 10% | 2700 | 59.30 | 59.59 | 179 | 187 | -0.42 |
| FastVID 5% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 5% | 2700 | 56.63 | 57.59 | 184 | 210 | -1.31 |
| FastVID 2% (ma chinh thuc, ghep vao cau hinh cua du an) | Uniform 2% | 2700 | 52.56 | 53.41 | 179 | 202 | -1.18 |
| HoliTom w/o M 40% (port sang Qwen2.5-VL) | Uniform 40% | 2700 | 65.26 | 64.07 | 118 | 86 | +2.24 |
| HoliTom w/o M 40% (port sang Qwen2.5-VL) | FastVID 40% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700 | 65.26 | 65.11 | 115 | 111 | +0.27 |
| HoliTom w/o M 25% (port sang Qwen2.5-VL) | Uniform 25% | 2700 | 63.85 | 63.07 | 145 | 124 | +1.28 |
| HoliTom w/o M 25% (port sang Qwen2.5-VL) | FastVID 25% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700 | 63.85 | 63.00 | 155 | 132 | +1.36 |
| HoliTom w/o M 15% (port sang Qwen2.5-VL) | Uniform 15% | 2700 | 62.26 | 61.07 | 184 | 152 | +1.75 |
| HoliTom w/o M 15% (port sang Qwen2.5-VL) | FastVID 15% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700 | 62.26 | 61.19 | 175 | 146 | +1.62 |
| HoliTom w/o M 10% (port sang Qwen2.5-VL) | Uniform 10% | 2700 | 61.33 | 59.59 | 199 | 152 | +2.51 |
| HoliTom w/o M 10% (port sang Qwen2.5-VL) | FastVID 10% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700 | 61.33 | 59.30 | 219 | 164 | +2.81 |
| HoliTom w/o M 5% (port sang Qwen2.5-VL) | Uniform 5% | 2700 | 58.70 | 57.59 | 212 | 182 | +1.51 |
| HoliTom w/o M 5% (port sang Qwen2.5-VL) | FastVID 5% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700 | 58.70 | 56.63 | 220 | 164 | +2.86 |
| HoliTom w/o M 2% (port sang Qwen2.5-VL) | Uniform 2% | 2700 | 55.37 | 53.41 | 247 | 194 | +2.52 |
| HoliTom w/o M 2% (port sang Qwen2.5-VL) | FastVID 2% (ma chinh thuc, ghep vao cau hinh cua du an) | 2700 | 55.37 | 52.56 | 250 | 174 | +3.69 |

## Loi trong log

- `dsta_v2_train.log`: 254 dong SKIP/Traceback
- `fastvid_calib_levels.log`: 11 dong SKIP/Traceback
- `sweep_dyseg_r040.log`: 1 dong SKIP/Traceback

## Nhan xet (Claude viet tay)

**06/10 10:15 UTC.** Hang doi 5 buoc da khoi dong (xem "Trang thai hang doi"). Phan nay Claude cap nhat moi khi mot buoc xong; cac bang phia tren tu dong lam moi 5 phut/lan.

- Uniform 25% dang hon KiToKe 25% ro rang (z ~ +4.7 tren ~2500 cau chung). Neu giu den het thi moc can vuot o 25% la Uniform, khong phai KiToKe. Chua kiem tra vi sao KiToKe kem Uniform tren Qwen2.5-VL (nghi van: sieu tham so cua paper chinh cho backbone khac, hoac buoc gop token lam hong feature); day la viec nen lam sau.
- 18 cau dau cua thu nghiem attention la tu lan chay bi sap tren may cua ban; hang doi se chay tiep tu cau thu 19.

**07/10 06:56 UTC.** Hang doi tu dong hom 06/10 bi treo o buoc WAIT (tien trinh 328214 thanh zombie, `kill -0` van bao song) roi container bi tat sau 15:17 UTC. Tu 07/10 cac buoc duoc chay tay, tung buoc mot; bang phia tren chi lam moi khi mot buoc xong.

- Attention 25% xong (2700/2700, khong SKIP). Ban khong thay cau hoi (loqo) bang dung Uniform 25% (63.07 vs 63.07, z = 0.00); ban thay cau hoi (self) hon Uniform 25% 1.45 diem (z = +2.14) va hon loqo (z = +3.26).
- 256 frame Uniform 50% (nguoi dung chay tren card khac) xong: 65.96 vs Dense 128 frame 65.15, z = +1.31 (chua co y nghia thong ke); rieng long 57.00 vs 54.11.
- Dang chay: Uniform 40%. Con lai: Dense hf_default.

**07/10 16:25 UTC.** Bo moc co dinh (Qwen2.5-VL-7B, 128 frame, ~16K token, time_mode=lmms) da du:
Dense 65.15 | Uniform 40% 64.07 (98.3%) | 25% 63.07 (96.8%) | 15% 61.07 (93.7%) | 10% 59.59 (91.5%).
Dense time_mode=hf_default (chay tren card khac) = 64.26: loi trung vi tri thoi gian lam mat 0.89 diem tong, tap trung o short (73.22 vs 76.89).

- Dang chay tren card nay: train lai DSTA scorer (outputs/dsta_scorer_2a_v2.pt, log dsta_v2_train.log), 2000 step tren 1049 cap video-caption dai 1-3 phut (LLaVA-Video-178K 1_2_m + 2_3_m), 128 frame, lmms. Sau do: danh gia 25% tren 2700 cau (pred_videomme_f128_dsta_v2_r025).
- Card khac: DySeg+STPrune o 25/10/15/40% (pred_videomme_f128_dyseg_rXXX), do agent ben do chay.
- Script da sua 07/10: run_inference_qwen25vl_dyseg.py, train_dsta_scorer_2a.py, infer_dsta_scorer_videomme.py (them max_frames, time_mode mac dinh lmms, retention). Ban goc: outputs/kitoke_baseline/*.bak_20261007.

**07/10 20:35 UTC.** Train DSTA v2 xong: 2000 step, 3 gio 54 phut, 934 video khac nhau (115 cap ego4d bi bo vi khong decode duoc: torchvision.io.read_video khong ton tai), checkpoint outputs/dsta_scorer_2a_v2.pt.
Phan thuong KHONG cai thien qua qua trinh train: reward trung binh -1.394 (250 step dau) -> -1.376 (250 step cuoi), CE caption 1.313 -> 1.300, reward_std 0.043 -> 0.032, ti le giu ~0.40, entropy 0.630 -> 0.575.
Dang chay: danh gia 25% tren 2700 cau (pred_videomme_f128_dsta_v2_r025), sau do doi chung chon unit cach deu 25% (pred_videomme_f128_unituniform_r025). Moc so sanh: Uniform theo token 25% = 63.07.

**08/10 05:35 UTC.** Ket qua o 25% tren bo moc (Uniform 25% = 63.07):
- DSTA v2: 60.22 (z = -3.63 so voi Uniform). Doi chung chon unit cach deu 25%: 61.48; DSTA khong hon doi chung (z = -1.56), va chon theo unit von thua chon theo token (z = -2.21).
- DySeg+STPrune 25% (fill_budget=on, ti le giu thuc te 24.99%): 61.63 (70.78/61.33/52.78), thua Uniform 1.44 diem (z = -2.23), hon KiToKe 25% (z = +2.46).
- DySeg 40% dang chay tren card khac (container vlm-hoang03). GPU card nay dang trong.

**08/10 09:10 UTC.** DySeg+STPrune tren bo moc, du 3 muc (fill_budget=on):
40%: 63.19 vs Uniform 64.07 (z = -1.46) | 25%: 61.63 vs 63.07 (z = -2.23) | 10%: 58.41 vs 59.59 (z = -1.64).
Thua tap trung o short (z = -2.84 / -1.57 / -3.51); medium va long ngang Uniform. Card khac dang chay bien the salience_mode=grid o 25%.

**08/10 10:55 UTC — hang doi qua dem.** `outputs/kitoke_baseline/night_queue.sh` dang chay tach khoi phien Claude (setsid nohup, log `night_queue.log`, tien do trong `overnight_status.txt`). Thu tu: (cho xong Uniform/KiToKe 5% va 2%) -> DySeg 5% -> DySeg 2% -> Attention 10% (loqo = tran query-agnostic) -> KiToKe 10% -> DySeg 15% -> KiToKe 15% -> Attention 5% -> KiToKe 25% seed 1,2 -> KiToKe 10% seed 1,2. Tat ca 128 frame, lmms. Neu container bi tao lai: chay lai dung lenh o dau file night_queue.sh (buoc da du 2700 cau tu bo qua, buoc do dang tu resume). Lich su chat duoc sao luu sang `/home/hoang/claude_backup/` sau moi buoc.
```

### kitoke_baseline/REPORT_notes.md

```
**06/10 10:15 UTC.** Hang doi 5 buoc da khoi dong (xem "Trang thai hang doi"). Phan nay Claude cap nhat moi khi mot buoc xong; cac bang phia tren tu dong lam moi 5 phut/lan.

- Uniform 25% dang hon KiToKe 25% ro rang (z ~ +4.7 tren ~2500 cau chung). Neu giu den het thi moc can vuot o 25% la Uniform, khong phai KiToKe. Chua kiem tra vi sao KiToKe kem Uniform tren Qwen2.5-VL (nghi van: sieu tham so cua paper chinh cho backbone khac, hoac buoc gop token lam hong feature); day la viec nen lam sau.
- 18 cau dau cua thu nghiem attention la tu lan chay bi sap tren may cua ban; hang doi se chay tiep tu cau thu 19.

**07/10 06:56 UTC.** Hang doi tu dong hom 06/10 bi treo o buoc WAIT (tien trinh 328214 thanh zombie, `kill -0` van bao song) roi container bi tat sau 15:17 UTC. Tu 07/10 cac buoc duoc chay tay, tung buoc mot; bang phia tren chi lam moi khi mot buoc xong.

- Attention 25% xong (2700/2700, khong SKIP). Ban khong thay cau hoi (loqo) bang dung Uniform 25% (63.07 vs 63.07, z = 0.00); ban thay cau hoi (self) hon Uniform 25% 1.45 diem (z = +2.14) va hon loqo (z = +3.26).
- 256 frame Uniform 50% (nguoi dung chay tren card khac) xong: 65.96 vs Dense 128 frame 65.15, z = +1.31 (chua co y nghia thong ke); rieng long 57.00 vs 54.11.
- Dang chay: Uniform 40%. Con lai: Dense hf_default.

**07/10 16:25 UTC.** Bo moc co dinh (Qwen2.5-VL-7B, 128 frame, ~16K token, time_mode=lmms) da du:
Dense 65.15 | Uniform 40% 64.07 (98.3%) | 25% 63.07 (96.8%) | 15% 61.07 (93.7%) | 10% 59.59 (91.5%).
Dense time_mode=hf_default (chay tren card khac) = 64.26: loi trung vi tri thoi gian lam mat 0.89 diem tong, tap trung o short (73.22 vs 76.89).

- Dang chay tren card nay: train lai DSTA scorer (outputs/dsta_scorer_2a_v2.pt, log dsta_v2_train.log), 2000 step tren 1049 cap video-caption dai 1-3 phut (LLaVA-Video-178K 1_2_m + 2_3_m), 128 frame, lmms. Sau do: danh gia 25% tren 2700 cau (pred_videomme_f128_dsta_v2_r025).
- Card khac: DySeg+STPrune o 25/10/15/40% (pred_videomme_f128_dyseg_rXXX), do agent ben do chay.
- Script da sua 07/10: run_inference_qwen25vl_dyseg.py, train_dsta_scorer_2a.py, infer_dsta_scorer_videomme.py (them max_frames, time_mode mac dinh lmms, retention). Ban goc: outputs/kitoke_baseline/*.bak_20261007.

**07/10 20:35 UTC.** Train DSTA v2 xong: 2000 step, 3 gio 54 phut, 934 video khac nhau (115 cap ego4d bi bo vi khong decode duoc: torchvision.io.read_video khong ton tai), checkpoint outputs/dsta_scorer_2a_v2.pt.
Phan thuong KHONG cai thien qua qua trinh train: reward trung binh -1.394 (250 step dau) -> -1.376 (250 step cuoi), CE caption 1.313 -> 1.300, reward_std 0.043 -> 0.032, ti le giu ~0.40, entropy 0.630 -> 0.575.
Dang chay: danh gia 25% tren 2700 cau (pred_videomme_f128_dsta_v2_r025), sau do doi chung chon unit cach deu 25% (pred_videomme_f128_unituniform_r025). Moc so sanh: Uniform theo token 25% = 63.07.

**08/10 05:35 UTC.** Ket qua o 25% tren bo moc (Uniform 25% = 63.07):
- DSTA v2: 60.22 (z = -3.63 so voi Uniform). Doi chung chon unit cach deu 25%: 61.48; DSTA khong hon doi chung (z = -1.56), va chon theo unit von thua chon theo token (z = -2.21).
- DySeg+STPrune 25% (fill_budget=on, ti le giu thuc te 24.99%): 61.63 (70.78/61.33/52.78), thua Uniform 1.44 diem (z = -2.23), hon KiToKe 25% (z = +2.46).
- DySeg 40% dang chay tren card khac (container vlm-hoang03). GPU card nay dang trong.

**08/10 09:10 UTC.** DySeg+STPrune tren bo moc, du 3 muc (fill_budget=on):
40%: 63.19 vs Uniform 64.07 (z = -1.46) | 25%: 61.63 vs 63.07 (z = -2.23) | 10%: 58.41 vs 59.59 (z = -1.64).
Thua tap trung o short (z = -2.84 / -1.57 / -3.51); medium va long ngang Uniform. Card khac dang chay bien the salience_mode=grid o 25%.

**08/10 10:55 UTC — hang doi qua dem.** `outputs/kitoke_baseline/night_queue.sh` dang chay tach khoi phien Claude (setsid nohup, log `night_queue.log`, tien do trong `overnight_status.txt`). Thu tu: (cho xong Uniform/KiToKe 5% va 2%) -> DySeg 5% -> DySeg 2% -> Attention 10% (loqo = tran query-agnostic) -> KiToKe 10% -> DySeg 15% -> KiToKe 15% -> Attention 5% -> KiToKe 25% seed 1,2 -> KiToKe 10% seed 1,2. Tat ca 128 frame, lmms. Neu container bi tao lai: chay lai dung lenh o dau file night_queue.sh (buoc da du 2700 cau tu bo qua, buoc do dang tu resume). Lich su chat duoc sao luu sang `/home/hoang/claude_backup/` sau moi buoc.
```

### kitoke_baseline/abl/ABL_REPORT.md

```
# Bao cao ablation qua dem (container vlm-hoang03)

Cap nhat: 2026-10-09 01:30:22 UTC. VideoMME khong subtitle, Qwen2.5-VL-7B, 128 frame, time_mode=lmms, fill_budget=on.
Diem Short/Medium/Long/Overall la tu script cham chinh thuc (chi co khi run du 2700 cau).
Cot 'so voi' = chenh lech diem tren cac cau chung (McNemar z; |z|>1.96 la co y nghia). Run chua du 2700 cau: so tam.

Moc Uniform: 40% 64.07 | 25% 63.07 | 15% 61.07 | 10% 59.59. DySeg+STPrune goc: 40% 63.19 | 25% 61.63.

| Muc giu | Run | Mo ta | Da chay | Short | Medium | Long | Overall | Giu thuc te % | So voi Uniform cung muc | So voi DySeg goc cung muc |
|---|---|---|---|---|---|---|---|---|---|---|
| 25% | abl_dyseg_delta_nomerge_r025 | DySeg+STPrune goc nhung DTM KHONG gop token (alpha=1) | 2700/2700 | 70.78 | 60.89 | 52.00 | 61.22 | 24.99 | -1.81 (z=-2.73, n=2700) | -0.37 (z=-1.27, n=2700) |
| 25% | abl_dyseg_grid_dtm0_r025 | DySeg chia ngan sach + chon cach deu, KHONG DTM | 2700/2700 | 73.44 | 62.56 | 53.44 | 63.15 | 25.00 | -0.04 (z=-0.06, n=2700) | +1.41 (z=+2.28, n=2700) |
| 25% | abl_dyseg_grid_nomerge_r025 | DySeg + chon cach deu + DTM khong gop | 2700/2700 | 71.78 | 62.11 | 52.44 | 62.11 | 24.99 | -1.00 (z=-1.53, n=2700) | +0.44 (z=+0.77, n=2700) |
| 25% | abl_len_delta_r025 | Chia theo do dai doan + STPrune goc (delta + DTM gop) | 2700/2700 | 71.78 | 62.44 | 53.22 | 62.48 | 24.99 | -0.59 (z=-0.96, n=2700) | +0.85 (z=+1.58, n=2700) |
| 25% | abl_len_grid_dtm0_r025 | Chia theo do dai doan + chon cach deu, khong DTM (doi chung ~Uniform) | 2700/2700 | 73.56 | 61.89 | 53.22 | 62.89 | 25.00 | -0.30 (z=-0.49, n=2700) | +1.15 (z=+1.75, n=2700) |
| 25% | abl_vcast_r025 | Kieu V-CAST (tu cai lai theo paper): do cong + chon theo frame, khong gop | 2700/2700 | 71.89 | 61.33 | 53.89 | 62.37 | 25.00 | -0.63 (z=-0.97, n=2700) | +0.81 (z=+1.23, n=2700) |

## Trang thai hang doi

```
2026-10-08 11:07:47 QUEUE START (pid 329382)
2026-10-08 11:07:47 START abl_dyseg_grid_dtm0_r025 (lan 1, da co 0/2700)
2026-10-08 13:29:15 DONE abl_dyseg_grid_dtm0_r025 (2700/2700) overall 2700 1705 0 0.6315
2026-10-08 13:29:22 START abl_vcast_r025 (lan 1, da co 0/2700)
2026-10-08 15:46:50 DONE abl_vcast_r025 (2700/2700) overall 2700 1684 0 0.6237
2026-10-08 15:46:57 START abl_dyseg_delta_nomerge_r025 (lan 1, da co 0/2700)
2026-10-08 18:09:21 DONE abl_dyseg_delta_nomerge_r025 (2700/2700) overall 2700 1653 0 0.6122
2026-10-08 18:09:24 START abl_len_grid_dtm0_r025 (lan 1, da co 0/2700)
2026-10-08 20:28:51 DONE abl_len_grid_dtm0_r025 (2700/2700) overall 2700 1698 0 0.6289
2026-10-08 20:28:57 START abl_len_delta_r025 (lan 1, da co 0/2700)
2026-10-08 22:50:27 DONE abl_len_delta_r025 (2700/2700) overall 2700 1687 0 0.6248
2026-10-08 22:50:36 START abl_dyseg_grid_nomerge_r025 (lan 1, da co 0/2700)
2026-10-09 01:30:12 DONE abl_dyseg_grid_nomerge_r025 (2700/2700) overall 2700 1677 0 0.6211
```
```

### rap_elastic/web14/numbers.json (so lieu idea 14 tren web, gom ca ket qua queue6 server moi)

```json
{
 "acc": {
  "uniform002": {
   "n": 2700,
   "short": 61.56,
   "medium": 51.78,
   "long": 46.89,
   "overall": 53.41
  },
  "uniform005": {
   "n": 2700,
   "short": 67.89,
   "medium": 54.78,
   "long": 50.11,
   "overall": 57.59
  },
  "uniform010": {
   "n": 2700,
   "short": 70.56,
   "medium": 56.89,
   "long": 51.33,
   "overall": 59.59
  },
  "uniform015": {
   "n": 2700,
   "short": 71.33,
   "medium": 60.0,
   "long": 51.89,
   "overall": 61.07
  },
  "uniform025": {
   "n": 2700,
   "short": 72.67,
   "medium": 63.22,
   "long": 53.33,
   "overall": 63.07
  },
  "uniform040": {
   "n": 2700,
   "short": 76.11,
   "medium": 61.78,
   "long": 54.33,
   "overall": 64.07
  },
  "holitom002": {
   "n": 2700,
   "short": 64.67,
   "medium": 53.11,
   "long": 48.33,
   "overall": 55.37
  },
  "holitom005": {
   "n": 2700,
   "short": 68.33,
   "medium": 57.22,
   "long": 50.56,
   "overall": 58.7
  },
  "holitom010": {
   "n": 2700,
   "short": 71.0,
   "medium": 59.56,
   "long": 53.44,
   "overall": 61.33
  },
  "holitom015": {
   "n": 2700,
   "short": 72.22,
   "medium": 61.56,
   "long": 53.0,
   "overall": 62.26
  },
  "holitom025": {
   "n": 2700,
   "short": 74.33,
   "medium": 62.0,
   "long": 55.22,
   "overall": 63.85
  },
  "holitom040": {
   "n": 2700,
   "short": 76.0,
   "medium": 64.67,
   "long": 55.11,
   "overall": 65.26
  },
  "fastvid002": {
   "n": 2700,
   "short": 61.11,
   "medium": 50.33,
   "long": 46.22,
   "overall": 52.56
  },
  "fastvid005": {
   "n": 2700,
   "short": 66.0,
   "medium": 53.89,
   "long": 50.0,
   "overall": 56.63
  },
  "fastvid010": {
   "n": 2700,
   "short": 69.56,
   "medium": 57.56,
   "long": 50.78,
   "overall": 59.3
  },
  "fastvid015": {
   "n": 2700,
   "short": 72.78,
   "medium": 59.22,
   "long": 51.56,
   "overall": 61.19
  },
  "fastvid025": {
   "n": 2700,
   "short": 74.22,
   "medium": 60.89,
   "long": 53.89,
   "overall": 63.0
  },
  "fastvid040": {
   "n": 2700,
   "short": 74.89,
   "medium": 64.56,
   "long": 55.89,
   "overall": 65.11
  },
  "vidcom2002": {
   "n": 2700,
   "short": 49.56,
   "medium": 49.56,
   "long": 44.56,
   "overall": 47.89
  },
  "vidcom2005": {
   "n": 2700,
   "short": 62.67,
   "medium": 55.67,
   "long": 47.11,
   "overall": 55.15
  },
  "vidcom2010": {
   "n": 2700,
   "short": 69.44,
   "medium": 58.56,
   "long": 51.78,
   "overall": 59.93
  },
  "vidcom2015": {
   "n": 2700,
   "short": 72.0,
   "medium": 58.78,
   "long": 52.89,
   "overall": 61.22
  },
  "vidcom2025": {
   "n": 2700,
   "short": 73.78,
   "medium": 63.67,
   "long": 54.44,
   "overall": 63.96
  },
  "vidcom2040": {
   "n": 2700,
   "short": 76.0,
   "medium": 65.89,
   "long": 54.44,
   "overall": 65.44
  },
  "dycoke025": {
   "n": 2700,
   "short": 71.33,
   "medium": 58.44,
   "long": 51.78,
   "overall": 60.52
  },
  "dycoke040": {
   "n": 2700,
   "short": 75.67,
   "medium": 63.11,
   "long": 54.33,
   "overall": 64.37
  },
  "dense": {
   "n": 2700,
   "short": 76.89,
   "medium": 64.44,
   "long": 54.11,
   "overall": 65.15
  },
  "runE025": {
   "n": 2700,
   "short": 73.33,
   "medium": 65.11,
   "long": 54.11,
   "overall": 64.19
  },
  "runD025": {
   "n": 2700,
   "short": 73.78,
   "medium": 65.11,
   "long": 54.33,
   "overall": 64.41
  },
  "R10E010": {
   "n": 2700,
   "short": 71.33,
   "medium": 61.0,
   "long": 52.44,
   "overall": 61.59
  },
  "R02E002": {
   "n": 2700,
   "short": 65.67,
   "medium": 54.0,
   "long": 48.67,
   "overall": 56.11
  },
  "R05E005": {
   "n": 2700,
   "short": 68.0,
   "medium": 59.56,
   "long": 52.44,
   "overall": 60.0
  },
  "R15E015": {
   "n": 2700,
   "short": 72.33,
   "medium": 63.11,
   "long": 52.56,
   "overall": 62.67
  },
  "R40E040": {
   "n": 2700,
   "short": 75.44,
   "medium": 63.89,
   "long": 53.67,
   "overall": 64.33
  },
  "R10D010": {
   "n": 2700,
   "short": 71.33,
   "medium": 59.56,
   "long": 52.67,
   "overall": 61.19
  },
  "V10E010": {
   "n": 2700,
   "short": 72.33,
   "medium": 60.56,
   "long": 53.33,
   "overall": 62.07
  },
  "V25E025": {
   "n": 2700,
   "short": 73.78,
   "medium": 65.33,
   "long": 53.67,
   "overall": 64.26
  },
  "ELA_r025": {
   "n": 2700,
   "short": 73.78,
   "medium": 65.11,
   "long": 53.67,
   "overall": 64.19
  },
  "ELA_r010": {
   "n": 2700,
   "short": 71.44,
   "medium": 59.0,
   "long": 53.0,
   "overall": 61.15
  },
  "ELA_s1_r025": {
   "n": 2700,
   "short": 73.56,
   "medium": 65.0,
   "long": 54.11,
   "overall": 64.22
  },
  "ELA_s1_r010": {
   "n": 2700,
   "short": 71.44,
   "medium": 58.89,
   "long": 52.78,
   "overall": 61.04
  },
  "R02W3_r002": {
   "n": 2700,
   "short": 65.89,
   "medium": 54.44,
   "long": 49.11,
   "overall": 56.48
  },
  "AS10U_r010": {
   "n": 2700,
   "short": 70.0,
   "medium": 57.22,
   "long": 52.56,
   "overall": 59.93
  },
  "AS10R_r010": {
   "n": 2700,
   "short": 71.78,
   "medium": 59.89,
   "long": 51.11,
   "overall": 60.93
  },
  "SU10_r010": {
   "n": 294,
   "short": 68.71,
   "medium": NaN,
   "long": NaN,
   "overall": 68.71
  },
  "SU25_r025": {
   "n": 2700,
   "short": 75.22,
   "medium": 62.22,
   "long": 53.33,
   "overall": 63.59
  },
  "C0_r025": {
   "n": 2700,
   "short": 73.67,
   "medium": 61.33,
   "long": 52.89,
   "overall": 62.63
  },
  "M3D-B_r025": {
   "n": 2700,
   "short": 73.33,
   "medium": 64.33,
   "long": 53.89,
   "overall": 63.85
  },
  "M3D-BD_r025": {
   "n": 1287,
   "short": 73.89,
   "medium": 66.15,
   "long": NaN,
   "overall": 71.56
  },
  "sub_DENSE256": {
   "n": 300,
   "short": 76.79,
   "medium": 65.06,
   "long": 57.14,
   "overall": 66.67
  }
 },
 "pairs": {
  "ELA_r025|runE025": {
   "d": 0.0,
   "z": 0.0,
   "n": 2700
  },
  "ELA_r025|uniform025": {
   "d": 1.11,
   "z": 2.0,
   "n": 2700
  },
  "ELA_r010|R10E010": {
   "d": -0.44,
   "z": -1.13,
   "n": 2700
  },
  "ELA_r010|uniform010": {
   "d": 1.56,
   "z": 2.48,
   "n": 2700
  },
  "R02W3_r002|R02E002": {
   "d": 0.37,
   "z": 0.88,
   "n": 2700
  },
  "R02W3_r002|uniform002": {
   "d": 3.07,
   "z": 4.14,
   "n": 2700
  },
  "R02W3_r002|holitom002": {
   "d": 1.11,
   "z": 1.3,
   "n": 2700
  },
  "R02E002|uniform002": {
   "d": 2.7,
   "z": 3.68,
   "n": 2700
  },
  "R05E005|uniform005": {
   "d": 2.41,
   "z": 3.4,
   "n": 2700
  },
  "AS10U_r010|uniform010": {
   "d": 0.33,
   "z": 0.5,
   "n": 2700
  },
  "AS10R_r010|R10E010": {
   "d": -0.67,
   "z": -1.01,
   "n": 2700
  },
  "SU10_r010|uniform010": {
   "d": -3.4,
   "z": -1.67,
   "n": 294
  },
  "SU25_r025|uniform025": {
   "d": 0.52,
   "z": 0.94,
   "n": 2700
  },
  "C0_r025|uniform025": {
   "d": -0.44,
   "z": -0.79,
   "n": 2700
  },
  "C0_r025|runE025": {
   "d": -1.56,
   "z": -2.42,
   "n": 2700
  },
  "M3D-B_r025|runE025": {
   "d": -0.33,
   "z": -1.1,
   "n": 2700
  },
  "M3D-B_r025|uniform025": {
   "d": 0.78,
   "z": 1.48,
   "n": 2700
  },
  "M3D-BD_r025|runE025": {
   "d": 0.16,
   "z": 0.33,
   "n": 1287
  },
  "M3D-BD_r025|uniform025": {
   "d": 1.32,
   "z": 1.66,
   "n": 1287
  },
  "M3D-BD_r025|M3D-B_r025": {
   "d": 0.39,
   "z": 0.93,
   "n": 1287
  },
  "V10E010|R10E010": {
   "d": 0.48,
   "z": 1.27,
   "n": 2700
  },
  "V10E010|uniform010": {
   "d": 2.48,
   "z": 3.86,
   "n": 2700
  },
  "V25E025|runE025": {
   "d": 0.07,
   "z": 0.23,
   "n": 2700
  },
  "V25E025|uniform025": {
   "d": 1.19,
   "z": 2.21,
   "n": 2700
  },
  "R10E010|uniform010": {
   "d": 2.0,
   "z": 3.1,
   "n": 2700
  },
  "runE025|uniform025": {
   "d": 1.11,
   "z": 2.04,
   "n": 2700
  },
  "sub_DENSE256|dense": {
   "d": -1.0,
   "z": -0.51,
   "n": 300
  },
  "R10D010|R10E010": {
   "d": -0.41,
   "z": -1.02,
   "n": 2700
  },
  "holitom010|uniform010": {
   "d": 1.74,
   "z": 2.51,
   "n": 2700
  },
  "holitom025|uniform025": {
   "d": 0.78,
   "z": 1.28,
   "n": 2700
  }
 }
}
```

## 3. So cap tung cau (cac file *_vs_*.txt)

**rap/pred_videomme_f128_rap_R10E_r010_vs_uniform.txt**
```
RAP-R10E vs Uniform010: 2700 cau chung (A co 2700, B co 2700)
nhom         n       RAP-R10E     Uniform010  chi A  chi B      z
short      900          71.33          70.56     47     40  +0.75
medium     900          61.00          56.89     76     39  +3.45
long       900          52.44          51.33     56     46  +0.99
overall   2700          61.59          59.59    179    125  +3.10
```

**rap/pred_videomme_f128_rap_runA_r025_vs_uniform.txt**
```
RAP-runA vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n       RAP-runA      Uniform25  chi A  chi B      z
short      900          73.44          72.67     51     44  +0.72
medium     900          62.33          63.22     53     61  -0.75
long       900          52.89          53.33     53     57  -0.38
overall   2700          62.89          63.07    157    162  -0.28
```

**rap/pred_videomme_f128_rap_runB_noise_r025_vs_uniform.txt**
```
RAP-runB_noise vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n RAP-runB_noise      Uniform25  chi A  chi B      z
short      900          72.78          72.67      6      5  +0.30
medium     900          63.56          63.22     13     10  +0.63
long       900          53.33          53.33      8      8  +0.00
overall   2700          63.22          63.07     27     23  +0.57
```

**rap/pred_videomme_f128_rap_runB_r025_vs_uniform.txt**
```
RAP-runB vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n       RAP-runB      Uniform25  chi A  chi B      z
short      900          73.67          72.67     34     25  +1.17
medium     900          65.11          63.22     49     32  +1.89
long       900          53.22          53.33     35     36  -0.12
overall   2700          64.00          63.07    118     93  +1.72
```

**rap/pred_videomme_f128_rap_runC_r025_vs_uniform.txt**
```
RAP-runC vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n       RAP-runC      Uniform25  chi A  chi B      z
short      900          74.00          72.67     37     25  +1.52
medium     900          64.33          63.22     36     26  +1.27
long       900          53.56          53.33     31     29  +0.26
overall   2700          63.96          63.07    104     80  +1.77
```

**rap/pred_videomme_f128_rap_runD_r025_vs_uniform.txt**
```
RAP-runD vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n       RAP-runD      Uniform25  chi A  chi B      z
short      900          73.78          72.67     32     22  +1.36
medium     900          65.11          63.22     49     32  +1.89
long       900          54.33          53.33     38     29  +1.10
overall   2700          64.41          63.07    119     83  +2.53
```

**rap/pred_videomme_f128_rap_runE_r025_vs_uniform.txt**
```
RAP-runE vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n       RAP-runE      Uniform25  chi A  chi B      z
short      900          73.33          72.67     34     28  +0.76
medium     900          65.11          63.22     53     36  +1.80
long       900          54.11          53.33     36     29  +0.87
overall   2700          64.19          63.07    123     93  +2.04
```

**rap/pred_videomme_f128_rap_runG_r025_vs_uniform.txt**
```
RAP-runG vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n       RAP-runG      Uniform25  chi A  chi B      z
short      900          72.67          72.67     32     32  +0.00
medium     900          65.22          63.22     50     32  +1.99
long       900          53.22          53.33     32     33  -0.12
overall   2700          63.70          63.07    114     97  +1.17
```

**rap/pred_videomme_f128_rap_runH_r025_vs_uniform.txt**
```
RAP-runH vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n       RAP-runH      Uniform25  chi A  chi B      z
short      900          73.67          72.67     34     25  +1.17
medium     900          64.89          63.22     49     34  +1.65
long       900          53.11          53.33     32     34  -0.25
overall   2700          63.89          63.07    115     93  +1.53
```

**rap/sub300_runA_vs_uniform.txt**
```
RAP-runA vs Uniform25: 300 cau chung (A co 300, B co 2700)
nhom         n       RAP-runA      Uniform25  chi A  chi B      z
short      112          67.86          70.54      5      8  -0.83
medium      83          67.47          66.27      4      3  +0.38
long       105          55.24          55.24      7      7  +0.00
overall    300          63.33          64.00     16     18  -0.34
```

**rap/sub300_runB_vs_uniform.txt**
```
RAP-runB vs Uniform25: 300 cau chung (A co 300, B co 2700)
nhom         n       RAP-runB      Uniform25  chi A  chi B      z
short      112          72.32          70.54      4      2  +0.82
medium      83          67.47          66.27      4      3  +0.38
long       105          54.29          55.24      5      6  -0.30
overall    300          64.67          64.00     13     11  +0.41
```

**rap/sub300_runC_vs_uniform.txt**
```
RAP-runC vs Uniform25: 300 cau chung (A co 300, B co 2700)
nhom         n       RAP-runC      Uniform25  chi A  chi B      z
short      112          73.21          70.54      6      3  +1.00
medium      83          66.27          66.27      1      1  +0.00
long       105          55.24          55.24      5      5  +0.00
overall    300          65.00          64.00     12      9  +0.65
```

**rap_next/pred_videomme_f128_rap_R02E_r002_vs_fastvid.txt**
```
R02E vs fastvid002: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R02E     fastvid002  chi A  chi B      z
short      900          65.67          61.11    104     63  +3.17
medium     900          54.00          50.33    110     77  +2.41
long       900          48.67          46.22     82     60  +1.85
overall   2700          56.11          52.56    296    200  +4.31
```

**rap_next/pred_videomme_f128_rap_R02E_r002_vs_holitom.txt**
```
R02E vs holitom002: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R02E     holitom002  chi A  chi B      z
short      900          65.67          64.67     94     85  +0.67
medium     900          54.00          53.11     95     87  +0.59
long       900          48.67          48.33     82     79  +0.24
overall   2700          56.11          55.37    271    251  +0.88
```

**rap_next/pred_videomme_f128_rap_R02E_r002_vs_uniform.txt**
```
R02E vs uniform002: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R02E     uniform002  chi A  chi B      z
short      900          65.67          61.56     85     48  +3.21
medium     900          54.00          51.78     86     66  +1.62
long       900          48.67          46.89     62     46  +1.54
overall   2700          56.11          53.41    233    160  +3.68
```

**rap_next/pred_videomme_f128_rap_R02E_r002_vs_vidcom2.txt**
```
R02E vs vidcom2002: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R02E     vidcom2002  chi A  chi B      z
short      900          65.67          49.56    206     61  +8.87
medium     900          54.00          49.56    132     92  +2.67
long       900          48.67          44.56    107     70  +2.78
overall   2700          56.11          47.89    445    223  +8.59
```

**rap_next/pred_videomme_f128_rap_R05E_r005_vs_fastvid.txt**
```
R05E vs fastvid005: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R05E     fastvid005  chi A  chi B      z
short      900          68.00          66.00     92     74  +1.40
medium     900          59.56          53.89    118     67  +3.75
long       900          52.44          50.00     84     62  +1.82
overall   2700          60.00          56.63    294    203  +4.08
```

**rap_next/pred_videomme_f128_rap_R05E_r005_vs_holitom.txt**
```
R05E vs holitom005: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R05E     holitom005  chi A  chi B      z
short      900          68.00          68.33     71     74  -0.25
medium     900          59.56          57.22     96     75  +1.61
long       900          52.44          50.56     80     63  +1.42
overall   2700          60.00          58.70    247    212  +1.63
```

**rap_next/pred_videomme_f128_rap_R05E_r005_vs_uniform.txt**
```
R05E vs uniform005: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R05E     uniform005  chi A  chi B      z
short      900          68.00          67.89     51     50  +0.10
medium     900          59.56          54.78     93     50  +3.60
long       900          52.44          50.11     71     50  +1.91
overall   2700          60.00          57.59    215    150  +3.40
```

**rap_next/pred_videomme_f128_rap_R05E_r005_vs_vidcom2.txt**
```
R05E vs vidcom2005: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R05E     vidcom2005  chi A  chi B      z
short      900          68.00          62.67    124     76  +3.39
medium     900          59.56          55.67    118     83  +2.47
long       900          52.44          47.11    111     63  +3.64
overall   2700          60.00          55.15    353    222  +5.46
```

**rap_next/pred_videomme_f128_rap_R10D_r010_vs_R10E.txt**
```
R10D vs R10E_seed0: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R10D     R10E_seed0  chi A  chi B      z
short      900          71.33          71.33     17     17  +0.00
medium     900          59.56          61.00     19     32  -1.82
long       900          52.67          52.44     17     15  +0.35
overall   2700          61.19          61.59     53     64  -1.02
```

**rap_next/pred_videomme_f128_rap_R10D_r010_vs_fastvid.txt**
```
R10D vs fastvid010: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R10D     fastvid010  chi A  chi B      z
short      900          71.33          69.56     78     62  +1.35
medium     900          59.56          57.56     95     77  +1.37
long       900          52.67          50.78     74     57  +1.49
overall   2700          61.19          59.30    247    196  +2.42
```

**rap_next/pred_videomme_f128_rap_R10D_r010_vs_holitom.txt**
```
R10D vs holitom010: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R10D     holitom010  chi A  chi B      z
short      900          71.33          71.00     62     59  +0.27
medium     900          59.56          59.56     79     79  +0.00
long       900          52.67          53.44     59     66  -0.63
overall   2700          61.19          61.33    200    204  -0.20
```

**rap_next/pred_videomme_f128_rap_R10D_r010_vs_uniform.txt**
```
R10D vs uniform010: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R10D     uniform010  chi A  chi B      z
short      900          71.33          70.56     48     41  +0.74
medium     900          59.56          56.89     64     40  +2.35
long       900          52.67          51.33     54     42  +1.22
overall   2700          61.19          59.59    166    123  +2.53
```

**rap_next/pred_videomme_f128_rap_R10D_r010_vs_vidcom2.txt**
```
R10D vs vidcom2010: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R10D     vidcom2010  chi A  chi B      z
short      900          71.33          69.44     84     67  +1.38
medium     900          59.56          58.56     99     90  +0.65
long       900          52.67          51.78     69     61  +0.70
overall   2700          61.19          59.93    252    218  +1.57
```

**rap_next/pred_videomme_f128_rap_R15E_r015_vs_fastvid.txt**
```
R15E vs fastvid015: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R15E     fastvid015  chi A  chi B      z
short      900          72.33          72.78     55     59  -0.37
medium     900          63.11          59.22     99     64  +2.74
long       900          52.56          51.56     65     56  +0.82
overall   2700          62.67          61.19    219    179  +2.01
```

**rap_next/pred_videomme_f128_rap_R15E_r015_vs_holitom.txt**
```
R15E vs holitom015: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R15E     holitom015  chi A  chi B      z
short      900          72.33          72.22     53     52  +0.10
medium     900          63.11          61.56     78     64  +1.17
long       900          52.56          53.00     55     59  -0.37
overall   2700          62.67          62.26    186    175  +0.58
```

**rap_next/pred_videomme_f128_rap_R15E_r015_vs_uniform.txt**
```
R15E vs uniform015: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R15E     uniform015  chi A  chi B      z
short      900          72.33          71.33     47     38  +0.98
medium     900          63.11          60.00     72     44  +2.60
long       900          52.56          51.89     41     35  +0.69
overall   2700          62.67          61.07    160    117  +2.58
```

**rap_next/pred_videomme_f128_rap_R15E_r015_vs_vidcom2.txt**
```
R15E vs vidcom2015: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R15E     vidcom2015  chi A  chi B      z
short      900          72.33          72.00     67     64  +0.26
medium     900          63.11          58.78    106     67  +2.97
long       900          52.56          52.89     60     63  -0.27
overall   2700          62.67          61.22    233    194  +1.89
```

**rap_next/pred_videomme_f128_rap_R40E_r040_vs_fastvid.txt**
```
R40E vs fastvid040: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R40E     fastvid040  chi A  chi B      z
short      900          75.44          74.89     40     35  +0.58
medium     900          63.89          64.56     49     55  -0.59
long       900          53.67          55.89     28     48  -2.29
overall   2700          64.33          65.11    117    138  -1.32
```

**rap_next/pred_videomme_f128_rap_R40E_r040_vs_holitom.txt**
```
R40E vs holitom040: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R40E     holitom040  chi A  chi B      z
short      900          75.44          76.00     33     38  -0.59
medium     900          63.89          64.67     39     46  -0.76
long       900          53.67          55.11     28     41  -1.57
overall   2700          64.33          65.26    100    125  -1.67
```

**rap_next/pred_videomme_f128_rap_R40E_r040_vs_uniform.txt**
```
R40E vs uniform040: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R40E     uniform040  chi A  chi B      z
short      900          75.44          76.11     18     24  -0.93
medium     900          63.89          61.78     45     26  +2.25
long       900          53.67          54.33     24     30  -0.82
overall   2700          64.33          64.07     87     80  +0.54
```

**rap_next/pred_videomme_f128_rap_R40E_r040_vs_vidcom2.txt**
```
R40E vs vidcom2040: 2700 cau chung (A co 2700, B co 2700)
nhom         n           R40E     vidcom2040  chi A  chi B      z
short      900          75.44          76.00     41     46  -0.54
medium     900          63.89          65.89     40     58  -1.82
long       900          53.67          54.44     35     42  -0.80
overall   2700          64.33          65.44    116    146  -1.85
```

**rap_seeds/pred_videomme_f128_rap_runD_s1_r025_vs_seed0.txt**
```
runD_s1 vs seed0: 2700 cau chung (A co 2700, B co 2700)
nhom         n        runD_s1          seed0  chi A  chi B      z
short      900          73.00          73.78      4     11  -1.81
medium     900          65.00          65.11     14     15  -0.19
long       900          54.44          54.33     12     11  +0.21
overall   2700          64.15          64.41     30     37  -0.86
```

**rap_seeds/pred_videomme_f128_rap_runD_s1_r025_vs_uniform.txt**
```
runD_s1 vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n        runD_s1      Uniform25  chi A  chi B      z
short      900          73.00          72.67     32     29  +0.38
medium     900          65.00          63.22     45     29  +1.86
long       900          54.44          53.33     40     30  +1.20
overall   2700          64.15          63.07    117     88  +2.03
```

**rap_seeds/pred_videomme_f128_rap_runD_s2_r025_vs_seed0.txt**
```
runD_s2 vs seed0: 2700 cau chung (A co 2700, B co 2700)
nhom         n        runD_s2          seed0  chi A  chi B      z
short      900          74.44          73.78     14      8  +1.28
medium     900          65.00          65.11     17     18  -0.17
long       900          53.33          54.33      7     16  -1.88
overall   2700          64.26          64.41     38     42  -0.45
```

**rap_seeds/pred_videomme_f128_rap_runD_s2_r025_vs_uniform.txt**
```
runD_s2 vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n        runD_s2      Uniform25  chi A  chi B      z
short      900          74.44          72.67     40     24  +2.00
medium     900          65.00          63.22     48     32  +1.79
long       900          53.33          53.33     30     30  +0.00
overall   2700          64.26          63.07    118     86  +2.24
```

**rap_seeds/pred_videomme_f128_rap_runE_s1_r025_vs_seed0.txt**
```
runE_s1 vs seed0: 2700 cau chung (A co 2700, B co 2700)
nhom         n        runE_s1          seed0  chi A  chi B      z
short      900          73.11          73.33      8     10  -0.47
medium     900          65.44          65.11     17     14  +0.54
long       900          54.89          54.11     20     13  +1.22
overall   2700          64.48          64.19     45     37  +0.88
```

**rap_seeds/pred_videomme_f128_rap_runE_s1_r025_vs_uniform.txt**
```
runE_s1 vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n        runE_s1      Uniform25  chi A  chi B      z
short      900          73.11          72.67     34     30  +0.50
medium     900          65.44          63.22     54     34  +2.13
long       900          54.89          53.33     45     31  +1.61
overall   2700          64.48          63.07    133     95  +2.52
```

**rap_seeds/pred_videomme_f128_rap_runE_s2_r025_vs_seed0.txt**
```
runE_s2 vs seed0: 2700 cau chung (A co 2700, B co 2700)
nhom         n        runE_s2          seed0  chi A  chi B      z
short      900          73.67          73.33     11      8  +0.69
medium     900          64.44          65.11     14     20  -1.03
long       900          54.11          54.11     15     15  +0.00
overall   2700          64.07          64.19     40     43  -0.33
```

**rap_seeds/pred_videomme_f128_rap_runE_s2_r025_vs_uniform.txt**
```
runE_s2 vs Uniform25: 2700 cau chung (A co 2700, B co 2700)
nhom         n        runE_s2      Uniform25  chi A  chi B      z
short      900          73.67          72.67     36     27  +1.13
medium     900          64.44          63.22     47     36  +1.21
long       900          54.11          53.33     35     28  +0.88
overall   2700          64.07          63.07    118     91  +1.87
```

**rap_vjepa/pred_videomme_f128_rap_V10E_r010_vs_R10E.txt**
```
V10E vs R10E: 2700 cau chung (A co 2700, B co 2700)
nhom         n           V10E           R10E  chi A  chi B      z
short      900          72.33          71.33     20     11  +1.62
medium     900          60.56          61.00     20     24  -0.60
long       900          53.33          52.44     19     11  +1.46
overall   2700          62.07          61.59     59     46  +1.27
```

**rap_vjepa/pred_videomme_f128_rap_V10E_r010_vs_fastvid.txt**
```
V10E vs fastvid010: 2700 cau chung (A co 2700, B co 2700)
nhom         n           V10E     fastvid010  chi A  chi B      z
short      900          72.33          69.56     78     53  +2.18
medium     900          60.56          57.56    101     74  +2.04
long       900          53.33          50.78     79     56  +1.98
overall   2700          62.07          59.30    258    183  +3.57
```

**rap_vjepa/pred_videomme_f128_rap_V10E_r010_vs_holitom.txt**
```
V10E vs holitom010: 2700 cau chung (A co 2700, B co 2700)
nhom         n           V10E     holitom010  chi A  chi B      z
short      900          72.33          71.00     67     55  +1.09
medium     900          60.56          59.56     90     81  +0.69
long       900          53.33          53.44     65     66  -0.09
overall   2700          62.07          61.33    222    202  +0.97
```

**rap_vjepa/pred_videomme_f128_rap_V10E_r010_vs_uniform.txt**
```
V10E vs uniform010: 2700 cau chung (A co 2700, B co 2700)
nhom         n           V10E     uniform010  chi A  chi B      z
short      900          72.33          70.56     52     36  +1.71
medium     900          60.56          56.89     75     42  +3.05
long       900          53.33          51.33     57     39  +1.84
overall   2700          62.07          59.59    184    117  +3.86
```

**rap_vjepa25/pred_videomme_f128_rap_V25E_r025_vs_fastvid.txt**
```
V25E vs fastvid025: 2700 cau chung (A co 2700, B co 2700)
nhom         n           V25E     fastvid025  chi A  chi B      z
short      900          73.78          74.22     53     57  -0.38
medium     900          65.33          60.89    102     62  +3.12
long       900          53.67          53.89     49     51  -0.20
overall   2700          64.26          63.00    204    170  +1.76
```

**rap_vjepa25/pred_videomme_f128_rap_V25E_r025_vs_holitom.txt**
```
V25E vs holitom025: 2700 cau chung (A co 2700, B co 2700)
nhom         n           V25E     holitom025  chi A  chi B      z
short      900          73.78          74.33     48     53  -0.50
medium     900          65.33          62.00     79     49  +2.65
long       900          53.67          55.22     43     57  -1.40
overall   2700          64.26          63.85    170    159  +0.61
```

**rap_vjepa25/pred_videomme_f128_rap_V25E_r025_vs_runE.txt**
```
V25E vs runE: 2700 cau chung (A co 2700, B co 2700)
nhom         n           V25E           runE  chi A  chi B      z
short      900          73.78          73.33     10      6  +1.00
medium     900          65.33          65.11     17     15  +0.35
long       900          53.67          54.11     13     17  -0.73
overall   2700          64.26          64.19     40     38  +0.23
```

**rap_vjepa25/pred_videomme_f128_rap_V25E_r025_vs_uniform.txt**
```
V25E vs uniform025: 2700 cau chung (A co 2700, B co 2700)
nhom         n           V25E     uniform025  chi A  chi B      z
short      900          73.78          72.67     35     25  +1.29
medium     900          65.33          63.22     48     29  +2.17
long       900          53.67          53.33     38     35  +0.35
overall   2700          64.26          63.07    121     89  +2.21
```

## 4. Toc do (bench_speed_rap.py)

**speed/bench_f128.jsonl**
```
30 video, vision encoder = 1892 ms, 84.8 TFLOPs (giong nhau moi cau hinh)

cau hinh | token | nen ms | prefill ms | prefill x | TTFT ms | TTFT x | e2e ms | peak GB | LLM TFLOPs | LLM % | nen TFLOPs | tong %
dense | 16064 | 0.0 | 1441 | 1.0 | 3333 | 1.00 | 3646 | 22.62 | 336.1 | 100.0 | 0.00 | 100.0
uniform 40% | 6425 | 0.3 | 526 | 2.7 | 2419 | 1.38 | 2771 | 19.29 | 110.5 | 32.9 | 0.00 | 46.4
rap 40% | 6425 | 19.7 | 526 | 2.7 | 2438 | 1.37 | 2790 | 19.29 | 110.5 | 32.9 | 0.93 | 46.6
uniform 25% | 4016 | 0.3 | 339 | 4.2 | 2232 | 1.49 | 2593 | 18.43 | 65.9 | 19.6 | 0.00 | 35.8
rap 25% | 4016 | 16.0 | 346 | 4.2 | 2254 | 1.48 | 2610 | 18.43 | 65.9 | 19.6 | 0.78 | 36.0
uniform 15% | 2409 | 0.2 | 201 | 7.2 | 2093 | 1.59 | 2470 | 17.86 | 38.7 | 11.5 | 0.00 | 29.3
rap 15% | 2409 | 14.1 | 201 | 7.2 | 2108 | 1.58 | 2478 | 17.86 | 38.7 | 11.5 | 0.68 | 29.5
uniform 10% | 1606 | 0.1 | 138 | 10.4 | 2030 | 1.64 | 2405 | 17.58 | 25.9 | 7.7 | 0.00 | 26.3
rap 10% | 1606 | 12.8 | 147 | 9.8 | 2052 | 1.62 | 2417 | 17.58 | 25.9 | 7.7 | 0.63 | 26.5
uniform 5% | 803 | 0.1 | 95 | 15.1 | 1988 | 1.68 | 2354 | 17.28 | 13.6 | 4.1 | 0.00 | 23.4
rap 5% | 803 | 12.1 | 83 | 17.3 | 1988 | 1.68 | 2375 | 17.28 | 13.6 | 4.1 | 0.58 | 23.5
uniform 2% | 321 | 0.1 | 45 | 32.4 | 1937 | 1.72 | 2317 | 17.11 | 6.5 | 1.9 | 0.00 | 21.7
rap 2% | 321 | 11.6 | 43 | 33.3 | 1947 | 1.71 | 2329 | 17.11 | 6.5 | 1.9 | 0.55 | 21.8
```

**speed/bench_f128_baselines.jsonl**
```
12 video, vision encoder = 1944 ms, 83.4 TFLOPs (giong nhau moi cau hinh)

cau hinh | token | nen ms | prefill ms | prefill x | TTFT ms | TTFT x | e2e ms | peak GB | LLM TFLOPs | LLM % | nen TFLOPs | tong %
dense | 16058 | 0.0 | 1438 | 1.0 | 3383 | 1.00 | 3699 | 21.62 | 328.0 | 100.0 | 0.00 | 100.0
uniform 40% | 6423 | 0.3 | 525 | 2.7 | 2470 | 1.37 | 2827 | 18.29 | 108.0 | 32.9 | 0.00 | 46.5
fastvid 40% | 5658 | 147.6 | 463 | 3.1 | 2555 | 1.32 | 2914 | 18.01 | 88.3 | 26.9 | 0.65 | 41.9
holitom 40% | 6452 | 183.2 | 525 | 2.7 | 2653 | 1.28 | 3013 | 18.30 | 108.5 | 33.1 | 0.80 | 46.8
uniform 25% | 4014 | 0.3 | 338 | 4.3 | 2283 | 1.48 | 2650 | 17.43 | 64.3 | 19.6 | 0.00 | 35.9
fastvid 25% | 3846 | 94.8 | 321 | 4.5 | 2360 | 1.43 | 2726 | 17.37 | 60.3 | 18.4 | 0.66 | 35.1
holitom 25% | 4047 | 180.9 | 346 | 4.2 | 2471 | 1.37 | 2831 | 17.44 | 64.8 | 19.8 | 0.80 | 36.2
uniform 15% | 2408 | 0.2 | 203 | 7.1 | 2147 | 1.58 | 2519 | 16.85 | 37.7 | 11.5 | 0.00 | 29.4
fastvid 15% | 2332 | 103.4 | 196 | 7.3 | 2244 | 1.51 | 2620 | 16.83 | 36.2 | 11.0 | 0.65 | 29.2
holitom 15% | 2439 | 154.3 | 202 | 7.1 | 2301 | 1.47 | 2677 | 16.87 | 38.2 | 11.6 | 0.80 | 29.8
uniform 10% | 1605 | 0.1 | 138 | 10.4 | 2083 | 1.62 | 2461 | 16.57 | 25.2 | 7.7 | 0.00 | 26.4
fastvid 10% | 1588 | 54.0 | 136 | 10.5 | 2135 | 1.58 | 2514 | 16.56 | 25.0 | 7.6 | 0.64 | 26.5
holitom 10% | 1636 | 154.3 | 138 | 10.4 | 2237 | 1.51 | 2615 | 16.58 | 25.7 | 7.8 | 0.80 | 26.7
uniform 5% | 802 | 0.1 | 84 | 17.1 | 2029 | 1.67 | 2409 | 16.28 | 13.1 | 4.0 | 0.00 | 23.5
fastvid 5% | 765 | 48.2 | 75 | 19.2 | 2068 | 1.64 | 2449 | 16.27 | 12.6 | 3.8 | 0.63 | 23.5
holitom 5% | 834 | 118.2 | 84 | 17.2 | 2146 | 1.58 | 2529 | 16.29 | 13.6 | 4.1 | 0.80 | 23.8
uniform 2% | 321 | 0.1 | 44 | 32.5 | 1989 | 1.70 | 2374 | 16.11 | 6.2 | 1.9 | 0.00 | 21.8
fastvid 2% | 318 | 35.0 | 44 | 32.6 | 2024 | 1.67 | 2410 | 16.11 | 6.1 | 1.9 | 0.63 | 21.9
holitom 2% | 353 | 120.2 | 46 | 31.1 | 2111 | 1.60 | 2497 | 16.12 | 6.6 | 2.0 | 0.80 | 22.1
```

**speed/smoke.jsonl**
```
3 video, vision encoder = 2199 ms, 84.8 TFLOPs (giong nhau moi cau hinh)

cau hinh | token | nen ms | prefill ms | prefill x | TTFT ms | TTFT x | e2e ms | peak GB | LLM TFLOPs | LLM % | nen TFLOPs | tong %
dense | 16128 | 0.0 | 1437 | 1.0 | 3635 | 1.00 | 4030 | 21.81 | 335.8 | 100.0 | 0.00 | 100.0
uniform 10% | 1612 | 0.3 | 234 | 6.1 | 2433 | 1.49 | 2727 | 16.74 | 25.8 | 7.7 | 0.00 | 26.3
rap 10% | 1612 | 144.2 | 137 | 10.5 | 2480 | 1.47 | 2867 | 16.74 | 25.8 | 7.7 | 0.63 | 26.4
```

## 5. So lieu validation cuoi cung cua tung lan train (dong cuoi *.metrics.jsonl)

- `rap/R10E.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7957, pseudo/ce_uniform=1.8721, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0739, pseudo/kl_uniform=0.1545, pseudo/erank_out=21.6574, pseudo/erank_anchor=21.6600, pseudo/cos_out=0.2904, pseudo/cos_anchor=0.2914, real/ce_rap=1.2962, real/ce_uniform=1.3635, real/ce_dense=1.2682, real/kl_rap=0.0376, real/kl_uniform=0.0940, real/erank_out=22.1103, real/erank_anchor=22.0826, real/cos_out=0.2855, real/cos_anchor=0.2978, pseudo/gap_closed=0.5556, real/gap_closed=0.7058
- `rap/runA.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.4026, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/erank_out=49.2560, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2907, pseudo/cos_anchor=0.2984, real/ce_rap=1.0537, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/erank_out=50.3868, real/erank_anchor=50.4631, real/cos_out=0.2903, real/cos_anchor=0.3020, pseudo/gap_closed=4.2898, real/gap_closed=5.8602
- `rap/runB.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7416, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0584, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6583, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2975, pseudo/cos_anchor=0.2984, real/ce_rap=1.2854, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0175, real/kl_uniform=0.0371, real/erank_out=50.5196, real/erank_anchor=50.4631, real/cos_out=0.2960, real/cos_anchor=0.3020, pseudo/gap_closed=0.9299, real/gap_closed=0.6109
- `rap/runC.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7387, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0597, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6984, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2974, pseudo/cos_anchor=0.2984, real/ce_rap=1.2765, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0205, real/kl_uniform=0.0371, real/erank_out=50.5660, real/erank_anchor=50.4631, real/cos_out=0.2926, real/cos_anchor=0.3020, qa/ce_rap=0.0992, qa/ce_uniform=0.4402, qa/ce_dense=0.4764, qa/kl_rap=0.0545, qa/kl_uniform=0.0137, qa/acc_rap=0.9634, qa/acc_uniform=0.9185, qa/acc_dense=0.9095, qa/erank_out=50.8773, qa/erank_anchor=50.7874, qa/cos_out=0.2827, qa/cos_anchor=0.2920, pseudo/gap_closed=0.9592, real/gap_closed=0.8112, qa/gap_closed=-9.4285
- `rap/runD.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7426, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0597, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6495, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2965, pseudo/cos_anchor=0.2984, real/ce_rap=1.2865, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0177, real/kl_uniform=0.0371, real/erank_out=50.5114, real/erank_anchor=50.4631, real/cos_out=0.2965, real/cos_anchor=0.3020, pseudo/gap_closed=0.9199, real/gap_closed=0.5846
- `rap/runE.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7752, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0734, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6801, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2967, pseudo/cos_anchor=0.2984, real/ce_rap=1.2819, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0168, real/kl_uniform=0.0371, real/erank_out=50.5103, real/erank_anchor=50.4631, real/cos_out=0.2954, real/cos_anchor=0.3020, pseudo/gap_closed=0.5968, real/gap_closed=0.6898
- `rap/runG.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7602, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0684, pseudo/kl_uniform=0.1285, pseudo/erank_out=48.5705, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.3408, pseudo/cos_anchor=0.2984, real/ce_rap=1.2795, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0156, real/kl_uniform=0.0371, real/erank_out=49.3861, real/erank_anchor=50.4631, real/cos_out=0.3357, real/cos_anchor=0.3020, pseudo/gap_closed=0.7459, real/gap_closed=0.7436
- `rap/runH.pt.metrics.jsonl` step 3000: pseudo/ce_rap=1.7698, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0725, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6516, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2985, pseudo/cos_anchor=0.2984, real/ce_rap=1.2827, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0170, real/kl_uniform=0.0371, real/erank_out=50.5134, real/erank_anchor=50.4631, real/cos_out=0.2949, real/cos_anchor=0.3020, pseudo/gap_closed=0.6505, real/gap_closed=0.6721
- `rap_elastic/ELA.pt.metrics.jsonl` step 6000: real@25/ce_rap=1.2788, real@25/ce_uniform=1.3123, real@25/ce_dense=1.2682, real@25/kl_rap=0.0176, real@25/kl_uniform=0.0371, real@25/erank_out=50.5232, real@25/erank_anchor=50.4631, real@25/cos_out=0.2989, real@25/cos_anchor=0.3020, real@10/ce_rap=1.2946, real@10/ce_uniform=1.3635, real@10/ce_dense=1.2682, real@10/kl_rap=0.0392, real@10/kl_uniform=0.0940, real@10/erank_out=22.0986, real@10/erank_anchor=22.0826, real@10/cos_out=0.2921, real@10/cos_anchor=0.2978, real@10/gap_closed=0.7234, real@25/gap_closed=0.7590
- `rap_elastic/ELA_s1.pt.metrics.jsonl` step 6000: real@25/ce_rap=1.2828, real@25/ce_uniform=1.3123, real@25/ce_dense=1.2682, real@25/kl_rap=0.0170, real@25/kl_uniform=0.0371, real@25/erank_out=50.5057, real@25/erank_anchor=50.4631, real@25/cos_out=0.2990, real@25/cos_anchor=0.3020, real@10/ce_rap=1.2968, real@10/ce_uniform=1.3635, real@10/ce_dense=1.2682, real@10/kl_rap=0.0391, real@10/kl_uniform=0.0940, real@10/erank_out=22.0928, real@10/erank_anchor=22.0826, real@10/cos_out=0.2929, real@10/cos_anchor=0.2978, real@10/gap_closed=0.6998, real@25/gap_closed=0.6686
- `rap_elastic/M3D-B.pt.metrics.jsonl` step 4000: real/ce_rap=1.2841, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0163, real/kl_uniform=0.0371, real/erank_out=50.5256, real/erank_anchor=50.4631, real/cos_out=0.2954, real/cos_anchor=0.3020, real/gap_closed=0.6406
- `rap_elastic/M3D-BD.pt.metrics.jsonl` step 4000: real/ce_rap=1.2826, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0169, real/kl_uniform=0.0371, real/erank_out=50.5152, real/erank_anchor=50.4631, real/cos_out=0.2945, real/cos_anchor=0.3020, real/gap_closed=0.6733
- `rap_elastic/R02W3.pt.metrics.jsonl` step 4000: real/ce_rap=1.3382, real/ce_uniform=1.4731, real/ce_dense=1.2682, real/kl_rap=0.1048, real/kl_uniform=0.2106, real/erank_out=4.8493, real/erank_anchor=4.8565, real/cos_out=0.3097, real/cos_anchor=0.3173, real/gap_closed=0.6582
- `rap_next/R02E.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.9317, pseudo/ce_uniform=2.0278, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.2824, pseudo/kl_uniform=0.4129, pseudo/erank_out=4.8368, pseudo/erank_anchor=4.8693, pseudo/cos_out=0.2977, pseudo/cos_anchor=0.2697, real/ce_rap=1.3411, real/ce_uniform=1.4731, real/ce_dense=1.2682, real/kl_rap=0.1076, real/kl_uniform=0.2106, real/erank_out=4.8413, real/erank_anchor=4.8565, real/cos_out=0.3142, real/cos_anchor=0.3173, pseudo/gap_closed=0.3278, real/gap_closed=0.6444
- `rap_next/R05E.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.8053, pseudo/ce_uniform=1.9087, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.1293, pseudo/kl_uniform=0.2440, pseudo/erank_out=11.2043, pseudo/erank_anchor=11.2453, pseudo/cos_out=0.3122, pseudo/cos_anchor=0.3115, real/ce_rap=1.3074, real/ce_uniform=1.4171, real/ce_dense=1.2682, real/kl_rap=0.0678, real/kl_uniform=0.1557, real/erank_out=11.5741, real/erank_anchor=11.5722, real/cos_out=0.2828, real/cos_anchor=0.2966, pseudo/gap_closed=0.5939, real/gap_closed=0.7368
- `rap_next/R10D.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7592, pseudo/ce_uniform=1.8721, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0583, pseudo/kl_uniform=0.1545, pseudo/erank_out=21.6649, pseudo/erank_anchor=21.6600, pseudo/cos_out=0.2892, pseudo/cos_anchor=0.2914, real/ce_rap=1.2935, real/ce_uniform=1.3635, real/ce_dense=1.2682, real/kl_rap=0.0401, real/kl_uniform=0.0940, real/erank_out=22.1345, real/erank_anchor=22.0826, real/cos_out=0.2878, real/cos_anchor=0.2978, pseudo/gap_closed=0.8212, real/gap_closed=0.7341
- `rap_next/R15E.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7670, pseudo/ce_uniform=1.8368, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0494, pseudo/kl_uniform=0.1095, pseudo/erank_out=31.4292, pseudo/erank_anchor=31.4467, pseudo/cos_out=0.2906, pseudo/cos_anchor=0.2913, real/ce_rap=1.2994, real/ce_uniform=1.3480, real/ce_dense=1.2682, real/kl_rap=0.0330, real/kl_uniform=0.0704, real/erank_out=32.0352, real/erank_anchor=31.9993, real/cos_out=0.2897, real/cos_anchor=0.2992, pseudo/gap_closed=0.6828, real/gap_closed=0.6085
- `rap_next/R40E.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7379, pseudo/ce_uniform=1.7978, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0156, pseudo/kl_uniform=0.0431, pseudo/erank_out=74.9459, pseudo/erank_anchor=74.8615, pseudo/cos_out=0.2928, pseudo/cos_anchor=0.2945, real/ce_rap=1.2773, real/ce_uniform=1.2879, real/ce_dense=1.2682, real/kl_rap=0.0078, real/kl_uniform=0.0186, real/erank_out=76.6608, real/erank_anchor=76.6084, real/cos_out=0.2989, real/cos_anchor=0.3023, pseudo/gap_closed=0.9466, real/gap_closed=0.5396
- `rap_night/R01E.pt.metrics.jsonl` step 4000: real/ce_rap=1.3767, real/ce_uniform=1.5270, real/ce_dense=1.2682, real/kl_rap=0.1611, real/kl_uniform=0.2917, real/erank_out=2.4737, real/erank_anchor=2.4885, real/cos_out=0.2902, real/cos_anchor=0.2659, real/gap_closed=0.5810
- `rap_night/R01W3.pt.metrics.jsonl` step 750: real/ce_rap=1.3787, real/ce_uniform=1.5270, real/ce_dense=1.2682, real/kl_rap=0.1747, real/kl_uniform=0.2917, real/erank_out=2.4578, real/erank_anchor=2.4885, real/cos_out=0.3288, real/cos_anchor=0.2659, real/gap_closed=0.5732
- `rap_night/R10KA.pt.metrics.jsonl` step 4000: real/ce_rap=1.2945, real/ce_uniform=1.3635, real/ce_dense=1.2682, real/kl_rap=0.0384, real/kl_uniform=0.0940, real/erank_out=22.1174, real/erank_anchor=22.0826, real/cos_out=0.2855, real/cos_anchor=0.2978, real/gap_closed=0.7244
- `rap_place/P10.pt.metrics.jsonl` step 1000: prior=2.3350, kl_uniform_anchor=0.0377, kl_greedy=0.0377, moved_frac=0.0000, n=8
- `rap_place/old_fixedprior/P10.pt.metrics.jsonl` step 500: kl_uniform_anchor=0.0377, kl_greedy=0.0377, moved_frac=0.0000, n=8
- `rap_seeds/runD_s1.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7407, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0605, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6490, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2949, pseudo/cos_anchor=0.2984, real/ce_rap=1.2822, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0172, real/kl_uniform=0.0371, real/erank_out=50.5269, real/erank_anchor=50.4631, real/cos_out=0.2940, real/cos_anchor=0.3020, pseudo/gap_closed=0.9393, real/gap_closed=0.6818
- `rap_seeds/runD_s2.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7412, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0590, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6550, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2945, pseudo/cos_anchor=0.2984, real/ce_rap=1.2865, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0176, real/kl_uniform=0.0371, real/erank_out=50.5381, real/erank_anchor=50.4631, real/cos_out=0.2944, real/cos_anchor=0.3020, pseudo/gap_closed=0.9345, real/gap_closed=0.5843
- `rap_seeds/runE_s1.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7745, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0725, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6383, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2984, pseudo/cos_anchor=0.2984, real/ce_rap=1.2849, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0175, real/kl_uniform=0.0371, real/erank_out=50.5063, real/erank_anchor=50.4631, real/cos_out=0.2951, real/cos_anchor=0.3020, pseudo/gap_closed=0.6045, real/gap_closed=0.6226
- `rap_seeds/runE_s2.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7750, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0733, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.7099, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2954, pseudo/cos_anchor=0.2984, real/ce_rap=1.2810, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0173, real/kl_uniform=0.0371, real/erank_out=50.5016, real/erank_anchor=50.4631, real/cos_out=0.2949, real/cos_anchor=0.3020, pseudo/gap_closed=0.5990, real/gap_closed=0.7103
- `rap_vjepa/V10E.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.8146, pseudo/ce_uniform=1.8721, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0816, pseudo/kl_uniform=0.1545, pseudo/erank_out=21.6563, pseudo/erank_anchor=21.6600, pseudo/cos_out=0.2919, pseudo/cos_anchor=0.2914, real/ce_rap=1.2999, real/ce_uniform=1.3635, real/ce_dense=1.2682, real/kl_rap=0.0396, real/kl_uniform=0.0940, real/erank_out=22.1060, real/erank_anchor=22.0826, real/cos_out=0.2887, real/cos_anchor=0.2978, pseudo/gap_closed=0.4184, real/gap_closed=0.6676
- `rap_vjepa/V10_s1.pt.metrics.jsonl` step 3000: rec=0.6064, rec_anchor_only=0.6103, delta_ratio=0.1550, n=16
- `rap_vjepa25/V25E.pt.metrics.jsonl` step 4000: pseudo/ce_rap=1.7710, pseudo/ce_uniform=1.8355, pseudo/ce_dense=1.7346, pseudo/kl_rap=0.0725, pseudo/kl_uniform=0.1285, pseudo/erank_out=49.6538, pseudo/erank_anchor=49.6299, pseudo/cos_out=0.2969, pseudo/cos_anchor=0.2984, real/ce_rap=1.2838, real/ce_uniform=1.3123, real/ce_dense=1.2682, real/kl_rap=0.0175, real/kl_uniform=0.0371, real/erank_out=50.5182, real/erank_anchor=50.4631, real/cos_out=0.2950, real/cos_anchor=0.3020, pseudo/gap_closed=0.6391, real/gap_closed=0.6460
- `rap_vjepa25/V25_s1.pt.metrics.jsonl` step 3000: rec=0.6061, rec_anchor_only=0.6106, delta_ratio=0.1582, n=16

## 6. Dong '[train...] done' trong log train

- `dsta_scorer_2a_train.log`: [train_dsta_scorer_2a] done -> saved scorer to /home/hoang/Long_Video_Understanding/KTV/outputs/dsta_scorer_2a.pt
- `dsta_scorer_2a_train2.log`: [train_dsta_scorer_2a] done -> saved scorer to /home/hoang/Long_Video_Understanding/KTV/outputs/dsta_scorer_2a.pt
- `kitoke_baseline/dsta_v2_train.log`: [train_dsta_scorer_2a] done -> saved scorer to /home/hoang/Long_Video_Understanding/KTV/outputs/dsta_scorer_2a_v2.pt
- `rap/R10E.train.log`: [train_rap] done step=4000 -> outputs/rap/R10E.pt
- `rap/runA.train.log`: [train_rap] done step=4000 -> outputs/rap/runA.pt
- `rap/runB.train.log`: [train_rap] done step=4000 -> outputs/rap/runB.pt
- `rap/runC.train.log`: [train_rap] done step=4000 -> outputs/rap/runC.pt
- `rap/runD.train.log`: [train_rap] done step=4000 -> outputs/rap/runD.pt
- `rap/runE.train.log`: [train_rap] done step=4000 -> outputs/rap/runE.pt
- `rap/runG.train.log`: [train_rap] done step=4000 -> outputs/rap/runG.pt
- `rap/runH.train.log`: [train_rap] done step=3000 -> outputs/rap/runH.pt
- `rap_elastic/ELA.train.log`: [train_rap] done step=6000 -> outputs/rap_elastic/ELA.pt
- `rap_elastic/ELA_s1.train.log`: [train_rap] done step=6000 -> outputs/rap_elastic/ELA_s1.pt
- `rap_elastic/M3D-B.train.log`: [train_rap] done step=4000 -> outputs/rap_elastic/M3D-B.pt
- `rap_elastic/M3D-BD.train.log`: [train_rap] done step=4000 -> outputs/rap_elastic/M3D-BD.pt
- `rap_elastic/R02W3.train.log`: [train_rap] done step=4000 -> outputs/rap_elastic/R02W3.pt
- `rap_next/R02E.train.log`: [train_rap] done step=4000 -> outputs/rap_next/R02E.pt
- `rap_next/R05E.train.log`: [train_rap] done step=4000 -> outputs/rap_next/R05E.pt
- `rap_next/R10D.train.log`: [train_rap] done step=4000 -> outputs/rap_next/R10D.pt
- `rap_next/R15E.train.log`: [train_rap] done step=4000 -> outputs/rap_next/R15E.pt
- `rap_next/R40E.train.log`: [train_rap] done step=4000 -> outputs/rap_next/R40E.pt
- `rap_night/R01E.train.log`: [train_rap] done step=4000 -> outputs/rap_night/R01E.pt
- `rap_night/R10KA.train.log`: [train_rap] done step=4000 -> outputs/rap_night/R10KA.pt
- `rap_seeds/runD_s1.train.log`: [train_rap] done step=4000 -> outputs/rap_seeds/runD_s1.pt
- `rap_seeds/runD_s2.train.log`: [train_rap] done step=4000 -> outputs/rap_seeds/runD_s2.pt
- `rap_seeds/runE_s1.train.log`: [train_rap] done step=4000 -> outputs/rap_seeds/runE_s1.pt
- `rap_seeds/runE_s2.train.log`: [train_rap] done step=4000 -> outputs/rap_seeds/runE_s2.pt
- `rap_vjepa/V10E.train.log`: [train_rap] done step=4000 -> outputs/rap_vjepa/V10E.pt
- `rap_vjepa/V10_s1.train.log`: [train_rap_vjepa] done step=3000 -> outputs/rap_vjepa/V10_s1.pt
- `rap_vjepa25/V25E.train.log`: [train_rap] done step=4000 -> outputs/rap_vjepa25/V25E.pt
- `rap_vjepa25/V25_s1.train.log`: [train_rap_vjepa] done step=3000 -> outputs/rap_vjepa25/V25_s1.pt
- `train_perceiver_stage2.log`: [train_perceiver_stage2] done -> saved policy to /home/hoang/Long_Video_Understanding/KTV/outputs/perceiver_stage2_smoke.pt
- `train_rl_gating_2stage_exp1.log`: [train_rl_gating_2stage] done -> saved policy to /home/hoang/Long_Video_Understanding/KTV/outputs/rl_gating_2stage_exp1.pt
- `train_rl_gating_2stage_exp2.log`: [train_rl_gating_2stage] done -> saved policy to /home/hoang/Long_Video_Understanding/KTV/outputs/rl_gating_2stage_exp2.pt
- `train_rl_gating_2stage_exp3.log`: [train_rl_gating_2stage] done -> saved policy to /home/hoang/Long_Video_Understanding/KTV/outputs/rl_gating_2stage_exp3.pt
- `train_rl_gating_exp4_resume.log`: [train_rl_gating] done -> saved policy to /home/hoang/Long_Video_Understanding/KTV/outputs/rl_gating_policy_exp4.pt
- `train_rl_gating_run1.log`: [train_rl_gating] done -> saved policy to /home/hoang/Long_Video_Understanding/KTV/outputs/rl_gating_policy_run1.pt
