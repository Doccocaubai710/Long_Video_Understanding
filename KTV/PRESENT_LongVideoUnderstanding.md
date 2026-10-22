# Long-Form Video Understanding — Tổng quan để trình bày

---

## 0. Mở đầu — Bài toán

Khi đưa một video dài vào các mô hình Video-LLM hiện đại (ví dụ Qwen2.5-VL), video được mã hóa thành **rất nhiều token thị giác** — có thể lên tới hàng chục nghìn token cho một video vài phút.

**Vấn đề:**
- Mô hình ngôn ngữ không kham nổi số lượng token đó (giới hạn bộ nhớ, chi phí tính toán tăng rất nhanh theo số token).
- Không thể đưa hết mọi chi tiết vào — buộc phải **nén**: bỏ bớt token, chỉ giữ lại phần "đáng giá".

**Câu hỏi nghiên cứu trung tâm:**
> Nén thế nào để bỏ đi nhiều nhất mà vẫn giữ được đủ thông tin trả lời đúng câu hỏi?

---

## 1. Các phương pháp xử lý Long Video Understanding hiện có

Có thể can thiệp giảm số token ở **ba tầng** khác nhau trong pipeline:

```
Video → [1. Chọn khung hình] → [2. Nén token sau vision encoder] → LLM → [3. Cắt tỉa động trong lúc LLM suy luận] → Trả lời
```

| Tầng can thiệp | Ý tưởng | Ví dụ phương pháp |
|---|---|---|
| **1. Chọn khung hình** | Chọn trước một số khung hình đại diện, bỏ các khung còn lại | KTV, AKS, F2C |
| **2. Nén token sau vision encoder** | Giữ nguyên các khung hình, giảm số token mô tả mỗi khung — *nhóm mình đang theo hướng này* | FastVID, VisionZip, PruneVID, SCORE |
| **3. Cắt tỉa động trong LLM** | Trong lúc mô hình đang suy luận, động loại bỏ token không còn cần thiết | FastV, DyCoke, AdaTP |

Một trục phân loại khác cũng quan trọng: **"query-agnostic"** (nén một lần cho cả video, không cần biết câu hỏi trước, dùng lại được cho mọi câu hỏi) vs **"query-aware"** (nén khác nhau tùy câu hỏi, phải làm lại cho mỗi câu hỏi — tốn hơn). Toàn bộ hướng đang theo đuổi là **query-agnostic xuyên suốt** — nén xong dùng lại được cho mọi câu hỏi của cùng video đó, không cần tính lại.

*(KTV — phương pháp nền tảng ban đầu của nhóm trước khi chuyển sang FastVID — thực ra là mô hình **lai**: bước chọn khung hình query-agnostic, nhưng bước quyết định "mỗi khung hình được bao nhiêu token" lại query-aware, dựa trên độ tương đồng CLIP giữa khung hình và câu hỏi — nên phải tính lại cho từng câu hỏi, không cache được. Đây là điểm khác biệt quan trọng với hướng hiện tại: hướng hiện tại cố tình làm bước "phân bổ ngân sách" cũng query-agnostic để cache được, đổi lại phải tìm tín hiệu khác (không dùng câu hỏi) để đo "đoạn nào đáng token hơn" — xem mục 2.)*

**Một hướng đã được đề xuất nhưng không theo:** dùng Reinforcement Learning kết hợp V-JEPA để huấn luyện một "policy" chọn khung hình, với phần thưởng (reward) là khả năng tái tạo lại hình ảnh gốc từ các khung được chọn. Không theo hướng này vì: các phương pháp SOTA thực tế trong nhánh nén token (SCORE) dùng tín hiệu **đúng/sai câu trả lời** làm phần thưởng, không phải khả năng tái tạo hình ảnh — và điều này đã được kiểm chứng thực nghiệm là hiệu quả hơn.

---

## 2. Ý tưởng nền tảng: FastVID và điểm muốn cải thiện

Pipeline hiện tại xây dựng trên nền một phương pháp đã công bố — **FastVID** — rồi cải thiện một điểm cụ thể.

### Pipeline FastVID gốc

```
Video → Chia thành các ĐOẠN → Ngân sách token chia ĐỀU theo độ dài đoạn → Chọn token trong mỗi đoạn → LLM
                                        (không quan tâm đoạn nào
                                         nhiều/ít thông tin)
```

Bước "chọn token trong mỗi đoạn" gồm 2 phần bổ trợ nhau:
- **Token "đại diện"** — tìm một vài token tiêu biểu cho mỗi vùng nội dung lặp lại (nền, bầu trời...), gộp cả vùng đó lại thành 1-2 token thay vì giữ tất cả.
- **Token "nổi bật"** — giữ riêng những token mà mô hình tự đánh giá là quan trọng (dựa vào tín hiệu attention nội bộ).

### Hạn chế của FastVID — chính cộng đồng cũng chỉ ra

FastVID cấp ngân sách token **đều nhau** cho mọi đoạn, bất kể đoạn đó là một cảnh tĩnh nhàm chán hay một đoạn hành động dồn dập nhiều thông tin. Một nghiên cứu đánh giá lại (KiToke) phê bình đúng điểm này: FastVID không phân biệt được đoạn nào giàu thông tin, đoạn nào nghèo — vấn đề càng nghiêm trọng khi ngân sách token càng ít.

### Ý tưởng cải thiện

**Thay bước "chia đều" bằng bước "chia theo độ giàu thông tin của từng đoạn"** — đoạn nhiều nội dung được cấp nhiều token hơn, đoạn đơn điệu được cấp ít hơn (nhưng không bao giờ về 0).

```
FastVID gốc:            [Chia đoạn] → [Chia ĐỀU]        → [Chọn token trong đoạn] → LLM
Pipeline hiện tại:       [Chia đoạn] → [Chia KHÔNG ĐỀU]  → [Chọn token trong đoạn] → LLM
                                         ▲
                                    ĐÓNG GÓP MỚI — phần
                                    còn lại giữ nguyên
                                    như FastVID
```

Đây chính xác là phần mới của nhóm — mọi phần khác của pipeline giữ nguyên theo thiết kế gốc của FastVID.

*(Một nguồn tham khảo khác ảnh hưởng thiết kế: SCORE — dùng tín hiệu về mức độ thay đổi giữa các khung hình làm một phần đầu vào cho việc chọn token. Ý tưởng này sẽ được nhắc lại ở mục 5.)*

### So sánh nhanh: "phân bổ ngân sách không đều" — KTV làm khác gì hướng hiện tại

Thú vị là KTV (mục 1) cũng có bước "phân bổ ngân sách không đều theo khung hình" giống ý tưởng cải thiện ở trên — nhưng đi theo hướng ngược lại:

| | Dựa vào gì để phân bổ | Query-aware? | Cache được không? |
|---|---|---|---|
| KTV | Độ tương đồng CLIP giữa khung hình và **câu hỏi** | Có | Không — tính lại mỗi câu hỏi |
| Hướng hiện tại | Độ "giàu nội dung" của đoạn (không cần câu hỏi) | Không | Có — tính 1 lần/video |

Hai cách tiếp cận hợp lý ở hai điểm cân bằng khác nhau: KTV đánh đổi chi phí tính toán lấy độ chính xác theo đúng câu hỏi; hướng hiện tại đánh đổi ngược lại — chấp nhận không "biết" câu hỏi để đổi lấy chi phí rẻ hơn nhiều khi một video có nhiều câu hỏi (như VideoMME, trung bình 3 câu hỏi/video).

---

## 3. Dataset & Setup hiện tại

| | |
|---|---|
| **Bộ dữ liệu đánh giá** | VideoMME — 2.700 câu hỏi trắc nghiệm (4 đáp án), trải trên 900 video |
| **Phân nhóm** | 3 nhóm theo độ dài video: ngắn / trung bình / dài (900 câu mỗi nhóm) |
| **Mô hình nền (backbone)** | Qwen2.5-VL-7B-Instruct |

Một bước quan trọng đã hoàn tất trước khi tin bất kỳ số liệu nào: **kiểm chứng lại toàn bộ quy trình đánh giá khớp đúng với chuẩn mà các paper khác dùng để báo cáo kết quả** (cùng cách hỏi, cùng cách chấm điểm, cùng cách lấy mẫu video). Nếu bỏ qua bước này, mọi so sánh sau đó đều có thể sai lệch mà không biết — đây là nền tảng để các kết luận ở mục 4 đáng tin cậy.

---

## 4. Các vấn đề hiện tại

### 4.1 Tin tốt: nén sâu gần như "miễn phí"

Ngay cả cách nén đơn giản nhất — lấy token **cách đều nhau**, không cần thuật toán thông minh gì — cũng giữ được gần như toàn bộ độ chính xác so với không nén gì cả, ở mức ngân sách vừa phải (~36% số token). Điều này cho thấy: video có độ dư thừa thông tin rất lớn, hướng đi "nén" nói chung là khả thi.

### 4.2 Vấn đề: cách làm "thông minh" hiện tại lại thua cách làm đơn giản

Đây là phát hiện quan trọng nhất cần trình bày:

| Cách chọn token | Ngân sách ~36% | Ngân sách ~7% (nén sâu hơn) |
|---|---|---|
| Không nén (dùng 100% token) | 64.3% | — |
| Lấy đều (không cần thuật toán) | **64.4%** | **60.3%** |
| Pipeline hiện tại (chia đoạn không đều + chọn thông minh) | 62.0% | 58.7% |

**Ở cả hai mức ngân sách, cách làm "thông minh" đều thua cách lấy đều đơn giản.** Đây là tín hiệu cần điều tra kỹ, không phải kết luận vội "ý tưởng sai" — cần tìm xem vấn đề nằm ở khâu nào trong pipeline.

**Thử tắt từng phần một để khoanh vùng** (ở mức ngân sách ~36%, so với mốc lấy đều 64.4%):

| Cấu hình thử nghiệm | Overall |
|---|---|
| Lấy đều (mốc so sánh) | **64.4%** |
| Pipeline đầy đủ | 62.0% |
| Tắt riêng phần "chọn token đại diện" (DTM) | 62.2% |
| Tắt riêng phần "phân bổ ngân sách không đều" (dùng ngân sách gần như đều) | 62.4% |
| Tắt riêng phần "xếp hạng token nổi bật" (chọn ngẫu nhiên trong đoạn thay vì xếp hạng) | 62.4% |

**Kết quả bất ngờ:** tắt riêng lẻ từng phần đều chỉ cải thiện rất ít (62.0% → 62.2-62.4%) — không phần nào, khi tắt một mình, giải thích được phần lớn khoảng cách với mốc 64.4%. Điều này cho thấy vấn đề không nằm gọn ở một bước, mà cần nhìn sâu hơn — dẫn đến bước chẩn đoán trực quan ở mục 4.3.

### 4.3 Chẩn đoán: tìm ra thành phần gây lỗi cụ thể

Bằng cách trực quan hóa **chính xác token nào bị giữ, token nào bị bỏ** trên từng khung hình thật, và đo đạc trên nhiều video, phát hiện được:

```
┌─────────────────────────┐
│  ▓▓▓▓▓▓  (nền/góc ảnh)  │   ← Token "đại diện" (DTM) có xu hướng
│  ▓▓░░▓▓                 │      RƠI VÀO ĐÂY — vùng ít thông tin
│  ▓▓░👤░▓▓ (chủ thể)      │
│  ▓▓░░▓▓                 │   ← Token "nổi bật" (salience) lại
│  ▓▓▓▓▓▓                 │      tập trung ĐÚNG vùng trung tâm — ổn
└─────────────────────────┘
```

Đo trên 30 video, tính khoảng cách trung bình từ token được giữ đến tâm khung hình (0 = giữa ảnh, càng lớn càng gần góc):

| Nhóm token | Khoảng cách tới tâm ảnh |
|---|---|
| Trung bình mọi vị trí trong ảnh (mốc so sánh) | 0.38 |
| Token do cơ chế "đại diện" (DTM) chọn | **0.46 — lệch rõ về góc/rìa** |
| Token do cơ chế "nổi bật" (salience) chọn | **0.33 — lệch về giữa, hợp lý** |

**Kết luận chẩn đoán:** một trong hai cơ chế chọn token (cơ chế tìm token "đại diện") đang có xu hướng chọn nhầm vùng nền/góc ảnh ít giá trị, thay vì vùng trung tâm quan trọng. Cơ chế còn lại (chọn token "nổi bật") thực ra hoạt động hợp lý — ban đầu nghi ngờ cả hai, nhưng số liệu đo được cho thấy chỉ một trong hai là vấn đề thật.

Ngoài ra còn phát hiện: với những video có nội dung thay đổi liên tục, việc "chia đoạn" bị vỡ vụn thành quá nhiều đoạn nhỏ, khiến bước "phân bổ không đều" gần như mất tác dụng cho loại video đó.

### 4.4 Vấn đề còn mở

Có một khoảng cách khoảng 4 điểm phần trăm với kết quả một paper cùng hướng đã công bố, hiện chưa xác định được nguyên nhân chính xác — đang có giả thuyết (khác biệt về số lượng khung hình lấy mẫu) nhưng chưa kiểm chứng.

---

## 5. Hướng đi tiếp theo: thử nghiệm V-JEPA

### Vì sao cần một tín hiệu tốt hơn

Cơ chế "chọn token nổi bật" hiện tại đo **mức độ thay đổi giữa các khung hình liên tiếp** (chuyển động). Tín hiệu này đơn giản, rẻ, nhưng có giới hạn: nó đo *chuyển động*, không đo *ý nghĩa*. Một vật thể quan trọng nhưng đứng yên (chữ trên biển hiệu, người đang ngồi nói chuyện) có thể bị đánh giá thấp chỉ vì nó không "động".

### V-JEPA là gì (giải thích trực giác, không công thức)

V-JEPA là một mô hình học cách hiểu video theo cách sau: **che một phần lớn của video lại, rồi yêu cầu mô hình đoán xem phần bị che "trông như thế nào" — không phải ở mức điểm ảnh, mà ở mức ý nghĩa/biểu diễn nội dung.**

```
Video gốc:        [thấy được] [thấy được] [BỊ CHE] [thấy được] [BỊ CHE]
                        │           │         │          │        │
                        └───────────┴────→ Mô hình đoán phần bị che
                                              dựa trên toàn bộ phần
                                              còn lại (không chỉ 1
                                              khung ngay trước)
```

**Ý tưởng dùng cho việc chọn token:** vùng nào mô hình **đoán sai nhiều** (khó đoán từ ngữ cảnh xung quanh) → vùng đó chứa thông tin "bất ngờ", đáng giữ lại. Vùng nào đoán đúng dễ dàng → dư thừa, có thể bỏ.

**Khác biệt cốt lõi so với tín hiệu hiện tại:** tín hiệu cũ chỉ so sánh với đúng 1 khung ngay trước đó; V-JEPA "nhìn" toàn bộ video xung quanh (nhiều khung trước/sau, nhiều vị trí không gian) trước khi đoán — nên phân biệt được "chuyển động đều đặn, dễ đoán" với "sự kiện thật sự mới, khó đoán".

### Dự định tích hợp

Không chỉ thay cho cơ chế "chọn token nổi bật" — vì đã xác định cơ chế đó không phải vấn đề chính — mà còn cân nhắc dùng cho chính cơ chế "chọn token đại diện" (nơi vừa phát hiện có thiên lệch về góc ảnh), vì tín hiệu V-JEPA đo được "độ quan trọng" theo ngữ nghĩa, có thể sửa đúng đúng lỗi vừa tìm ra.

### Trạng thái hiện tại

Đã hoàn tất phần code (tính toán tín hiệu V-JEPA cho từng video và tích hợp vào pipeline chọn token) — đang ở bước chạy thử nghiệm để đánh giá.

---

## 6. Tóm tắt nhanh

| Đã làm | Đang ở đâu | Bước tiếp theo |
|---|---|---|
| Kiểm chứng lại toàn bộ quy trình đánh giá khớp chuẩn literature | Xác nhận: nén sâu gần như miễn phí, nhưng cách chọn token "thông minh" hiện tại thua cách lấy đều đơn giản | Thử nghiệm tín hiệu V-JEPA cho cả hai cơ chế chọn token |
| Trực quan hóa + đo đạc định lượng, tìm ra chính xác thành phần gây lỗi (không phải đoán) | Đã cô lập được: lỗi nằm ở cơ chế "đại diện", không phải "nổi bật" | Kiểm chứng lại toàn bộ pipeline sau khi sửa, so sánh lại với cách lấy đều đơn giản |

---

## Phụ lục A — Bảng số liệu đầy đủ

*(Dành cho phần hỏi-đáp sau trình bày — nội dung chính ở mục 4 đã đủ để hiểu bức tranh tổng thể, không cần đọc phụ lục này khi thuyết trình.)*

### A.1 Toàn bộ các lần chạy, chia theo 3 nhóm độ dài video

| Cấu hình | Ngân sách | Overall | Short | Medium | Long |
|---|---|---|---|---|---|
| Không nén (dense) | 100% (~14-16K token) | 63.33% | 73.33 | 63.22 | 53.44 |
| Pipeline hiện tại, ngân sách gần đầy (~95-100% retention) | ~14-16K→16384 | 61.78% | 71.56 | 61.00 | 52.78 |
| Không nén (dense, pool lớn hơn) | 100% (~42-48K token) | 64.26% | 73.67 | 65.00 | 54.11 |
| **Lấy đều (Uniform)** | 36% (→16384) | **64.41%** | 73.00 | 66.00 | 54.22 |
| Pipeline hiện tại, đầy đủ | 36% (→16384) | 61.96% | 70.00 | 61.67 | 54.22 |
| Pipeline hiện tại, bỏ phần "đại diện" (DTM) | 36% (→16384) | 62.15% | 69.22 | 62.11 | 55.11 |
| **Lấy đều (Uniform)** | ~7.2% (→3240) | **60.30%** | 68.11 | 61.44 | 51.33 |
| Pipeline hiện tại, đầy đủ | ~7.2% (→3240) | 58.70% | 65.44 | 58.56 | 52.11 |

**Đọc bảng:** ở mọi mức ngân sách thử nghiệm, "lấy đều" luôn bằng hoặc nhỉnh hơn pipeline hiện tại. Đáng chú ý nhất: cột **Long** (video dài) là bucket DUY NHẤT nơi pipeline hiện tại liên tục bằng hoặc nhỉnh hơn Uniform (54.22 vs 54.22, rồi 55.11 vs 54.22, và 52.11 vs 51.33) — lặp lại nhất quán ở cả 2 mức ngân sách khác nhau. Ngược lại, Short/Medium luôn là nơi pipeline hiện tại mất điểm rõ rệt nhất. Đây là tín hiệu tích cực duy nhất, đáng để khai thác tiếp trong hướng đi kế tiếp.

Mức ~7.2% (`tokens_num=3240`) được chọn vì khớp đúng số token tuyệt đối mà FastVID báo cáo trong paper của họ (3240.7 token) — để so sánh được ở đúng cùng một điểm vận hành với literature.

### A.2 So với chính paper FastVID — vì sao số tuyệt đối "trông có vẻ" thua

Bảng gốc của FastVID (Table 4 trong paper, đo trên cùng backbone Qwen2.5-VL, cùng cấu hình "tối đa 16384 token, tối đa 768 khung hình"):

| | Số token | Overall | % giữ so với bản không nén của chính họ |
|---|---|---|---|
| Không nén (Vanilla, paper FastVID) | 13447.1 | 67.3% | 100% |
| FastVID (paper) | 3240.7 (24.1%) | 62.8% | 93.3% |
| PruneVID (paper) | 3294.5 (24.5%) | 59.3% | 88.1% |

**Điểm mấu chốt:** số token "không nén" của FastVID (13447) gần khớp với số mình tự đo được (~14-16K) — tức cùng một cấu hình cơ bản. Nhưng **kết quả "không nén" của họ (67.3%) cao hơn của mình (63.33%) tới ~4 điểm**, dù đã kiểm chứng kỹ quy trình đánh giá. Đây là một khoảng cách riêng, **chưa tìm ra nguyên nhân**, và **tách biệt hoàn toàn** khỏi câu hỏi "pipeline hiện tại có thắng nổi cách lấy đều không" — hai vấn đề độc lập, không nên gộp chung khi trình bày.

Vì vậy, cách so sánh công bằng hơn giữa các bên là tính theo **% giữ được so với chính mốc "không nén" của bên đó** (loại bỏ khoảng cách ~4 điểm nói trên khỏi phép so sánh):

| | Số token thực tế | Mức nén | Pool "không nén" dùng làm mẫu số | % giữ so với "không nén" của chính mình |
|---|---|---|---|---|
| FastVID (paper) | 3240.7 | 24.1% | ~13.4K (Vanilla của FastVID) | 93.3% |
| PruneVID (paper) | 3294.5 | 24.5% | ~13.4K (Vanilla của FastVID) | 88.1% |
| Lấy đều — của mình | 16384 | 36% | ~42-48K (pool 3x của mình) | **100.2%** |
| Pipeline hiện tại — của mình | 16384 | 36% | ~42-48K (pool 3x của mình) | 96.4-96.7% |
| Lấy đều — của mình | 3240 | ~7.2% | ~42-48K (pool 3x của mình) | 93.8% |
| Pipeline hiện tại — của mình | 3240 | ~7.2% | ~42-48K (pool 3x của mình) | 91.4% |

**⚠️ Lưu ý khi đọc bảng này:** cột "Mức nén" (24.1% vs 36% vs 7.2%) **KHÔNG cùng thang đo** — mỗi bên tính trên mẫu số "pool không nén" khác nhau (cột thứ 4), nên không nên so trực tiếp "7.2%" với "24.1%" và kết luận mình nén sâu hơn 3 lần. Điểm thực sự so sánh được là **số token tuyệt đối** (cột 2): ở đúng cùng ~3240 token, so được trực tiếp FastVID (93.3%) với của mình (91.4%). Cột cuối ("% giữ so với không nén của chính mình") là cột duy nhất tự-chuẩn-hóa, dùng được để so sánh chéo giữa các bên một cách công bằng.

**Đọc theo cách này:** ở mức 16384 token (36% pool của mình), pipeline hiện tại giữ % còn tốt hơn cả FastVID publish (96-97% > 93.3%) — dù mốc "không nén" tuyệt đối của mình thấp hơn FastVID ~4 điểm (xem trên). Nhưng ở đúng ~3240 token tuyệt đối — nơi so sánh trực tiếp được với FastVID — pipeline hiện tại lại giữ % thấp hơn (91.4% < 93.3%). Khoảng cách nới rộng khi ngân sách token tuyệt đối càng ít — bằng chứng gián tiếp cho thấy: khi ngân sách càng eo hẹp, những hạn chế của cơ chế "đại diện"/"nổi bật" hiện tại càng lộ rõ hơn.
