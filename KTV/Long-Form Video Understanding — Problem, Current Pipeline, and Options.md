---
title: 'Long-Form Video Understanding — Problem, Current Pipeline, and Options'

---

# Long-Form Video Understanding — Problem, Current Pipeline, and Options

# 1 . The Problem

Video LLMs answer questions about videos by encoding each frame into visual tokens (LLaVA: 576 tokens/frame) and feeding them into an LLM alongside the question. For long videos this breaks down:

- **Token explosion** — a 1-hour video at 1 FPS produces millions of visual tokens, far beyond any LLM context window.
- **Quadratic cost** — attention scales O(n²) in sequence length.
- **Attention dilution** — even if tokens fit, too many of them makes the model worse at finding what matters.
- **Lost temporal continuity** — sampling 32 frames from a 30-min video leaves ~50s gaps, destroying motion and event progression.

**Formally:** given video `V = {f_1...f_N}` and question `Q`, produce `A = f_VLM(Select(V), Q)` where `Select` must reduce N frames to K ≪ N. The research question is how to design `Select`.

# 2. The Design Space (three layers, two philosophies)

Redundancy can be removed at **three layers**:

| Layer                              | What it does                       | Example methods          |
| ---------------------------------- | ---------------------------------- | ------------------------ |
| **1. Frame selection**             | N frames → K frames/clips          | KTV, MaxInfo, AKS, F2C   |
| **2. Token reduction (per frame)** | 576 tokens → fewer                 | LLaVA-PruMerge, CDPruner |
| **3. Token pruning (inside LLM)**  | prune across LLM layers / decoding | AdaTP, DyCoke, FastV     |

And there are **two competing philosophies**:
**A. Keyframe Selection** — commit to a small frame set *before* the LLM runs. Cheap, simple. **Risk: recall.** If the selector misses the evidence, it's gone forever — no recovery. *This is where our pipeline sits.*
**B. Load-many + Compress** — keep most frames, compress tokens inside the LLM. **Risk: compression quality**, but has a safety net (pruned tokens can be reactivated). Optimizes FLOPs, not accuracy.
Also orthogonal: **query-aware** (uses the question to select) vs **query-agnostic** (selects once per video, question-independent).


# 3. Our Current Pipeline

**Backbone:** LLaVA-1.6-Vicuna-7B (CLIP ViT-L/14-336 vision encoder + Vicuna-7B LLM).
**Base:** KTV (AAAI 2026) + F2C clip idea + our Temporal-Chain proposal.
**Token budget:** 1872 tokens total (a KTV preset; = 2×288 + 8×144 + 2×72).
**Stage 0 — Offline feature extraction.** DINOv2 features per frame, precomputed and cached. (Separate from the LLaVA backbone.)
**Stage 1 — Temporal-Chain Anchor Selection (query-agnostic).** Greedy selection of K anchors maximizing:

    G(t|S) = R_cluster + λ_event·R_event + α_gap·T_gap − β_red·R_red

combining representativeness (k-means centroid proximity), eventness (motion/scene-change), timeline coverage, and redundancy penalty. **K is currently hard-coded to 12.**
**Stage 2 — F2C Clip Expansion.** Each anchor expands into a short clip (up to 4 frames), trading spatial resolution for temporal context at constant token budget: a 4-frame clip gets `budget/4` tokens per frame (288→72, 144→36, 72→18).
**Stage 3 — Token pruning.** KTV's `cls_new_token_sim` (inherited from LLaVA-PruMerge): keep top-K tokens per frame by `cls_attention × rate + (1−redundancy) × (1−rate)`.
**Key property:** fully query-agnostic — selection runs **once per video**, reused across all questions. Cheaper than query-aware reranking (which must re-run CLIP per question).


# 4. Key Finding: The Fixed K=12 Is Wrong (measured on our own data)

We analyzed DINO features of 30 VideoMME videos (10 short / 10 medium / 10 long):

| Metric                           | Result                                  |
| -------------------------------- | --------------------------------------- |
| Videos where optimal k ≈ 12 (±3) | **6.7%**                                |
| Optimal k (silhouette), median   | **28** (vs our 12)                      |
| Effective rank, median / max     | **141 / 419**                           |
| Effective rank by bucket         | short ~70 → medium ~150 → **long ~300** |

**Interpretation:**

- The natural cluster count of VideoMME videos is **26–30+**, more than double what we use. Only 7% of videos fit K=12.
- **Effective rank grows monotonically with video length** — confirming longer videos genuinely need more clusters.
- But variance is huge (one near-static video needs only 2 clusters). So **cluster count should adapt to** ***content*****, not to duration** — a finding also reported by QCA, which showed that scaling segments linearly with length actually *fails*.

**Caveats to state honestly:** our silhouette scan was capped at k=30 and many videos hit that ceiling, so 28 is a *lower bound*. Also MaxInfo and event-boundary in our current config are not actually adaptive (stuck at their 64 / 6 caps) — a config issue to fix before drawing conclusions from them.

![](/static/img/pixel.gif)

![](https://paper-attachments.dropboxusercontent.com/s_4502225969474342BACC038CBEA802E9B2012BC4031107C94FF5F7C26B6FA330_1784776079436_download.png)
![](/static/img/pixel.gif)

# 5. Landscape of Approaches (with backbones — numbers are NOT directly comparable)

**Frame Selection (our camp):**

| Method         | Backbone       | Mechanism                                     | Query    |
| -------------- | -------------- | --------------------------------------------- | -------- |
| KTV (our base) | LLaVA-1.6-7B   | frame feature clustering                      | agnostic |
| MaxInfo        | LLaVA-Video-7B | SVD + MaxVol (diversity), adaptive count      | agnostic |
| AKS (CVPR'25)  | LLaVA-Video-7B | recursive judge-and-split, relevance+coverage | aware    |
| AdaRD-Key      | LLaVA-Video    | relevance + diversity                         | aware    |
| F2C            | Qwen2.5-VL-7B  | watershed + adaptive clip length              | aware    |
| VideoTree      | —              | hierarchical, expand only relevant branches   | aware    |

**Token Compression (other camp):**

| Method         | Backbone          | Mechanism                              | Query    |
| -------------- | ----------------- | -------------------------------------- | -------- |
| LLaVA-PruMerge | LLaVA-1.5         | [CLS]-attention + IQR + merge          | agnostic |
| DyCoke         | LLaVA-OV/Video-7B | temporal merge + dynamic KV cache      | agnostic |
| AdaTP          | LLaVA-OV-7B       | segment + attention-bias correction    | hybrid   |
| CDPruner       | LLaVA             | DPP: diversity + instruction relevance | aware    |




# 6. Paper anh Khánh đề xuất: Query-Agnostic Frame Selection via RL + V-JEPA Latent Loss


## 1. Vấn đề  

Khi video dài từ vài giây lên hàng giờ, số visual token bùng nổ theo cấp số nhân. Ngay cả Qwen2.5-VL với context window 131,072 token cũng không thể nạp một bộ phim hay video giám sát 2 giờ — vượt giới hạn phần cứng hoặc chi phí suy luận không chấp nhận được.
Nó phê phán **cả hai** nhánh hiện có:

- **Uniform sampling:** tạo đứt gãy thời gian, rủi ro cực cao bỏ lỡ sự kiện quan trọng diễn ra chớp nhoáng → thiếu bằng chứng cốt lõi để suy luận.
- **Query-conditioned selection:** tối ưu được không gian đầu vào nhưng **xem video như tập hợp ảnh tĩnh độc lập**, phá vỡ tính liên tục không-thời gian, và dễ bị hallucination bởi câu hỏi chứa thông tin sai lệch hoặc suy luận đa bước.

→ **Chuyển bài toán từ "tìm khung hình phù hợp với văn bản" sang "tìm tập khung hình tối thiểu có khả năng tái cấu trúc lại thực tại vật lý của toàn bộ video".**
Cơ chế: che khuất (mask) các frame bị loại bỏ, đo **lượng thông tin mất mát trong không gian tiềm ẩn**. Nếu frame được giữ chứa đủ bằng chứng vật lý, model có thể nội suy tốt phần bị che → mất mát nhỏ → frame đó giá trị cao. Đây là **query-agnostic tuyệt đối** vì thước đo là *vật lý nội tại của video*, không phải câu hỏi.


## 2. Kiến trúc( 3 khối)

**2.1 Khối phân đoạn:** KiToke (Kernel-based Interval-aware Token Compression)
Báo cáo bác bỏ cắt đoạn cố định (60s/đoạn) vì sự kiện thực tế hiếm khi tuân theo ranh giới cứng — cắt đôi một cú nhảy hay xe chuyển hướng làm đứt gãy chuỗi nhân quả.
KiToke là **training-free**, ước tính độ đa dạng token toàn cục qua **kernel-based redundancy measure**, xây dựng temporal intervals thích ứng nội dung bằng phân tích khác biệt token giữa frame liên tiếp. Bất kỳ sai lệch đột ngột nào so với động lực học thời gian cục bộ (chuyển cảnh, đổi góc máy, sự kiện mới) → tự động là boundary. Kết quả: clip độ dài biến thiên **10–30s** tùy nhịp độ sự kiện.
Bảng so sánh trong báo cáo:

| Phương pháp                         | Ưu                                                 | Nhược                                                           |
| ----------------------------------- | -------------------------------------------------- | --------------------------------------------------------------- |
| Fixed length                        | dễ code, tiền xử lý ~0                             | cắt ngang sự kiện, đứt gãy nhân quả                             |
| **KiToke**                          | bảo toàn scene continuity, gắn kết không-thời gian | chi phí ban đầu tính khoảng cách L2 giữa token                  |
| Phân đoạn bằng ngôn ngữ (VLM mô tả) | ranh giới có ý nghĩa với người                     | phụ thuộc chất lượng sinh text, **chi phí cực đắt**, dễ ảo giác |

2.2 Khối action: VLM agent nhỏ
**Policy model:** Qwen2.5-VL-3B-Instruct (~6–7GB VRAM khi lượng tử hóa, ~16GB full precision) — đủ nhẹ làm agent trong vòng lặp RL.

- **State:** clip sự kiện + system prompt kiểu *"Hãy phân tích đoạn video này và xác định một tập hợp tối thiểu các khung hình chứa đựng những khoảnh khắc quan trọng nhất... Trả về danh sách chỉ số."*
- **Action:** sinh tự hồi quy ra danh sách chỉ số nguyên, ví dụ `[0, 5, 8, 14]`.
- Điều khiển số frame qua **constrained decoding** hoặc hình phạt trong reward.
- Không gian hành động: **rời rạc tổ hợp** → bắt buộc phải dùng RL vì việc chọn indices là **non-differentiable**.

2.3 Khối chấm điểm — V-JEPA 2.1
**Vì sao V-JEPA chứ không phải VideoMAE:** các model tự giám sát kiểu pixel reconstruction lãng phí tài nguyên khổng lồ vào chi tiết bề mặt không quan trọng (gợn sóng, lá rung, nhiễu số). V-JEPA bỏ hẳn tái tạo pixel, thay bằng **dự đoán biểu diễn trong không gian tiềm ẩn** — buộc mạng nắm bắt cấu trúc nhân quả, động lực học chuyển động, tương tác vật lý (tiếp xúc, biến dạng).
V-JEPA 2.1 thêm **Dense Predictive Loss** + **Deep Self-Supervision**: cả visible token lẫn masked token đều đóng góp vào loss, ngăn context token hoạt động như bộ tổng hợp toàn cục.
**Quy trình tính reward (4 bước):**

- **Target representation:** clip gốc → Target Encoder (bản EMA của context encoder) → "ground truth" trong latent space.
- **Masking:** frame không được chọn → thay bằng learnable mask tokens.
- **Latent prediction:** context encoder xử lý frame được chọn + mask tokens → Predictor nội suy lại biểu diễn phần bị che.
- **Information loss:** khoảng cách giữa dự đoán và ground truth trên vùng bị che (có stop-gradient).

**Hàm reward tổng thể** gồm hai thành phần đối nghịch:

- Thành phần 1: thưởng lớn khi V-JEPA khôi phục chính xác thông tin bị mất → chứng tỏ frame được chọn đủ dữ kiện vật lý.
- Thành phần 2: **hình phạt tuyến tính theo số frame** → ép sparsity tối đa.

2.4  Tối ưu chính sách — GRPO thay PPO
**Vì sao không PPO:** PPO cần Critic Network kích thước tương đương Policy. Tải đồng thời Policy VLM + Critic VLM + V-JEPA lớn vào GPU là **nút thắt không thể vượt qua** với hầu hết lab.
**GRPO** loại bỏ hoàn toàn Critic: sinh một **nhóm 4–8 tập frame** cho mỗi clip, tính mean và std của reward trong nhóm, dùng thống kê đó chuẩn hóa để ra **advantage tương đối**. Cộng KL penalty để policy mới không trôi quá xa reference (tránh thoái hóa khả năng ngôn ngữ của VLM).

|             | PPO                                | **GRPO (đề xuất)**           |
| ----------- | ---------------------------------- | ---------------------------- |
| Advantage   | Critic + GAE                       | mean/std nội nhóm            |
| VRAM        | rất cao (Policy+Ref+Critic+Reward) | thấp hơn đáng kể (bỏ Critic) |
| Phù hợp LVU | không khả thi khi GPU hạn chế      | lý tưởng                     |


**2**.5 Rủi ro:
**Rủi ro cụ thể:** VLM có thể phát hiện lối tắt — luôn chọn frame đầu, giữa, cuối của mọi clip (bất kể nội dung), V-JEPA Predictor vẫn đủ thông tin nội suy tuyến tính để đạt điểm "chấp nhận được" ở cảnh ít biến động. Hội tụ vào cực tiểu cục bộ này **phá hủy hoàn toàn ý nghĩa** của việc trích xuất tự động.
**Giải pháp giảm nhẹ:**

- Đa dạng hóa kịch bản động lực học trong data (thể thao nhịp độ cao, cắt cảnh liên tục).
- Dùng clip **độ dài biến thiên từ KiToke** → khung thời gian luôn đổi → agent không học thuộc vị trí cố định.
- Kết hợp GRPO với **DPP (Determinantal Point Processes)** để phạt dư thừa nội nhóm, ép VLM đề xuất đa dạng mẫu.

**2.6 So sánh:** 

|                         | Pipeline hiện tại (KTV+F2C)                    | Báo cáo này (RL+V-JEPA)                           |
| ----------------------- | ---------------------------------------------- | ------------------------------------------------- |
| **Phân đoạn**           | k-means, K=12 fix cứng                         | KiToke, interval adaptive 10–30s                  |
| **Tín hiệu chọn frame** | heuristic: eventness + gap − redundancy (DINO) | **học được**: latent reconstruction loss (V-JEPA) |
| **Training**            | training-free                                  | SFT + GRPO                                        |
| **Số frame**            | fix 12                                         | adaptive qua sparsity penalty                     |
| **Backbone infer**      | LLaVA-1.6-7B                                   | Qwen2.5-VL-72B                                    |
| **Query**               | agnostic                                       | agnostic (mạnh hơn: dựa trên vật lý)              |

    
# 7. Ý tưởng đề xuất
## 7.1 Ý tưởng 

Báo cáo V-JEPA có một reframe rất hay: *"chọn tập frame tối thiểu có thể tái cấu trúc lại toàn bộ video"*. Nhưng họ hiện thực nó bằng RL + V-JEPA + GRPO — quá nặng và có rủi ro reward hacking.
→ Xấp xỉ chính xác cùng mục tiêu đó bằng đại số tuyến tính trên feature DINO, không cần huấn luyện gì cả.
Cụ thể: coi ma trận feature `F ∈ R^{N×D}` (N frame) là thứ cần được "tái tạo". Chọn tập con S sao cho **mọi frame còn lại tái tạo được từ S với sai số nhỏ nhất** (linear reconstruction / column subset selection). Đây chính là "nếu che frame này đi, tôi có khôi phục được nó từ các frame đã giữ không?" — đúng reward của V-JEPA, nhưng thay predictor học được bằng phép chiếu tuyến tính, nên **rẻ và training-free**.

Chạy greedy: mỗi bước thêm frame có **residual lớn nhất** (frame bị tái tạo tệ nhất = mới nhất về thông tin). Điều kỳ diệu là **một quá trình greedy này cho ra CẢ BỐN quyết định** mà pipeline hiện tại của bạn đang dùng 3–4 heuristic rời rạc:

- **Bao nhiêu frame** → dừng khi residual biên < τ ⇒ **số cluster tự adaptive theo nội dung** (khớp đúng phát hiện effective-rank của bạn).
- **Frame nào** → frame giảm sai số tái tạo toàn video nhiều nhất.
- **Token budget mỗi frame** → tỉ lệ với **lượng residual nó giải thích được** (frame mang nhiều thông tin độc nhất → nhiều token). Thay cho việc xếp hạng bằng eventness.
- **Clip dài bao nhiêu** → mở rộng anchor sang hàng xóm **khi hàng xóm còn residual cao** (vùng chuyển động, frame kề mang thông tin mới) và dừng khi hàng xóm đã redundant (vùng tĩnh). ⇒ clip dài ở chỗ động, ngắn ở chỗ tĩnh — đúng tinh thần F2C nhưng **có nguyên lý**, không phải λ chỉnh tay.
## 7.2 Bài toán

**Đặt bài toán**

## 5. Tầng 1 — RECON-Select

### Ý tưởng
Reframe từ báo cáo V-JEPA: *"chọn tập frame tối thiểu tái cấu trúc lại toàn bộ video"*. Thay predictor học được (V-JEPA) bằng **phép chiếu tuyến tính** trên feature DINO → **training-free, không RL, không reward hacking**.

### Đặt bài toán
Feature DINO `F ∈ R^{N×D}`, chuẩn hoá hàng `x_i = f_i/||f_i||`, `X ∈ R^{N×D}`. Chọn `S ⊆ {1..N}`:

$$\min_{S}\; \sum_{i=1}^{N}\big\| x_i - P_S x_i \big\|^2 \;+\; \lambda\,|S|$$

với P_S = phép chiếu trực giao lên span của `{x_j : j∈S}`. Trực giác: che frame i → khôi phục bằng chiếu lên frame đã giữ; sai số = "thông tin độc nhất của i mà S chưa có". Bản linear của reward V-JEPA.

### Thuật toán greedy (QR có pivot)
Khởi tạo residual `R^{(0)} = X`. Tại bước t:

**(1) Chọn anchor** (frame residual lớn nhất = novel nhất):
$$a_t = \arg\max_i \big\| r_i^{(t)} \big\|_2$$

**(2) Ghi giá trị thông tin** (dùng cho token allocation):
$$v_{a_t} = \big\| r_{a_t}^{(t)} \big\|_2$$

**(3) Cập nhật residual** (Gram-Schmidt một bước, `q = r_{a_t}^{(t)}/\|r_{a_t}^{(t)}\|`):
$$r_i^{(t+1)} = r_i^{(t)} - \langle r_i^{(t)}, q\rangle\, q,\quad \forall i$$

**(4) Dừng** (số frame tự adaptive):
$$\max_i \|r_i^{(t+1)}\| < \tau \quad\text{hoặc}\quad |S| \ge K_{\max}$$
với `τ = ρ · median_i ||x_i||`, `ρ ∈ [0.1, 0.3]`.

> Về mặt toán học đây là **QR pivot theo cột** trên `Xᵀ`; K khi dừng ≈ **effective rank** — nên K của RECON tự khớp phân bố effective-rank đã đo (một xác nhận đẹp cho ablation).

### Chọn K theo năng lượng (energy stopping) — giải concern "1872 cố định"
Chuỗi `v_1 ≥ v_2 ≥ …` (residual mỗi frame giải thích) tự nói khi nào đủ. Giữ γ% năng lượng:

$$K^* = \min\Big\{ k : \frac{\sum_{t\le k} v_t^2}{\sum_{t} v_t^2} \ge \gamma \Big\},\quad \gamma \approx 0.90\text{–}0.95$$

Kẹp bởi trần **vật lý** (context window), không phải trần tuỳ ý:
$$K = \text{clip}\Big(K^*,\; K_{\min},\; \big\lfloor B_{\max}/\bar b \big\rfloor \Big)$$

### Mở rộng clip theo residual (thay adaptive_length dò λ tay)
Với anchor a, thêm hàng xóm j (a±1, a±2,…) nếu nó **vẫn còn residual cao** given S (vùng động → frame kề mang thông tin mới):
$$\big\| r_j^{(\text{final})}\big\| > \theta\, v_a,\quad \theta \in [0.3, 0.6],\quad |clip(a)| \le l_{\max}$$
→ clip dài ở chỗ chuyển động, ngắn ở chỗ tĩnh — **có nguyên lý**.

### Metric đánh giá không cần VLM — Reconstruction Coverage
$$\text{RC}(S) = 1 - \frac{\sum_i \| x_i - P_S x_i\|^2}{\sum_i \|x_i\|^2} \in [0,1]$$
Vẽ **RC vs budget** cho uniform / k-means / MaxInfo / RECON. RECON Pareto-dominate = bằng chứng coverage tốt hơn, **rẻ, độc lập backbone/query**. Đây giải trực tiếp nỗi đau "backbone LLaVA-1.6 yếu nên không so số tuyệt đối".

### Một mục tiêu, bốn quyết định
Toàn bộ có thể viết như một mục tiêu greedy đang xấp xỉ:
$$\min_{S,\{l_a\}}\; \underbrace{\sum_i \|x_i - P_S x_i\|^2}_{\text{coverage}}\; +\; \lambda|S|\; -\; \mu \sum_{a\in S}\sum_{j\in clip(a)} \|r_j\|^2$$
→ (1) **số frame** K (energy stop), (2) **frame nào** (residual max), (3) **token/clip** ∝ `v_a`, (4) **độ dài clip** (residual hàng xóm). Thay 4 heuristic bằng **1 đại lượng: residual**.

---

## 6. Tầng 2 — QTA (Query-Aware Token Allocation)

Tầng 1 chạy **một lần/video** (cache). Tầng 2 chạy **mỗi câu hỏi**, chỉ phân bổ lại token trên đúng tập anchor S — rất rẻ.

**Bước 1 — relevance từ CLIP** (image embedding của anchor đã cache; chỉ encode text câu hỏi):
$$s_a = \cos\big(E_{\text{img}}(f_a),\; E_{\text{txt}}(Q)\big),\quad a\in S$$

**Bước 2 — điểm kết hợp** (relevance câu hỏi × importance nội tại từ Tầng 1):
$$w_a = (\tilde s_a)^{\alpha}\cdot(\tilde v_a)^{\beta}$$
với `s̃, ṽ` min-max về [0,1]. `β=0` → thuần aware; `α=0` → agnostic (LC1). Hai cờ ablation.

**Bước 3 — phân bổ token có sàn** (mọi cluster vẫn "có mặt"):
$$B_a = B_{\min} + \big(B - K B_{\min}\big)\cdot \frac{\exp(w_a/T)}{\sum_{a'}\exp(w_{a'}/T)}$$
`B_min` = sàn (vd 18), `T` = nhiệt độ (nhỏ → dồn top). Ràng buộc an toàn: `B_min ≤ B/(2K)`.

**Bước 4 — lượng tử hoá** `B_a` về `{288,144,72,36,18}` (khớp nhánh token), rồi mỗi frame trong clip nhận `B_a / |clip(a)|`.

**Chi phí:** CLIP-image-encode 30 anchor = **một lần/video** (cache). Mỗi câu hỏi = 1 text-encode + 30 cosine + softmax ≈ mili-giây. Gần như rẻ bằng agnostic thuần.

**Giá trị khởi đầu đề xuất:** `α=1.0, β=0.5, T=0.5, B_min=18, γ=0.90`.

---

## 7. Ngân sách token

**Ba lựa chọn khi K lớn (vd 30) mà không muốn giữ cứng 1872:**

- **Cách A — nâng budget theo K:** `B = K·b̄` (b̄≈144 → B_30≈4320). Nhưng **bị chặn** bởi context LLaVA-1.6 (~4K) → chỉ khả thi nếu đổi backbone dài (Qwen2.5-VL 128K).
- **Cách B — giữ B, phân bổ hình chóp (khuyến nghị):** 5–8 cluster quan trọng giữ 144–288, phần còn lại 18–36 (thumbnail "điểm danh"). Tổng vẫn ~B. Chính là công thức Tầng 2.
- **Cách C — hierarchical (VideoTree):** 30 cluster ở tầng phủ (rẻ), nén xuống ~12 đại diện đưa vào VLM (đắt).

**Trần vật lý:** `B_max = context_window − text_tokens − system_tokens`. Không được vượt, nếu không tràn context/OOM.

**Nguyên tắc chốt:** *budget nên tỉ lệ với độ giàu nội dung (đo bằng residual/effective rank), chặn bởi context window — không phải hằng số.* Với backbone hiện tại → **Cách B**.

---

## 8. Kiến trúc tổng thể

```
VIDEO ─┐
       ▼
┌──────────────────────────────────────────────┐
│ TẦNG 1 — RECON-Select (một-lần/video, AGNOSTIC)│
│  DINO features → greedy QR-pivot reconstruction │
│  → S = {anchor} (K adaptive theo residual/energy)│
│  → v_a (giá trị thông tin), clip_frames         │
│  → cache CLIP image-embed của anchor            │  ← cache theo video_name
└──────────────────────────────────────────────┘
       │  (dùng lại cho MỌI câu hỏi của video)
       ▼
┌──────────────────────────────────────────────┐
│ TẦNG 2 — QTA (mỗi-câu-hỏi, AWARE)              │
│  s_a = cos(CLIP_img(f_a), CLIP_txt(Q))         │
│  w_a = s_a^α · v_a^β                            │
│  B_a = softmax(w_a/T) có sàn B_min             │
│  → xuất [frame, token_budget] theo question_id │
└──────────────────────────────────────────────┘
       ▼
   VLM (LLaVA-1.6) + cls_new_token_sim (GIỮ NGUYÊN) → Answer
```

- **DINO** cho Tầng 1 (coverage), **CLIP** cho Tầng 2 (relevance) — hai encoder khác nhau.
- Cache Tầng 1 theo `video_name`; JSON cuối theo `question_id`.

---

## 9. Nguồn gốc ý tưởng

| Thành phần trong CAS-QTA | Lấy ý tưởng từ |
|---|---|
| Reconstruction-as-value (chọn frame tái tạo được video) | **Báo cáo V-JEPA/GRPO** (upload) — ta làm bản *linear, training-free* |
| QR-pivot / max-volume / adaptive count | **MaxInfo** (SVD + rect MaxVol) |
| Clip expansion + đổi resolution↔token | **F2C** (From Frames to Clips) |
| `cls_new_token_sim` token pruning (Stage 3) | **KTV** ← **LLaVA-PruMerge** ([CLS] attn + IQR + merge) |
| Phân đoạn theo biến động thời gian | **KiToke**, **shot/scene**, bước segment của **AdaTP** |
| Query rerank ở Tầng 2 | **KTV-rank**, **AKS**, **F2C** (query relevancy) |
| Phân bổ token theo importance (hình chóp/softmax) | **AdaTP** (budget theo đoạn), **DyCoke** (token merge) |
| Diversity mạnh hơn k-means (DPP option) | **CDPruner** (DPP: diversity + relevance) |
| Adaptive count + chi phí có trần | **MaxInfo**, **TimeMarker** (fps theo độ dài), **VideoTree** (coarse-to-fine) |
| "Nội dung chứ không độ dài" | **QCA** (scale theo length không giúp) |
| Lợi ích bão hoà log theo số frame | **LiteFrame** |
| Effective rank làm tín hiệu số cluster | **MaxInfo** + phân tích của ta |

---

## 10. Files, config, lệnh

### Files cần có
| File | Vai trò | Trạng thái |
|---|---|---|
| `recon_select.py` | Tầng 1: RECON greedy + clip + cache anchor/CLIP embed | **cần viết** |
| `qta_allocate.py` | Tầng 2: CLIP relevance + softmax token allocation | **cần viết** |
| `analyze_videomme_clusters.py` | Phân tích cấu trúc cluster (đã dùng chứng minh K=12 sai) | **đã có** |
| `ktv/llava/model/llava_arch.py` | Patch `order>=12 → dùng token budget trực tiếp` | **đã patch** |
| `cluster_and_rank_keyframes.py` | Helper load tensor, CLIP, get_original_frame_number | có sẵn |

### Patch bắt buộc trong `llava_arch.py`
```python
if keyframe_order:
    order = keyframe_order[frame_idx]
    if order >= 12:                    # CAS-QTA: value là token budget/frame
        remain_tokens_num = order      # dùng trực tiếp
    elif num_frames == 6:
        ...
```
Giữ nhánh cũ để file rank 0–11 vẫn chạy; file mới (budget ≥16) đi thẳng.

### Config mẫu (`configs/keyframe_cluster/experiment/vmme_casqta.yaml`)
```yaml
# @package _global_
json_path: playground/gt_qa_files/Videomme/val_qa.json
video_path: datasets/Video-MME/data
video_frame_tensor_path: ktv/save_tensor/Video-MME
save_cluster_path: outputs/vmme_casqta_kf
combined_output_path: outputs/vmme_casqta.json
dataset: Video-MME
device: auto

# Tầng 1 (RECON)
gamma: 0.90            # energy stop -> K adaptive
rho: 0.2              # nguong tau tuong doi
k_min: 8
k_max: 48
theta: 0.4           # nguong mo clip theo residual
l_max: 4
fps_downsample: 1    # nen ha ve ~1 FPS truoc khi chay
# Tầng 2 (QTA)
alpha: 1.0
beta: 0.5
temperature: 0.5
b_min: 18
tokens_num: 1872     # B; hoac dat adaptive B = K * b_bar
b_bar: 144
```

### Lệnh chạy (3 bước)
```bash
# 0) Phan tich cau truc (da co) — chung minh K=12 sai
python analyze_videomme_clusters.py --per_bucket 10 --out_dir outputs/cluster_analysis

# 1) Tang 1: RECON selection (mot lan/video) — nho xoa cache truoc
rm -rf outputs/vmme_casqta_kf outputs/vmme_casqta.json
python recon_select.py experiment=vmme_casqta      # cache anchor + CLIP embed theo video

# 2) Tang 2: QTA allocation (moi cau hoi) -> JSON [frame, token_budget]
python qta_allocate.py experiment=vmme_casqta \
  alpha=1.0 beta=0.5 temperature=0.5

# 3) Inference + eval (giu nguyen prune)
python run_inference_multiple_choice_qa.py experiment=videomme_6keyframe \
  output_name=pred_casqta key_frame_path=outputs/vmme_casqta.json \
  prune_mode=cls_new_token_sim rate=0.2 tokens_num=1872
python eval/compute_accuracy.py outputs/pred_casqta.json
```

> **Bẫy cache:** khối `if os.path.exists(out_path): continue` sẽ nạp lại file cũ. Đổi tham số → phải `rm -rf` thư mục `*_kf` + JSON tổng. Mỗi experiment dùng `save_cluster_path` riêng.

---

## 11. Đánh giá & ablation

### Không cần VLM (rẻ, độc lập backbone)
- **RC vs budget** cho uniform / k-means / MaxInfo / RECON, **tách theo short/medium/long**. Kỳ vọng: RECON Pareto-dominate; nhóm long cần nhiều token hơn để đạt RC=0.9 → xác nhận budget nên co giãn.
- So **K adaptive của RECON** với **effective rank** đã đo (kỳ vọng khớp).

### Cần VLM (accuracy)
Ablation tách đóng góp từng thành phần (cùng tập anchor):
| Biến thể | α | β | Ý nghĩa |
|---|---|---|---|
| Agnostic alloc (LC1) | 0 | 1 | token theo importance nội tại |
| Aware thuần | 1 | 0 | token theo relevance câu hỏi |
| **CAS-QTA (đề xuất)** | 1 | 0.5 | kết hợp |
| Uniform / KTV-rank | — | — | baseline |

- Quét budget `1872 / 2880 / 4320` trên subset nhóm **long** → trả lời "budget lớn hơn có giúp không, ở nhóm nào".
- **Benchmark:** VideoMME **và** LongVideoBench + MLVU (frame-selection tỏa sáng ở LVB/MLVU; VideoMME lợi thế mỏng — nếu chỉ đo VideoMME sẽ *đánh giá thấp* phương pháp).
- **Báo cáo dạng "% so với dense của chính backbone"**, không so tuyệt đối với F2C 63.4 (khác backbone).

---

## 12. Rủi ro & lưu ý

1. **Linear reconstruction ≠ vật lý/nhân quả** như V-JEPA — bắt coverage/diversity tốt nhưng không mô hình hoá tương tác vật lý. Là điểm yếu *và* là lộ trình mở rộng (mục 13).
2. **Feature quá dày (2000–5400 frame/video)** → residual dồn vào nhiễu ánh sáng. **Down-sample về ~1 FPS** trước khi chạy.
3. **Greedy QR nhạy chuẩn hoá** → L2-normalize (và cân nhắc trừ mean) trước khi chạy.
4. **CLIP relevance nhiễu bởi nền trội** (điểm yếu AKS mà AdaRD-Key chỉ ra) → số hạng `v_a` phần nào chống lại; theo dõi.
5. **Câu hỏi global** ("video nói về gì") → rerank theo relevance có thể *hại*; nên giảm α cho câu global.
6. **Nhãn phương pháp:** đây là *agnostic selection + aware allocation* — **KHÔNG** phải thuần query-agnostic. Mô tả đúng khi trình leader/viết paper.
7. **Trần context LLaVA-1.6 (~4K)** chặn cứng budget — muốn K lớn + token đầy đủ phải đổi backbone dài.

Bối cảnh bài toán
Mỗi frame video được biểu diễn bằng vector đặc trưng DINO, sau đó chuẩn hoá về độ dài 1 (x_i = f_i / ||f_i||). Chuẩn hoá để mọi frame "bình đẳng" về độ lớn — ta chỉ so sánh hướng, không so sánh độ lớn.

Mục tiêu: chọn ra một tập con S các frame (anchor) sao cho:

$$\min_{S}; \sum_{i=1}^{N}\big| x_i - P_S x_i \big|^2 ;+; \lambda,|S|$$

Đọc thế nào:

P_S x_i = hình chiếu của frame i lên không gian con được tạo bởi các frame đã chọn trong S. Hiểu nôm na: "nếu chỉ có các frame trong S, ta cố tái tạo lại frame i bằng tổ hợp tuyến tính của chúng — đây là bản tái tạo tốt nhất có thể".
x_i - P_S x_i = phần không tái tạo được, tức là thông tin riêng của frame i mà tập S chưa "biết".
‖x_i - P_S x_i‖² càng lớn → frame i càng khác biệt (novel) so với những gì đã chọn.
λ|S| là số hạng phạt: chọn càng nhiều frame thì càng tốn (giống Occam's razor — đừng chọn dư).
→ Insight cốt lõi: một frame "đáng chọn" không phải là frame đẹp/nét/độ tương phản cao, mà là frame không thể suy ra được từ các frame khác đã chọn. Đây chính là ý tưởng "chọn tập tối thiểu để tái tạo lại toàn bộ video" từ V-JEPA, nhưng thay vì dùng một mạng neural (predictor) học cách đoán, ở đây dùng phép chiếu tuyến tính đại số thuần tuý — nên không cần train, không cần RL.

Thuật toán greedy (bước lặp)
Vì bài toán trên là tổ hợp (chọn tập con) nên không giải trực tiếp được — dùng greedy từng bước, giống hệt thuật toán QR phân rã có pivot cột.

Khởi tạo: R⁽⁰⁾ = X (residual ban đầu = chính feature gốc, vì S rỗng chưa tái tạo được gì).

(1) Chọn anchor:
$$a_t = \arg\max_i |r_i^{(t)}|_2$$
→ Ở bước t, chọn frame nào đang có phần dư (chưa giải thích được) lớn nhất. Đây là frame "lạ" nhất so với các anchor đã có.

(2) Ghi giá trị thông tin:
$$v_{a_t} = |r_{a_t}^{(t)}|_2$$
→ Lưu lại độ lớn residual tại thời điểm được chọn — con số này về sau dùng để biết frame đó "đáng giá" bao nhiêu (dùng để phân bổ token ở Tầng 2).

(3) Cập nhật residual (Gram-Schmidt):
$$q = \frac{r_{a_t}^{(t)}}{|r_{a_t}^{(t)}|}, \qquad r_i^{(t+1)} = r_i^{(t)} - \langle r_i^{(t)}, q\rangle, q$$
→ Sau khi chọn anchor, "trừ đi" phần mà mọi frame khác có thể giải thích được nhờ anchor mới này (chiếu lên hướng q rồi trừ ra). Phần còn lại (r_i^{(t+1)}) là thông tin thực sự chưa ai giải thích được, kể cả anchor vừa chọn.

→ Đây đúng là một bước Gram-Schmidt/QR: mỗi anchor được chọn giống như chọn một cột "pivot" trong phân rã QR có pivot, và residual chính là phần trực giao còn lại.

(4) Điều kiện dừng:
$$\max_i |r_i^{(t+1)}| < \tau \quad\text{hoặc}\quad |S| \ge K_{\max}$$
với τ = ρ · median_i ||x_i||.
→ Dừng khi residual lớn nhất còn lại đã nhỏ hơn một ngưỡng (nghĩa là mọi frame còn lại gần như đã được "giải thích" đủ tốt bởi các anchor đã chọn), hoặc đã chạm số lượng frame tối đa cho phép.

Ý nghĩa toán học sâu hơn: đây thực chất là QR pivot theo cột trên ma trận Xᵀ. Số anchor cần thiết để đạt ngưỡng dừng xấp xỉ effective rank của tập feature — tức "độ đa dạng nội tại thực sự" của video đó. Video tĩnh (nói chuyện, ít chuyển động) → effective rank thấp → dừng sớm, ít frame. Video nhiều cảnh/chuyển động → effective rank cao → cần nhiều frame hơn. Đây là lý do số frame K tự động thích nghi theo nội dung, không phải hằng số cố định.

Chọn K theo "năng lượng" (energy stopping)
Thay vì dùng ngưỡng τ hơi khó diễn giải, có cách khác trực quan hơn dựa trên dãy giá trị thông tin đã lưu v_1 ≥ v_2 ≥ ... (residual giảm dần theo từng bước chọn):

$$K^* = \min\Big{ k : \frac{\sum_{t\le k} v_t^2}{\sum_{t} v_t^2} \ge \gamma \Big}$$

→ Đọc như "phương sai giải thích được" trong PCA: v_t² là "năng lượng" mà anchor thứ t đóng góp. Cộng dồn năng lượng của k anchor đầu tiên, chia cho tổng năng lượng toàn bộ quá trình — khi tỷ lệ này đạt γ (vd 90–95%) thì dừng. Tức là: giữ đủ anchor để giải thích được γ% "thông tin mới" toàn video, phần còn lại (100-γ%) coi là dư thừa/trùng lặp, bỏ qua được.

Sau đó kẹp bởi giới hạn vật lý (không phải số tùy ý):
$$K = \text{clip}\Big(K^*,; K_{\min},; \lfloor B_{\max}/\bar b \rfloor \Big)$$
→ K không được nhỏ hơn K_min (đảm bảo tối thiểu), và không được vượt quá số frame mà ngân sách token vật lý (context window của VLM) cho phép.

Mở rộng "clip" quanh mỗi anchor
$$|r_j^{(\text{final})}| > \theta, v_a$$

→ Với mỗi anchor a đã chọn, xét các frame lân cận j (a-1, a+1, a-2,...). Nếu residual còn lại của frame lân cận đó (sau khi đã trừ hết các anchor) vẫn còn lớn (so với chính giá trị thông tin v_a của anchor, theo tỷ lệ θ) thì gộp j vào cùng "clip" với anchor a. → Ý nghĩa: vùng đang chuyển động mạnh thì các frame lân cận cũng mang thông tin riêng (chưa được anchor giải thích hết) nên clip nên dài ra; vùng tĩnh thì các frame lân cận gần như giống hệt anchor (residual thấp) nên clip ngắn. Đây thay thế việc phải dò tay tham số adaptive_length.

Công thức tổng hợp (mục tiêu chung)
$$\min_{S,{l_a}}; \underbrace{\sum_i |x_i - P_S x_i|^2}{\text{coverage}}; +; \lambda|S|; -; \mu \sum{a\in S}\sum_{j\in clip(a)} |r_j|^2$$

→ Đây là cách viết lại "gộp" cả 4 quyết định (số frame, frame nào, phân token, độ dài clip) thành một hàm mục tiêu duy nhất, mà thuật toán greedy ở trên chính là nghiệm xấp xỉ từng bước của nó. Insight chốt: tất cả 4 quyết định thiết kế đều quy về đúng một đại lượng — residual (phần thông tin chưa được giải thích). Đây là điểm mạnh về mặt "đẹp toán học": không phải 4 heuristic rời rạc, mà là 4 hệ quả của cùng một nguyên lý.

Tóm một câu: RECON-Select không hỏi "frame nào đẹp/quan trọng" mà hỏi "frame nào KHÔNG THỂ đoán được từ những frame đã chọn" — và trả lời bằng đại số tuyến tính (chiếu trực giao) thay vì mạng neural học được, nên vừa rẻ vừa không cần huấn luyện.

Bạn muốn mình đi sâu thêm vào phần nào — ví dụ tại sao dùng QR-pivot thay vì SVD, hay cách v_a liên kết sang Tầng 2 (QTA)?