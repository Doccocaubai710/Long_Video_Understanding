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
| -------------------------------- | ---------------------------------------- |
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


# 6. Paper anh Khánh đề xuất: Query-Agnostic Frame Selection via RL + V-JEPA Latent Loss (tóm tắt, và vì sao không dùng)

**Reframe cốt lõi:** thay vì "tìm khung hình phù hợp với câu hỏi", chuyển bài toán thành "tìm tập khung hình tối thiểu có khả năng tái cấu trúc lại thực tại vật lý của toàn bộ video" — che (mask) các frame bị loại, đo lượng thông tin mất mát trong không gian tiềm ẩn. Nếu frame giữ lại đủ bằng chứng vật lý để nội suy tốt phần bị che → mất mát nhỏ → frame giá trị cao. Query-agnostic tuyệt đối vì thước đo là vật lý nội tại video, không phải câu hỏi.

**Kiến trúc 3 khối:**
1. **Phân đoạn — KiToke**: training-free, ước lượng đa dạng token qua kernel-based redundancy measure, tạo interval thích ứng nội dung (10–30s/đoạn), thay vì cắt cố định.
2. **Action — VLM agent nhỏ**: Qwen2.5-VL-3B-Instruct làm policy, sinh tự hồi quy danh sách chỉ số frame `[0, 5, 8, 14]`. Không gian hành động rời rạc tổ hợp → bắt buộc RL vì non-differentiable.
3. **Chấm điểm — V-JEPA 2.1**: dự đoán biểu diễn trong không gian tiềm ẩn (không tái tạo pixel) cho phần frame bị che; reward = độ chính xác nội suy − hình phạt tuyến tính theo số frame (ép sparsity).

**Tối ưu — GRPO thay PPO**: bỏ hẳn Critic Network (vốn cần tải thêm 1 VLM ngang Policy, không khả thi với GPU hạn chế), thay bằng chuẩn hóa advantage tương đối trong nhóm 4–8 rollout.

**Vì sao không dùng:** phần kiến trúc (frozen backbone + policy head nhỏ + GRPO) **đúng hướng** — hai paper SOTA độc lập (TSPO, SCORE, xem §7.6.3/Phụ lục C của `idea.md`) đã hội tụ về đúng shape này, nên trực giác kiến trúc của đề xuất này được xác nhận. Nhưng phần **reward** là chỗ lệch: V-JEPA reconstruction loss đo được *độ đầy đủ vật lý* (physical completeness) của tập frame giữ lại, không đo *độ đủ ngữ nghĩa để trả lời câu hỏi*. Bằng chứng trực tiếp, đo được trên chính dữ liệu của chúng ta: RECON-Select — bản training-free, rẻ, của đúng ý tưởng reconstruction này — khi kiểm chứng thực nghiệm cho kết quả phẳng, không vượt được các heuristic đơn giản đã có (§8.1, Bảng 5). Trong khi đó, phương pháp SOTA hiện tại trong nhánh nén token — **SCORE** — dùng tín hiệu CE loss/độ đúng câu trả lời làm reward (không phải reconstruction), và vượt cả model không nén (65.3% > Vanilla 63.8% trên VideoMME, Qwen2.5-VL) — đúng loại tín hiệu mà reconstruction reward không cung cấp được.

**Kết luận:** giữ nguyên kiến trúc RL + policy-head nhỏ cho một giai đoạn sau (nếu dữ liệu cho thấy cần), nhưng đổi tín hiệu reward sang độ đúng câu trả lời (như TSPO/SCORE), không dùng V-JEPA reconstruction làm reward chính. Ý tưởng đề xuất ở §7 đi theo hướng khác nữa: xấp xỉ *chính bất đối xứng* giữa "chọn frame" (cần câu hỏi, cứng) và "phân bổ token" (không cần câu hỏi, mềm) — xem lập luận đầy đủ ở §7.0.


# 7. Ý tưởng đề xuất: Phân bổ Ngân sách Token Không Đồng Nhất, Độc lập với Câu hỏi

*(Toàn văn thiết kế đầy đủ — bao gồm quy trình thực nghiệm, sổ trích dẫn, phụ lục survey nhánh token-compression, và đối chiếu chi tiết với đề xuất RL/V-JEPA ở mục 6 — nằm ở `idea.md`. Phần dưới đây là Phần 0–6: lý do thiết kế, thuật toán, và định vị/novelty.)*

<a name="phần-0"></a>
## PHẦN 0 — VÌ SAO `Rel(x_i | Q)` XUẤT HIỆN TRONG MỘT PHƯƠNG PHÁP QUERY-AGNOSTIC

Đây là chỗ dễ hiểu nhầm nhất của toàn bộ thiết kế, nên đặt lên đầu.

Câu trả lời ngắn: **`Rel` được định nghĩa để chứng minh rằng ta KHÔNG cần nó.** Nó là đối tượng bị loại bỏ, không phải đối tượng được sử dụng.

### 0.1 Phân biệt hai loại xuất hiện của Q

| | Nơi xuất hiện | Có phá vỡ query-agnostic không? |
|---|---|---|
| **(A)** Trong **phân tích lý thuyết** — để định nghĩa khái niệm và chứng minh mệnh đề | ❌ Không |
| **(B)** Trong **thuật toán chạy lúc inference** — thuật toán đọc Q để ra quyết định | ✅ Có |

`Rel(x_i | Q)` chỉ xuất hiện ở (A). Không có dòng nào trong pseudo-code từ bước 1 đến bước 15 đọc `Q`. Thuật toán chỉ nhận `Q` ở bước cuối cùng, khi đưa vào LLM — nhưng lúc đó việc nén đã hoàn tất từ lâu và đã được cache.

**So sánh:** khi chứng minh một thuật toán sắp xếp là đúng, bạn phải định nghĩa "hoán vị sai" và "hoán vị đúng". Việc định nghĩa "hoán vị sai" không có nghĩa thuật toán tạo ra hoán vị sai — nó có nghĩa bạn cần khái niệm đó để chứng minh thuật toán tránh được nó.

### 0.2 Cấu trúc lập luận đầy đủ

Định nghĩa hai đại lượng ở mức token:

```
R(x_i | X)     = mức độ x_i có thể được dự đoán từ X \ {x_i}     (ĐỘ DƯ THỪA)
Rel(x_i | Q)   = mức độ x_i đóng góp vào việc trả lời Q          (ĐỘ LIÊN QUAN)
```

Lý tưởng nhất, ta muốn giữ token có `Rel` cao. Nhưng `Rel` cần `Q`, mà `Q` thay đổi theo từng câu hỏi → **không cache được**.

> **Mệnh đề.** `R(x_i | X) ⊥ Q` — độ dư thừa độc lập với câu hỏi.

**Chứng minh trực giác.** `R` chỉ phụ thuộc vào phân bố chung của các token trong chính video đó. Một patch tường tĩnh có thể dự đoán được từ các patch lân cận **bất kể ai đang hỏi gì**. Loại bỏ một token dư thừa hoàn toàn là thao tác bảo toàn thông tin **dưới mọi câu hỏi**, vì token đó khôi phục được từ phần còn lại.

> **Hệ quả.** Nếu một bộ nén **chỉ** loại bỏ token dư thừa (không cố gắng ước lượng `Rel`), thì:
> 1. Nó không cần `Q` → tính một lần, cache, dùng cho mọi câu hỏi của video.
> 2. Đầu ra hợp lệ cho mọi `Q`, không chỉ `Q` đã thấy.

**Đây chính là chỗ `Rel` phát huy tác dụng trong lập luận:** phải định nghĩa `Rel` mới nói được câu "chúng tôi tối ưu `R` chứ không phải `Rel`, và đó là lý do phương pháp cache được." Không có `Rel` trong phần setup thì mệnh đề trên không phát biểu được.

### 0.3 Bỏ `Rel` thì mất gì?

Đặt bài toán chọn tối ưu ở hai chế độ:

```
S*(Q) = argmax_{|S|=K}  A(X_S, Q)              (query-aware: mỗi câu hỏi một tập)
S*    = argmax_{|S|=K}  E_Q [ A(X_S, Q) ]      (query-agnostic: một tập chung)
```

Chi phí của việc bỏ `Q`:

```
Gap = E_Q[ A(X_{S*(Q)}) ] − E_Q[ A(X_{S*}) ]  ≥ 0
```

Gap luôn không âm. Câu hỏi là **lớn bao nhiêu** — và câu trả lời phụ thuộc hoàn toàn vào việc bạn đang **chọn frame** hay **phân bổ token**.

### 0.4 ⭐ Bất đối xứng cốt lõi — luận điểm chính của bài báo

**Chọn frame là quyết định recall CỨNG và KHÔNG THỂ ĐẢO NGƯỢC.**
Bỏ frame `t` → đóng góp của nó bằng 0 vĩnh viễn. Nếu bằng chứng nằm ở đó, không có gì cứu được. Và bạn **không thể biết** frame nào chứa bằng chứng nếu không biết câu hỏi. Nên `Gap` lớn.

**Phân bổ token là điều chỉnh MỀM về độ chi tiết.**
Với sàn `B_min > 0`, **mọi đoạn đều hiện diện**. Bạn chỉ quyết định đoạn nào được xem ở độ nét cao, đoạn nào chỉ thumbnail. Bằng chứng bị giảm chất lượng chứ không bị xóa. Nên `Gap` nhỏ.

**Ba dữ kiện độc lập cùng khớp:**

| Bằng chứng | Kết quả |
|---|---|
| QCA ablation: bỏ hạng tử content-deviation (query-agnostic) | VideoMME **tăng** 70.1 → 70.4 |
| KTV (chọn frame query-agnostic) trên LLaVA-1.6 | 42.48 < uniform-24 **43.30** |
| SCORE (nén token query-agnostic) trên Qwen2.5-VL | **65.3 > Vanilla 63.8** |

Cùng một triết lý query-agnostic: **thất bại ở tầng frame, SOTA ở tầng token.**

> **Câu văn cho bài báo:**
> *Độ dư thừa độc lập với câu hỏi; độ liên quan thì không. Chọn frame buộc phải phá hủy thông tin nên cần câu hỏi. Phân bổ token chỉ điều chỉnh độ chi tiết nên không cần.*

### 0.5 Bằng chứng từ literature: nhánh thành công đều query-agnostic

| Method | Thấy câu hỏi? | Qwen2.5-VL / VideoMME |
|---|---|---|
| FastVID | ✗ | 63.3 @ 25% |
| HoliTom | ✗ | 61.6 @ 40% |
| VidCom² | ✗ | 64.6 @ 25% |
| **SCORE** | ✗ (input chỉ là `[X; ΔX]`) | **65.3 @ 25%** |
| PruneVid | ✓ (query attention ở layer 10) | yếu nhất nhóm |

Policy network của SCORE **không bao giờ nhìn thấy text**. Nó nhận token embedding và residual liên frame, xuất xác suất giữ/bỏ. Và nó vượt cả model không nén. Phương pháp query-aware duy nhất trong nhóm lại là phương pháp kém nhất.

**Kết luận:** query-agnostic không phải hạn chế. Nó là paradigm chủ đạo của nhánh này. Ý tưởng ban đầu đúng — chỉ là trước đây áp dụng ở sai tầng (frame thay vì token).

---

<a name="phần-1"></a>
## PHẦN 1 — ĐẶT BÀI TOÁN

### 1.1 Setup

Video được sample thành `F` frame, mỗi frame mã hóa thành `N` token bởi vision encoder đóng băng:

```
X ∈ R^{F × N × D},    X[t] ∈ R^{N × D}
```

Chi phí prefill của LLM bị chi phối bởi self-attention:

```
O( (FN + |Q| + |Y|)² ) ≈ O( (FN)² )
```

vì `FN >> |Q|, |Y|`. Trên Qwen2.5-VL-7B, 128 frame: `FN = 42,566` token thị giác so với 42,657 token tổng — **text chỉ chiếm 0.2%**. (SCORE Table 3)

### 1.2 Mục tiêu ràng buộc

SCORE §3, Eq. (1):

```
min_{X_comp ⊆ X}  |X_comp|
        s.t.       A(X_comp) ≥ A(X) − δ
```

Giảm token tối đa với ràng buộc accuracy không tụt quá `δ`. Không giải trực tiếp được: không gian tập con là `2^{FN}` và `A(·)` không khả vi theo lựa chọn rời rạc.

### 1.3 ⭐ Phân rã hai tầng — bước đi cấu trúc quan trọng nhất

```
     {B_s}_{s=1..S},  Σ_s B_s ≤ B              T_s = Select(X_s, B_s)
   └────────────────────────────────┘      └──────────────────────────┘
   TẦNG 1: phân bổ ngân sách                TẦNG 2: chọn trong đoạn
   (ĐÓNG GÓP CỦA TA)                        (mượn FastVID)
```

**Luận điểm:** literature đã tối ưu Tầng 2 đến cạn kiệt và để Tầng 1 ở nghiệm tầm thường `B_s = ρ·|X_s|` (tỉ lệ thuận với độ dài đoạn). Chúng tôi chỉ ra Tầng 1 mới là chỗ còn dư địa.

---

<a name="phần-2"></a>
## PHẦN 2 — TẦNG 1: PHÂN ĐOẠN THỜI GIAN ĐỘNG

### 2.1 Tại sao phải phân đoạn

FastVID §3.2 đưa hai lý do:

**Toàn vẹn thời gian.** Xáo trộn thứ tự frame hoặc bỏ frame → hiểu sai nội dung. Phải prune *trong* đoạn liên tục, không bao giờ xuyên đoạn.

**Dư thừa trong đoạn.** Merge chỉ an toàn giữa các token rất giống nhau. Đoạn phải tối đa hóa độ tương đồng nội bộ.

**Ablation — FastVID Table 6** (LLaVA-OV, r=10%):

| Phân đoạn | MVBench | LongVideoBench | MLVU | VideoMME | Avg Acc |
|---|---|---|---|---|---|
| Fixed-interval | 55.1 | 53.6 | 61.7 | 55.4 | 95.3% |
| Cluster-based | 53.2 | 53.0 | 61.7 | 54.8 | 93.9% |
| **DySeg** | **55.9** | **56.3** | **62.7** | **57.3** | **98.0%** |

> ⚠️ **Clustering TỆ HƠN cả chia đều** (93.9 < 95.3). Lý do FastVID đưa ra: clustering bỏ qua thứ tự thời gian, gộp các frame ở những thời điểm khác nhau vào cùng một đoạn.
>
> Điều này vừa **xác nhận gợi ý #2 trong báo cáo** (chia clip ngắn 10–30s), vừa **cảnh báo đừng dùng clustering**.

### 2.2 Thuật toán DySeg

Đặc trưng frame toàn cục `f_i` (mean-pool token, hoặc DINOv2 CLS). Độ tương đồng transition:

```
t_i = cos(f_i, f_{i+1}),    i = 1, ..., F−1
T   = {t_1, ..., t_{F−1}}
```

Tập biên:

```
S₁ = argmin_{c−1} T            (c−1 transition thấp nhất)
S₂ = { i : t_i < τ }           (mọi transition dưới ngưỡng)
S  = S₁ ∪ S₂
```

- `S₁`: đảm bảo tối thiểu `c` đoạn ngay cả với video mượt, không có cắt cảnh.
- `S₂`: bắt cắt cảnh thật.

**Tự thích nghi:** video đơn giản → `S₂ = ∅`, `S₁` chia mịn các thay đổi tinh tế. Video phức tạp → `S₁ ⊂ S₂`, `S₂` chi phối.

**Siêu tham số:** `c = 8`, `τ = 0.9`. Ablation Fig. 6 của FastVID: ổn định với `1 ≤ c ≤ 16`, tụt ở `c = 32` (một frame/đoạn phá hủy quan hệ thời gian).

**Chi phí:** 0.5 ms (FastVID Table 3), so với 5.2 ms cho clustering của PruneVid.

### 2.3 Điều chỉnh cho setting hiện tại

Ở `F = 128` trên video 60 phút, một frame ≈ 28 giây. Mục tiêu 10–30s tương ứng 1–2 frame/đoạn — quá mịn. Hai lựa chọn:

- **(a)** Sample FPS cao hơn chỉ cho khâu phân đoạn (rẻ: chỉ forward DINOv2), rồi phân bổ frame cho đoạn ở Tầng 2.
- **(b)** Giữ `F = 128`, "đoạn" nghĩa là "khoảng liên tục cùng shot", thường 4–16 frame. Đặt `c` sao cho `F/c ≈ 8–16`.

> **Khuyến nghị (b)** cho bản đầu — trùng khớp setting FastVID nên bước reproduce kiểm chứng được trực tiếp.

### 2.4 Cache

`{S, f}` độc lập với câu hỏi → tính một lần/video. VideoMME có ~3 câu hỏi/video → **khấu hao 3×**.

---

<a name="phần-3"></a>
## PHẦN 3 — TẦNG 2: PHÂN BỔ NGÂN SÁCH KHÔNG ĐỒNG NHẤT (ĐÓNG GÓP CHÍNH)

### 3.1 Khoảng trống, phát biểu chính xác

FastVID §3.3 viết: *"Với một đoạn gồm P frame, chúng tôi giữ lại rPN token, trong đó N là số token mỗi frame và r là tỉ lệ giữ lại."*

`r` là **hằng số toàn cục**:

```
B_s^FastVID = r · P_s · N
```

> Một đoạn 16 frame quay người ngồi nói và một đoạn 16 frame hành động dày đặc nhận **ngân sách y hệt nhau**.

- **HoliTom, VisionZip, DyCoke**: đều vậy.
- **VidCom²**: điều chỉnh theo từng frame nhưng **không** phân bổ xuyên đoạn thời gian.
- **SCORE**: có phân bổ không đồng nhất nhưng chỉ *ngầm* qua global top-K học được — không phát biểu thành nguyên lý, không ablate.

> **Chưa ai đặt câu hỏi: bản thân tỉ lệ giữ lại có nên không đồng nhất theo thời gian không, và mức nào?**

### 3.2 Đo mật độ thông tin của đoạn

Cần `v_s` = mật độ thông tin của đoạn `s`, query-agnostic. Ba ước lượng.

#### 3.2.1 Ước lượng A — Residual tái cấu trúc (RECON, dùng lại đúng chỗ)

Với `F_s = [f_{s,1}, ..., f_{s,P_s}] ∈ R^{D × P_s}`, chạy greedy QR-pivot. Ở bước `k`:

```
R_k = F_s − F_{s,P_k} · F_{s,P_k}^† · F_s = (I − Π_{P_k}) F_s
```

trong đó `Π_{P_k}` là phép chiếu trực giao lên `span(F_{s,P_k})`.

Đường cong tái cấu trúc:

```
RC_s(k) = 1 − ‖R_k‖²_F / ‖F_s‖²_F

v_s^RC  = min{ k : RC_s(k) ≥ γ } / P_s
```

Diễn giải: đoạn tĩnh đạt `γ = 0.9` với 1 pivot (`v_s` nhỏ); đoạn động cần nhiều pivot (`v_s` lớn).

> ### ⭐ Tại sao lần này RECON hoạt động còn trước đây thì không
>
> RECON thất bại ở chọn frame vì `argmax_k ‖r_k‖` bị chi phối bởi **thay đổi ngoại hình toàn cục** — cắt cảnh, ánh sáng, camera pan — chứ không phải bằng chứng QA (thường nhỏ về không gian, thấp về năng lượng).
>
> Nhưng ở đây **ta không dùng residual để CHỌN gì cả**. Ta dùng *tốc độ suy giảm tổng hợp trên một khoảng* làm một con số vô hướng tóm tắt lượng nội dung độc lập trong khoảng đó. **Chế độ hỏng (chọn nhầm frame outlier) biến mất về mặt cấu trúc vì không có phép argmax nào được thực hiện.**
>
> *Đây là câu trả lời của bài báo cho câu hỏi "tại sao hướng reconstruction trước đây thất bại?"*

#### 3.2.2 Ước lượng B — Effective rank

```
erank(F_s) = exp( − Σ_i p_i log p_i ),    p_i = σ_i / Σ_j σ_j

v_s^er = erank(F_s) / P_s
```

với `σ_i` là các trị kỳ dị của `F_s`. (Roberts, Varshavsky & Wolf 2007 — entropy của phổ chuẩn hóa, một nới lỏng liên tục của hạng ma trận.)

> **Cứu lại phát hiện effective rank.** Trước đây đo erank ≈ 141 trên toàn video và hiểu là số sự kiện. Không phải — chuyển động camera liên tục cũng đẩy erank lên cao trong `R^768`.
>
> **Nhưng trong một đoạn ngắn cùng shot, erank là thước đo hợp lệ của độ đa dạng thị giác**, vì chuyển động camera bị chặn trong đoạn. Chia cho `P_s` và **so sánh tương đối giữa các đoạn cùng video** loại bỏ hoàn toàn vấn đề thang tuyệt đối.

#### 3.2.3 Ước lượng C — Năng lượng residual thời gian (surprise signal của SCORE)

SCORE §4.2, Eq. (2)–(3):

```
ΔX[t] = X[t] − X[t−1] ∈ R^{N × D},   X[0] = 0
H[t]  = [ X[t] ; ΔX[t] ] ∈ R^{N × 2D}
```

Tổng hợp về mức đoạn:

```
v_s^Δ = (1 / (P_s · N)) · Σ_{t ∈ s} Σ_{i=1..N} ‖ΔX[t,i]‖₂
```

**Lý do SCORE đưa ra (§4.2):** embedding các frame liền kề tương quan rất cao do liên tục thời gian; chỉ dùng `X[t]` khiến policy gán điểm gần như y hệt qua các frame và không tìm được vùng thay đổi. `ΔX` là **bộ lọc thông cao theo thời gian**.

**SCORE Table 5** (VideoMME, Qwen2.5-VL):

| | R | Short | Medium | Long | Overall |
|---|---|---|---|---|---|
| Vanilla | 100% | 74.4 | 64.2 | 52.8 | 63.8 |
| không residual | 40% | 74.4 | 65.3 | 54.3 | 64.7 |
| **có residual** | 40% | 75.0 | 65.0 | 55.1 | **65.0** |
| không residual | 25% | 74.8 | 64.9 | 54.8 | 64.5 |
| **có residual** | 25% | 74.8 | 65.4 | 55.7 | **65.3** |
| không residual | 10% | 71.4 | 62.0 | 52.6 | 62.0 |
| **có residual** | 10% | 73.6 | 63.1 | 53.9 | **63.6** |

> Chênh lệch **nới rộng khi ngân sách siết**: +0.3 @40%, +0.8 @25%, **+1.6 @10%**.
>
> Đây là kết quả chuyển giao được nhất trong survey — tín hiệu residual quan trọng nhất **đúng ở vùng ta muốn hoạt động**.

#### 3.2.4 Kết hợp khuyến nghị

```
v_s = λ · ṽ_s^RC + (1−λ) · ṽ_s^Δ       (ṽ = min-max chuẩn hóa TRONG video)
```

- `v^RC` bắt **đa dạng không gian/ngoại hình** (bao nhiêu "vẻ ngoài" khác nhau)
- `v^Δ` bắt **động lực thời gian** (bao nhiêu chuyển động)

Bổ trợ nhau: một đoạn có thể đa dạng nhưng tĩnh (pan chậm qua bức tranh tường) hoặc đồng nhất nhưng động (người khoa tay trước tường trơn).

> ⚠️ **Chuẩn hóa TRONG video, không bao giờ toàn cục.** Độ lớn residual tuyệt đối biến thiên rất mạnh giữa các video (độ phân giải, codec, nội dung), và ta chỉ cần so các đoạn của **cùng một video** với nhau.

### 3.3 Phân bổ với sàn phủ

```
┌───────────────────────────────────────────────────────────────┐
│                                    exp(ṽ_s / T)               │
│  B_s = B_min + (B − S·B_min) · ─────────────────────          │
│                                  Σ_{s'} exp(ṽ_{s'} / T)       │
└───────────────────────────────────────────────────────────────┘

ràng buộc:  Σ_s B_s = B   và   B_min ≤ B_s ≤ B_max
```

**Ba thành phần:**

| Thành phần | Vai trò |
|---|---|
| `B_min` | **Sàn phủ.** Mọi đoạn được đảm bảo mức tối thiểu. Đảm bảo `∀s: B_s > 0` — không đoạn nào bị xóa hẳn. |
| `T` | **Nhiệt độ.** `T → ∞` **khôi phục đúng phân bổ đều** (chính là baseline FastVID); `T → 0` dồn toàn bộ ngân sách tùy ý vào đoạn có `v` cao nhất. |
| `B_max` | Trần, chống một đoạn nuốt hết. |

> `T → ∞` khiến **phân bổ đều trở thành trường hợp đặc biệt** của phương pháp này — một dòng ablation cực sạch.

#### ⭐ Tại sao sàn phủ là bắt buộc — dữ liệu của chính chúng ta

Từ Bảng 2 (LLaVA-1.6-Vicuna-7B, xem §8.1):

| Điều kiện | Overall | Short | Medium | Long |
|---|---|---|---|---|
| Uniform 12 frame | **43.15** | 52.78 | 41.56 | 35.11 |
| Oracle, K=30 → top-12 theo đáp án đúng | 42.81 | 52.78 | 40.44 | 35.22 |

> Một bộ chọn **có quyền truy cập đáp án đúng**, chọn 12 từ pool 30, lại **thua** `np.linspace`.
>
> Giải thích duy nhất mạch lạc: **dồn cục** — xếp hạng theo relevance tập trung các lựa chọn vào một cửa sổ thời gian hẹp, và mất độ phủ tốn hơn cái được từ relevance.
>
> **Vì ngay cả oracle cũng không thắng nổi uniform khi không có ràng buộc phủ, không bộ chọn khả thi nào làm được.** Đây là dữ liệu gốc của chúng ta và nó thuộc về hình minh họa động lực của bài báo.

**Phát biểu hình thức:** gọi `L_cov` là mất mát do bỏ sót bằng chứng hoàn toàn, `G_det` là cái được từ độ chi tiết thêm. Kết quả trên cho thấy `L_cov > G_det` khi phân bổ không ràng buộc. **Sàn chặn `L_cov` theo cấu trúc.**

**Giá trị gợi ý** (`B` = 15% dense):
- `B_min ≈ 18–36` token/frame-tương-đương
- `B_max ≈ 4×` trung bình
- `T ∈ [0.5, 2.0]`

### 3.4 Cổng entropy ở mức video

Nhiệt độ tối ưu không cố định giữa các video. Định nghĩa **độ tập trung thông tin**:

```
H_v = − Σ_{s=1..S} p_s log p_s,    p_s = exp(ṽ_s/T₀) / Σ_{s'} exp(ṽ_{s'}/T₀)

Ĥ_v = H_v / log S  ∈ [0, 1]
```

| `Ĥ_v` | Ý nghĩa | Hành động |
|---|---|---|
| → 1 | Thông tin trải đều (phim tài liệu liên tục, một cảnh dài) | Không có gì để khai thác → **làm phẳng về uniform** |
| → 0 | Thông tin tập trung ở vài đoạn (phần lớn tĩnh, thỉnh thoảng có hoạt động) | **Làm sắc phân bổ** |

```
T(Ĥ_v) = T₀ · exp( κ (Ĥ_v − H₀) )
```

Đây là **thích nghi bậc hai**: không chỉ "phân bổ không đồng nhất" mà "thích nghi *mức độ* không đồng nhất theo từng video". Query-agnostic (chỉ phụ thuộc `{v_s}`), chi phí một phép entropy, và cache cùng video.

> **Chưa ai có cái này.** FastVID: `r` cố định. HoliTom: cố định. VidCom²: theo frame, không theo cấu trúc video. SCORE: ngầm qua top-K học được, không bao giờ phơi bày hay ablate.

Nó còn sinh ra **dự đoán kiểm chứng được**: mức tăng phải tương quan với `1 − Ĥ_v` → biểu đồ scatter gain theo độ tập trung, một kết quả ở **mức cơ chế** chứ không chỉ một con số.

---

<a name="phần-4"></a>
## PHẦN 4 — TẦNG 3: CHỌN TOKEN TRONG ĐOẠN

> Mượn nguyên STPrune của FastVID. **Đừng phát minh lại** — nó đã được ablate kỹ và ta không được gì khi làm khác.

### 4.1 Hai loại token bổ trợ

```
B_s^DTM = d · B_s        (đại diện — representative)
B_s^ATS = (1−d) · B_s    (nổi bật — salient)
```

**FastVID Table 4** (LLaVA-OV, r=10%):

| d | MVBench | LongVideoBench | MLVU | VideoMME | Avg Acc |
|---|---|---|---|---|---|
| 0.0 (chỉ ATS) | 55.6 | 54.5 | 62.0 | 56.2 | 96.3% |
| **0.4** | **55.9** | **56.3** | **62.7** | **57.3** | **98.0%** |
| 1.0 (chỉ DTM) | 54.1 | 50.1 | 61.5 | 55.9 | 93.5% |

Kết hợp hơn ATS-đơn **1.8 điểm** và DTM-đơn **4.5 điểm**.

### 4.2 DTM — Density-based Token Merging

Density peaks clustering (Rodriguez & Laio, *Science* 2014). Với mỗi token `v_i` trong anchor frame:

```
ρ_i = exp( −(1/k) Σ_{v_j ∈ kNN(v_i)} d(v_i, v_j)² )         (mật độ cục bộ)

       ┌ min_{j: ρ_j > ρ_i} d(v_i, v_j)    nếu ∃j: ρ_j > ρ_i
δ_i = ─┤                                                     (độ tách biệt)
       └ max_j d(v_i, v_j)                 ngược lại
```

**Điểm = ρ_i × δ_i**: bao quanh bởi nhiều token tương tự (**đại diện**) VÀ xa các token mật độ cao khác (**phân biệt**). Token điểm cao nhất trở thành anchor.

**Gán:** mỗi token còn lại trong đoạn về anchor gần nhất theo cosine.

**Tổng hợp (FastVID Eq. 5):**

```
a* = α · a + ((1−α)/n) · Σ_{i=1..n} b_i,    α = 0.6
```

**Anchor frame:** chỉ lấy anchor từ frame cách nhau `p = 4`, **không** lấy từ toàn đoạn — vì chọn anchor xuyên toàn đoạn sẽ phá vỡ quan hệ không gian (anchor của các phần khác nhau của cùng vật thể rơi vào các frame khác nhau).

### 4.3 ⚠️ Ràng buộc positional — YÊU CẦU CỨNG

FastVID Fig. 4 & Table 5. Cluster-based merging nối các centroid, **vứt bỏ index vị trí gốc**. RoPE mã hóa vị trí tương đối; không có index đúng thì không biểu diễn được cấu trúc không-thời gian.

**FastVID Table 5:**

| Merging | MVBench | LongVideoBench | MLVU | VideoMME | Avg Acc |
|---|---|---|---|---|---|
| *LLaVA-OneVision* | | | | | |
| Uniform (VisionZip) | 55.0 | 54.7 | 62.0 | 56.4 | 96.2% |
| Cluster-based | 55.6 | 55.2 | 62.4 | 57.3 | 97.2% |
| **DTM** | **55.9** | **56.3** | **62.7** | **57.3** | **98.0%** |
| *LLaVA-Video* | | | | | |
| Uniform | 55.4 | 54.6 | 62.9 | 59.8 | 91.5% |
| Cluster-based | 55.3 | 55.1 | 62.9 | 60.1 | 91.8% |
| **DTM** | **58.5** | **56.5** | **64.9** | **60.7** | **94.7%** |

DTM hơn cluster-based **2.9 điểm** trên LLaVA-Video. SCORE §4.5 độc lập đi đến cùng yêu cầu: *giữ nguyên index vị trí gốc của token được chọn, bảo toàn thứ tự không-thời gian mà không thay đổi positional encoding.*

> **Hai paper, độc lập, cùng xác định bảo toàn vị trí là then chốt. Coi đây là ràng buộc cứng.**
> Trên Qwen2.5-VL còn quan trọng hơn vì mRoPE mã hóa **thời gian tuyệt đối**.

### 4.4 Thay thế ATS trên Qwen2.5-VL — bắt buộc, và là một delta chính đáng

ATS dùng `[CLS]` attention. FastVID §3.3 ghi nhận Video LLM dùng SigLIP lấy layer áp chót, bỏ qua head sinh `[CLS]` — nên họ **tích hợp lại một SigLIP head pretrain** (15.2M tham số).

**Qwen2.5-VL không có `[CLS]` token.** Vision encoder của nó dùng window attention với native dynamic resolution. Nên **phải** thay ATS.

Thay thế query-agnostic tự nhiên chính là residual thời gian, đã được SCORE kiểm chứng:

```
sal(t, i) = ‖ΔX[t,i]‖₂ = ‖X[t,i] − X[t−1,i]‖₂
```

Chọn top `(1−d)·B_s` token theo `sal`. **Diễn giải:** token có nội dung thay đổi so với frame trước mang thông tin mới; token tĩnh thì không.

> Đây là đóng góp thật, không phải giải pháp chữa cháy — ta chứng minh một tín hiệu salience **thuần thời gian, độc lập kiến trúc**, có thể thay attention proxy phụ thuộc kiến trúc. Nó tổng quát cho mọi backbone.
>
> Đưa vào ablation: **ATS ([CLS], chỉ LLaVA) vs. residual salience (phổ quát)**.

---

<a name="phần-5"></a>
## PHẦN 5 — THUẬT TOÁN ĐẦY ĐỦ

```
INPUT: video V, ngân sách token B
       c=8, τ=0.9, d=0.4, p=4, α=0.6, λ=0.5, T₀=1.0, κ, B_min, B_max

═══ GIAI ĐOẠN 0: MÃ HÓA (một lần/video, cache) ═══
 1  frames ← sample(V, F=128)
 2  X ← VisionEncoder(frames)              # [F,N,D], đóng băng
 3  f ← mean_pool(X, dim=token)            # [F,D]
 4  ΔX ← X[1:] − X[:−1];  ΔX[0] ← X[0]     # SCORE Eq.(2)

═══ TẦNG 1: PHÂN ĐOẠN (cache) ═══                          # FastVID §3.2
 5  t_i ← cos(f_i, f_{i+1})
 6  S₁ ← argmin_{c−1}(T);  S₂ ← {i: t_i<τ};  S ← S₁ ∪ S₂
 7  segments ← split(F, boundaries=S)

═══ TẦNG 2: PHÂN BỔ NGÂN SÁCH (cache) ═══                  # ĐÓNG GÓP CỦA TA
 8  for mỗi đoạn s:
 9      v_s^RC ← greedy_QR_pivot_curve(F_s, γ) / P_s
10      v_s^Δ  ← mean(‖ΔX[t,i]‖) với t ∈ s
11  ṽ ← minmax_normalize({ λ·v^RC + (1−λ)·v^Δ })            # TRONG video
12  Ĥ ← entropy(softmax(ṽ/T₀)) / log(S)
13  T ← T₀ · exp(κ(Ĥ − H₀))                                 # cổng entropy
14  B_s ← B_min + (B − S·B_min) · softmax(ṽ/T)_s
15  B_s ← clip(B_s, B_min, B_max); chuẩn hóa lại Σ B_s = B

═══ TẦNG 3: PRUNE TRONG ĐOẠN ═══                           # FastVID §3.3
16  for mỗi đoạn s:
17      anchor_frames ← mỗi frame thứ p trong s
18      ρ_i ← exp(−(1/k) Σ_{kNN} d(v_i,v_j)²)
19      δ_i ← min_{ρ_j>ρ_i} d(v_i,v_j)
20      anchors ← top-(d·B_s / ⌈P_s/p⌉) theo ρ_i·δ_i
21      gán non-anchor → anchor gần nhất (cosine)
22      a* ← α·a + ((1−α)/n)·Σ b_i
23      salient ← top-((1−d)·B_s) theo ‖ΔX[t,i]‖₂            # thay ATS
24      keep(s) ← anchors* ∪ salient
25      ⚠️ GIỮ NGUYÊN INDEX VỊ TRÍ GỐC

═══ OUTPUT ═══
26  X_comp ← ghép các đoạn, đúng thứ tự thời gian
27  answer ← FrozenLLM(X_comp, Q)          ← Q CHỈ XUẤT HIỆN Ở ĐÂY
```

| | |
|---|---|
| **Cache được** | Dòng 1–15 — toàn bộ **trước khi thấy `Q`** |
| **Mỗi câu hỏi** | **KHÔNG GÌ CẢ.** Tập token nén phục vụ mọi câu hỏi của video. |

---

<a name="phần-6"></a>
## PHẦN 6 — ĐỊNH VỊ VÀ NOVELTY

### 6.1 ⭐ Khoảng trống, bằng chính lời của SOTA

FastVID §C.1 (Limitations) thừa nhận: trên model lấy nhiều frame (LLaVA-Video 64, Qwen2-VL 768), accuracy tụt rõ so với model gốc, vì nhiều frame → ít token/frame → mỗi token quan trọng hơn → prune mạnh có nguy cơ bỏ thông tin then chốt. Và họ kết luận rằng **pruning query-agnostic của họ không làm nổi bật được thông tin liên quan câu hỏi vốn phân bố thưa thớt trong chuỗi frame dài**, đề xuất hướng tương lai là keyframe selection dẫn dắt bằng text.

> **Đọc kỹ — đây là cửa của chúng ta, nhưng KHÔNG theo hướng họ đề xuất.**
>
> Họ chẩn đoán đúng vấn đề (prune đồng nhất trên nhiều frame phá hủy thông tin thưa) nhưng đề xuất cách sửa query-aware.
>
> **Luận điểm của chúng ta: vấn đề giải được KHÔNG CẦN câu hỏi.** Lỗi nằm ở chỗ *ngân sách* đồng nhất, không phải ở chỗ *pruning* query-agnostic. Nếu đoạn nhiều thông tin nhận nhiều token hơn, thông tin thưa sẽ sống sót mà không cần nhìn câu hỏi.

Đây là một bất đồng trực tiếp, có cơ sở, với future work của một paper NeurIPS'25. Reviewer thích điều này.

### 6.2 Bảng đóng góp

| | Trạng thái |
|---|---|
| **C1.** Ngân sách token không đồng nhất xuyên đoạn, theo độ khó tái cấu trúc | **Mới.** Mọi công trình trước dùng `r` hằng số. |
| **C2.** Sàn phủ `B_min`, có động lực từ hiện tượng dồn cục đã đo được | **Mới**, hậu thuẫn bởi thí nghiệm oracle của chính chúng ta. |
| **C3.** Cổng entropy mức video điều chỉnh độ sắc của phân bổ | **Mới.** Không ai thích nghi độ quyết liệt theo cấu trúc video. |
| **C4.** Salience dựa residual thay `[CLS]` attention | **Mới**, độc lập backbone, cho phép chạy trên Qwen2.5-VL. |
| **C5.** Bất đối xứng dư thừa/liên quan → phân bổ token cache được, chọn frame thì không | **Framing mới**, hậu thuẫn bởi bằng chứng liên nhánh. |

### 6.3 Related Work — viết theo cách này

> **DyCoke** chia thành cửa sổ 4 frame cố định và giữ token của frame đầu — content-agnostic và bị chặn sàn ở 25% retention. **PruneVid** cluster token và dùng query attention bên trong LLM, nhưng clustering vứt bỏ thông tin vị trí và tốn latency cao. **FastVID** đóng góp phân đoạn động và density-based pruning trong đoạn, nhưng áp tỉ lệ giữ lại *hằng số* cho mọi đoạn. **HoliTom** phân đoạn theo dư thừa toàn cục và merge spatio-temporal, cũng ở tỉ lệ đồng nhất. **VidCom²** điều chỉnh cường độ nén theo tính độc đáo của từng frame nhưng không phân bổ xuyên đoạn thời gian. **SCORE** học chính sách giữ token bằng RL; global top-K của nó tạo ra phân bổ không đồng nhất một cách ngầm định, nhưng nguyên lý này không được phát biểu cũng không được ablate, và nó cần ~67K mẫu huấn luyện với re-annotation bằng teacher model.
>
> Chúng tôi trực giao với tất cả những công trình trên **ở tầng phân bổ**: chúng tôi giữ nguyên phân đoạn và chọn token trong đoạn của FastVID, và chỉ ra rằng làm cho bản thân ngân sách không đồng nhất — theo độ khó tái cấu trúc, với sàn phủ và độ sắc thích nghi theo video — mang lại thêm cải thiện, hoàn toàn training-free.

**Về QCA.** Cite nổi bật. QCA (arXiv 2607.00983) chung tác giả với KTV (Song, Peng, Zhou, Ji — Đại học Hạ Môn), nên họ nhiều khả năng sẽ review bài này. Định vị rõ: *QCA phân bổ **frame** cho đoạn dùng query matching; chúng tôi phân bổ **token** cho đoạn dùng độ khó tái cấu trúc, không cần query và cache theo video.*

### 6.4 Bảng kết quả mục tiêu

Qwen2.5-VL-7B, VideoMME w/o subtitles, 128 frame:

| Method | R | Query? | Train? | Overall |
|---|---|---|---|---|
| Vanilla | 100% | — | — | 63.8 |
| HoliTom | 40% | ✗ | ✗ | 61.6 |
| FastVID | 40% | ✗ | ✗ | 63.7 |
| VidCom² | 25% | ✗ | ✗ | 64.6 |
| SCORE | 25% | ✗ | **✓** | 65.3 |
| **Ours** | **15%** | ✗ | ✗ | **≥64.6** |

**Luận điểm bán:** *đạt ngang phương pháp training-free tốt nhất ở mức retention thấp hơn đáng kể, không cần huấn luyện.*

*(Kết quả đo được thực tế — chưa đạt bảng mục tiêu này — nằm ở §8.3.)*

### 6.5 Bảng chi phí — luận điểm "con gà quả trứng"

| | Chi phí/video | Chi phí/câu hỏi |
|---|---|---|
| AKS / QCA | mã hóa frame | **BLIP-2 ITM trên mọi frame** |
| SCORE | mã hóa + policy | policy forward |
| **Ours** | mã hóa + phân đoạn + phân bổ | **KHÔNG** |

Với ~3 câu hỏi/video trên VideoMME → **khấu hao 3×** toàn bộ giai đoạn chọn lọc.

> Đây là phiên bản mạnh hơn của phê phán "con gà quả trứng": nó áp dụng cho cả **relevance scorer** (BLIP-2), không chỉ VLM.

*(Quy trình thực nghiệm đầy đủ — thí nghiệm cổng G1/G2/G3, bảng ablation từng dòng, sổ trích dẫn, survey nhánh token-compression, và đối chiếu chi tiết với đề xuất RL/V-JEPA của mục 6 — xem `idea.md` Phần 7–9 và Phụ lục A/B/C.)*


# 8. Kết quả thực nghiệm

## 8.1 Phase 0 — Kiểm chứng thực nghiệm trên pipeline hiện tại (2026-08-04 → 08-06)

Trước khi đầu tư vào ý tưởng ở §7, chúng tôi chạy hai thí nghiệm Phase-0 rẻ trên pipeline hiện tại (§3): **(1) một oracle upper bound** (xếp hạng đúng K=12 candidate KMeans/DINO theo độ tương đồng CLIP với *đáp án đúng* thay vì câu hỏi — giữ nguyên token budget/pruning downstream) và **(2) một sweep frame-vs-resolution** (uniform sampling, không chọn keyframe, tổng budget B=1872 cố định, đổi số frame). Mọi lượt chạy dưới đây là full VideoMME val (n=2700, 900/bucket), cùng `cls_new_token_sim` pruning, `rate=0.2`, `tokens_num=1872` trừ khi có ghi chú khác.

### Table 1 — Prior baselines (pre-Phase-0, reported by team)

| Method | VideoMME acc. |
| --- | --- |
| KTV (our reproduction) | 42.48% |
| KTV (paper, different backbone/setup — not directly comparable) | 44.00% |
| Temporal-Chain (best acc.) | 43.48% |
| Temporal-Chain (default) | 43.70% |
| Temporal-Chain + rerank | 42.59% |
| Temporal-Chain + F2C | 39.96% |

### Table 2 — Phase 0 on the current backbone (LLaVA-1.6-Vicuna-7B)

| Condition | Overall | Short | Medium | Long |
| --- | --- | --- | --- | --- |
| Oracle, K=12 candidates (rank by GT answer) | 43.59% | 53.22 | 41.78 | 35.78 |
| Oracle, K=30→top-12 candidates (rank by GT answer) | 42.81% | 52.78 | 40.44 | 35.22 |
| Uniform 8 frames (234 tok/frame) | 41.56% | 50.78 | 40.00 | 33.89 |
| Uniform 12 frames (156 tok/frame) | 43.15% | 52.78 | 41.56 | 35.11 |
| Uniform 16 frames (117 tok/frame) | 41.89% | 51.78 | 39.22 | 34.67 |
| Uniform 24 frames (78 tok/frame) | 43.30% | 53.44 | 41.33 | 35.11 |
| Uniform 32 frames (58 tok/frame) | 42.48% | 51.56 | 42.00 | 33.89 |

**Reading it:** every condition — different selection strategies, 4× range in frame count/resolution, even a "cheating" oracle — lands within **41.56–43.59%** (≤2 points spread), with **no monotonic trend** in the resolution sweep (noise-dominated, consistent with `temperature=0.2` sampling). Widening the oracle's candidate pool 12→30 did *not* help either (43.59%→42.81%, a *z*≈0.58 difference — statistical noise). This matches the pre-registered kill switch: *"if accuracy moves 1–2 points, the backbone is the bottleneck, not the selector."* Verdict: **backbone-bound**, not selection-bound, on this checkpoint.

### Table 3 — Abandoned: LLaVA-NeXT-Video-7B (`lmms-lab/LLaVA-NeXT-Video-7B`)

30-question smoke subsets (n=30, not full runs — used to diagnose before committing a full pass):

| Condition | Acc. (n=30) | Note |
| --- | --- | --- |
| ChatML conv_mode, KTV F2C keyframes | 26.67% | ~random chance (25%) |
| Vicuna conv_mode, KTV F2C keyframes | 33.33% | conv_mode helps a little, not enough |
| Vicuna conv_mode, dense 32-frame, `prune_mode=null` | 23.33% (13/30 unparsed) | **invalid run** — 32×576=18,432 tokens overflowed the 8192 context, truncating the question/options off the prompt entirely |
| Vicuna conv_mode, dense 32-frame, `prune_mode=cls_new_token_sim` (context fixed) | 26.67% | still ~random after fixing overflow |
| Vicuna conv_mode, raw 12-frame, **no pruning at all** | 30.00% | still ~random even with untouched CLIP tokens |

**Root cause (confirmed by reading `clip_encoder.py`):** `CLIPVisionTower.load_model()` unconditionally re-downloads vanilla `openai/clip-vit-large-patch14-336` from HF Hub — it never loads the vision-tower weights bundled in the checkpoint itself. Harmless for `llava-v1.6-vicuna-7b` (vision tower frozen at vanilla CLIP during its training), but silently wrong for LLaVA-NeXT-Video-7B if its vision tower was fine-tuned — the mm_projector ends up reading features it was never trained on. Fixing this needs real code (re-apply the checkpoint's own `vision_tower.*` state dict after load) with no guaranteed payoff. **Decision: abandoned, pivoted to Qwen2.5-VL-7B instead** (bigger integration cost, but a clean rewrite rather than inheriting this codebase's loading bug).

### Table 4 — Phase 0 on the new backbone (Qwen2.5-VL-7B-Instruct)

Standalone script (`run_inference_qwen25vl.py`), separate `transformers>=4.49` env — architecturally incompatible with the `ktv/llava` fork (native dynamic-resolution vision encoder, not CLIP-ViT; token budget controlled via `min_pixels=max_pixels` on the processor, ≈28×28 px per merged visual token).

| Condition | Overall | Short | Medium | Long |
| --- | --- | --- | --- | --- |
| Uniform 8 frames (234 tok/frame) | 52.33% | 62.89 | 50.33 | 43.78 |
| Uniform 12 frames (156 tok/frame) | 54.11% | 63.78 | 50.78 | 47.78 |
| Uniform 16 frames (117 tok/frame) | 54.67% | 65.22 | 52.44 | 46.33 |
| Uniform 24 frames (78 tok/frame) | 54.63% | 65.33 | 51.89 | 46.67 |
| Uniform 32 frames (58 tok/frame) | 55.52% | 66.33 | 52.22 | 48.00 |
| Oracle, K=12 (rank by GT answer, same keyframe file as Table 2) | 56.19% | 67.11 | 52.11 | 49.33 |

*Caveat on the K=12 oracle row above (found later): `run_inference_qwen25vl.py` sorts loaded frames by index and applies one uniform `min_pixels` to all of them, so the CLIP-answer rank is discarded — and since candidates=keyframes=12 here, CLIP never filters either. That row ends up measuring "KMeans-diverse set vs uniform," not relevance. See Table 5 for the corrected test.*

### Table 5 — CAS-QTA re-verification + corrected selection-vs-oracle test on Qwen2.5-VL-7B

*(CAS-QTA = RECON-Select + QTA, the earlier proposal that used to occupy §7 of this document before it was replaced by the current idea; kept here only as an experimental result.)*

**CAS-QTA (`recon_select.py`+`qta_allocated.py`) re-verified on LLaVA-1.6-Vicuna-7B**, `order>=12` patch confirmed present, `verify_token_budget.py` clean (0 crashes/flattening/overflow) — the earlier ~0.40 result was *not* a bug. 5 clean full runs (unparsed=0) across different hyperparameters land at **36.78–41.67%** (best: `l_max=2, top_m=4`). A fixed-1872-token run (40.22%) ≈ the adaptive-budget default (40.00%, mean 5737 tok/question) — rules out token budget as the explanation. Best CAS-QTA config is statistically tied with the KTV baseline (*z*≈0.60, p≈0.55) — **no clear win, but not a clear loss either.**

**Corrected 3-way decomposition on Qwen2.5-VL-7B** (num_candidates=30→12, so CLIP *actually* filters; K=12, B=1872 throughout):

| Condition | Overall | Short | Medium | Long |
| --- | --- | --- | --- | --- |
| Uniform 12 frames | 54.11% | 63.78 | 50.78 | 47.78 |
| Wide-diverse, ranked by **question** (deployable) | 54.56% | 63.44 | 52.33 | 47.89 |
| Wide-diverse, ranked by **answer** (true oracle) | 53.89% | 62.78 | 50.44 | 48.44 |

All three pairwise diffs are noise (*z*≈0.16–0.49, p>0.6 throughout) — **even a true, unconfounded oracle does not beat uniform sampling.**

### Key findings (Phase 0)

1. **Backbone was indeed the bottleneck.** Same frame count, same token budget (uniform 12 frames, 156 tok/frame): LLaVA-1.6-Vicuna-7B 43.15% → Qwen2.5-VL-7B 54.11%, **+10.96 points** (two-proportion *z*≈8.06, p<10⁻¹⁵ — decisive, not noise). Confirms the Phase-0 diagnosis and the decision to switch backbones over continuing to invest in frame-selection algorithms on the old checkpoint.
2. **Unlike the old backbone, Qwen2.5-VL shows a real frame-count trend**, not noise: 8 vs 32 frames, 52.33%→55.52% (*z*≈2.35, p≈0.019). This backbone benefits from denser temporal sampling even at lower per-frame resolution — a genuinely different characteristic from the old backbone's flat/noisy response to the same sweep.
3. **Frame-selection quality does not help, on either backbone, once measured cleanly.** CAS-QTA (reconstruction-signal, old backbone) ties existing heuristics at best. On Qwen2.5-VL, a properly-isolated true oracle (Table 5) is statistically indistinguishable from plain uniform sampling (*z*≈0.16). Two independent relevance signals (CLIP-similarity, reconstruction-residual), two backbones, same conclusion: **the frame-selection lever, as currently operationalized, has no measurable headroom left to capture on VideoMME at K=12.**
4. **The `long` bucket still benefits most from more frames** (43.78%→48.00% across the resolution sweep) — but not from selection quality (Table 5's long-bucket spread across uniform/question/oracle is only 47.78→48.44, noise-level).

**Implication for the RL vs. distillation debate (§6):** finding 3 undercuts the shared premise that training a policy (via RL or distillation) to approximate a relevance signal pays off — it only does if that signal beats naive baselines by a real margin, and it currently doesn't, on two different signals and two backbones.

### Table 6 — Token/frame budget scaling on Qwen2.5-VL-7B (complete)

K=12 and B=1872 are KTV-legacy numbers with no principled basis on this backbone — Qwen3-VL's own reported dense video config is ~128 frames / 16,384 tokens (per KiToke, arXiv: Qwen3-VL-8B dense ≈68.5 avg., 10–25%-budget pruning methods still at 56–66%). Testing whether scaling past 1872 closes the gap, holding ~128–150 tok/frame and increasing frame count (uniform sampling, `key_frame_path=null`, same backbone/env as Tables 4–5):

| num_frames | tokens_num | tok/frame | Overall | Short | Medium | Long |
| --- | --- | --- | --- | --- | --- | --- |
| 32 (reference, Table 4) | 1872 | 58 | 55.52% | 66.33 | 52.22 | 48.00 |
| 48 | 7200 | 150 | 58.07% | 69.33 | 55.56 | 49.33 |
| 64 | 9600 | 150 | 58.74% | 69.89 | 57.89 | 48.44 |
| **96** | **14400** | **150** | **59.41%** | 69.44 | 58.67 | 50.11 |
| 128 | 16384 | 128 | 59.19% | 69.67 | 57.78 | 50.11 |

**Verdict — rises then plateaus, doesn't close the gap to reported dense numbers.**
- 32→96 (peak): +3.89 pts, *z*≈2.89, p≈0.004 — the real, significant gain, confirming 1872 was leaving accuracy on the table.
- 48→64→96→128: **every pairwise diff among these four is noise** (*z*<1, p>0.4 throughout; 96 vs 128 even dips slightly, within noise). The curve is flat from ~7200 tokens onward — **budget scaling captured its gain by ~48 frames / 7200 tokens and saturated**, all the way out to 16,384 (the field's own "dense" reference point for this model class).
- Even at that ceiling, 59.19–59.41% remains **~9 points below** the ~68.5 dense average KiToke reports for Qwen3-VL-8B — a different (newer, larger) model generation, so the residual gap is more likely a **backbone-generation gap** (Qwen2.5 vs Qwen3) than anything left to capture via budget or selection on Qwen2.5-VL-7B specifically.
- `unparsed` stays low throughout (3–6 of 2700) — not a formatting/overflow artifact.

**Where this left the three levers tested pre-idea (frame selection, total budget, backbone):**

| Lever | Tested as | Result |
| --- | --- | --- |
| Frame **selection** (Tables 2, 5) | CLIP-relevance, reconstruction-residual (CAS-QTA), 2 backbones | No measurable benefit over uniform/naive baselines |
| Total **budget** (Table 6) | 1872→16,384 tokens on Qwen2.5-VL-7B | Real gain to ~7,200, then flat — saturated, not a route to closing the dense-number gap |
| **Backbone** (Tables 2 vs 4) | LLaVA-1.6-Vicuna-7B → Qwen2.5-VL-7B | +10.96 pts, by far the largest lever found |

Backbone swap dominated everything else tested at the time. Neither smarter frame selection nor more budget moved the needle nearly as much — pointing at the two untested levers per §2's three-layer framing: **token-level pruning/compression** (layer 2/3 — the direction §7's idea now pursues) and a further backbone upgrade.

## 8.2 Gate G1 — Xác minh input video-native cho Qwen2.5-VL (2026-08-12 → 08-13)

Đây là "Bước 0" trong lộ trình thực nghiệm của ý tưởng ở §7 (`idea.md` Phần 7.2, gate G1): trước khi đo bất kỳ phương pháp nén nào, phải xác nhận số "Vanilla" (không nén) của harness khớp với số Vanilla mà literature dùng làm mốc so sánh (63.8% cho Qwen2.5-VL-7B, VideoMME, 128 frame).

Test trực tiếp hai giả thuyết chính về nguyên nhân lệch: dựng script riêng (`run_inference_qwen25vl_native.py`), truyền frame dạng `{"type":"video","video":frames,"fps":sample_fps}` qua `qwen_vl_utils.process_vision_info` thay vì N `{"type":"image"}` riêng lẻ, bỏ hẳn ép `min_pixels=max_pixels`.

**Xác minh bằng chẩn đoán trực tiếp (không suy diễn từ pixel math), 1 câu mẫu:**
```
input_keys có 'video_grid_thw' + 'second_per_grid_ts'  → đúng nhánh video, không phải nhánh ảnh
video_grid_thw = [64, 40, 72]   (T=64 khớp 128 frame / temporal_patch_size=2)
visual_tokens sau spatial merge (÷4) = 46,080  →  360 tok/frame
```
**360 tok/frame gần khớp 333 tok/frame của SCORE Vanilla** (chênh <8%) — xác nhận đã sửa đúng cả hai giả thuyết: (a) độ phân giải bị ép, (b) thiếu metadata fps cho mRoPE.

**Nhưng full 2700 câu lại KHÔNG cải thiện:**

| Cấu hình | tok/frame | Overall | Short | Medium | Long |
|---|---|---|---|---|---|
| SCORE Vanilla (mục tiêu gate) | ~333 | **63.8** | 74.4 | 64.2 | 52.8 |
| Cũ — `f128_t16384` (ép min=max_pixels) | 128 | 59.19 | 69.67 | 57.78 | 50.11 |
| **Mới — native video, không ép pixel** | **~360** | **57.85** | 68.78 | 56.11 | 48.67 |

Mới vs Cũ: *z*≈1.0 (p≈0.32) — không có ý nghĩa thống kê, và về số tuyệt đối **thấp hơn**, dù dùng token nhiều hơn ~2.8×.

> **Kết luận: Gate G1 KHÔNG đạt** (cách 63.8% mục tiêu ~6 điểm). Giả thuyết (a)+(b) — dù xác nhận đã sửa đúng bằng chẩn đoán trực tiếp — **không giải thích được phần lớn khoảng cách với SCORE Vanilla**. Theo đúng nguyên tắc gate của `idea.md` ("Không qua thì không đi tiếp"), lẽ ra chưa nên sang bước reproduce FastVID.
>
> **Nghi phạm còn lại, chưa kiểm tra:** prompt template khác `lmms-eval` (đã sửa ở §8.3, chưa re-verify riêng cho dense/no-prune), cách sample frame (linspace đều vs fps-cap sampling của `lmms-eval`/Qwen). Tỉ lệ unparsed đã loại (2/2700 = 0.07%, không đáng kể).

## 8.3 Triển khai đầy đủ Tầng 1+2+3 (`run_inference_qwen25vl_dyseg.py`, 2026-08-13 → 08-14)

**Quyết định lệch khỏi lộ trình gốc:** thay vì đi tuần tự Bước 0(gate)→1(reproduce FastVID)→2(Tầng1+3)→3(Tầng2), triển khai **thẳng cả 3 tầng cùng lúc** (DySeg + phân bổ non-uniform + cổng entropy + STPrune, §7 Phần 2–5) trên nền prompt `lmms-eval` đã sửa, không chờ gate G1 đạt, không reproduce FastVID trước. Đánh đổi có ý thức: nhanh có con số để báo cáo, nhưng mất khả năng quy kết chính xác phần cải thiện đến từ tầng nào (không có ablation từng tầng theo đúng khuyến nghị của `idea.md`).

**Xác minh bắt buộc trước khi tin số liệu** — `verify_mode=true` (giữ nguyên 100% token, chỉ test cơ chế dựng lại `inputs_embeds`/`position_ids` cho tập con không liên tục): **30/30 khớp tuyệt đối** với `run_inference_qwen25vl_native.py` cùng điều kiện (`temperature=0`) → xác nhận "ràng buộc cứng" bảo toàn vị trí (§7 Phần 4.3) được tuân thủ đúng.

**Kết quả full 2700 câu**, `tokens_num=16384` (mục tiêu, thực tế đạt ~76-95%/đoạn do dedup giữa DTM và salience — xem "Hạn chế" bên dưới):

| Điều kiện | Overall | Short | Medium | Long |
|---|---|---|---|---|
| Cũ — `f128_t16384` (uniform, ép độ phân giải) | 59.19% | 69.67 | 57.78 | 50.11 |
| Native dense, không cắt (§8.2, prompt cũ) | 57.85% | 68.78 | 56.11 | 48.67 |
| **DySeg + QTA + STPrune (đầy đủ)** | **60.33%** | 68.89 | 59.78 | 52.33 |

**Kiểm định thống kê (so với `f128_t16384`, mốc yêu cầu):**
```
z ≈ 0.85 (p≈0.39) — vượt số tuyệt đối (+1.14 pts) nhưng CHƯA đạt ý nghĩa thống kê ở n=2700
```
Theo bucket (cùng so với `f128_t16384`): Short *z*≈−0.36 (nhích xuống, không đáng kể), Medium *z*≈0.86 (nhích lên), Long *z*≈0.94 (nhích lên nhiều nhất). Không bucket nào riêng lẻ có ý nghĩa thống kê, nhưng **hướng nhất quán cả 3 bucket** (Medium/Long tăng, Short đứng yên/giảm nhẹ) — khớp đúng dự đoán của chính triết lý phân bổ non-uniform: đoạn/video giàu thông tin (Medium, Long — effective rank cao hơn theo phát hiện §4) hưởng lợi nhiều hơn từ việc được cấp thêm ngân sách có chọn lọc, video đơn giản (Short) không cần thêm mà cũng không mất.

**Hạn chế cần ghi nhận trung thực:**
- **Gate G1 chưa từng đạt** (57.85% ở §8.2, chưa re-verify dense/no-prune với prompt mới) — nên con số 60.33% này **không có mốc "dense chuẩn" đáng tin để tính % retention** so sánh với SCORE/FastVID (§7.6.4 kỳ vọng ≥64.6 ở 15% retention — 60.33% ở đây thấp hơn, và retention thật cũng không rõ ràng vì actual_total hụt mục tiêu).
- **Chưa ablate từng thành phần** (DySeg riêng, DTM riêng, salience riêng, cổng entropy riêng) — không biết phần nào trong 60.33% đến từ đâu.
- **actual_total hụt mục tiêu** (12507–15621 / 16384 quan sát được, ~76-95%) do vòng lặp salience bỏ qua token trùng với DTM mà không tìm ứng viên thay thế — cần sửa nếu muốn ngân sách chính xác.
- So với native-dense (§8.2, nhưng khác prompt nên không sạch 100%): *z*≈1.85 (p≈0.064) — gợi ý nhiều hơn nhưng vẫn chưa chắc chắn.

**Đối chiếu với luận điểm cốt lõi của ý tưởng (§7 Phần 0.4):** kết quả **phù hợp hướng** với luận điểm "phân bổ token (mềm) nên hoạt động tốt hơn chọn frame (cứng)" — đây là lần đầu tiên trong toàn bộ quá trình thực nghiệm một cơ chế non-uniform vượt qua uniform ở cùng ngân sách (dù chưa đạt ý nghĩa thống kê chặt), khác hẳn mọi thử nghiệm chọn-frame trước đó (§8.1: KTV oracle, CAS-QTA) đều hòa hoặc thua. Cần thêm dữ liệu (ablation từng tầng, gate G1, benchmark khác) trước khi khẳng định chắc chắn — xem `idea.md` cho lộ trình tiếp theo.


# 9. Cập nhật (2026-08-14 → 2026-08-20): sửa harness, kiểm định lại, và phát hiện bug DTM

Phần §8.3 ở trên (60.33%) được đo trên harness **chưa đóng Gate G1** (57.85% dense, thấp hơn Vanilla literature 63.8% ~6 điểm — nguyên nhân là 4 lỗi harness, không phải backbone). Mục này ghi lại việc đóng Gate G1, kiểm định lại toàn bộ ý tưởng trên harness đã sửa, và một phát hiện quan trọng làm thay đổi cách đọc mọi kết luận trước đó: **thành phần DTM trong code hiện tại không làm đúng việc nó được thiết kế để làm.**

## 9.1 Đóng Gate G1

Diff trực tiếp `lmms_eval/models/qwen2_5_vl.py` + `lmms_eval/tasks/videomme/{utils.py,videomme.yaml}` (clone từ repo FastVID chính chủ) với script tự viết, tìm ra 4 chỗ lệch: (1) sampling frame/độ phân giải — native harness để `qwen_vl_utils` tự chọn số frame (tối đa 768) thay vì ép cứng 128 frame; (2) generation config — greedy tuyệt đối, `max_new_tokens=16`, không phải `temperature=0.2`; (3) có system prompt `"You are a helpful assistant."`; (4) `post_prompt` đúng là `"\nAnswer with the option's letter from the given choices directly."`, không phải `"The best answer is:"`. Sửa cả 4 → **Native (dense, không nén) đạt 63.33%, khớp SCORE Vanilla 63.8%** (chênh 0.47 điểm, không có ý nghĩa thống kê). Gate G1 coi như đóng.

Chi tiết kỹ thuật, log, và quá trình chẩn đoán từng bước nằm trong docstring của `run_inference_qwen25vl_native.py`.

## 9.2 Bảng kết quả đầy đủ (tất cả các lần chạy trên harness đã sửa)

Hai regime token: **1x** (`MAX_PIXELS=16384×28×28`, pool dense ~14-16K token/video — gần bằng `tokens_num`, gần như không có margin nén thật) và **3x** (`MAX_PIXELS=16384×3×28×28`, pool dense ~42-48K token/video, nén thật xuống 16384 ≈ 36% retention).

| # | Cấu hình | Regime | Overall | Short | Medium | Long |
|---|---|---|---|---|---|---|
| 1 | Native (dense 100%) | 1x | 63.33% | 73.33 | 63.22 | 53.44 |
| 2 | DySeg + STPrune, `d=0.4`, `tokens_num=16384` (mặc định) | 1x | 61.78% | 71.56 | 61.00 | 52.78 |
| 3 | Native (dense 100%) | 3x | 64.26% | 73.67 | 65.00 | 54.11 |
| 4 | **Uniform** (lấy đều, không DySeg/STPrune) | 3x (→16384) | **64.41%** | 73.00 | 66.00 | 54.22 |
| 5 | DySeg + STPrune, `d=0.4` (đầy đủ) | 3x (→16384) | 61.96% | 70.00 | 61.67 | 54.22 |
| 6 | DySeg (Tầng 1+2) + salience thuần, `d=0` | 3x (→16384) | 62.15% | 69.22 | 62.11 | 55.11 |
| 7 | DySeg `d=0`, `tokens_num=b_max=999999` (ép retention≈100%) | 1x | **63.33%** | 73.33 | 63.22 | 53.44 |
| 8 | DySeg `d=0.4`, `tokens_num=b_max=999999` (ép retention≈100%) | 1x | 63.52% | 73.56 | 63.22 | 53.78 |
| 9 | `temperature0=<cực lớn>` (T→∞, Tầng 2 suy biến về đều, Tầng 3 vẫn `d=0.4`) | 3x (→16384) | 62.41% | 69.89 | 63.11 | 54.22 |
| 10 | `salience_mode=grid` (bỏ xếp hạng salience, lấy đều trong đoạn; Tầng 2 + DTM `d=0.4` vẫn bật) | 3x (→16384) | 62.41% | 70.11 | 62.67 | 54.44 |

Hàng 7 khớp hàng 1 **tuyệt đối** — không chỉ overall mà cả 3 bucket đều trùng khít đến 2 chữ số thập phân, gần như chắc chắn là cùng một tập dự đoán. Hàng 8 nhỉnh hơn hàng 1 (+0.19, không có ý nghĩa).

**Đọc bảng này:**
- Dòng 4: **nén 36% gần như miễn phí** so với dense (64.41 ≥ 64.26, z≈0.11 — không có ý nghĩa). Pool dense ~42-48K rất dư thừa.
- Dòng 7 vs dòng 1: khi ép ngân sách chắc chắn phủ hết pool (`tokens_num`/`b_max` cực lớn), cơ chế Tầng 1+2 (chia đoạn, phân bổ theo `v_s`) + gather vị trí **tái tạo đúng y hệt Native** — loại trừ bug ở khâu bảo toàn vị trí và ở khâu tính toán ngân sách khi ngân sách đủ.
- Dòng 8 vs dòng 7: DTM merge, ở retention thật sự ~100%, **không gây hại** (+0.19, không âm).
- Vậy khoản lỗ 1.55 điểm ở dòng 2 (so với dòng 1, `tokens_num=16384` mặc định, KHÔNG ép huge) phải đến từ việc một số video có pool dense **thật sự vượt 16384** (dải quan sát được ở 1x: 13,680–16,272) — với các video đó, phân bổ theo `v_s` có cắt thật một phần.
- **Dòng 9, 10 — bất ngờ quan trọng nhất:** tắt RIÊNG Tầng 2 (dòng 9) hoặc tắt RIÊNG salience-ranking (dòng 10) đều chỉ nhích lên **cùng một mức 62.41%** — cách Uniform ~2 điểm, gần như không đổi so với dòng 5 (61.96%). Cả 4 cấu hình DySeg (5, 6, 9, 10) dồn vào một dải hẹp 61.96–62.41, bất kể tắt yếu tố nào. **Điều này làm YẾU ĐI giả thuyết "DTM-trong-unit là thủ phạm chính"** ở §9.3.3: nếu đúng, dòng 6 (`d=0`, bỏ hẳn DTM) phải cải thiện rõ hơn nhiều so với các dòng còn giữ DTM — nhưng dòng 6 (62.15%) gần như không khác dòng 9, 10 (đều giữ DTM). Xem §9.3.6 để đọc lại giả thuyết cho đúng dữ liệu.

## 9.3 Đối chiếu chi tiết với FastVID gốc — phát hiện lỗi DTM

*(Toàn bộ phần này là một phân tích kỹ thuật chi tiết, đối chiếu từng dòng thiết kế của FastVID với code hiện tại, xác minh trực tiếp trong `run_inference_qwen25vl_dyseg.py`.)*

### 9.3.1 Hai loại dư thừa trong video

Token thị giác `X ∈ R^{T×N×D}` (T = unit thời gian sau gộp `temporal_patch_size=2`, N = token không gian/unit) có hai loại dư thừa:

- **Dư thừa không gian** (trong 1 unit): các patch cùng thuộc một vùng đồng nhất (bầu trời, cỏ...) giống nhau.
- **Dư thừa thời gian** (giữa các unit): một vùng tĩnh (bầu trời, tường) lặp lại gần như y hệt qua nhiều unit liên tiếp.

**Dư thừa thời gian lớn hơn nhiều dư thừa không gian.** Một phương pháp chỉ nén trong-unit giữ được ít dư thừa không gian nhưng vẫn giữ nguyên dư thừa thời gian (ví dụ: giảm 50→5 token bầu trời mỗi unit, nhưng vẫn nhân với số unit — 5×4=20 qua 4 unit). Một phương pháp nén xuyên-unit giảm dư thừa thời gian trực tiếp (còn 5 tổng cộng cho cả 4 unit).

### 9.3.2 Thiết kế DTM của FastVID — anchor frame + gán xuyên đoạn

FastVID §3.3: DTM chỉ chạy thuật toán ρ×δ (density-peaks) trên các **anchor unit** — cách nhau `p_anchor` bước (mặc định 4), KHÔNG chạy trên mọi unit. Sau khi có anchor, **mọi token của TẤT CẢ unit trong đoạn** (kể cả các unit không phải anchor) được gán về anchor gần nhất theo cosine similarity — bất kể anchor đó nằm ở unit nào. Đây chính là bước loại bỏ dư thừa thời gian: bầu trời của unit 1, 2, 3 được gán chung về đúng 1 anchor bầu trời ở unit 0 (anchor frame), thay vì mỗi unit giữ token bầu trời riêng.

Vì sao chỉ lấy anchor từ một số unit thay vì toàn đoạn: nếu để anchor tự do chọn từ mọi unit, các phần khác nhau của cùng một vật thể có thể rơi vào các unit khác nhau về mặt "làm đại diện", phá vỡ tính nhất quán không gian tại một thời điểm. Cố định anchor theo từng unit mốc (`p_anchor`) giữ mỗi "ảnh chụp nhanh" nhất quán, đồng thời vẫn cho phép gán xuyên đoạn.

Bằng chứng giá trị của cơ chế này — FastVID Table 5 (LLaVA-Video, retention 10%): merge uniform per-frame (không xuyên frame) đạt 91.5% avg-acc, merge cluster-based đạt 91.8%, **DTM xuyên frame đạt 94.7%** — chênh **3.2 điểm** so với xử lý từng frame độc lập.

### 9.3.3 Code hiện tại — xác minh trực tiếp DTM chỉ chạy trong-unit

`stprune_segment()`, `run_inference_qwen25vl_dyseg.py`:

```python
for t in anchor_units:
    toks = X[t]                                    # CHỈ N token của unit t
    rho, delta = density_peaks(toks)                # density-peaks TRONG unit t
    ...
    toks_unit = normalize(toks, ...)
    sim = toks_unit @ toks_unit[anchor_local_idx].T # so sánh CHỈ trong unit t
    assign = sim.argmax(dim=1)                      # gán CHỈ N token của unit t
```

`toks = X[t]` — chỉ N token của riêng unit `t`. `sim` so sánh token của unit `t` với chính anchor của unit `t`. Không có dòng nào đưa `X[t+1], X[t+2], ...` (các unit không phải anchor giữa hai mốc `p_anchor`) vào phép gán này. **Các unit không phải anchor hoàn toàn không được DTM xử lý** — token của chúng chỉ có cơ hội sống sót qua nhánh salience (xếp hạng riêng, độc lập), không hề được "hấp thụ" vào bất kỳ anchor nào.

Đây khớp với mô tả trong `idea.md` Phần 4.2 chỉ ở phần "anchor lấy theo `p_anchor`", nhưng KHÔNG khớp phần "gán mỗi token còn lại **trong đoạn** về anchor gần nhất" — code hiện tại chỉ gán trong unit, không phải trong đoạn. Đây là một lỗi cài đặt cụ thể, xác minh được trực tiếp, không phải khác biệt cách diễn giải.

**Hệ quả:** DTM trong code hiện tại về bản chất là nén ảnh áp dụng độc lập lên từng unit — chỉ xử lý dư thừa không gian, hoàn toàn bỏ qua dư thừa thời gian (theo §9.3.1, đây là phần dư thừa lớn hơn). Khoảng chênh 3.2 điểm mà FastVID đo được cho đúng cơ chế bị thiếu này (§9.3.2) và khoản lỗ 2.45 điểm quan sát được ở §9.2 (dòng 5 vs dòng 4) nằm cùng độ lớn — không chắc chắn là cùng nguyên nhân, nhưng đủ gần để là nghi phạm hàng đầu.

**Vì sao ablation `d=0` (bỏ DTM, §9.2 dòng 6) không cứu được:** `d=0` tắt hẳn DTM, nhưng DTM vốn đã không làm việc nó được thiết kế để làm (không loại bỏ dư thừa thời gian) — tắt một cơ chế không hoạt động đúng thì không thay đổi nhiều là hợp lý. Kết luận trước đó "đã loại trừ giả thuyết DTM merge làm hỏng token" **cần rút lại một phần**: đã loại trừ đúng giả thuyết hẹp (trộn giá trị ở retention gần 100% không gây hại — xác nhận lại ở §9.2 dòng 7-8), nhưng CHƯA loại trừ được liệu một DTM cài đúng (xuyên đoạn) có giúp gì không, vì DTM hiện tại chưa từng được test đúng thiết kế.

### 9.3.4 Salience đo chuyển động, không đo ngữ nghĩa — vẫn là nghi phạm riêng biệt

FastVID dùng attention của token `[CLS]` (ATS) — đo **ngữ nghĩa**: vùng nào encoder cho là quan trọng, bất kể tĩnh hay động. Code hiện tại dùng `sal(t,i) = ‖X[t,i] − X[t-1,i]‖` — đo **chuyển động**: vùng nào vừa thay đổi. Hai đại lượng khác nhau và nhiều tình huống trái ngược nhau: chữ trên slide/mặt người đang nói (tĩnh nhưng quan trọng) → ATS cao, `‖ΔX‖` ≈ 0 → bị vứt. Nền rung do camera (động nhưng không quan trọng) → ATS thấp, `‖ΔX‖` cao → được giữ.

Khớp đúng với mẫu hình bucket đã quan sát (§9.2 dòng 4-6): lỗ tập trung hết ở Short/Medium (−3.0 đến −4.3 điểm), Long không mất (0 đến +0.9). Video ngắn trên VideoMME nhiều nội dung tĩnh-nhưng-quan-trọng (slide, biển hiệu, người ngồi nói) — `‖ΔX‖` vứt sạch loại nội dung này.

SCORE (paper gốc dùng ý tưởng residual) ghép `[X[t]; ΔX[t]]` làm input cho một policy **học được** — X mang ngữ nghĩa, ΔX mang động lực, cả hai cùng vào. Code hiện tại bỏ X, chỉ giữ ΔX làm tiêu chí chọn trực tiếp (không qua policy học) — đây là một thay thế thiếu một nửa tín hiệu, không tương đương.

### 9.3.5 Tầng 2 (đóng góp chính, phân bổ theo `v_s`) — chưa được test công bằng

Ý tưởng cấp ngân sách không đều theo đoạn (thay vì đều theo độ dài đoạn như FastVID) là hợp lý và trùng với chính phê phán của KiToke §C.1.3 dành cho FastVID (không phân biệt đoạn nhiều/ít thông tin). Nhưng test hiện tại (§9.2 dòng 5-6) đo Tầng 2 **cộng gộp với** một Tầng 3 có 2 vấn đề đã nêu (§9.3.3, §9.3.4) — không tách bạch được phần lỗ đến từ đâu. Ngoài ra, ở 36% retention Uniform đã gần chạm trần dense (§9.2 dòng 4) — không còn nhiều dư địa để bất kỳ cơ chế nào thắng; FastVID/SCORE thường report ở vùng retention thấp hơn nhiều (10-25%).

### 9.3.6 Cập nhật sau ablation đơn-yếu-tố (dòng 9, 10): giả thuyết DTM-per-unit bị yếu đi

Đã chạy 2 ablation đơn-yếu-tố tách biệt: tắt riêng Tầng 2 (dòng 9, `T→∞`) và tắt riêng salience-ranking (dòng 10, `salience_mode=grid`) — cả hai đều **giữ nguyên DTM `d=0.4`** (vẫn per-unit, theo §9.3.3). Kết quả: cả hai đều dừng ở đúng **62.41%**, gần như không khác dòng 5 (61.96%, đầy đủ) hay dòng 6 (62.15%, `d=0` bỏ hẳn DTM).

**Điều này ngược với kỳ vọng.** Nếu DTM-per-unit thật sự là thủ phạm chính, dòng 6 (DTM tắt hẳn) phải cải thiện rõ rệt so với dòng 5, 9, 10 (DTM vẫn bật). Thực tế cả 4 dòng dồn vào dải hẹp 61.96–62.41 — **tắt DTM hoàn toàn cũng không giúp gì hơn tắt riêng Tầng 2 hoặc riêng salience-ranking.** Ba giả thuyết đơn-yếu-tố (Tầng 2 lệch, DTM per-unit, salience sai loại tín hiệu) đều bị làm yếu đi bởi cùng một dữ liệu này — không cái nào, khi tắt riêng lẻ, giải thích được phần lớn khoảng cách 2 điểm với Uniform.

**Cách đọc khả dĩ hơn:** có thể không phải MỘT thành phần cụ thể là thủ phạm, mà là đặc điểm chung của việc **chọn token cục bộ trong ranh giới đoạn/unit** (dù chọn theo DTM, theo salience, hay theo grid-trong-đoạn) — khác với Uniform vốn lấy phẳng trên toàn bộ pool T×N, không bị ràng buộc bởi bất kỳ ranh giới nào. Cần một test loại bỏ ĐỒNG THỜI cả 3 yếu tố để kiểm tra giả thuyết này.

**Test quyết định, chưa chạy:** `dtm_ratio=0 temperature0=<cực lớn> salience_mode=grid` — bỏ hẳn DTM, Tầng 2 suy biến gần-đều, salience-ranking thay bằng lấy đều trong đoạn. Đây là cấu hình gần nhất có thể có với "chỉ còn Tầng 1 (chia đoạn) + lấy đều trong mỗi đoạn", khác Uniform đúng một điểm: vẫn tôn trọng ranh giới đoạn khi lấy mẫu, Uniform thì không. Nếu cấu hình này **vẫn** thua Uniform ~2 điểm → xác nhận bản thân việc tôn trọng ranh giới đoạn/unit lúc chọn token là điều gây hại, một phát hiện gốc rễ hơn nhiều so với sửa riêng DTM. Nếu nó **khớp lại gần 64.4** → ba yếu tố cộng dồn (không phải một yếu tố đơn lẻ) mới là lời giải thích đúng, và cần sửa cả DTM lẫn Tầng 2 lẫn salience cùng lúc, không chỉ một trong số đó.

### 9.3.7 Kiểm tra ở mức nén sâu hơn (~7.2%, khớp absolute token count của FastVID)

Mọi test ở §9.2/9.3.6 đều ở `tokens_num=16384` (36% retention) — vùng mà chính Uniform đã gần chạm trần dense (§9.3.5), không còn nhiều dư địa cho bất kỳ cơ chế nào thể hiện. Chạy thêm ở `tokens_num=3240` (~7.2% retention, khớp đúng số token tuyệt đối 3240.7 mà FastVID paper báo cáo) — vùng nén sâu, nơi cả Uniform lẫn dense đều phải trả giá thật:

| Cấu hình | Regime | Overall | Short | Medium | Long |
|---|---|---|---|---|---|
| Uniform | 3x (→3240, ~7.2%) | 60.30% | 68.11 | 61.44 | 51.33 |
| DySeg `d=0.4` (đầy đủ) | 3x (→3240, ~7.2%) | 58.70% | 65.44 | 58.56 | 52.11 |

**DySeg vẫn thua Uniform (−1.60 điểm, z≈−1.20, không có ý nghĩa thống kê) ngay cả ở vùng nén sâu này** — nơi lẽ ra phân bổ không đều phải có nhiều cơ hội thể hiện giá trị nhất (càng ít token, sai chỗ phân bổ càng đắt). Điều này không ủng hộ giả thuyết "chỉ cần nén sâu hơn thì DySeg mới thắng". Đáng chú ý: cả hai đều tụt mạnh so với 16384 token (Uniform: 64.41→60.30, −4.11; DySeg: 61.96→58.70, −3.26) — xác nhận 3240 token mới thật sự là vùng "nén có giá", khác hẳn 16384 (vùng "nén miễn phí", §9.2).

**Mẫu hình bucket lặp lại lần thứ ba, nhất quán:** Short −2.67, Medium −2.88, **Long +0.78** (DySeg vẫn nhỉnh hơn Uniform ở bucket Long, dù không có ý nghĩa thống kê). Đây là lần thứ 3 (sau §9.2 dòng 5, 6) mẫu hình này lặp lại y hệt — Short/Medium luôn mất, Long luôn hòa hoặc nhỉnh — ở cả hai mức retention rất khác nhau (36% và 7.2%). Độ lặp lại này, dù từng lần riêng lẻ không đạt ý nghĩa thống kê, tự nó là một tín hiệu đáng tin hơn một lần đo đơn lẻ.

So với FastVID paper (93.3% giữ so với chính Vanilla của họ ở retention tương đương absolute token count): DySeg của mình chỉ giữ 58.70/64.26 = **91.4%** — thấp hơn FastVID publish ở mức nén sâu, trong khi ở 16384 (36%) lại NHỈNH HƠN (96.4-96.7% > 93.3%, §9.1 cũ). Gợi ý: khoảng cách với FastVID publish nới rộng khi nén càng sâu — khớp hướng với giả thuyết DTM/salience (vốn chỉ hoạt động đúng thiết kế ở FastVID, §9.3.3-9.3.4) ngày càng quan trọng khi ngân sách càng eo hẹp.

## 9.4 Kế hoạch thực nghiệm ưu tiên

| # | Việc | Công sức ước tính | Trả lời câu hỏi gì |
|---|---|---|---|
| 1 | **Sửa DTM: gán token xuyên cả đoạn** (không chỉ trong unit) theo đúng §9.3.2/idea.md Phần 4.2 | ~1 ngày | Đây là việc ưu tiên cao nhất — nếu đúng nghi ngờ §9.3.3, có thể cải thiện 2-3 điểm |
| 2 | Salience → `‖X[t,i]‖` (proxy ngữ nghĩa thô) hoặc `‖X‖+β‖ΔX‖`; đã có sẵn `salience_mode=grid`/`norm` để test nhanh | đã cài, chỉ cần chạy | Xác nhận giả thuyết §9.3.4 |
| 3 | ~~Unit test retention≈100% ở 1x~~ | **Đã làm (§9.2 dòng 7-8)** | Đã xác nhận: không phải bug ở vị trí/ngân sách, không phải DTM-merge-tại-100%-retention |
| 4 | Chạy FastVID gốc (đã có sẵn code, `../FastVID/fastvid_qwen25vl`) trong đúng harness này ở 25%/10% retention | ~1 ngày | Có baseline THẬT (không phải số từ paper khác backbone/harness) để so trực tiếp |
| 5 | `temperature0=<số cực lớn>` (T→∞, Tầng 2 suy biến về đều, Tầng 3 vẫn bật) | 1 run (đã có config, đang chờ chạy) | Tách bạch Tầng 2 vs Tầng 3 — hiện đang lẫn |
| 6 | Chuyển sang retention 10-15% (giảm `tokens_num` hoặc tăng `MAX_PIXELS` hơn nữa) | — | Vùng Uniform chưa bão hòa, nơi phân bổ không đều mới thật sự có đất diễn |

Bảng ablation mục tiêu sau khi làm xong việc 1, 4, 5, 6 (mỗi dòng chỉ khác đúng 1 thành phần so với dòng trên):

| Method | 25% | 15% | 10% |
|---|---|---|---|
| Uniform | ? | ? | ? |
| FastVID gốc (reproduce, harness này) | ? | ? | ? |
| FastVID + Tầng 2 (`v_s`) thay ngân sách đều | ? | ? | ? |

Dòng 3 khác dòng 2 đúng một chỗ (cơ chế phân bổ ngân sách) — đây mới là phép đo trực tiếp cho câu hỏi "phân bổ không đều có giá trị không", tách hoàn toàn khỏi vấn đề DTM/salience đang lẫn ở các thí nghiệm §9.2.

## 9.5 Kết luận hiện tại — cập nhật, thay thế một phần kết luận trước đó

**Kết luận vẫn đứng vững:** nén xuống 36% gần như miễn phí so với dense (Uniform 64.41 ≥ Native 64.26, §9.2 dòng 3-4) — so sánh sạch, cùng harness.

**Kết luận cần rút lại/làm rõ hơn:** "DySeg đầy đủ thua Uniform ~2.45 điểm nên phân bổ không đều không hiệu quả" — **kết luận này hiện đo được ba thứ cộng gộp, không phải một**: (a) phân bổ không đều theo `v_s` (Tầng 2, đóng góp thật của ý tưởng), (b) DTM chỉ xử lý dư thừa không gian trong-unit thay vì xuyên đoạn (xác nhận là lỗi cài đặt, §9.3.3), (c) salience đo chuyển động thay vì ngữ nghĩa (rất có thể là thay thế không tương đương, §9.3.4). Số liệu hiện tại chỉ cho biết tổ hợp (a)+(b)+(c) không hiệu quả — **chưa đủ cơ sở để kết luận riêng (a) sai**, vì (a) chưa từng được test trên một Tầng 3 làm đúng thiết kế.

**Việc cần làm trước khi viết kết luận cuối cùng cho ý tưởng:** hoàn thành mục 1, 4, 5 ở §9.4. Nếu sau khi sửa DTM đúng thiết kế (xuyên đoạn) và tách bạch Tầng 2/Tầng 3 mà phân bổ không đều **vẫn** không thắng nổi FastVID gốc + ngân sách đều — lúc đó mới có đủ cơ sở kết luận âm tính, và đó vẫn là một kết quả có giá trị công bố (KiToke §C.1.3 phê phán FastVID không phân biệt đoạn nhiều/ít thông tin nhưng chưa ai công bố thực nghiệm cho thấy phân biệt được cũng không giúp gì).
