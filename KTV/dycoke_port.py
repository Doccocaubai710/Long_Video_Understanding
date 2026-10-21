"""
dycoke_port.py -- DyCoke, CHI giai doan TTM (Temporal Token Merging, nen truoc LLM), cho Qwen2.5-VL.

Nguon: Long_Video_Understanding/DyCoke (commit dd74634), llava/model/llava_arch.py::dycole_ttm (dong 35-91).
Repo goc chi ho tro LLaVA-OneVision. Ham duoi day chep logic cua dycole_ttm, them 1 danh sach chi so chay song song
voi danh sach feature de biet token nao duoc giu (ban goc chi tra feature). Khac ban goc: so sanh cosine tinh
bang float32; va video co <= 4 unit thi bo qua vong "odd" (ban goc se loi torch.stack([])).
KHONG chay giai doan 2 cua DyCoke (cat KV cache dong o layer 3 trong LLM): no dua vao attention luc sinh cau tra loi
nen phu thuoc cau hoi, va khong cung dinh nghia ngan sach token voi Uniform.
TTM: moi nhom 4 unit giu 1 unit day du + 3 unit con (1-k) token -> ti le giu ~ (1 + 3(1-k))/4, thap nhat ~25% (k=1).
"""
import torch


def k_for_retention(r):
    """dycoke_k sao cho ti le giu ~ r (r >= 0.25)."""
    return min(1.0, max(0.0, 1.0 - (4.0 * r - 1.0) / 3.0))


@torch.no_grad()
def dycoke_ttm_select(video_embeds, video_grid_thw, merging_ratio):
    num_frames = int(video_grid_thw[0, 0])
    num_tokens_per_frame = video_embeds.shape[0] // num_frames
    image_feature = video_embeds.float()
    all_idx = torch.arange(video_embeds.shape[0], device=video_embeds.device)
    merging_ratio = 1 - merging_ratio
    # Calculate similarities between adjacent even frames
    similarities = []
    for i in range(0, num_frames - 1, 2):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]
        frame1_norm = torch.nn.functional.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2_tokens, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)
    similarities = torch.stack(similarities)

    # Process even frames
    modified_idx = []
    for i in range(0, num_frames - 1, 2):
        avg_similarity = similarities[i // 2]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        modified_idx.append(all_idx[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])
        modified_idx.append(all_idx[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep])

    # Process odd frames
    odd_similarities = []
    for i in range(0, num_frames - 4, 4):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame: (i + 3) * num_tokens_per_frame]
        similarity = torch.nn.functional.cosine_similarity(frame1_tokens, frame2_tokens, dim=1)
        odd_similarities.append(similarity)
    for i in range(0, num_frames - 4, 4):
        avg_similarity = odd_similarities[i // 4]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        modified_idx[i] = all_idx[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        modified_idx[i + 2] = all_idx[(i + 2) * num_tokens_per_frame: (i + 3) * num_tokens_per_frame][tokens_to_keep]

    keep = torch.sort(torch.cat(modified_idx, dim=0)).values
    return keep, video_embeds[keep], {"mode": "dycoke_ttm", "dycoke_k": 1 - merging_ratio,
                                      "frames_used": 2 * len(similarities), "num_frames": num_frames}
