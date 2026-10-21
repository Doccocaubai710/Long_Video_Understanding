"""
holitom_port.py -- HoliTom (NeurIPS'25), ban "w/o M" (chi nen TRUOC LLM), cho Qwen2.5-VL trong cau hinh danh gia cua du an.

Nguon: Long_Video_Understanding/HoliTom (commit e9b2972), holitom/llava_arch.py. Repo goc chi ho tro
LLaVA-OneVision / LLaVA-Video nen phai port:
  - CHEP NGUYEN VAN (chi bo `self`): cluster_dpc_knn, select_static_windows, merge_tokens_by_clustering,
    merge_tokens_by_attention_density, merge_tokens_by_density, get_static_dynamic_features, va khoi dieu phoi
    trong prepare_inputs_labels_for_multimodal (dong 375-407).
  - VIET LAI: (1) attention cua vision tower. Ban goc: attention lop cuoi SigLIP, trung binh theo head va theo
    query, roi get_2dPool ve luoi token. O day: attention block cuoi cua ViT Qwen2.5-VL (full attention trong
    tung unit thoi gian), softmax theo tung query, trung binh theo head + query, roi trung binh 4 patch cua moi
    nhom merge 2x2. (2) ghep token: ban goc xep [static, dynamic] theo tung segment (LLaVA khong co vi tri thoi
    gian rieng); o day moi token giu chi so (unit, vi tri) goc de lay MRoPE: token static (da gop trung binh ca
    cua so) dat o unit DAU cua so -- giong nhanh mm_newline_position="grid" cua ban goc.
Sieu tham so = scripts/eval_*_holitom.sh: T(tau)=0.80 (0.65 o muc 10%), BETA=0.6, D=0, K=7, MAX_WINDOW_SIZE=1024.
"""
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
from transformers.vision_utils import get_vision_window_index


def holitom_params(args):
    r = float(getattr(args, "retention", 0.15))
    tau_default = 0.65 if r <= 0.10 + 1e-9 else 0.80
    return SimpleNamespace(
        retain_ratio=r, tau=float(getattr(args, "holitom_tau", tau_default)),
        Beta=float(getattr(args, "holitom_beta", 0.6)), D=float(getattr(args, "holitom_D", 0.0)),
        K=int(getattr(args, "holitom_K", 7)), max_window_size=int(getattr(args, "holitom_max_window", 1024)),
    )


@torch.no_grad()
def holitom_frame_attn(llm, pixel_values_videos, video_grid_thw, chunk=4):
    """Tra ve (video_embeds [T*N, D], attn [T, N]): attn = attention ma moi token nhan duoc o block cuoi vision tower."""
    visual = llm.visual
    captured = {}

    def hook(module, args, kwargs):
        captured["hidden_states"] = args[0] if args else kwargs["hidden_states"]
        captured["position_embeddings"] = kwargs["position_embeddings"]

    handle = visual.blocks[-1].attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        feat = llm.get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw)
    finally:
        handle.remove()
    video_embeds = feat.pooler_output
    if isinstance(video_embeds, (list, tuple)):
        video_embeds = torch.cat(video_embeds, dim=0)

    attn = visual.blocks[-1].attn
    hidden_states = captured["hidden_states"]
    cos, sin = captured["position_embeddings"]
    L = hidden_states.shape[0]
    q, k, _ = attn.qkv(hidden_states).reshape(L, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    T, gh, gw = int(video_grid_thw[0, 0]), int(video_grid_thw[0, 1]), int(video_grid_thw[0, 2])
    P = gh * gw
    q = q.transpose(0, 1).reshape(attn.num_heads, T, P, attn.head_dim)
    k = k.transpose(0, 1).reshape(attn.num_heads, T, P, attn.head_dim)
    out = []
    for s in range(0, T, chunk):  # [heads, chunk, P, P]
        w = torch.matmul(q[:, s:s + chunk], k[:, s:s + chunk].transpose(-2, -1)) / math.sqrt(attn.head_dim)
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32)
        out.append(w.mean(dim=0).mean(dim=1))  # trung binh theo head, roi theo query -> [chunk, P]
    attn_weights = torch.cat(out, dim=0).view(T, P // 4, 4).mean(-1)  # 4 patch cua 1 nhom merge 2x2 -> [T, N]

    window_index, _ = get_vision_window_index(
        video_grid_thw, spatial_merge_size=visual.spatial_merge_size, window_size=visual.window_size,
        patch_size=visual.patch_size, kwargs={},
    )
    reverse_indices = torch.argsort(window_index.to(attn_weights.device))
    attn_weights = attn_weights.reshape(-1)[reverse_indices].reshape(T, -1)
    return video_embeds, attn_weights


# ====== BAT DAU khoi chep nguyen van tu holitom/llava_arch.py (chi bo `self`) ======
def cluster_dpc_knn(x, cluster_num, k=7):
    with torch.no_grad():
        batch_size, seq_len, embed_dim = x.shape

        dist_matrix = torch.cdist(x.float(), x.float()) / (embed_dim ** 0.5)    # (batch_size, seq_len, seq_len)

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k, dim=-1, largest=False) # (batch_size, seq_len, k)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp() # (batch_size, seq_len)
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        # get distance indicator
        mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select the cluster center according to the score
        score = dist * density
        _, index_center = score.topk(cluster_num, dim=-1)

        return index_center, dist_matrix

def select_static_windows(feature_sim, batch_size, tau, max_window_size):
    # pruned_static_count[s,e]
    def get_pruned_static_count_vectorized(feature_sim, batch_size, tau):
        similarity_matrix = torch.ones((batch_size, batch_size, feature_sim.shape[1]), device=feature_sim.device)

        for start in range(batch_size-1):
            cum_similarity = torch.cumprod(feature_sim[start:] > tau, dim=0)
            similarity_matrix[start, start+1:start+1+len(cum_similarity)] = cum_similarity

        window_lengths = torch.arange(batch_size, device=feature_sim.device).unsqueeze(0) - \
                       torch.arange(batch_size, device=feature_sim.device).unsqueeze(1)
        window_lengths = window_lengths.clamp(min=0)

        pruned_static_count = (similarity_matrix.sum(dim=-1) * window_lengths).float()
        return pruned_static_count

    pruned_static_count = get_pruned_static_count_vectorized(feature_sim, batch_size, tau)

    dp = torch.zeros(batch_size, device=pruned_static_count.device)
    prev = torch.zeros(batch_size, dtype=torch.long, device=pruned_static_count.device)
    # [prev[i], i]

    for i in range(batch_size):
        max_val = dp[i-1] if i > 0 else 0
        best_j = i

        for window_size in range(2, min(i+1, max_window_size) + 1):
            j = i - window_size
            current_val = (dp[j] if j>=0 else 0) + pruned_static_count[j+1, i]  # [-, j] + [j+1, i]
            if current_val > max_val:
                max_val = current_val
                best_j = j+1

        dp[i] = max_val
        prev[i] = best_j    # [best_j, i]

    selected_frames = []
    i = batch_size - 1
    while i >= 0:
        selected_frames.append((prev[i].item(), i))
        i = prev[i].item() - 1

    selected_frames = selected_frames[::-1]
    total_reduced = dp[-1].item()

    return selected_frames, total_reduced

def merge_tokens_by_clustering(feat, target_indices, dist_matrix, cluster_num, Beta):
    batch_size, seq_len, embed_dim = feat.shape
    all_indices = torch.arange(seq_len, device=feat.device)
    all_indices = all_indices.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
    non_target_indices = torch.zeros((batch_size, seq_len-cluster_num), dtype=torch.long, device=feat.device)
    for b in range(batch_size):
        non_target_mask = ~torch.isin(all_indices[b], target_indices[b])
        non_target_indices[b] = all_indices[b][non_target_mask]
    # non_target_indices (batch_size, seq_len-cluster_num)

    non_target_feat = torch.gather(
        feat,
        dim=1,
        index=non_target_indices.unsqueeze(-1).expand(-1, -1, feat.size(-1))
    )   # (batch_size, seq_len-cluster_num, embed_dim)

    dist_matrix = torch.gather(
        dist_matrix,
        dim=1,
        index=non_target_indices.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))
    )   # (batch_size, seq_len-cluster_num, seq_len)
    dist_matrix = torch.gather(
        dist_matrix,
        dim=2,
        index=target_indices.unsqueeze(1).expand(-1, dist_matrix.size(1), -1)
    )   # (batch_size, seq_len-cluster_num, cluster_num)

    idx_cluster = torch.argmin(dist_matrix, dim=-1) # (batch_size, seq_len-cluster_num)

    cluster_tokens = []
    for b in range(batch_size):
        batch_tokens = []
        for i in range(cluster_num):
            mask = (idx_cluster[b] == i)
            if mask.any():
                cluster_features = non_target_feat[b][mask]
                import os
                if os.environ.get("NO_BETA", "0") == "0":
                    # rank0_print("USE_BETA")
                    cluster_means = cluster_features.mean(dim=0)
                    batch_tokens.append(Beta * feat[b][target_indices[b][i]] + (1 - Beta) * cluster_means)
                else:
                    # rank0_print("NO_BETA")
                    all_features = torch.cat([feat[b][target_indices[b][i]].unsqueeze(0), cluster_features], dim=0)
                    batch_tokens.append(all_features.mean(dim=0))
            else:
                batch_tokens.append(feat[b][target_indices[b][i]])
        cluster_tokens.append(torch.stack(batch_tokens))
    cluster_tokens = torch.stack(cluster_tokens)  # shape: (batch_size, cluster_num, embed_dim)

    return cluster_tokens

def merge_tokens_by_attention_density(feat, attn, pos, retain_ratio, D, Beta, K):
    batch_size, seq_len, embed_dim = feat.shape
    dominant_num = round(math.ceil(seq_len * retain_ratio) * (1-D))
    contextual_num = math.ceil(seq_len * retain_ratio) - dominant_num

    ## Dominant Visual Tokens
    if dominant_num > 0:
        all_indices = attn.topk(dominant_num, dim=1).indices
        mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device).scatter_(1, all_indices, False)  # (batch_size, seq_len) False means retained tokens
        # finally, (batch_size, dominant_num, embed_dim) compare with feat
        dominant_tokens = feat.masked_select(~mask.unsqueeze(-1)).view(batch_size, dominant_num, embed_dim)
        dominant_pos = pos.masked_select(~mask).view(batch_size, dominant_num)
    else:
        mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device)
        dominant_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        dominant_pos = torch.empty((batch_size, 0), device=feat.device)

    ## Contextual Visual Tokens
    if contextual_num > 0:
        ### Filter
        # feat_filtered: (batch_size, seq_len-dominant_num, embed_dim)
        feat_filtered = feat.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num, embed_dim)
        contextual_pos = pos.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num)
        target_indices, dist_matrix = cluster_dpc_knn(feat_filtered, contextual_num, k=min(K,contextual_num))
        target_indices = torch.sort(target_indices, dim=-1)[0]
        contextual_pos = torch.stack([contextual_pos[b][target_indices[b]] for b in range(batch_size)]) # (batch_size, contextual_num)
        # target_indices (batch_size, contextual_num)
        # dist_matrix (batch_size, seq_len-dominant_num, seq_len-dominant_num)
        # assign tokens to the nearest center

        contextual_tokens = merge_tokens_by_clustering(feat_filtered, target_indices, dist_matrix, contextual_num, Beta)
    else:
        contextual_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        contextual_pos = torch.empty((batch_size, 0), device=feat.device)

    image_feat = []
    image_pos = []
    for b in range(batch_size):
        batch_tokens = torch.cat([dominant_tokens[b], contextual_tokens[b]], dim=0)
        batch_pos = torch.cat([dominant_pos[b], contextual_pos[b]], dim=0)
        image_feat.append(batch_tokens)
        image_pos.append(batch_pos)
    image_feat = torch.stack(image_feat)  # shape: (batch_size, dominant_num + contextual_num, embed_dim)
    image_pos = torch.stack(image_pos)

    return image_feat, image_pos

def merge_tokens_by_density(feat, pos, retain_ratio, Beta, K):
    batch_size, seq_len, embed_dim = feat.shape
    cluster_num = round(seq_len * retain_ratio)
    if cluster_num > 0:
        target_indices, dist_matrix = cluster_dpc_knn(feat, cluster_num, k=min(K,cluster_num))
        target_indices = torch.sort(target_indices, dim=-1)[0]
        image_pos = torch.stack([pos[b][target_indices[b]] for b in range(batch_size)])

        cluster_tokens = merge_tokens_by_clustering(feat, target_indices, dist_matrix, cluster_num, Beta)
        image_feat = cluster_tokens
    else:
        image_feat = torch.empty((batch_size, 0, embed_dim), device=feat.device)
        image_pos = torch.empty((batch_size, 0), device=feat.device)

    return image_feat, image_pos

def get_static_dynamic_features(image_feat, attn_weights, selected_frames, feature_sim, tau):
    # attn_weights: (batch_size, seq_len)
    batch_size, seq_len, embed_dim = image_feat.shape
    static_feat_list, dynamic_feat_list, _, dynamic_attn_list  = [], [], [], []
    static_pos_list, dynamic_pos_list = [], []
    for start,end in selected_frames:
        all_indices = torch.arange(seq_len, device=image_feat.device).unsqueeze(0)   # (1, seq_len)
        if start == end:
            static_feat_list.append(torch.empty((0, embed_dim), device=image_feat.device))
            # static_attn_list.append(torch.empty((0,), device=attn_weights.device))
            dynamic_feat_list.append(image_feat[start:end+1])
            dynamic_attn_list.append(attn_weights[start:end+1])

            static_pos_list.append(torch.empty((0, seq_len), device=image_feat.device))
            dynamic_pos_list.append(all_indices)
        else:
            windows_size = end-start+1
            mask = torch.all(
                feature_sim[start:end, :] > tau, dim=0
            )
            static_feat = image_feat[start:end+1, mask]
            # static_attn = attn_weights[start:end+1, mask]
            dynamic_feat = image_feat[start:end+1, ~mask]
            dynamic_attn = attn_weights[start:end+1, ~mask]

            static_feat_list.append(static_feat.mean(dim=0))
            # static_attn_list.append(static_attn.mean(dim=0))
            dynamic_feat_list.append(dynamic_feat)
            dynamic_attn_list.append(dynamic_attn)

            static_pos_list.append(all_indices[:, mask].expand(1,-1))
            dynamic_pos_list.append(all_indices[:, ~mask].expand(windows_size,-1))

    return static_feat_list, dynamic_feat_list, _, dynamic_attn_list, static_pos_list, dynamic_pos_list
# ====== KET THUC khoi chep nguyen van ======


@torch.no_grad()
def holitom_select(video_embeds, frame_attn_weights, video_grid_thw, p):
    """video_embeds [T*N, D] -> (kept_indices tang dan [K], kept_embeds [K, D], diag)."""
    T = int(video_grid_thw[0, 0])
    out_dtype = video_embeds.dtype
    pooled_image_feat = video_embeds.reshape(T, -1, video_embeds.shape[-1])
    attn_weights = frame_attn_weights.to(pooled_image_feat.device)
    retain_ratio, tau, Beta, D, K, max_window_size = p.retain_ratio, p.tau, p.Beta, p.D, p.K, p.max_window_size

    # --- dong 375-396 cua ban goc ---
    batch_size, seq_len, embed_dim = pooled_image_feat.shape

    pooled_image_feat_normed = torch.nn.functional.normalize(pooled_image_feat, p=2, dim=-1)
    feature_sim = torch.nn.functional.cosine_similarity(
        pooled_image_feat_normed[:-1], pooled_image_feat_normed[1:], dim=-1
    )   # (batch_size-1, seq_len')

    selected_frames, total_reduced = select_static_windows(
        feature_sim, batch_size, tau, max_window_size
    )

    total_tokens = batch_size * seq_len
    retain_ratio = min(retain_ratio /
                       ((total_tokens-total_reduced)/total_tokens), 1)

    static_feat, dynamic_feat, _, dynamic_attn, static_pos, dynamic_pos = get_static_dynamic_features(
        pooled_image_feat, attn_weights, selected_frames,
        feature_sim, tau
    )

    # --- dong 398-407 + ham holitom() cua ban goc; khac: giu chi so (unit, vi tri) thay vi noi theo segment ---
    feats, idxs = [], []
    n_static = 0
    for idx, (start, end) in enumerate(selected_frames):
        window_size = end - start + 1
        d_feat, d_pos = merge_tokens_by_attention_density(
            dynamic_feat[idx], dynamic_attn[idx], dynamic_pos[idx], retain_ratio, D, Beta, K)
        unit = torch.arange(start, end + 1, device=d_pos.device).unsqueeze(1)
        feats.append(d_feat.flatten(0, 1).float())
        idxs.append((unit * seq_len + d_pos.long()).flatten())
        if window_size > 1:
            s_feat, s_pos = merge_tokens_by_density(static_feat[idx].unsqueeze(0), static_pos[idx], retain_ratio, Beta, K)
            feats.append(s_feat.flatten(0, 1).float())
            idxs.append((start * seq_len + s_pos.long()).flatten())  # token static dat o unit dau cua so
            n_static += int(s_pos.numel())
    kept_embeds = torch.cat(feats, dim=0)
    kept_indices = torch.cat(idxs, dim=0)
    assert kept_indices.unique().numel() == kept_indices.numel(), "holitom: trung chi so token"
    kept_indices, order = torch.sort(kept_indices)
    kept_embeds = kept_embeds[order].to(out_dtype)
    diag = {
        "mode": "holitom", "num_segments": len(selected_frames), "tau": tau,
        "static_reduced": float(total_reduced), "retain_ratio_after_static": float(retain_ratio),
        "n_static_kept": n_static, "max_window": max(e - s + 1 for s, e in selected_frames),
    }
    return kept_indices, kept_embeds, diag
