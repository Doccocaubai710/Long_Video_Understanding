"""
fastvid_port.py -- FastVID (NeurIPS'25) cho Qwen2.5-VL, ghep vao cau hinh danh gia cua du an.

Nguon: Long_Video_Understanding/FastVID (commit a40a109),
  fastvid_qwen25vl/lmms-eval/lmms_eval/models/qwenvlutils/modeling_qwen2_5_vl.py
  - dong 1899-2090: khoi DySeg + STPrune + DTM trong forward()  -> fastvid_select(), CHEP NGUYEN VAN
    (chi doi self.fastvid_* -> p.*, va chan k cua topk khi video co it hon c unit -- ban goc se crash).
  - Qwen2_5_VLVisionFlashAttention2.forward(): cach tinh attn_weights o block cuoi cua vision tower
    -> fastvid_frame_attn(), viet lai bang hook vi transformers 5.x khong tra attn ra ngoai.
Sieu tham so mac dinh = scripts/eval.sh cua ho (c=8, tau=0.84, ignore=0.95, d=0.4, p=4, beta=0.6).
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
from transformers.vision_utils import get_vision_window_index


def fastvid_params(args):
    g = lambda k, d: type(d)(getattr(args, k, d))
    return SimpleNamespace(
        retention_ratio=float(getattr(args, "retention", 0.27)),
        DySeg_c=g("fastvid_DySeg_c", 8), DySeg_tau=g("fastvid_DySeg_tau", 0.84), DySeg_ignore=g("fastvid_DySeg_ignore", 0.95),
        STPrune_d=g("fastvid_STPrune_d", 0.4), DTM_p=g("fastvid_DTM_p", 4), DTM_beta=g("fastvid_DTM_beta", 0.6),
    )


@torch.no_grad()
def fastvid_frame_attn(llm, pixel_values_videos, video_grid_thw):
    """Tra ve (video_embeds [T*N, D], frame_attn_weights [T, N]) giong self.visual(...) cua ban FastVID."""
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
    seq_length = hidden_states.shape[0]
    q, k, v = attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    # --- tu day chep tu Qwen2_5_VLVisionFlashAttention2.forward cua FastVID ---
    q_attn, k_attn = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q_attn = q_attn.transpose(0, 1)
    k_attn = k_attn.transpose(0, 1)
    grid_thw = video_grid_thw
    grid_t, grid_h, grid_w = grid_thw[0,0], grid_thw[0,1], grid_thw[0,2]
    q_attn = q_attn.view(attn.num_heads, grid_t, grid_h * grid_w, attn.head_dim).mean(-2, keepdim=True)
    k_attn = k_attn.view(attn.num_heads, grid_t, grid_h * grid_w // 4, 4, attn.head_dim).mean(-2)
    attn_weights = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_attn.dtype)
    attn_weights = attn_weights.mean(2).mean(0)
    # --- va tu Qwen2_5_VisionTransformerPretrainedModel.forward cua FastVID ---
    window_index, _ = get_vision_window_index(
        grid_thw, spatial_merge_size=visual.spatial_merge_size, window_size=visual.window_size,
        patch_size=visual.patch_size, kwargs={},
    )
    reverse_indices = torch.argsort(window_index.to(attn_weights.device))
    attn_weights = attn_weights.view(-1)
    attn_weights = attn_weights[reverse_indices]
    attn_weights = attn_weights.reshape(grid_thw[0, 0], -1)
    return video_embeds, attn_weights


@torch.no_grad()
def fastvid_select(video_embeds, frame_attn_weights, video_grid_thw, p):
    """video_embeds [T*N, D] (da ep ve dtype cua inputs_embeds) -> (kept_indices tang dan [K], kept_embeds [K, D], diag)."""
    frame_num, frame_w, frame_h = video_grid_thw[0,0].item(), video_grid_thw[0,1].item() // 2, video_grid_thw[0,2].item() // 2
    frame_global_features = video_embeds.view(video_grid_thw[0,0], video_grid_thw[0,1] * video_grid_thw[0,2] // 4, -1).mean(1)
    device_type = video_embeds.device
    prompt_start_len = 0
    video_hidden_states = video_embeds.reshape(frame_num, frame_w * frame_h, -1)
    keep_indexs = ()

    # ====== BAT DAU khoi chep nguyen van (dong 1899-2090 cua file goc) ======
    frame_global_features = frame_global_features / frame_global_features.norm(dim=1, keepdim=True) 
    similarity_matrix = (frame_global_features[:-1] * frame_global_features[1:]).sum(dim=1)

    cut_indices = torch.topk(similarity_matrix, min(p.DySeg_c-1, similarity_matrix.shape[0]), largest=False).indices
    cos_indices = torch.nonzero(similarity_matrix < p.DySeg_tau).squeeze(1)
    cut_indices = torch.unique(torch.cat([cut_indices, cos_indices]))
    cut_indices = cut_indices.sort().values

    segment_sizes = [cut_indices[0].item() + 1]  
    for i in range(1, cut_indices.shape[0]):
        segment_sizes.append(cut_indices[i].item() - cut_indices[i-1].item())  
    segment_sizes.append(frame_num - cut_indices[-1].item() - 1)

    segments_hidden_states = torch.split(video_hidden_states, segment_sizes)
    segments_attn_weights = torch.split(frame_attn_weights, segment_sizes)
    segment_num = len(segment_sizes)

    segments_global_features = torch.split(frame_global_features, segment_sizes)

    vision_tokens = []

    frame_retain_num = int(frame_w * frame_h * p.retention_ratio)

    seg_index_start = prompt_start_len
    for seg_i in range(segment_num):
        cur_hidden_states = segments_hidden_states[seg_i]
        cur_attn_weights = segments_attn_weights[seg_i]
        cur_global_features = segments_global_features[seg_i]
        cur_seg_len, frm_len = cur_attn_weights.shape

        seg_retain_num = frame_retain_num * cur_seg_len
        seg_context_num = max(int(seg_retain_num * p.STPrune_d), 1)
        seg_salient_num = seg_retain_num - seg_context_num

        if cur_seg_len == 1:
            frm_salient_num = [seg_salient_num]
            frm_context_num = [seg_context_num]
        else:
            cur_global_similarity_matrix = (cur_global_features[:-1] * cur_global_features[1:]).sum(dim=1)
            cur_cut_indices = torch.nonzero(cur_global_similarity_matrix < p.DySeg_ignore).squeeze(1)
            cur_cut_indices = cur_cut_indices.sort().values
            if cur_cut_indices.shape[0] > 0:
                cur_segment_sizes = [cur_cut_indices[0].item() + 1]  
                for i in range(1, cur_cut_indices.shape[0]):
                    cur_segment_sizes.append(cur_cut_indices[i].item() - cur_cut_indices[i-1].item())  
                cur_segment_sizes.append(cur_seg_len - cur_cut_indices[-1].item() - 1)
            else:
                cur_segment_sizes = [cur_seg_len]

            valid_seg_len = len(cur_segment_sizes)

            chunk_size = seg_salient_num // valid_seg_len
            remainder = seg_salient_num % valid_seg_len
            frm_salient_num = [chunk_size + (1 if i < remainder else 0) for i in range(valid_seg_len)]

            temp_num = (valid_seg_len + p.DTM_p - 1) //  p.DTM_p

            chunk_size = seg_context_num // temp_num
            remainder = seg_context_num % temp_num
            temp_context_num = [chunk_size + (1 if i < remainder else 0) for i in range(temp_num)]

            tmp_ctx_idx = 0
            frm_context_num = []
            for tmp_i in range(valid_seg_len):
                if tmp_i % p.DTM_p == 0:
                    frm_context_num.append(min(temp_context_num[tmp_ctx_idx], frm_len//2))
                    if temp_context_num[tmp_ctx_idx] + frm_context_num[-1] > frm_len:
                        temp_context_num[tmp_ctx_idx] = frm_len - frm_context_num[-1]
                    tmp_ctx_idx += 1
                else:
                    frm_context_num.append(0)
            frm_context_num.reverse()

            valid_frm_salient_num = []
            valid_frm_context_num = []
            for tmp_tmp_tmp_i in range(valid_seg_len):
                valid_frm_salient_num.extend([0] * (cur_segment_sizes[tmp_tmp_tmp_i] - 1))
                valid_frm_context_num.extend([0] * (cur_segment_sizes[tmp_tmp_tmp_i] - 1))

                if frm_context_num[tmp_tmp_tmp_i] == 0:
                    valid_frm_salient_num.append(min(frm_salient_num[tmp_tmp_tmp_i], frm_len))
                    valid_frm_context_num.append(0)
                elif frm_context_num[tmp_tmp_tmp_i] > 0:
                    valid_frm_context_num.append(min(frm_context_num[tmp_tmp_tmp_i], frm_len//2))
                    valid_frm_salient_num.append(min(frm_salient_num[tmp_tmp_tmp_i], frm_len - valid_frm_context_num[-1]))
                else:
                    valid_frm_salient_num.append(frm_len-int(frm_len * p.STPrune_d))
                    valid_frm_context_num.append(int(frm_len * p.STPrune_d))

            frm_salient_num = valid_frm_salient_num
            frm_context_num = valid_frm_context_num

        cur_salient_indexs = []
        cur_context_indexs = []

        for frame_i in range(cur_seg_len):
            ### Get Salient Indexs
            if frm_salient_num[frame_i] > 0:
                frm_top_indexes = torch.topk(cur_attn_weights[frame_i], frm_salient_num[frame_i]).indices
            else:
                frm_top_indexes = None

            ### Get Context Indexs
            if frm_context_num[frame_i] > 0:
                frm_all_indexs = torch.arange(frm_len, device=device_type)
                if frm_top_indexes is not None:
                    tmp_mask = torch.isin(frm_all_indexs, frm_top_indexes)
                    remaining_indexs = frm_all_indexs[~tmp_mask]  
                else:
                    remaining_indexs = frm_all_indexs

                tmp_frm_hidden_states = cur_hidden_states[frame_i]
                tmp_frm_hidden_states = tmp_frm_hidden_states[remaining_indexs].unsqueeze(0)
                _, tmp_N, tmp_D = tmp_frm_hidden_states.shape

                dist_matrix = torch.cdist(tmp_frm_hidden_states.float(), tmp_frm_hidden_states.float()) / (tmp_D ** 0.5)

                # get local density
                dist_nearest, index_nearest = torch.topk(dist_matrix, k=4, dim=-1, largest=False)
                density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
                density = density + torch.rand(
                    density.shape, device=device_type, dtype=density.dtype) * 1e-6

                # get distance indicator
                mask = density[:, None, :] > density[:, :, None]
                mask = mask.type(tmp_frm_hidden_states.dtype)
                dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
                dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

                # select clustering center according to score
                score = dist * density
                _, sampled_indexs = torch.topk(score, k=frm_context_num[frame_i], dim=-1)


                sampled_indexs = sampled_indexs[0].sort().values

                frm_context_indexes = remaining_indexs[sampled_indexs]
            else:
                frm_context_indexes = None

            ### Index Process
            if frm_top_indexes is not None:
                cur_salient_indexs.append(frm_top_indexes + frame_i * frm_len)
            if frm_context_indexes is not None:
                cur_context_indexs.append(frm_context_indexes + frame_i * frm_len)

        cur_tome_hidden_states = cur_hidden_states.reshape(cur_seg_len * frm_len, -1)

        cur_salient_indexs = torch.cat(cur_salient_indexs)
        cur_salient_hidden_states = cur_tome_hidden_states[cur_salient_indexs]

        ### Context Merge
        cur_all_indexs = torch.arange(cur_seg_len * frm_len, device=device_type)
        cur_tome_tokens = cur_tome_hidden_states / cur_tome_hidden_states.norm(dim=-1, keepdim=True) 
        cur_context_hidden_states = []

        for tmp_context_indexs in cur_context_indexs:
            tmp_retain_index = torch.cat([cur_salient_indexs, tmp_context_indexs])
            tmp_mask = torch.isin(cur_all_indexs, tmp_retain_index)
            tmp_merge_indexs = cur_all_indexs[~tmp_mask] 

            tmp_target_tokens = cur_tome_tokens[tmp_context_indexs]
            tmp_to_merge_tokens = cur_tome_tokens[tmp_merge_indexs]

            similarity = torch.mm(tmp_to_merge_tokens, tmp_target_tokens.T)
            assign_one_hot = torch.zeros(tmp_to_merge_tokens.shape[0], tmp_context_indexs.shape[0], dtype=cur_hidden_states.dtype, device=device_type)
            assign_one_hot.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)

            avg_weights = (1 / (assign_one_hot.sum(dim=0).unsqueeze(-1) + 1)).clamp(min=p.DTM_beta)
            counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
            hidden_to_merge = cur_tome_hidden_states[tmp_merge_indexs]
            aggregated_hidden = torch.mm(assign_one_hot.T, hidden_to_merge) / counts
            target_hidden = cur_tome_hidden_states[tmp_context_indexs] 
            target_hidden = avg_weights * target_hidden + (1 - avg_weights) * aggregated_hidden

            cur_context_hidden_states.append(target_hidden)

        cur_context_indexs = torch.cat(cur_context_indexs)
        cur_context_hidden_states = torch.cat(cur_context_hidden_states, dim=0)

        ### segment final hidden_states & indexs
        cur_index_combined = torch.cat([cur_salient_indexs, cur_context_indexs])
        cur_hidden_combined = torch.cat([cur_salient_hidden_states, cur_context_hidden_states], dim=0)

        sorted_indexs = torch.argsort(cur_index_combined)
        cur_index_combined = cur_index_combined[sorted_indexs]
        cur_hidden_combined = cur_hidden_combined[sorted_indexs]

        vision_tokens.append(cur_hidden_combined)
        keep_indexs += (cur_index_combined + seg_index_start,)
        seg_index_start = seg_index_start + cur_seg_len * frm_len

    # ====== KET THUC khoi chep nguyen van ======
    vision_tokens = torch.cat(vision_tokens, dim=0)
    keep_indexs = torch.cat(keep_indexs)
    order = torch.argsort(keep_indexs)
    assert torch.equal(order, torch.arange(order.numel(), device=order.device)), "keep_indexs khong tang dan nhu ban goc gia dinh"
    assert keep_indexs.unique().numel() == keep_indexs.numel(), "trung chi so"
    return keep_indexs, vision_tokens, {"num_segments": int(segment_num), "frame_retain_num": int(frame_retain_num)}
