"""
analyze_perceiver_tokens.py

Phan tich token nen T2 cua Perceiver Resampler (train_perceiver_stage1/2.py):
  A. Do da dang cua T2 (cosine tung cap, hang hieu dung, phu thuoc dau vao).
  B. T2 bi trung binh hoa hay phan cuc (entropy attention, norm, do phu).
  C. Kha nang bieu dien truoc/sau nen (tai tao T1, caption CE voi nhieu bien
     the doi chung, bo tung token, logit lens).
Moi chi so tinh song song tren cac bo 64 token doi chung lay tu CUNG T1:
uniform64, pool64 (anh), kmeans64, random64.

Chi DOC checkpoint/du lieu, ghi ket qua vao outputs/perceiver_analysis/.

Usage:
  python analyze_perceiver_tokens.py --n_images 200 --n_videos 40
"""

import argparse
import collections
import json
import os
import random
import time

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vision_process_fastvid import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import run_inference_qwen25vl_dyseg as dm
from run_inference_qwen25vl_dyseg import dyseg_segments
from train_perceiver_stage1 import CAPTION_PROMPT, MIN_PIXELS, SYSTEM_PROMPT, PerceiverResampler

K = 64


# ---------------------------------------------------------------- helpers
def offdiag_mean_cos(X):
    Xn = F.normalize(X.float(), dim=-1)
    S = Xn @ Xn.t()
    n = S.shape[0]
    return float((S.sum() - S.diag().sum()) / max(n * (n - 1), 1))


def spectrum_stats(X):
    """Hang hieu dung (exp entropy pho phuong sai) + participation ratio."""
    Xc = (X - X.mean(0, keepdim=True)).double()
    s = torch.linalg.svdvals(Xc)
    lam = s ** 2
    p = lam / lam.sum().clamp(min=1e-30)
    erank = float(torch.exp(-(p * torch.log(p.clamp(min=1e-30))).sum()))
    pr = float(lam.sum() ** 2 / (lam ** 2).sum().clamp(min=1e-30))
    return erank, pr, p.float().cpu().numpy()


def subspace_r2(S, T1, mu):
    """Ti le phuong sai cua T1 (da tru mu) nam trong khong gian con sinh boi S - mu."""
    Q, _ = torch.linalg.qr((S - mu).double().t())  # [D, k]
    T1c = (T1 - mu).double()
    proj = T1c @ Q
    return float((proj ** 2).sum() / (T1c ** 2).sum().clamp(min=1e-30))


def kmeans(X, k, iters=10):
    idx = torch.linspace(0, X.shape[0] - 1, k).round().long().to(X.device)
    C = X[idx].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(dim=1)
        newC = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.zeros(k, device=X.device).index_add_(0, a, torch.ones(X.shape[0], device=X.device))
        ok = cnt > 0
        C[ok] = newC[ok] / cnt[ok].unsqueeze(-1)
    return C


def set_metrics(S, T1, mu, prefix, row):
    """S: [k, D] bo token can danh gia; T1: [n, D] token goc; mu: trung binh T1."""
    S = S.float()
    row[f"{prefix}/raw_cos"] = offdiag_mean_cos(S)
    row[f"{prefix}/cen_cos"] = offdiag_mean_cos(S - mu)
    er, pr, _ = spectrum_stats(S)
    row[f"{prefix}/erank"] = er
    row[f"{prefix}/part_ratio"] = pr
    nrm = S.norm(dim=-1)
    row[f"{prefix}/norm_mean"] = float(nrm.mean())
    row[f"{prefix}/norm_cv"] = float(nrm.std() / nrm.mean())
    row[f"{prefix}/cos_to_T1mean"] = float(F.cosine_similarity(S, mu.expand_as(S), dim=-1).mean())
    sim = F.normalize(S, dim=-1) @ F.normalize(T1, dim=-1).t()  # [k, n]
    mx, am = sim.max(dim=1)
    row[f"{prefix}/anchor_maxcos"] = float(mx.mean())
    row[f"{prefix}/anchor_unique"] = int(am.unique().numel())
    row[f"{prefix}/cover_raw"] = float(sim.max(dim=0).values.mean())
    simc = F.normalize(S - mu, dim=-1) @ F.normalize(T1 - mu, dim=-1).t()
    row[f"{prefix}/cover_cen"] = float(simc.max(dim=0).values.mean())
    row[f"{prefix}/r2_subspace"] = subspace_r2(S, T1, mu)


def perceiver_forward(p, T1):
    """Giong PerceiverResampler.forward nhung tra them attention TUNG HEAD
    va ti le norm (dong gop cua cross-attention so voi query)."""
    q = p.query.unsqueeze(0)
    kv = T1.unsqueeze(0)
    maps, ratios = [], []
    for i in range(len(p.self_attns)):
        qn = p.ln_self[i](q)
        so, _ = p.self_attns[i](qn, qn, qn)
        q = q + so
        kvn = p.ln_kv[i](kv)
        co, w = p.cross_attns[i](p.ln_q[i](q), kvn, kvn, need_weights=True, average_attn_weights=False)
        maps.append(w[0].detach())  # [heads, Q, n]
        ratios.append(float(co.norm(dim=-1).mean() / q.norm(dim=-1).mean()))
        q = q + co
        q = q + p.ffns[i](q)
    return p.final_ln(q).squeeze(0), maps, ratios


def entropy(P):
    return -(P * torch.log(P.clamp(min=1e-30))).sum(-1)


def attn_metrics(maps, n, prefix, row):
    for l, A in enumerate(maps):
        Am = A.mean(0)  # [Q, n]
        effn = torch.exp(entropy(Am))
        effn_head = torch.exp(entropy(A)).mean(0)
        row[f"{prefix}/L{l}/effn_mean"] = float(effn.mean())
        row[f"{prefix}/L{l}/effn_median"] = float(effn.median())
        row[f"{prefix}/L{l}/effn_min"] = float(effn.min())
        row[f"{prefix}/L{l}/effn_frac_of_n"] = float(effn.mean() / n)
        row[f"{prefix}/L{l}/effn_perhead_mean"] = float(effn_head.mean())
        row[f"{prefix}/L{l}/effn_perhead_frac"] = float(effn_head.mean() / n)
        row[f"{prefix}/L{l}/maxw_mean"] = float(Am.max(-1).values.mean())
        row[f"{prefix}/L{l}/top5_mass"] = float(Am.topk(min(5, n), dim=-1).values.sum(-1).mean())
        mm = Am.mean(0)
        row[f"{prefix}/L{l}/cover_eff_frac"] = float(torch.exp(entropy(mm)) / n)
        row[f"{prefix}/L{l}/dead_frac"] = float((mm < 0.1 / n).float().mean())
        row[f"{prefix}/L{l}/query_map_cos"] = offdiag_mean_cos(Am)
        row[f"{prefix}/L{l}/argmax_unique"] = int(Am.argmax(-1).unique().numel())


def ce_seq(model, pre_ids, post_ids, cap_ids, toks, tok_id):
    """Caption CE (teacher-forced) khi thay token anh bang toks [B, k, D],
    position_ids TUAN TU (giong het luc train Perceiver). Tra ve [B]."""
    B, k, D = toks.shape
    dev = pre_ids.device
    ids = torch.cat([pre_ids, torch.full((k,), tok_id, dtype=pre_ids.dtype, device=dev), post_ids, cap_ids])
    input_ids = ids.unsqueeze(0).expand(B, -1)
    emb = model.get_input_embeddings()(input_ids).clone()
    if k > 0:
        emb[input_ids == tok_id] = toks.reshape(-1, D).to(emb.dtype)
    L = ids.shape[0]
    pos = torch.arange(L, device=dev).view(1, 1, L).expand(3, B, L)
    out = model(inputs_embeds=emb, position_ids=pos, attention_mask=torch.ones_like(input_ids), return_dict=True)
    C = cap_ids.shape[0]
    logits = out.logits[:, L - C - 1: L - 1].float()
    tgt = cap_ids.unsqueeze(0).expand(B, -1)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(B, C)
    return loss.mean(1)


def agg(rows):
    keys = collections.OrderedDict()
    for r in rows:
        for k in r:
            if isinstance(r[k], (int, float)):
                keys.setdefault(k, []).append(float(r[k]))
    return {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                    p90=float(np.percentile(v, 90)), n=len(v)) for k, v in keys.items()}


def load_perceiver(path, device):
    t0 = time.time()
    ckpt = torch.load(path, map_location="cpu")
    p = PerceiverResampler(ckpt["dim"], num_queries=ckpt["num_queries"], num_layers=ckpt["num_layers"]).float()
    p.load_state_dict(ckpt["state_dict"])
    meta = {k: ckpt[k] for k in ckpt if k not in ("state_dict", "optimizer_state")}
    del ckpt
    print(f"loaded {path} in {time.time()-t0:.0f}s meta={meta}", flush=True)
    return p.to(device).eval(), meta


def cross_sample_stats(X):
    """X: [S, Q, D] (cung 1 query/vi tri qua S mau). Tach phuong sai:
    phan do DANH TINH QUERY (khong phu thuoc dau vao) vs phan do NOI DUNG."""
    X = X.float()
    S, Q, D = X.shape
    grand = X.mean((0, 1), keepdim=True)
    tot = ((X - grand) ** 2).sum()
    qmean = X.mean(0, keepdim=True)  # [1, Q, D]
    smean = X.mean(1, keepdim=True)  # [S, 1, D]
    out = {
        "var_frac_query_identity": float(S * ((qmean - grand) ** 2).sum() / tot),
        "var_frac_sample_identity": float(Q * ((smean - grand) ** 2).sum() / tot),
    }
    Xn = F.normalize(X - grand, dim=-1)
    # cung query, khac mau
    same_q = torch.einsum("sqd,tqd->qst", Xn, Xn)
    mask = ~torch.eye(S, dtype=torch.bool)
    out["cos_same_query_diff_sample"] = float(same_q[:, mask].mean())
    # cung mau, khac query
    same_s = torch.einsum("sqd,srd->sqr", Xn, Xn)
    maskq = ~torch.eye(Q, dtype=torch.bool)
    out["cos_same_sample_diff_query"] = float(same_s[:, maskq].mean())
    out["per_query_input_var_frac"] = (((X - qmean) ** 2).sum((0, 2)) / (((X - grand) ** 2).sum((0, 2)))).tolist()
    return out


# ---------------------------------------------------------------- images
def analyze_images(model, processor, p1, p2, En, args, device, out_dir):
    d = json.load(open(args.coco_captions))
    img_by_id = {im["id"]: im["file_name"] for im in d["images"]}
    cap_by_img = {}
    for ann in d["annotations"]:
        cap_by_img.setdefault(img_by_id[ann["image_id"]], ann["caption"].strip())
    names = sorted(cap_by_img)
    random.Random(0).shuffle(names)
    names = names[: args.n_images]

    llm = model.model if hasattr(model, "model") else model
    tok_id = model.config.image_token_id
    tk = processor.tokenizer
    max_pixels = args.image_units * 28 * 28

    rows, T2_all, T2s2_all, pool_all, uni_all, maps_all, loo_all = [], [], [], [], [], [], []
    spectra = collections.defaultdict(list)
    lens_T2, lens_T1 = collections.Counter(), collections.Counter()
    prev_T2 = None
    for si, name in enumerate(names):
        try:
            path = os.path.join(args.coco_dir, name)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": path, "max_pixels": max_pixels, "min_pixels": MIN_PIXELS},
                    {"type": "text", "text": CAPTION_PROMPT},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ii, vi, _ = process_vision_info(messages, return_video_kwargs=True)
            inp = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(device)
            feat = llm.get_image_features(inp["pixel_values"], image_grid_thw=inp["image_grid_thw"])
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            T1 = (feat if not isinstance(feat, (list, tuple)) else torch.cat(feat, dim=0)).float()
            thw = inp["image_grid_thw"][0].tolist()
            h, w = int(thw[1]) // 2, int(thw[2]) // 2
            n = T1.shape[0]
            if n != h * w or n < 100:
                continue
            mu = T1.mean(0, keepdim=True)
            row = {"name": name, "n": n, "h": h, "w": w}

            T2, maps, ratios = perceiver_forward(p1, T1)
            T2s2, maps2, ratios2 = perceiver_forward(p2, T1)
            g = torch.Generator(device="cpu").manual_seed(si)
            sets = {
                "T2": T2,
                "T2s2": T2s2,
                "uniform64": T1[torch.linspace(0, n - 1, K).round().long()],
                "random64": T1[torch.randperm(n, generator=g)[:K].sort().values],
                "pool64": F.adaptive_avg_pool2d(T1.view(h, w, -1).permute(2, 0, 1).unsqueeze(0), (8, 8))[0].flatten(1).t(),
                "kmeans64": kmeans(T1, K),
            }
            for nm, S in sets.items():
                set_metrics(S, T1, mu, nm, row)
                spectra[nm].append(spectrum_stats(S.float())[2][:63])
            # T1 goc
            sub = T1[torch.randperm(n, generator=g)[:512]]
            row["T1/raw_cos"] = offdiag_mean_cos(sub)
            row["T1/cen_cos"] = offdiag_mean_cos(sub - mu)
            er, pr, sp = spectrum_stats(T1)
            row["T1/erank"], row["T1/part_ratio"] = er, pr
            row["T1/pca64_share"] = float(sp[:K].sum())
            nrm = T1.norm(dim=-1)
            row["T1/norm_mean"], row["T1/norm_cv"] = float(nrm.mean()), float(nrm.std() / nrm.mean())
            row["T1/cos_to_T1mean"] = float(F.cosine_similarity(T1, mu.expand_as(T1), dim=-1).mean())

            attn_metrics(maps, n, "attn", row)
            attn_metrics(maps2, n, "attn_s2", row)
            for l, r in enumerate(ratios):
                row[f"attn/L{l}/cross_over_query_norm"] = r

            # do nhay voi dau vao
            T2_zero, _, _ = perceiver_forward(p1, torch.zeros_like(T1))
            T2_noise, _, _ = perceiver_forward(p1, torch.randn_like(T1) * T1.std() + T1.mean())
            T2_shuf, _, _ = perceiver_forward(p1, T1[torch.randperm(n, generator=g)])
            cs = lambda a, b: float(F.cosine_similarity(a, b, dim=-1).mean())
            row["sens/cos_real_vs_zeroinput"] = cs(T2, T2_zero)
            row["sens/cos_real_vs_noiseinput"] = cs(T2, T2_noise)
            row["sens/cos_real_vs_shuffledinput"] = cs(T2, T2_shuf)
            row["sens/cos_real_vs_query_param"] = cs(T2, p1.query)
            if prev_T2 is not None:
                row["sens/cos_real_vs_prev_image"] = cs(T2, prev_T2)
                row["sens/cos_s1_vs_s2"] = cs(T2, T2s2)

            # logit lens: tu vung gan nhat
            def lens(X, counter):
                s = F.normalize(X.float(), dim=-1).to(En.dtype) @ En.t()
                mx, am = s.float().max(dim=-1)
                for t in am.tolist():
                    counter[t] += 1
                return float(mx.mean()), int(am.unique().numel())
            row["lens/T2_maxcos"], row["lens/T2_unique_words"] = lens(T2, lens_T2)
            row["lens/T1_maxcos"], row["lens/T1_unique_words_of64"] = lens(sets["uniform64"], lens_T1)

            # ------------- caption CE
            full_ids = inp["input_ids"][0]
            m = full_ids == tok_id
            nz = m.nonzero(as_tuple=True)[0]
            pre_ids, post_ids = full_ids[: nz[0]], full_ids[nz[-1] + 1:]
            cap_ids = torch.tensor(tk(cap_by_img[name], add_special_tokens=False).input_ids + [tk.eos_token_id],
                                   dtype=full_ids.dtype, device=device)
            C = cap_ids.shape[0]
            # native: T1 day du, mRoPE 2D that
            ids_nat = torch.cat([full_ids, cap_ids]).unsqueeze(0)
            out = model(input_ids=ids_nat, attention_mask=torch.ones_like(ids_nat), pixel_values=inp["pixel_values"],
                        image_grid_thw=inp["image_grid_thw"], return_dict=True)
            Ln = ids_nat.shape[1]
            row["ce/T1_full_native"] = float(F.cross_entropy(out.logits[0, Ln - C - 1: Ln - 1].float(), cap_ids))
            row["ce/T1_full_seqpos"] = float(ce_seq(model, pre_ids, post_ids, cap_ids, T1.unsqueeze(0), tok_id)[0])
            row["ce/blind_no_image"] = float(ce_seq(model, pre_ids, post_ids, cap_ids, T1[:0].unsqueeze(0), tok_id)[0])

            T2c = T2 - T2.mean(0, keepdim=True)
            U, Sv, Vh = torch.linalg.svd(T2c.double(), full_matrices=False)
            lowrank = lambda r: ((U[:, :r] * Sv[:r]) @ Vh[:r]).float() + T2.mean(0, keepdim=True)
            variants = collections.OrderedDict([
                ("T2", T2), ("T2s2", T2s2), ("uniform64", sets["uniform64"]), ("random64", sets["random64"]),
                ("pool64", sets["pool64"]), ("kmeans64", sets["kmeans64"]),
                ("T2_noise", torch.randn_like(T2) * T2.std() + T2.mean()), ("T2_zeros", torch.zeros_like(T2)),
                ("T2_order_shuffled", T2[torch.randperm(K, generator=g)]),
                ("T2_mean_x64", T2.mean(0, keepdim=True).expand(K, -1)),
                ("T1mean_x64", mu.expand(K, -1)),
                ("T2_rank1", lowrank(1)), ("T2_rank4", lowrank(4)), ("T2_rank16", lowrank(16)),
                ("T2_from_zero_input", T2_zero), ("T2_query_param_only", p1.query.detach()),
            ])
            if prev_T2 is not None:
                variants["T2_of_other_image"] = prev_T2
            ces = ce_seq(model, pre_ids, post_ids, cap_ids, torch.stack(list(variants.values())).float(), tok_id)
            for nm, v in zip(variants, ces.tolist()):
                row[f"ce/{nm}"] = v
            # so token it hon nua
            for kk in (16, 32):
                row[f"ce/T2_first{kk}"] = float(ce_seq(model, pre_ids, post_ids, cap_ids, T2[:kk].unsqueeze(0), tok_id)[0])
                row[f"ce/uniform{kk}"] = float(ce_seq(model, pre_ids, post_ids, cap_ids,
                                                      T1[torch.linspace(0, n - 1, kk).round().long()].unsqueeze(0), tok_id)[0])
            # bo tung token T2 (leave-one-out)
            if len(loo_all) < args.n_loo:
                keep = ~torch.eye(K, dtype=torch.bool, device=device)
                loo_toks = torch.stack([T2[keep[j]] for j in range(K)])
                loo = ce_seq(model, pre_ids, post_ids, cap_ids, loo_toks, tok_id) - row["ce/T2"]
                loo_u = ce_seq(model, pre_ids, post_ids, cap_ids,
                               torch.stack([sets["uniform64"][keep[j]] for j in range(K)]), tok_id) - row["ce/uniform64"]
                loo_all.append(loo.cpu().numpy())
                row["loo/T2_max_delta"], row["loo/T2_absmean_delta"] = float(loo.max()), float(loo.abs().mean())
                row["loo/uniform_max_delta"], row["loo/uniform_absmean_delta"] = float(loo_u.max()), float(loo_u.abs().mean())

            prev_T2 = T2.clone()
            T2_all.append(T2.half().cpu())
            T2s2_all.append(T2s2.half().cpu())
            pool_all.append(sets["pool64"].half().cpu())
            uni_all.append(sets["uniform64"].half().cpu())
            Am = maps[-1].mean(0).view(K, 1, h, w)
            m12 = F.adaptive_avg_pool2d(Am, (12, 12)).flatten(1)
            maps_all.append((m12 / m12.sum(-1, keepdim=True)).half().cpu())
            rows.append(row)
        except Exception as e:
            print(f"SKIP image {name}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        if len(rows) % 20 == 0:
            print(f"[images] {len(rows)} done ({si+1}/{len(names)})", flush=True)
            json.dump({"rows": rows, "agg": agg(rows)}, open(os.path.join(out_dir, "image_partial.json"), "w"))

    res = {"rows": rows, "agg": agg(rows)}
    res["cross_sample"] = {
        "T2": cross_sample_stats(torch.stack(T2_all)),
        "T2s2": cross_sample_stats(torch.stack(T2s2_all)),
        "pool64": cross_sample_stats(torch.stack(pool_all)),
        "uniform64": cross_sample_stats(torch.stack(uni_all)),
    }
    M = torch.stack(maps_all).float()  # [S, Q, 144]
    Mc = F.normalize(M - 1.0 / 144, dim=-1)
    S = M.shape[0]
    cons = torch.einsum("sqd,tqd->qst", Mc, Mc)[:, ~torch.eye(S, dtype=torch.bool)].mean(1)
    res["spatial"] = {
        "per_query_map_consistency_across_images": cons.tolist(),
        "mean_consistency": float(cons.mean()),
        "dataset_mean_map_12x12": M.mean((0, 1)).view(12, 12).tolist(),
    }
    res["spectra_mean"] = {k: np.mean(np.stack(v), 0).tolist() for k, v in spectra.items()}
    dec = lambda c: [(tk.decode([t]), cnt) for t, cnt in c.most_common(40)]
    res["lens"] = {"T2_top": dec(lens_T2), "T1_top": dec(lens_T1),
                   "T2_distinct_total": len(lens_T2), "T1_distinct_total": len(lens_T1)}
    if loo_all:
        L = np.stack(loo_all)
        res["loo"] = {"per_query_mean_delta": L.mean(0).tolist(), "n_images": int(L.shape[0]),
                      "frac_abs_lt_0.01": float((np.abs(L) < 0.01).mean()),
                      "frac_negative": float((L < 0).mean())}
    np.savez_compressed(os.path.join(out_dir, "image_arrays.npz"), per_query_maps=M.mean(0).view(K, 12, 12).numpy(),
                        T2_example=T2_all[0].float().numpy(), uniform_example=uni_all[0].float().numpy())
    json.dump(res, open(os.path.join(out_dir, "image_results.json"), "w"))
    print("[images] DONE", flush=True)
    return res


# ---------------------------------------------------------------- videos
def analyze_videos(model, processor, p2, args, device, out_dir):
    qa = json.load(open(args.videomme_gt))
    vids = []
    for s in qa:
        if s["video_name"] not in vids:
            vids.append(s["video_name"])
    vids = vids[:: max(1, len(vids) // args.n_videos)][: args.n_videos]
    llm = model.model if hasattr(model, "model") else model
    max_pixels = args.video_units * 28 * 28
    taus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    seg_rows, vid_rows = [], []
    for vi_, vname in enumerate(vids):
        try:
            messages = dm.build_messages(os.path.join(args.videomme_dir, vname), "Describe the video.", ["A. x"], max_pixels)
            messages[1]["content"][0]["max_frames"] = args.max_frames
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ii, vv, _ = process_vision_info(messages, return_video_kwargs=True)
            inp = processor(text=[text], images=ii, videos=vv, padding=True, return_tensors="pt").to(device)
            feat = llm.get_video_features(inp["pixel_values_videos"], video_grid_thw=inp["video_grid_thw"])
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            T1f = (feat if not isinstance(feat, (list, tuple)) else torch.cat(feat, dim=0)).float()
            thw = inp["video_grid_thw"][0].tolist()
            T, h, w = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
            N = h * w
            T1 = T1f.view(T, N, -1)
            segs = dyseg_segments(T1.mean(1), c=args.dyseg_c, tau=args.dyseg_tau)
            vmu = T1f.mean(0, keepdim=True)
            T2s, segmeans, kept = [], [], {t: 0 for t in taus}
            for (s, e) in segs:
                X = T1[s:e].reshape(-1, T1.shape[-1])
                n = X.shape[0]
                mu = X.mean(0, keepdim=True)
                row = {"video": vname, "seg_len": e - s, "n": n}
                T2, maps, ratios = perceiver_forward(p2, X)
                g = torch.Generator(device="cpu").manual_seed(vi_ * 1000 + s)
                sets = {"T2": T2, "uniform64": X[torch.linspace(0, n - 1, K).round().long()],
                        "random64": X[torch.randperm(n, generator=g)[:K].sort().values], "kmeans64": kmeans(X, K, iters=6)}
                for nm, S_ in sets.items():
                    set_metrics(S_, X, mu, nm, row)
                attn_metrics(maps, n, "attn", row)
                for l, r in enumerate(ratios):
                    row[f"attn/L{l}/cross_over_query_norm"] = r
                if e - s > 1:
                    per_unit = maps[-1].mean(0).view(K, e - s, N).sum(-1)  # [Q, len]
                    row["temporal/query_unit_entropy_norm"] = float((entropy(per_unit) / np.log(e - s)).mean())
                    mu_unit = per_unit.mean(0)
                    row["temporal/mean_unit_entropy_norm"] = float(entropy(mu_unit) / np.log(e - s))
                    row["temporal/max_unit_share"] = float(mu_unit.max())
                    row["temporal/uniform_share"] = 1.0 / (e - s)
                sim = F.normalize(T2, dim=-1) @ F.normalize(X, dim=-1).t()
                mpt = sim.max(dim=0).values
                for t in taus:
                    c = int((mpt >= t).sum())
                    row[f"point/frac_tau{t}"] = c / n
                    kept[t] += c
                row["point/maxsim_mean"] = float(mpt.mean())
                row["point/maxsim_p10"] = float(mpt.quantile(0.1))
                row["point/maxsim_p90"] = float(mpt.quantile(0.9))
                # token duoc point co tap trung vao it unit / it vi tri khong
                pm = (mpt >= 0.5).view(e - s, N).float()
                if pm.sum() > 0 and e - s > 1:
                    pu = pm.sum(1) / pm.sum()
                    row["point/unit_entropy_norm"] = float(entropy(pu) / np.log(e - s))
                seg_rows.append(row)
                T2s.append(T2)
                segmeans.append(mu[0])
            vrow = {"video": vname, "T": T, "N": N, "n_total": T * N, "n_segments": len(segs),
                    "n_one_unit_segments": sum(1 for s, e in segs if e - s == 1)}
            for t in taus:
                vrow[f"final_tokens_tau{t}"] = K * len(segs) + kept[t]
                vrow[f"retention_tau{t}"] = (K * len(segs) + kept[t]) / (T * N)
            vrow["T2_share_of_final_tau0.5"] = K * len(segs) / (K * len(segs) + kept[0.5])
            if len(segs) > 1:
                Z = torch.stack(T2s)  # [nseg, Q, D]
                Zc = F.normalize(Z - Z.mean((0, 1), keepdim=True), dim=-1)
                ns = Z.shape[0]
                mk = ~torch.eye(ns, dtype=torch.bool)
                vrow["T2_cos_same_query_diff_segment"] = float(torch.einsum("sqd,tqd->qst", Zc, Zc)[:, mk].mean())
                vrow["T2_cos_same_segment_diff_query"] = float(
                    torch.einsum("sqd,srd->sqr", Zc, Zc)[:, ~torch.eye(K, dtype=torch.bool)].mean())
                vrow["T2_raw_cos_same_query_diff_segment"] = float(
                    torch.einsum("sqd,tqd->qst", F.normalize(Z, dim=-1), F.normalize(Z, dim=-1))[:, mk].mean())
                SM = F.normalize(torch.stack(segmeans) - vmu, dim=-1)
                vrow["segmean_cos_diff_segment"] = float((SM @ SM.t())[mk].mean())
                allT2 = Z.reshape(-1, Z.shape[-1])
                vrow["allT2/erank"] = spectrum_stats(allT2)[0]
                vrow["allT2/raw_cos"] = offdiag_mean_cos(allT2)
                uni = T1f[torch.linspace(0, T1f.shape[0] - 1, allT2.shape[0]).round().long()]
                vrow["uniform_same_count/erank"] = spectrum_stats(uni)[0]
                vrow["uniform_same_count/raw_cos"] = offdiag_mean_cos(uni)
                vrow["allT2/r2_subspace"] = subspace_r2(allT2, T1f, vmu)
                vrow["uniform_same_count/r2_subspace"] = subspace_r2(uni, T1f, vmu)
            vid_rows.append(vrow)
        except Exception as ex:
            print(f"SKIP video {vname}: {type(ex).__name__}: {ex}", flush=True)
            torch.cuda.empty_cache()
            continue
        print(f"[videos] {len(vid_rows)}/{len(vids)} {vname} segs={len(segs)}", flush=True)
        if len(vid_rows) % 5 == 0:
            json.dump({"seg_agg": agg(seg_rows), "vid_agg": agg(vid_rows)}, open(os.path.join(out_dir, "video_partial.json"), "w"))
    one = [r for r in seg_rows if r["seg_len"] == 1]
    multi = [r for r in seg_rows if r["seg_len"] > 1]
    res = {"seg_rows": seg_rows, "vid_rows": vid_rows, "seg_agg": agg(seg_rows), "vid_agg": agg(vid_rows),
           "seg_agg_one_unit": agg(one), "seg_agg_multi_unit": agg(multi)}
    json.dump(res, open(os.path.join(out_dir, "video_results.json"), "w"))
    print("[videos] DONE", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--ckpt_s1", default="outputs/perceiver_stage1_selfattn.pt")
    ap.add_argument("--ckpt_s2", default="outputs/perceiver_stage2_v3.pt")
    ap.add_argument("--coco_dir", default="Datasets/COCO/val2017")
    ap.add_argument("--coco_captions", default="Datasets/COCO/annotations/captions_val2017.json")
    ap.add_argument("--videomme_dir", default="Datasets/Video-MME/data/data")
    ap.add_argument("--videomme_gt", default="playground/gt_qa_files/VideoMME/val_qa_subset300.json")
    ap.add_argument("--out_dir", default="outputs/perceiver_analysis")
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--n_videos", type=int, default=40)
    ap.add_argument("--n_loo", type=int, default=24)
    ap.add_argument("--image_units", type=int, default=1024)
    ap.add_argument("--video_units", type=int, default=16384)
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--dyseg_c", type=int, default=8)
    ap.add_argument("--dyseg_tau", type=float, default=0.9)
    ap.add_argument("--skip_images", action="store_true")
    ap.add_argument("--skip_videos", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    print("loading Qwen ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.image_units * 28 * 28, min_pixels=MIN_PIXELS)
    p1, meta1 = load_perceiver(args.ckpt_s1, device)
    p2, meta2 = load_perceiver(args.ckpt_s2, device)
    json.dump({"s1": meta1, "s2": meta2, "args": vars(args)}, open(os.path.join(args.out_dir, "meta.json"), "w"), default=str)

    with torch.no_grad():
        # thong ke tinh cua query hoc duoc + do lech tham so giua 2 checkpoint
        st = {"query_raw_cos_s1": offdiag_mean_cos(p1.query), "query_raw_cos_s2": offdiag_mean_cos(p2.query),
              "query_erank_s1": spectrum_stats(p1.query)[0], "query_erank_s2": spectrum_stats(p2.query)[0],
              "query_norm_s1": float(p1.query.norm(dim=-1).mean()), "query_norm_s2": float(p2.query.norm(dim=-1).mean()),
              "query_cos_s1_vs_s2": float(F.cosine_similarity(p1.query, p2.query, dim=-1).mean())}
        json.dump(st, open(os.path.join(args.out_dir, "static.json"), "w"))
        print("static:", st, flush=True)
        if not args.skip_images:
            E = model.get_input_embeddings().weight
            En = F.normalize(E.float(), dim=-1).to(torch.bfloat16)
            analyze_images(model, processor, p1, p2, En, args, device, args.out_dir)
            del En
            torch.cuda.empty_cache()
        if not args.skip_videos:
            processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.video_units * 28 * 28, min_pixels=dm.MIN_PIXELS)
            analyze_videos(model, processor, p2, args, device, args.out_dir)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
