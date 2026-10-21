"""
scene_model.py -- chia canh + token tom tat canh cho RAP (lay y tu HTP-LVU).

  Y 1 (S1 cua HTP-LVU): chia video thanh cac canh bang vi sai giua cac unit (alloc_model.unit_change),
      ranh gioi o unit t neu Delta_t > mu_local + kappa * sigma_local (thong ke tren +-local unit, khong tinh t).
      Them: canh dai toi da max_len unit (cat deu neu dai hon), toi thieu min_len unit.
      RAP dung scene_ids lam mask: neo chi nhin token goc CUNG CANH trong cua so +-window.
  Y 2 (residual bypass token cua HTP-LVU, dang gop kieu HoliTom): moi canh 1 token tom tat
      = trung binh moi token goc cua canh, chuan hoa ve do lon trung binh cua token trong canh.
      Lay cho tu ngan sach: tong token = dung K cua Uniform (K - S neo + S token tom tat).
      Token tom tat dung vi tri (t, hang, cot) cua mot token goc o giua canh nen mRoPE van dung.
"""

import torch
import torch.nn.functional as F

from alloc_model import unit_change
from rap_model import uniform_indices


@torch.no_grad()
def scene_ids(feat, H, W, kappa=2.0, local=8, min_len=2, max_len=8):
    """feat: [T, N, D] -> LongTensor [T]: chi so canh cua moi unit (0, 1, 2, ...)."""
    T = feat.shape[0]
    dev = feat.device
    d = unit_change(feat, H, W).float()
    cut = torch.zeros(T, dtype=torch.bool, device=dev)
    for t in range(1, T):
        lo, hi = max(1, t - local), min(T, t + local + 1)
        nb = torch.cat([d[lo:t], d[t + 1:hi]])
        if nb.numel() >= 2 and d[t] > nb.mean() + kappa * nb.std():
            cut[t] = True
    sid, cur, start = torch.zeros(T, dtype=torch.long, device=dev), 0, 0
    for t in range(1, T):
        if (cut[t] and t - start >= min_len) or t - start >= max_len:
            cur, start = cur + 1, t
        sid[t] = cur
    return sid


@torch.no_grad()
def summary_layout(feat, H, W, retention, sid):
    """Tra ve (anchor_idx [K-S], sum_idx [S], sum_emb [S, D]); chi so phang trong T*N."""
    T, N, D = feat.shape
    dev = feat.device
    K = int(uniform_indices(T * N, retention, dev).numel())
    S = int(sid.max()) + 1
    n_anchor = max(1, K - S)
    anchor_idx = torch.unique(torch.linspace(0, T * N - 1, n_anchor, device=dev).round().long(), sorted=True)
    taken = torch.zeros(T * N, dtype=torch.bool, device=dev)
    taken[anchor_idx] = True
    sum_idx, sum_emb = [], []
    for s in range(S):
        units = (sid == s).nonzero(as_tuple=True)[0]
        tok = feat[units].reshape(-1, D).float()
        m = tok.mean(0)
        m = m * (tok.norm(dim=-1).mean() / m.norm().clamp(min=1e-6))
        tm = int(units[len(units) // 2])
        base = tm * N + (H // 2) * W + W // 2
        for off in range(N):  # o giua canh; neu trung neo thi lui sang o ke tiep trong cung unit
            i = tm * N + (base - tm * N + off) % N
            if not taken[i]:
                break
        taken[i] = True
        sum_idx.append(i)
        sum_emb.append(m)
    return anchor_idx, torch.tensor(sum_idx, device=dev, dtype=torch.long), torch.stack(sum_emb)


@torch.no_grad()
def cluster_medoid_indices(feat, H, W, retention, sid, iters=10, lam=1.0):
    """C0 (y HTP-LVU 1.2 + tube MAP-3D 3.2): trong moi canh, k-means toan bo token (dac trung chuan hoa
    + lam * toa do (t, y, x) chuan hoa theo canh), khoi tao tu luoi deu; moi cum giu 1 medoid (token that gan tam
    nhat). So cum moi canh ti le so token cua canh, tong = dung K cua Uniform. Tra ve kept_idx da sap xep."""
    T, N, D = feat.shape
    dev = feat.device
    K = int(uniform_indices(T * N, retention, dev).numel())
    S = int(sid.max()) + 1
    n_units = torch.bincount(sid, minlength=S).float()
    share = n_units / n_units.sum() * K
    ks = share.floor().long().clamp(min=1)
    for i in torch.argsort(share - share.floor(), descending=True)[: max(0, K - int(ks.sum()))]:
        ks[i] += 1
    Fn = F.normalize(feat.float(), dim=-1)
    yy = (torch.arange(N, device=dev) // W).float() / max(1, H - 1)
    xx = (torch.arange(N, device=dev) % W).float() / max(1, W - 1)
    kept = []
    for s in range(S):
        units = (sid == s).nonzero(as_tuple=True)[0]
        Ts = units.numel()
        tt = (torch.arange(Ts, device=dev).float() / max(1, Ts - 1)).repeat_interleave(N)
        P = torch.cat([Fn[units].reshape(-1, D), lam * torch.stack([tt, yy.repeat(Ts), xx.repeat(Ts)], 1)], 1)
        k = min(int(ks[s]), P.shape[0])
        C = P[torch.linspace(0, P.shape[0] - 1, k, device=dev).round().long()].clone()
        for _ in range(iters):
            a = torch.cdist(P, C).argmin(1)
            cnt = torch.bincount(a, minlength=k)
            C = torch.where((cnt > 0).unsqueeze(1),
                            torch.zeros_like(C).index_add_(0, a, P) / cnt.clamp(min=1).unsqueeze(1).float(), C)
        a = torch.cdist(P, C).argmin(1)
        d = (P - C[a]).norm(dim=1)
        order = torch.argsort(a.double() * 1e3 + d.double())  # theo cum, trong cum gan tam truoc
        first = torch.ones_like(order, dtype=torch.bool)
        first[1:] = a[order][1:] != a[order][:-1]
        pick = order[first]  # 1 medoid / cum khong rong
        if pick.numel() < k:  # cum rong -> bu bang token xa tap da chon nhat (farthest point)
            chosen = torch.zeros(P.shape[0], dtype=torch.bool, device=dev)
            chosen[pick] = True
            md = torch.cdist(P, P[pick]).min(1).values.masked_fill(chosen, -1)
            pick = torch.cat([pick, torch.topk(md, k - pick.numel()).indices])
        kept.append(units[0] * N + pick)  # cac unit cua 1 canh lien tiep -> chi so phang = unit dau * N + vi tri trong canh
    return torch.unique(torch.cat(kept), sorted=True)


def merge_kept(anchor_idx, anchor_emb, sum_idx, sum_emb):
    """Ghep neo va token tom tat, sap xep theo chi so (build_pruned_sequence can thu tu tang dan)."""
    idx = torch.cat([anchor_idx, sum_idx])
    emb = torch.cat([anchor_emb, sum_emb.to(anchor_emb.dtype)])
    order = torch.argsort(idx)
    return idx[order], emb[order]
