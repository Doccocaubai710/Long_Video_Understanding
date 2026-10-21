"""
place_model.py -- hoc VI TRI dat neo cho RAP (AnchorPlacer), train bang GRPO (train_place.py).

RAP (rap_model.py) chi hoc NOI DUNG cua neo; vi tri neo luon la luoi Uniform.
AnchorPlacer cham diem moi token goc, va trong MOI unit chon dung so neo ma
Uniform dat o unit do (ngan sach theo thoi gian giu nguyen, khong the don het
neo vao vai frame -- loi "sup do" cua DSTA-Scorer truoc day).

  logit[t, n] = s_theta(token) + prior * [n la vi tri Uniform]
  s_theta     : LN(x) || LN(x_t - x_{t-1}) -> Linear -> 1 lop self-attn trong unit -> Linear (khoi tao 0)

Luc khoi tao s_theta = 0 nen chon greedy (top-k logit) TRUNG KHOP Uniform,
giong y tuong "delta khoi tao 0" cua RAP. Lay mau = Gumbel top-k (tuong duong
Plackett-Luce khong hoan lai), log-prob tinh dung theo thu tu lay.
"""

import torch
import torch.nn as nn

from rap_model import MAX_GRID, uniform_indices


class AnchorPlacer(nn.Module):
    def __init__(self, dim, d_model=512, num_heads=8, prior=4.0):
        super().__init__()
        self.dim, self.d_model, self.num_heads, self.prior = dim, d_model, num_heads, prior
        self.ln_x = nn.LayerNorm(dim)
        self.ln_d = nn.LayerNorm(dim)
        self.inp = nn.Linear(2 * dim, d_model)
        self.row_emb = nn.Embedding(MAX_GRID, d_model)
        self.col_emb = nn.Embedding(MAX_GRID, d_model)
        nn.init.normal_(self.row_emb.weight, std=0.02)
        nn.init.normal_(self.col_emb.weight, std=0.02)
        self.ln_a = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln_h = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.head[-1].weight)  # s = 0 luc khoi tao -> greedy == Uniform
        nn.init.zeros_(self.head[-1].bias)

    def config(self):
        return {"dim": self.dim, "d_model": self.d_model, "num_heads": self.num_heads, "prior": self.prior}

    def forward(self, X, W):
        """X: [T, N, D] float32. Tra ve diem hoc duoc s [T, N] (chua cong prior)."""
        T, N, _ = X.shape
        dev = X.device
        prev = torch.cat([X[:1], X[:-1]], dim=0)
        h = self.inp(torch.cat([self.ln_x(X), self.ln_d(X - prev)], dim=-1))
        rows = (torch.arange(N, device=dev) // W).clamp(max=MAX_GRID - 1)
        cols = (torch.arange(N, device=dev) % W).clamp(max=MAX_GRID - 1)
        h = h + (self.row_emb(rows) + self.col_emb(cols)).unsqueeze(0)
        a = self.ln_a(h)
        h = h + self.attn(a, a, a, need_weights=False)[0]
        return self.head(self.ln_h(h)).squeeze(-1)


def uniform_layout(T, N, retention, device):
    """Chi so Uniform, mat na Uniform [T, N] va so neo moi unit [T]."""
    idx = uniform_indices(T * N, retention, device)
    mask = torch.zeros(T * N, dtype=torch.bool, device=device)
    mask[idx] = True
    counts = torch.bincount(idx // N, minlength=T)
    return idx, mask.view(T, N), counts


def place(logits, counts, sample, gen=None):
    """Chon counts[t] token trong moi unit t.
    logits [T, N]. sample=False: top-k (greedy). sample=True: Gumbel top-k.
    Tra ve (kept_idx phang da sap xep [K], logp tong [] hoac None, so luot chon)."""
    T, N = logits.shape
    kmax = int(counts.max())
    if sample:
        u = torch.rand(logits.shape, device=logits.device, generator=gen).clamp_(1e-10, 1 - 1e-10)
        order = (logits.detach() - torch.log(-torch.log(u))).topk(kmax, dim=1).indices   # [T, kmax]
    else:
        order = logits.detach().topk(kmax, dim=1).indices
    valid = torch.arange(kmax, device=logits.device).unsqueeze(0) < counts.unsqueeze(1)   # [T, kmax]
    flat = (torch.arange(T, device=logits.device).unsqueeze(1) * N + order)[valid]
    kept_idx = flat.sort().values
    if not sample:
        return kept_idx, None, int(valid.sum())
    # log P(thu tu) = sum_j [l(a_j) - logsumexp(l tren cac token chua chon)]
    picked = logits.gather(1, order)                                   # [T, kmax]
    taken = torch.zeros_like(logits, dtype=torch.bool)
    lse = []
    for j in range(kmax):
        lse.append(torch.logsumexp(logits.masked_fill(taken, float("-inf")), dim=1))
        taken = taken.scatter(1, order[:, j:j + 1], True)
    lse = torch.stack(lse, dim=1)
    logp = ((picked - lse) * valid).sum()
    return kept_idx, logp, int(valid.sum())


def save_placer(path, placer, step, extra=None):
    torch.save({"state_dict": placer.state_dict(), "config": placer.config(), "step": step, **(extra or {})}, path)


def load_placer(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    placer = AnchorPlacer(**ckpt["config"])
    placer.load_state_dict(ckpt["state_dict"])
    return placer.to(device).float(), ckpt
