import math

import torch  # type: ignore[reportMissingImports]
from torch import nn

# a boolean tensor that broadcasts to the score shape (B, h, L_q, L_k); 
# True = "may attend", False = "hidden"

class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V."""

    def forward(self, q, k, v, mask=None):
        # q: (B, h, Lq, d_k)   k: (B, h, Lk, d_k)   v: (B, h, Lk, d_v)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, Lq, Lk)
        if mask is not None:
            # Most negative finite value of the current dtype: behaves like -inf
            # after softmax (exp underflows to exactly 0) but never produces NaN,
            # and stays valid under fp16 autocast.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn  # (B, h, Lq, d_v), (B, h, Lq, Lk)


class MultiHeadAttention(nn.Module):
    """MultiHead(Q, K, V) = Concat(head_1..head_h) W^O,
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).

    The h per-head projections are packed into one (d_model x h*d_k) matrix,
    which is mathematically identical to h separate (d_model x d_k) matrices.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.attn_weights = None  # last attention map (B, h, Lq, Lk), detached, for analysis

    def _split(self, x):
        b, l, _ = x.shape  # (B, L, d_model) -> (B, h, L, d_k)
        return x.view(b, l, self.h, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))
        out, attn = self.attention(q, k, v, mask)
        self.attn_weights = attn.detach()
        b, _, lq, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, lq, self.h * self.d_k)  # concat heads
        return self.w_o(out)
