"""Section 3.1 and 3.3: feed-forward network, encoder layer and decoder layer.

Post-norm residual blocks exactly as in the paper: LayerNorm(x + Dropout(Sublayer(x))).
"""
from torch import nn

from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2, applied to every position independently."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))


class EncoderLayer(nn.Module):
    """self-attention -> add & norm -> FFN -> add & norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """masked self-attention -> add & norm -> cross-attention -> add & norm -> FFN -> add & norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, y, memory, src_mask, tgt_mask):
        y = self.norm1(y + self.drop1(self.self_attn(y, y, y, tgt_mask)))
        y = self.norm2(y + self.drop2(self.cross_attn(y, memory, memory, src_mask)))
        y = self.norm3(y + self.drop3(self.ffn(y)))
        return y
