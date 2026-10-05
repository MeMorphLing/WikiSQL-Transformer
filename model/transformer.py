"""Full encoder-decoder Transformer (Vaswani et al., 2017) and its masks.

Inputs come from the starter InputLayer (scaled shared embedding + sinusoidal
positions + dropout). The same TokenEmbedding instance feeds the encoder, the
decoder and (transposed) the output projection: three-way weight sharing.
"""
import torch
from embeddings import (  # starter module (starter/ is on sys.path)
    InputLayer,
    TokenEmbedding,
)
from torch import nn

from .layers import DecoderLayer, EncoderLayer


# ----------------------------------------------------------------------- masks
def make_src_mask(src, pad_id: int = 0):
    """Padding mask over source keys: (B, 1, 1, S). Used by encoder
    self-attention and decoder cross-attention."""
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(length: int, device=None):
    """Lower-triangular mask (1, 1, T, T): position t sees positions <= t only."""
    m = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt, pad_id: int = 0):
    """Decoder self-attention mask: padding mask AND causal mask -> (B, 1, T, T)."""
    pad = (tgt != pad_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
    return pad & make_causal_mask(tgt.size(1), tgt.device)


# ---------------------------------------------------------------------- stacks
class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, y, memory, src_mask, tgt_mask):
        for layer in self.layers:
            y = layer(y, memory, src_mask, tgt_mask)
        return y


# ----------------------------------------------------------------- full model
class Transformer(nn.Module):
    def __init__(self, vocab_size=8000, d_model=256, n_heads=4, n_layers=3,
                 d_ff=1024, dropout=0.1, max_len=512, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.shared = TokenEmbedding(vocab_size, d_model, pad_id)
        self.src_input = InputLayer(self.shared, d_model, max_len, dropout)
        self.tgt_input = InputLayer(self.shared, d_model, max_len, dropout)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        self.generator.weight = self.shared.emb.weight  # weight sharing (same tensor)
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():  # tied weight is visited once
            if name == "shared.emb.weight":
                # N(0, d^-1/2): after the sqrt(d_model) scaling, embeddings have
                # unit variance, matching the scale of the positional encodings.
                nn.init.normal_(p, mean=0.0, std=self.d_model ** -0.5)
                with torch.no_grad():
                    p[self.pad_id].zero_()
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -- pieces used by decoding -------------------------------------------
    def encode(self, src, src_mask=None):
        src_mask = make_src_mask(src, self.pad_id) if src_mask is None else src_mask
        return self.encoder(self.src_input(src), src_mask), src_mask

    def decode(self, tgt_in, memory, src_mask):
        tgt_mask = make_tgt_mask(tgt_in, self.pad_id)
        return self.generator(self.decoder(self.tgt_input(tgt_in), memory, src_mask, tgt_mask))

    def forward(self, src, tgt_in):
        """src: (B, S), tgt_in: (B, T) -> logits (B, T, vocab)."""
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)

    def cross_attention_maps(self):
        """Cross-attention weights of every decoder layer from the last forward pass."""
        return [layer.cross_attn.attn_weights for layer in self.decoder.layers]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
