"""Transformer encoder with a pluggable attention bias. [CONCRETE]

A pre-norm transformer encoder (templated on DreaMS's ``tnq_transformer.py``) whose self-attention
adds an :class:`~gems.models.layers.attention_bias.AttentionBias` to the QKᵀ logits before softmax.
The attention bias is the single seam where the locked ``graph`` mechanism and the ``no_bias`` /
``dense_edge_bias`` sanity checks differ — the rest of the stack is identical. The readout is
**attention-pooling** over the final peak embeddings (no CLS master node, which would need edges to
all peaks and break the Δm-graph sparsity).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gems.models.layers.attention_bias import AttentionBias, NoBias
from gems.models.layers.feed_forward import FeedForward

NEG_INF = float("-inf")


class DeltaTransformerLayer(nn.Module):
    """One pre-norm encoder block: biased multi-head self-attention + feed-forward. [CONCRETE]

    Args:
        dim: model dimension.
        n_heads: attention heads.
        ff_mult: feed-forward expansion factor.
        dropout: dropout probability.
        attention_bias: the Δm attention-bias strategy applied to QKᵀ logits (default ``NoBias``).
    """

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0,
                 attention_bias: AttentionBias | None = None):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.attention_bias = attention_bias if attention_bias is not None else NoBias()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(dim, dim, hidden_dims=[dim * ff_mult], dropout=dropout)

    def _self_attention(self, x: torch.Tensor, delta_edges, key_padding_mask) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (b, heads, n, head_dim)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, heads, n, n)
        logits = self.attention_bias(logits, delta_edges, n)

        if key_padding_mask is not None:
            # key_padding_mask: (b, n) with True == padding (ignore as a key).
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], NEG_INF)

        attn = torch.softmax(logits, dim=-1)
        # A fully-padded query row would be all -inf -> NaN after softmax; zero those rows.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (b, heads, n, head_dim)
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor, delta_edges=None, key_padding_mask=None) -> torch.Tensor:
        x = x + self._self_attention(self.norm1(x), delta_edges, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DeltaTransformerEncoder(nn.Module):
    """Stack of :class:`DeltaTransformerLayer` + attention-pooling readout. [CONCRETE]

    Args:
        dim, n_layers, n_heads, ff_mult, dropout: standard transformer hyperparameters.
        attention_bias: shared bias module (same instance across layers).
        pooling: 'attention' (attention pooling — the locked readout). 'cls' is intentionally not a
            path here: a CLS master node would need edges to every peak and break Δm-graph sparsity.
    """

    def __init__(self, dim: int, n_layers: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.0, attention_bias: AttentionBias | None = None,
                 pooling: str = "attention"):
        super().__init__()
        if pooling != "attention":
            raise ValueError(f"pooling={pooling!r} unsupported; only 'attention' is implemented.")
        self.dim = dim
        self.pooling = pooling
        self.attention_bias = attention_bias
        self.layers = nn.ModuleList(
            DeltaTransformerLayer(dim, n_heads, ff_mult, dropout, attention_bias)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        # Attention-pooling: a learned query scores each peak; softmax over valid peaks → pooled z.
        self.pool_query = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.pool_query, std=dim ** -0.5)

    def forward(self, x: torch.Tensor, delta_edges=None, key_padding_mask=None):
        """Return (peak_embeddings (B,N,dim), pooled_embedding (B,dim)). [CONCRETE]"""
        for layer in self.layers:
            x = layer(x, delta_edges=delta_edges, key_padding_mask=key_padding_mask)
        peak_emb = self.final_norm(x)

        scores = (peak_emb @ self.pool_query) / math.sqrt(self.dim)  # (B, N)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, NEG_INF)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0).unsqueeze(-1)  # (B, N, 1)
        pooled = (weights * peak_emb).sum(dim=1)  # (B, dim)
        return peak_emb, pooled
