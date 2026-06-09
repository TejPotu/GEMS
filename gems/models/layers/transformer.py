"""Transformer encoder with a pluggable attention bias. [STUB]

A pre-norm transformer encoder (templated on DreaMS's ``tnq_transformer.py``) whose self-attention
adds an :class:`~gems.models.layers.attention_bias.AttentionBias` to the QKᵀ logits before
softmax. This is the single seam where Phase 1 (``NoBias``) and Phase 2 (``EdgeBias``/``SparseMask``)
differ — the rest of the stack is identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gems.models.layers.attention_bias import AttentionBias


class DeltaTransformerLayer(nn.Module):
    """One pre-norm encoder block: biased multi-head self-attention + feed-forward. [STUB]

    Args:
        dim: model dimension.
        n_heads: attention heads.
        ff_mult: feed-forward expansion factor.
        dropout: dropout probability.
        attention_bias: the Δm attention-bias strategy applied to QKᵀ logits.
    """

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0,
                 attention_bias: AttentionBias | None = None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.attention_bias = attention_bias
        # TODO[STUB]: norms, qkv projection, output projection, FeedForward(dim, dim, [dim*ff_mult]).

    def forward(self, x: torch.Tensor, delta_edges=None, key_padding_mask=None) -> torch.Tensor:
        """Self-attention (with Δm bias) + FFN, pre-norm residual. [STUB]

        Compute QKᵀ/√d → ``attention_bias(logits, delta_edges, n_peaks)`` → mask padding →
        softmax → weighted V → out-proj → residual; then pre-norm FFN + residual.
        """
        raise NotImplementedError("DeltaTransformerLayer.forward is a stub.")


class DeltaTransformerEncoder(nn.Module):
    """Stack of :class:`DeltaTransformerLayer` + pooling for a whole-spectrum embedding. [STUB]

    Args:
        dim, n_layers, n_heads, ff_mult, dropout: standard transformer hyperparameters.
        attention_bias: shared/strategy bias module (same instance across layers, or per-layer).
        pooling: 'cls' (prepend a learned token) or 'attention' (attention pooling).
    """

    def __init__(self, dim: int, n_layers: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.0, attention_bias: AttentionBias | None = None,
                 pooling: str = "cls"):
        super().__init__()
        self.dim = dim
        self.pooling = pooling
        self.attention_bias = attention_bias
        self.layers = nn.ModuleList(
            DeltaTransformerLayer(dim, n_heads, ff_mult, dropout, attention_bias)
            for _ in range(n_layers)
        )
        # TODO[STUB]: CLS parameter / attention-pooling head per `pooling`.

    def forward(self, x: torch.Tensor, delta_edges=None, key_padding_mask=None):
        """Return (peak_embeddings (B,N,dim), pooled_embedding (B,dim)). [STUB]"""
        raise NotImplementedError("DeltaTransformerEncoder.forward is a stub.")
