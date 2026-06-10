"""Transformer encoder with a pluggable attention bias. [STUB]

A pre-norm transformer encoder (templated on DreaMS's ``tnq_transformer.py``) whose self-attention
adds an :class:`~gems.models.layers.attention_bias.AttentionBias` to the QKᵀ logits before softmax.
The attention bias is the single seam where the locked ``graph`` mechanism and the ``no_bias`` /
``dense_edge_bias`` sanity checks differ — the rest of the stack is identical. The readout is
**attention-pooling** over the final peak embeddings (no CLS master node, which would need edges to
all peaks and break the Δm-graph sparsity).
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
        pooling: 'attention' (attention pooling — the locked readout). 'cls' is intentionally not a
            path here: a CLS master node would need edges to every peak and break Δm-graph sparsity.
    """

    def __init__(self, dim: int, n_layers: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.0, attention_bias: AttentionBias | None = None,
                 pooling: str = "attention"):
        super().__init__()
        self.dim = dim
        self.pooling = pooling
        self.attention_bias = attention_bias
        self.layers = nn.ModuleList(
            DeltaTransformerLayer(dim, n_heads, ff_mult, dropout, attention_bias)
            for _ in range(n_layers)
        )
        # TODO[STUB]: attention-pooling head (learned query over final peak embeddings).

    def forward(self, x: torch.Tensor, delta_edges=None, key_padding_mask=None):
        """Return (peak_embeddings (B,N,dim), pooled_embedding (B,dim)). [STUB]"""
        raise NotImplementedError("DeltaTransformerEncoder.forward is a stub.")
