"""Mass-difference attention bias — the thesis core. [STUB, NoBias CONCRETE]

The attention bias turns the per-spectrum Δm graph into a modification of the QKᵀ logits. The locked
design (BUILD_PLAN A3) is a single mechanism — ``GraphDeltaBias`` — that does **both** at once:

    A_ij = (qᵢ·kⱼ)/√d + b(Δm_type_ij, log_abund_ij)   for graph-linked (i, j), −∞ otherwise

i.e. attention is **masked to Δm edges** *and* each allowed pair gets a learned, abundance-weighted
per-building-block bias. The edge set already encodes "which Δm", so the mask and the bias come from
the same graph. ``NoBias`` (plain transformer) is the performance floor; ``DenseEdgeBias`` is a
reduced-N sanity check (dense edge bias with no sparsity), never the primary path.

Leakage guard: edges incident to a masked peak arrive tagged with ``vocab.masked_edge_index``, so the
bias for those edges comes from the reserved ``[masked-edge]`` row — the model learns a masked peak is
*related* to its neighbors, never *which* block links them.

Interface contract: ``forward(logits, delta_edges, n_peaks) -> logits'`` where ``logits`` is the
pre-softmax score tensor (batch, heads, n_peaks, n_peaks) and ``delta_edges`` is a
:class:`~gems.vocab.graph.DeltaEdges` (or None).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionBias(nn.Module):
    """Base strategy: given QKᵀ logits and a Δm graph, return modified logits."""

    def forward(self, logits: torch.Tensor, delta_edges, n_peaks: int) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError


class NoBias(AttentionBias):
    """Performance floor: identity — plain transformer attention. [CONCRETE]

    Returns the logits unchanged (ignores the Δm graph). This is the sanity floor the locked
    ``graph`` attention must beat; if masked-peak pretraining on top of ``NoBias`` already matched the
    KMD/van-Krevelen baseline, the Δm-attention mechanism would not be earning its keep.
    """

    def forward(self, logits: torch.Tensor, delta_edges=None, n_peaks: int | None = None) -> torch.Tensor:
        return logits


class GraphDeltaBias(AttentionBias):
    """Locked primary: sparse Δm-graph mask + abundance-weighted per-block edge bias. [STUB]

    For each edge (i, j) linked by building block ``b`` with corpus abundance weight ``w``, add a
    learned scalar ``bias[b, head]`` (× ``w`` if ``use_abundance_weight``) to ``logits[:, head, i, j]``;
    every non-edge pair is set to −∞ so softmax zeroes it. The learnable per-block bias lets
    mixture-specific blocks emerge; the abundance weighting is the novelty over DreaMS. The bias table
    reserves a final ``[masked-edge]`` row for leakage-guarded edges (size = ``vocab.n_embeddings``).

    Args:
        n_blocks: number of embedding rows = real blocks + 1 masked-edge sentinel (``vocab.n_embeddings``).
        n_heads: attention heads (per-head bias).
        use_abundance_weight: multiply the learned bias by the edge's abundance weight.
        allow_self: keep the diagonal (self-attention) unmasked.
    """

    def __init__(self, n_blocks: int, n_heads: int, use_abundance_weight: bool = True,
                 allow_self: bool = True):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.use_abundance_weight = use_abundance_weight
        self.allow_self = allow_self
        self.bias = nn.Parameter(torch.zeros(n_blocks, n_heads))  # learned per-block, per-head (+sentinel)

    def forward(self, logits: torch.Tensor, delta_edges, n_peaks: int) -> torch.Tensor:
        raise NotImplementedError(
            "GraphDeltaBias.forward is a stub: set non-edge logits to -inf (keep diagonal if "
            "allow_self), then scatter self.bias[delta_edges.block_idx] (× weight if "
            "use_abundance_weight) onto the surviving edge logits."
        )


class DenseEdgeBias(AttentionBias):
    """Reduced-N sanity check: dense Graphormer-style edge bias, NO sparsity. [STUB]

    Same learned per-block, per-head (abundance-weighted) bias as ``GraphDeltaBias`` but added to a
    fully dense attention map — only feasible for small ``n_peaks``. Kept to isolate "does the bias
    help?" from "does the sparsity help?"; not the primary path at FT-ICR scale.

    Args:
        n_blocks: embedding rows = real blocks + masked-edge sentinel (``vocab.n_embeddings``).
        n_heads: attention heads.
        use_abundance_weight: multiply the learned bias by the edge's abundance weight.
    """

    def __init__(self, n_blocks: int, n_heads: int, use_abundance_weight: bool = True):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.use_abundance_weight = use_abundance_weight
        self.bias = nn.Parameter(torch.zeros(n_blocks, n_heads))

    def forward(self, logits: torch.Tensor, delta_edges, n_peaks: int) -> torch.Tensor:
        raise NotImplementedError(
            "DenseEdgeBias.forward is a stub: scatter self.bias[delta_edges.block_idx] (× weight if "
            "use_abundance_weight) into a dense (heads, n_peaks, n_peaks) bias and add to logits "
            "(no -inf masking)."
        )


# Registry for config-driven dispatch. [CONCRETE]
ATTENTION_BIASES: dict[str, type[AttentionBias]] = {
    "no_bias": NoBias,             # performance floor
    "graph": GraphDeltaBias,       # locked primary (sparse mask + abundance-weighted edge bias)
    "dense_edge_bias": DenseEdgeBias,  # reduced-N sanity check
}


def build_attention_bias(cfg, vocab=None) -> AttentionBias:
    """Instantiate the attention-bias strategy named by ``cfg.variant``. [CONCRETE]

    Args:
        cfg: object/dict with ``variant`` in ``ATTENTION_BIASES`` and, for the biased variants,
            ``n_heads`` / ``use_abundance_weight`` / ``allow_self``.
        vocab: a :class:`DeltaVocabulary`; supplies ``n_embeddings`` (blocks + masked-edge sentinel).
    """
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    variant = get("variant", "graph")
    if variant not in ATTENTION_BIASES:
        raise KeyError(f"Unknown attention variant {variant!r}; choices: {sorted(ATTENTION_BIASES)}")

    if variant == "no_bias":
        return NoBias()

    n_blocks = vocab.n_embeddings if vocab is not None else int(get("n_blocks", 0))
    n_heads = int(get("n_heads", 1))
    use_w = bool(get("use_abundance_weight", True))
    if variant == "dense_edge_bias":
        return DenseEdgeBias(n_blocks=n_blocks, n_heads=n_heads, use_abundance_weight=use_w)
    return GraphDeltaBias(n_blocks=n_blocks, n_heads=n_heads, use_abundance_weight=use_w,
                          allow_self=bool(get("allow_self", True)))
