"""Mass-difference attention bias — the thesis core. [STUB, NoBias CONCRETE]

The attention bias is a swappable *strategy* that turns the per-spectrum Δm graph into either an
additive bias on the QKᵀ logits (Graphormer-style ``EdgeBias``) or a connectivity mask
(``SparseMask``). Phase 1 of the project uses ``NoBias`` (a plain transformer) to establish the
performance floor; Phase 2 swaps in ``EdgeBias`` / ``SparseMask`` via config — no other code change.

Interface contract: ``forward(logits, delta_edges, n_peaks) -> logits'`` where ``logits`` is the
pre-softmax attention score tensor (batch, heads, n_peaks, n_peaks) and ``delta_edges`` is a
:class:`~gems.vocab.delta_graph.DeltaEdges` (or None).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionBias(nn.Module):
    """Base strategy: given QKᵀ logits and a Δm graph, return modified logits."""

    def forward(self, logits: torch.Tensor, delta_edges, n_peaks: int) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError


class NoBias(AttentionBias):
    """Phase-1 baseline: identity — plain transformer attention. [CONCRETE]

    Returns the logits unchanged (ignores the Δm graph). This is what the Phase 1 → 2 ablation is
    measured against; if masked-peak pretraining on top of ``NoBias`` already beats the KMD/van
    Krevelen baseline, the Δm-attention mechanism must add measurable value to justify itself.
    """

    def forward(self, logits: torch.Tensor, delta_edges=None, n_peaks: int | None = None) -> torch.Tensor:
        return logits


class EdgeBias(AttentionBias):
    """Graphormer-style learnable per-building-block bias, scaled by Δm abundance weight. [STUB]

    For each edge (i, j) linked by building block ``b`` with corpus abundance weight ``w``, add a
    learned scalar ``bias[b, head]`` (optionally ``× w``) to ``logits[:, head, i, j]``. The learnable
    per-block bias means mixture-specific blocks can emerge; the abundance weighting is the novelty.

    Args:
        n_blocks: vocabulary size (number of building blocks).
        n_heads: attention heads (per-head bias).
        use_abundance_weight: multiply the learned bias by the edge's abundance weight.
    """

    def __init__(self, n_blocks: int, n_heads: int, use_abundance_weight: bool = True):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.use_abundance_weight = use_abundance_weight
        self.bias = nn.Parameter(torch.zeros(n_blocks, n_heads))  # learned per-block, per-head

    def forward(self, logits: torch.Tensor, delta_edges, n_peaks: int) -> torch.Tensor:
        raise NotImplementedError(
            "EdgeBias.forward is a stub: scatter self.bias[delta_edges.block_idx] (× weight if "
            "use_abundance_weight) into a (heads, n_peaks, n_peaks) bias and add to logits."
        )


class SparseMask(AttentionBias):
    """Restrict attention to peak pairs linked by a building-block Δm. [STUB]

    Adds ``-inf`` to logits for non-edges (so softmax zeroes them). Simultaneously the inductive bias
    and the tractability fix — turns dense O(n^2) attention into a sparse Δm-graph. Likely the primary
    path at full FT-ICR scale.
    """

    def __init__(self, allow_self: bool = True):
        super().__init__()
        self.allow_self = allow_self

    def forward(self, logits: torch.Tensor, delta_edges, n_peaks: int) -> torch.Tensor:
        raise NotImplementedError(
            "SparseMask.forward is a stub: build a boolean edge mask from delta_edges and set "
            "logits[~mask] = -inf (keeping the diagonal if allow_self)."
        )


# Registry for config-driven dispatch. [CONCRETE]
ATTENTION_BIASES: dict[str, type[AttentionBias]] = {
    "no_bias": NoBias,
    "edge_bias": EdgeBias,
    "sparse_mask": SparseMask,
}


def build_attention_bias(cfg, vocab=None) -> AttentionBias:
    """Instantiate the attention-bias strategy named by ``cfg.variant``. [CONCRETE]

    Args:
        cfg: object/dict with ``variant`` in ``ATTENTION_BIASES`` and, for edge_bias, ``n_heads``.
        vocab: a :class:`DeltaVocabulary`; supplies ``n_blocks`` for edge_bias.
    """
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    variant = get("variant", "no_bias")
    if variant not in ATTENTION_BIASES:
        raise KeyError(f"Unknown attention variant {variant!r}; choices: {sorted(ATTENTION_BIASES)}")

    if variant == "edge_bias":
        n_blocks = len(vocab) if vocab is not None else int(get("n_blocks", 0))
        return EdgeBias(n_blocks=n_blocks, n_heads=int(get("n_heads", 1)),
                        use_abundance_weight=bool(get("use_abundance_weight", True)))
    if variant == "sparse_mask":
        return SparseMask(allow_self=bool(get("allow_self", True)))
    return NoBias()
