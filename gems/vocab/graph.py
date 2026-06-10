"""Per-spectrum Δm graph (``vocab/graph.py``) — peak list → attention bias. [STUB]

For one spectrum, find peak pairs (i, j) whose mass difference matches a building block within a
ppm window, and tag each edge with the block index and that block's abundance weight. The result
(:class:`DeltaEdges`) drives both halves of the locked attention (BUILD_PLAN A3): it is the
connectivity **mask** *and* the source of the abundance-weighted **edge bias** — the same edge set
encodes "which Δm" and "how abundant".

Two graph-level commitments from the plan live here:
  - **Degree cap** ``k`` (default :data:`gems.definitions.DEFAULT_DEGREE_CAP`): keep only the top-k
    edges per node by Δm abundance, which is what guarantees linear (not O(n^2)) attention cost.
  - **Graph-induced selection**: the node set is *induced* by the edges — a peak is a token iff it
    sits on >=1 abundant edge (see :mod:`gems.data.peak_selection`). There is no Top-N stage.

This can reuse pyc2mc's pairwise machinery (``MassDifferencesCompute``) to avoid a naive O(n^2)
Python loop; at FT-ICR scale the sparse construction is mandatory, not optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gems.definitions import DEFAULT_DEGREE_CAP, DEFAULT_PPM_TOLERANCE
from gems.vocab.vocabulary import DeltaVocabulary


@dataclass
class DeltaEdges:
    """Sparse Δm-graph for one spectrum.

    Attributes:
        src, dst: paired peak indices (int arrays), one entry per matched edge.
        block_idx: vocabulary index of the building block linking each edge (``masked_edge_index``
            for edges stripped by the leakage guard).
        weight: abundance weight of that block (the novel abundance weighting).
        n_peaks: number of peaks (token count) the indices refer to.
    """

    src: np.ndarray
    dst: np.ndarray
    block_idx: np.ndarray
    weight: np.ndarray
    n_peaks: int

    def node_mask(self) -> np.ndarray:
        """Boolean (n_peaks,) — True for peaks that sit on >=1 edge. [STUB]

        The basis for graph-induced peak selection: a peak is a node iff it is incident to an edge.
        """
        raise NotImplementedError("DeltaEdges.node_mask is a stub (peaks touched by any edge).")

    def with_masked_edges(self, masked_nodes: np.ndarray, masked_edge_index: int) -> "DeltaEdges":
        """Apply the pre-training leakage guard (BUILD_PLAN A2/B1). [STUB]

        Return a copy where every edge incident to a peak in ``masked_nodes`` keeps its
        (src, dst) connectivity but has its ``block_idx`` rewritten to ``masked_edge_index`` and its
        ``weight`` zeroed — so the masked node still aggregates from neighbors, but a typed edge can
        never reveal *which* building block (hence the masked mass) links them. Edges between two
        unmasked peaks are unchanged.

        Args:
            masked_nodes: boolean (n_peaks,) or int index array of masked peaks.
            masked_edge_index: the sentinel block id (``vocab.masked_edge_index``).
        """
        raise NotImplementedError(
            "DeltaEdges.with_masked_edges is a stub: strip block_idx→sentinel and weight→0 on every "
            "edge touching a masked peak; leave unmasked-unmasked edges intact."
        )

    def to_dense_bias(self, n_heads: int):
        """Materialize an (n_heads, n_peaks, n_peaks) additive bias tensor. [STUB]

        For the reduced-N ``dense_edge_bias`` sanity check only. Sparse → dense; feasible just for
        small ``n_peaks``. The learned per-block, per-head bias is added by the attention module.
        """
        raise NotImplementedError("DeltaEdges.to_dense_bias is a stub.")

    def to_sparse_mask(self):
        """Return a sparse boolean connectivity mask (n_peaks, n_peaks). [STUB]

        For the locked ``graph`` attention: True where peaks are linked by a building-block Δm
        (plus the diagonal). Non-edges are masked out before softmax.
        """
        raise NotImplementedError("DeltaEdges.to_sparse_mask is a stub.")


def pairwise_delta_m(mz: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise |m_i - m_j| (or a sparse edge list at scale). [STUB]"""
    raise NotImplementedError("pairwise_delta_m is a stub (use pyc2mc MassDifferencesCompute at scale).")


def build_delta_graph(
    mz: np.ndarray,
    vocab: DeltaVocabulary,
    ppm_tol: float = DEFAULT_PPM_TOLERANCE,
    degree_cap: int = DEFAULT_DEGREE_CAP,
    include_c13: bool = True,
) -> DeltaEdges:
    """Build the Δm edge set for one spectrum's peaks. [STUB]

    Intended: for each candidate peak pair, compute Δm; if it matches a vocabulary block within the
    ppm window (``vocab.match``), emit an edge tagged with that block's index and abundance weight.
    Then apply the **degree cap**: keep only the top ``degree_cap`` edges per node by abundance
    weight (the linear-cost knob). Drop or flag ¹³C edges per ``include_c13``.
    """
    raise NotImplementedError(
        "build_delta_graph is a stub: match pairwise Δm against vocab within ppm_tol, tag with "
        "block index + abundance weight, cap to top-`degree_cap` edges/node → DeltaEdges."
    )
