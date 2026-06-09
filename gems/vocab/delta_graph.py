"""Per-spectrum Δm graph — the bridge from a peak list to the attention bias. [STUB]

For one spectrum, find peak pairs (i, j) whose mass difference matches a building block within a
ppm window, and tag each edge with the block index and that block's abundance weight. The result
(:class:`DeltaEdges`) is consumed by the attention bias: as an additive per-pair logit bias
(edge-bias variant) or as a connectivity mask (sparse variant).

This can reuse pyc2mc's pairwise machinery (``MassDifferencesCompute``) to avoid a naive O(n^2)
Python loop; at FT-ICR scale the sparse construction is mandatory, not optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gems.definitions import DEFAULT_PPM_TOLERANCE
from gems.vocab.building_blocks import DeltaVocabulary


@dataclass
class DeltaEdges:
    """Sparse Δm-graph for one spectrum.

    Attributes:
        src, dst: paired peak indices (int arrays), one entry per matched edge.
        block_idx: vocabulary index of the building block linking each edge.
        weight: abundance weight of that block (the novel abundance weighting).
        n_peaks: number of peaks (token count) the indices refer to.
    """

    src: np.ndarray
    dst: np.ndarray
    block_idx: np.ndarray
    weight: np.ndarray
    n_peaks: int

    def to_dense_bias(self, n_heads: int):
        """Materialize an (n_heads, n_peaks, n_peaks) additive bias tensor. [STUB]

        For the Graphormer-style edge-bias variant. Sparse → dense; only feasible for small
        ``n_peaks`` (dev). The learned per-block, per-head bias is added by the attention module.
        """
        raise NotImplementedError("DeltaEdges.to_dense_bias is a stub.")

    def to_sparse_mask(self):
        """Return a sparse boolean connectivity mask (n_peaks, n_peaks). [STUB]

        For the sparse-attention variant: True where peaks are linked by a building-block Δm.
        """
        raise NotImplementedError("DeltaEdges.to_sparse_mask is a stub.")


def pairwise_delta_m(mz: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise |m_i - m_j| (or a sparse edge list at scale). [STUB]"""
    raise NotImplementedError("pairwise_delta_m is a stub (use pyc2mc MassDifferencesCompute at scale).")


def build_delta_graph(
    mz: np.ndarray,
    vocab: DeltaVocabulary,
    ppm_tol: float = DEFAULT_PPM_TOLERANCE,
    include_c13: bool = True,
) -> DeltaEdges:
    """Build the Δm edge set for one spectrum's peaks. [STUB]

    Intended: for each candidate peak pair, compute Δm; if it matches a vocabulary block within the
    ppm window (``vocab.match``), emit an edge tagged with that block's index and weight. Drop or
    flag ¹³C edges per ``include_c13``.
    """
    raise NotImplementedError(
        "build_delta_graph is a stub: match pairwise Δm against vocab within ppm_tol → DeltaEdges."
    )
