"""Per-spectrum Δm graph (``vocab/graph.py``) — peak list → attention bias. [CONCRETE]

For one spectrum, find peak pairs (i, j) whose mass difference matches a building block within a
ppm window, and tag each edge with the block index and that block's abundance weight. The result
(:class:`DeltaEdges`) drives both halves of the locked attention (BUILD_PLAN A3): it is the
connectivity **mask** *and* the source of the abundance-weighted **edge bias**.

Two graph-level commitments from the plan live here:
  - **Degree cap** ``k`` (default :data:`gems.definitions.DEFAULT_DEGREE_CAP`): keep only the top-k
    edges per node by Δm abundance — what bounds attention cost.
  - **Graph-induced selection**: the node set is *induced* by the edges (see
    :mod:`gems.data.peak_selection`).

This builds the candidate pairs densely (upper triangle), which is fine after peak selection has
capped the token count. A sorted-window / pyc2mc ``MassDifferencesCompute`` construction is the
scale-up optimization (it avoids materializing the full O(n^2) triangle).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from gems.definitions import DEFAULT_DEGREE_CAP, DEFAULT_PPM_TOLERANCE
from gems.vocab.vocabulary import DeltaVocabulary


@dataclass
class DeltaEdges:
    """Sparse Δm-graph for one spectrum (undirected; each edge stored once with ``src < dst``).

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
        """Boolean (n_peaks,) — True for peaks that sit on >=1 edge. [CONCRETE]"""
        m = np.zeros(self.n_peaks, dtype=bool)
        m[self.src] = True
        m[self.dst] = True
        return m

    def with_masked_edges(self, masked_nodes: np.ndarray, masked_edge_index: int) -> "DeltaEdges":
        """Apply the pre-training leakage guard (BUILD_PLAN A2/B1). [CONCRETE]

        Every edge incident to a masked peak keeps its (src, dst) connectivity but has its
        ``block_idx`` rewritten to ``masked_edge_index`` and its ``weight`` zeroed — so the masked
        node still aggregates from neighbors, but a typed edge can never reveal *which* building block
        (hence the masked mass) links them.
        """
        masked = np.asarray(masked_nodes)
        if masked.dtype != bool:
            tmp = np.zeros(self.n_peaks, dtype=bool)
            tmp[masked] = True
            masked = tmp
        incident = masked[self.src] | masked[self.dst]
        block_idx = self.block_idx.copy()
        weight = self.weight.copy()
        block_idx[incident] = masked_edge_index
        weight[incident] = 0.0
        return DeltaEdges(self.src.copy(), self.dst.copy(), block_idx, weight, self.n_peaks)

    def to_dense_bias(self, n_heads: int):
        """Materialize an (n_heads, n_peaks, n_peaks) additive bias tensor. [STUB]

        Not needed by the attention modules (they scatter the per-edge bias directly); kept as a
        future convenience for the reduced-N ``dense_edge_bias`` debugging path.
        """
        raise NotImplementedError("DeltaEdges.to_dense_bias is unused (attention scatters directly).")

    def to_sparse_mask(self) -> np.ndarray:
        """Return a boolean connectivity mask (n_peaks, n_peaks), incl. the diagonal. [CONCRETE]"""
        mask = np.zeros((self.n_peaks, self.n_peaks), dtype=bool)
        mask[self.src, self.dst] = True
        mask[self.dst, self.src] = True
        np.fill_diagonal(mask, True)
        return mask


def pairwise_delta_m(mz: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise |m_i - m_j| as a flat array (i<j). [CONCRETE]"""
    mz = np.asarray(mz, dtype=np.float64)
    iu, ju = np.triu_indices(mz.shape[0], k=1)
    return np.abs(mz[ju] - mz[iu])


def build_delta_graph(
    mz: np.ndarray,
    vocab: DeltaVocabulary,
    ppm_tol: float = DEFAULT_PPM_TOLERANCE,
    degree_cap: int = DEFAULT_DEGREE_CAP,
    include_c13: bool = True,
) -> DeltaEdges:
    """Build the Δm edge set for one spectrum's peaks. [CONCRETE]

    For each peak pair, match Δm against the vocabulary within a ppm window (nearest block wins),
    tag the edge with the block index + abundance weight, then keep the top ``degree_cap`` edges per
    node by weight (union over endpoints, so connectivity is preserved).
    """
    mz = np.asarray(mz, dtype=np.float64)
    n = mz.shape[0]
    if n < 2:
        e = np.array([], dtype=int)
        return DeltaEdges(e, e, e, np.array([], dtype=float), n)

    iu, ju = np.triu_indices(n, k=1)
    d = np.abs(mz[ju] - mz[iu])

    names = [nm for nm in vocab.names if include_c13 or nm != "C13"]
    masses = {nm: vocab.masses[nm] for nm in names}
    max_mass = max(masses.values())
    within = d <= max_mass * (1.0 + ppm_tol * 1e-6) + 1e-9   # prune impossible pairs early
    iu, ju, d = iu[within], ju[within], d[within]

    block_idx = np.full(d.shape[0], -1, dtype=int)
    weight = np.zeros(d.shape[0], dtype=np.float64)
    for nm in names:                      # blocks are sub-ppm separated → first match is the match
        m = masses[nm]
        half = m * ppm_tol * 1e-6
        sel = (block_idx < 0) & (d >= m - half) & (d <= m + half)
        if sel.any():
            block_idx[sel] = vocab.index_of(nm)
            weight[sel] = float(vocab.weights.get(nm, 0.0))

    keep = block_idx >= 0
    src, dst = iu[keep].astype(int), ju[keep].astype(int)
    block_idx, weight = block_idx[keep], weight[keep]

    if degree_cap and degree_cap > 0 and src.size:
        src, dst, block_idx, weight = _apply_degree_cap(src, dst, block_idx, weight, degree_cap)

    return DeltaEdges(src, dst, block_idx, weight, n)


def _apply_degree_cap(src, dst, block_idx, weight, degree_cap):
    """Keep an edge if it ranks in the top-``degree_cap`` (by weight) of *either* endpoint."""
    incident: dict[int, list[int]] = defaultdict(list)
    for e in range(src.shape[0]):
        incident[int(src[e])].append(e)
        incident[int(dst[e])].append(e)

    keep = np.zeros(src.shape[0], dtype=bool)
    for edges in incident.values():
        top = sorted(edges, key=lambda e: weight[e], reverse=True)[:degree_cap]
        keep[top] = True
    return src[keep], dst[keep], block_idx[keep], weight[keep]
