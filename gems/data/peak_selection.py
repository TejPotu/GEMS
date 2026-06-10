"""Peak selection — graph-induced node set (BUILD_PLAN locked decision #1). [STUB + CONCRETE registry]

FT-ICR spectra carry 10^4–10^5 peaks; attention cannot consume them all. The locked decision is that
selection is **graph-induced**: a peak becomes a token iff it sits on >=1 abundant Δm edge. There is
no standalone Top-N stage — the Δm graph itself decides which peaks matter. ``TopNAbundance`` is kept
only as a config-switchable sanity check (the performance floor), not a milestone.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from gems.data.peaklist import SpectrumRecord


@runtime_checkable
class PeakSelector(Protocol):
    """Strategy interface: choose which peaks become tokens."""

    def select(self, spec: SpectrumRecord, n: int) -> np.ndarray:
        """Return up to ``n`` peak indices (into ``spec.mz``) to keep, as an int array."""
        ...


class GraphInduced:
    """Locked default: keep peaks that sit on >=1 abundant Δm edge. [STUB]

    The node set is *induced* by :func:`gems.vocab.graph.build_delta_graph`: build the (degree-capped)
    Δm graph over the spectrum, then keep exactly the peaks incident to an edge — i.e.
    ``DeltaEdges.node_mask()``. ``n`` caps the kept count (drop lowest-abundance nodes if it overflows;
    the degree cap usually keeps this far below the limit). Selection and graph construction share the
    same vocabulary, so this stays consistent with the attention bias.

    Args:
        vocab: a :class:`~gems.vocab.vocabulary.DeltaVocabulary` (supplies blocks + abundance weights).
        ppm_tol: Δm ↔ block match window.
        degree_cap: top-k edges per node by abundance (the linear-cost knob).
    """

    def __init__(self, vocab=None, ppm_tol: float | None = None, degree_cap: int | None = None,
                 pool_mult: int = 8, pool_cap: int = 2048):
        if vocab is None:
            raise ValueError("GraphInduced selection requires a DeltaVocabulary (got vocab=None).")
        self.vocab = vocab
        self.ppm_tol = ppm_tol
        self.degree_cap = degree_cap
        self.pool_mult = pool_mult     # candidate pool = top (pool_mult * n) abundant peaks
        self.pool_cap = pool_cap       # hard cap on the pool so the dense graph build stays tractable

    def select(self, spec: SpectrumRecord, n: int) -> np.ndarray:
        from gems.definitions import DEFAULT_DEGREE_CAP, DEFAULT_PPM_TOLERANCE
        from gems.vocab.graph import build_delta_graph

        mz = np.asarray(spec.mz)
        intensity = np.asarray(spec.intensity)

        # Candidate pool: the most abundant peaks, so the O(pool^2) graph build stays tractable.
        # (The scale-up path runs the graph over all peaks via a sorted-window / pyc2mc construction.)
        pool_size = min(mz.shape[0], min(self.pool_cap, max(n * self.pool_mult, n)))
        pool = np.argsort(intensity)[::-1][:pool_size]
        pool = pool[np.argsort(mz[pool])]                      # order pool by m/z

        edges = build_delta_graph(
            mz[pool], self.vocab,
            ppm_tol=self.ppm_tol if self.ppm_tol is not None else DEFAULT_PPM_TOLERANCE,
            degree_cap=self.degree_cap if self.degree_cap is not None else DEFAULT_DEGREE_CAP,
        )
        kept = pool[edges.node_mask()]                          # peaks sitting on >=1 Δm edge
        if kept.shape[0] > n:                                   # too many → keep most abundant
            kept = kept[np.argsort(intensity[kept])[::-1][:n]]
        return kept[np.argsort(mz[kept])]                       # return sorted by m/z


class TopNAbundance:
    """Sanity-check fallback: keep the ``n`` most abundant peaks (the floor). [STUB]

    Not the locked path — kept only as a config-switchable baseline to measure what the graph-induced
    selection buys. TODO: ``np.argsort(spec.intensity)[::-1][:n]``, then return sorted-by-m/z.
    """

    def select(self, spec: SpectrumRecord, n: int) -> np.ndarray:
        intensity = np.asarray(spec.intensity)
        keep = np.argsort(intensity)[::-1][:n]                 # top-n by intensity (descending)
        return keep[np.argsort(np.asarray(spec.mz)[keep])]     # return sorted by m/z


# Registry for config-driven dispatch. [CONCRETE]
PEAK_SELECTORS: dict[str, type] = {
    "graph_induced": GraphInduced,   # locked default
    "top_n": TopNAbundance,          # sanity-check floor
}


def build_peak_selector(cfg, vocab=None) -> PeakSelector:
    """Instantiate the selector named by ``cfg.strategy`` from the registry. [CONCRETE]

    Args:
        cfg: an object/dict with a ``strategy`` key in ``PEAK_SELECTORS`` (default ``graph_induced``).
        vocab: a :class:`DeltaVocabulary`, required by the graph-induced strategy.
    """
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    strategy = get("strategy", "graph_induced")
    if strategy not in PEAK_SELECTORS:
        raise KeyError(f"Unknown peak-selection strategy {strategy!r}; "
                       f"choices: {sorted(PEAK_SELECTORS)}")
    if strategy == "graph_induced":
        return GraphInduced(vocab=vocab, ppm_tol=get("ppm_tol"), degree_cap=get("degree_cap"))
    return TopNAbundance()
