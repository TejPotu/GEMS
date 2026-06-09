"""Peak selection ("special peaks") — a first-class, ablatable preprocessing stage. [STUB + CONCRETE registry]

FT-ICR spectra carry 10^4–10^5 peaks; O(n^2) attention cannot consume them all. Selecting which
peaks become tokens is a tunable knob the thesis must ablate (top-N by abundance vs series-anchored
vs assigned-formula). Each strategy returns the indices of the peaks to keep.
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


class TopNAbundance:
    """Keep the ``n`` most abundant peaks. The default, simplest strategy. [STUB]

    TODO: ``np.argsort(spec.intensity)[::-1][:n]``, then return sorted-by-m/z for stable ordering.
    """

    def select(self, spec: SpectrumRecord, n: int) -> np.ndarray:
        raise NotImplementedError(
            "TopNAbundance.select is a stub: take argsort of spec.intensity (descending), "
            "keep the top n, return indices sorted by m/z."
        )


class SeriesAnchored:
    """Prefer peaks that anchor abundant Kendrick / Δm series. [STUB]

    Intended to leverage pyc2mc ``KendrickSeries`` to find homologous-series members and bias
    selection toward them (so the Δm graph stays connected after subsampling).
    """

    def select(self, spec: SpectrumRecord, n: int) -> np.ndarray:
        raise NotImplementedError(
            "SeriesAnchored.select is a stub: detect Kendrick/Δm series (pyc2mc KendrickSeries) "
            "and prioritize series-member peaks, backfilling with TopNAbundance up to n."
        )


class AssignedFormula:
    """Keep peaks that received a confident formula assignment (post noise removal). [STUB]

    Uses pyc2mc peak attribution; introduces assignment bias, so treat as an explicit ablation arm,
    not the default (see PROJECT_IDEA.md "raw-peak vs formula-aware").
    """

    def select(self, spec: SpectrumRecord, n: int) -> np.ndarray:
        raise NotImplementedError(
            "AssignedFormula.select is a stub: run pyc2mc attribution and keep assigned peaks."
        )


# Registry for config-driven dispatch. [CONCRETE]
PEAK_SELECTORS: dict[str, type] = {
    "top_n": TopNAbundance,
    "series_anchor": SeriesAnchored,
    "assigned_formula": AssignedFormula,
}


def build_peak_selector(cfg) -> PeakSelector:
    """Instantiate the selector named by ``cfg.strategy`` from the registry. [CONCRETE]

    Args:
        cfg: an object/dict with a ``strategy`` key in ``PEAK_SELECTORS``.

    Returns:
        a :class:`PeakSelector` instance.
    """
    strategy = cfg["strategy"] if isinstance(cfg, dict) else getattr(cfg, "strategy")
    if strategy not in PEAK_SELECTORS:
        raise KeyError(f"Unknown peak-selection strategy {strategy!r}; "
                       f"choices: {sorted(PEAK_SELECTORS)}")
    return PEAK_SELECTORS[strategy]()
