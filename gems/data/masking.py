"""Spectrum-denoising corruption — the single shared corrupted view (BUILD_PLAN Part B). [STUB]

The three repair channels (masked-m/z, masked-intensity, replaced-peak) share **one** corrupted view:
different peaks get different corruptions, the encoder runs once, and per-peak heads repair each. This
module builds that view and the per-channel targets, plus the masked-node mask the Δm-graph leakage
guard consumes.

Corruptions (disjoint subsets of a spectrum's peaks):
  - **mask-m/z** (default 30%, sampled ∝ intensity; ~⅓ of masks are 2–3 *consecutive* homologous-
    series members): blank the m/z (keep intensity), record nominal+defect targets.
  - **mask-intensity** (separate subset): blank the intensity (keep m/z), record the intensity target.
  - **replace** (default 15%): swap in a plausible-but-wrong m/z (non-vocabulary Δm, or a vocabulary Δm
    that breaks defect consistency); keep intensity; label the peak fake.
Edges incident to mask-m/z peaks are stripped to the ``[masked-edge]`` sentinel downstream (the graph
sees ``masked_nodes``), so a typed edge can never reveal the masked mass.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from gems.definitions import (
    DEFAULT_MASK_PROB,
    REPLACED_PEAK_FRACTION,
    SERIES_SPAN_FRACTION,
)


class SpectrumDenoisingDataset(Dataset):
    """Wrap a spectrum dataset and emit one corrupted view + per-channel targets. [STUB]

    Each item carries the corrupted ``mz``/``intensity``, the channel target masks/values, the
    ``masked_nodes`` boolean (for the graph leakage guard), and the per-peak ``replaced_label``.

    Args:
        base: an underlying ``Dataset`` yielding fixed-size spectrum tensor dicts.
        mask_prob: fraction of peaks whose m/z is masked (∝ intensity).
        series_span_fraction: share of m/z masks placed on consecutive homologous-series members.
        replaced_fraction: fraction of peaks given a plausible-but-wrong m/z.
        vocab: a :class:`~gems.vocab.vocabulary.DeltaVocabulary`, used to draw plausible replacement Δm.
    """

    def __init__(self, base: Dataset, mask_prob: float = DEFAULT_MASK_PROB,
                 series_span_fraction: float = SERIES_SPAN_FRACTION,
                 replaced_fraction: float = REPLACED_PEAK_FRACTION, vocab=None):
        self.base = base
        self.mask_prob = mask_prob
        self.series_span_fraction = series_span_fraction
        self.replaced_fraction = replaced_fraction
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        raise NotImplementedError(
            "SpectrumDenoisingDataset.__getitem__ is a stub: draw the three disjoint corruption "
            "subsets, blank m/z (keep intensity) / blank intensity (keep m/z) / replace m/z, record "
            "nominal+defect+intensity targets and replaced_label, and the masked_nodes mask."
        )


# ---- corruption primitives -----------------------------------------------------------------

def intensity_weighted_mask(intensity: np.ndarray, frac: float,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a boolean mask over peaks, drawn ∝ intensity. [STUB]"""
    raise NotImplementedError("intensity_weighted_mask is a stub.")


def series_span_mask(mz: np.ndarray, n_spans: int, base: str = "C H2",
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Mask 2–3 *consecutive* members of a homologous (e.g. CH2) series. [STUB]

    Forces the model to extrapolate a series and its defect progression rather than interpolate one
    edge — where compositional reasoning forms.
    """
    raise NotImplementedError("series_span_mask is a stub (consecutive Kendrick/Δm-series members).")


def replace_with_implausible_mz(mz: np.ndarray, idx: np.ndarray, vocab=None,
                                rng: np.random.Generator | None = None) -> np.ndarray:
    """Shift selected peaks' m/z by a non-vocabulary Δm (or a defect-breaking one). [STUB]

    The replaced-peak (Electra) corruption: the new mass is plausible in nominal terms but
    chemically inconsistent with the rest of the mixture.
    """
    raise NotImplementedError("replace_with_implausible_mz is a stub.")
