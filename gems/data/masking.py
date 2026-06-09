"""Self-supervised masking + augmentations for FT-ICR spectra. [STUB]

Provides the dataset wrapper for masked-peak modeling (the workhorse SSL objective) and the
augmentation primitives used by the contrastive objective. Adapted from DreaMS's masked-spectrum
dataset, but masking targets here are the exact mass defect and/or intensity of FT-ICR peaks.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class MaskedPeakDataset(Dataset):
    """Wrap a spectrum dataset and mask peaks for masked-peak modeling. [STUB]

    BERT-style: select ``mask_prob`` of peaks; for each, mask its m/z and/or intensity (replace with
    a sentinel) and record the original value as the reconstruction target. The model predicts the
    masked peak's exact mass defect (and/or intensity) from its homologous-series neighbors.

    Args:
        base: an underlying ``Dataset`` yielding fixed-size spectrum tensor dicts.
        mask_prob: fraction of (valid) peaks to mask.
        mask_mz: mask the m/z channel.
        mask_intensity: mask the intensity channel.
    """

    def __init__(self, base: Dataset, mask_prob: float = 0.15,
                 mask_mz: bool = True, mask_intensity: bool = False):
        self.base = base
        self.mask_prob = mask_prob
        self.mask_mz = mask_mz
        self.mask_intensity = mask_intensity

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        raise NotImplementedError(
            "MaskedPeakDataset.__getitem__ is a stub: draw a Bernoulli(mask_prob) mask over valid "
            "peaks, blank the chosen m/z/intensity channels, and attach the targets + target mask."
        )


# ---- contrastive-invariance augmentations (SimCLR-style) -----------------------------------

def subsample_peaks(spec: dict, frac: float, rng: np.random.Generator | None = None) -> dict:
    """Randomly keep a fraction of peaks (robustness to peak-picking thresholds). [STUB]"""
    raise NotImplementedError("subsample_peaks is a stub.")


def jitter_intensity(spec: dict, sigma: float, rng: np.random.Generator | None = None) -> dict:
    """Multiply intensities by lognormal noise (abundance-measurement jitter). [STUB]"""
    raise NotImplementedError("jitter_intensity is a stub.")


def simulate_calibration_drift(spec: dict, ppm: float, rng: np.random.Generator | None = None) -> dict:
    """Apply a small ppm-scaled m/z shift to mimic cross-instrument calibration drift. [STUB]

    This is the key augmentation defending against the model shortcutting to acquisition signatures
    rather than chemistry (see PROJECT_IDEA.md "Design flags for sample-class labels").
    """
    raise NotImplementedError("simulate_calibration_drift is a stub.")


def contrastive_views(spec: dict, cfg) -> tuple[dict, dict]:
    """Produce two augmented views of a spectrum for a SimCLR/contrastive loss. [STUB]"""
    raise NotImplementedError("contrastive_views is a stub (compose the augmentations above).")
