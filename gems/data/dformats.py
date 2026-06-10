"""Data-format specs — normalize → select/trim → pad a spectrum to a fixed tensor. [STUB]

Mirrors DreaMS ``dformats.py``. A :class:`DataFormat` is the contract that turns a variable-length
:class:`~gems.data.peaklist.SpectrumRecord` into a fixed ``max_peaks`` tensor the model can batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gems.definitions import (
    DEV_MAX_PEAKS,
    FULL_MAX_PEAKS,
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_PEAKS,
)
from gems.data.peaklist import SpectrumRecord


@dataclass
class DataFormat:
    """Spectrum → fixed-size tensor contract.

    Attributes:
        min_peaks: spectra with fewer resolved peaks are rejected.
        max_peaks: hard cap on tokens (peaks beyond this are dropped by the peak selector).
        max_mz: peaks above this m/z are discarded.
        log_intensity: log1p-transform intensities before normalization.
        pad_value: value used to right-pad short spectra.
    """

    min_peaks: int = DEFAULT_MIN_PEAKS
    max_peaks: int = DEV_MAX_PEAKS
    max_mz: float = DEFAULT_MAX_MZ
    log_intensity: bool = True
    pad_value: float = 0.0

    def normalize_intensity(self, intensity: np.ndarray) -> np.ndarray:
        """Log1p (optional) + relative-abundance normalization to [0, 1]. [CONCRETE]"""
        x = np.asarray(intensity, dtype=np.float64)
        if self.log_intensity:
            x = np.log1p(np.clip(x, 0.0, None))
        peak = x.max() if x.size else 0.0
        return (x / peak) if peak > 0 else np.zeros_like(x)

    def trim(self, spec: SpectrumRecord, selector) -> np.ndarray:
        """Apply the m/z cutoff then delegate to ``selector`` for ≤ max_peaks indices. [CONCRETE]"""
        idx = selector.select(spec, self.max_peaks)
        mz = np.asarray(spec.mz)[idx]
        idx = idx[mz <= self.max_mz]
        return idx[: self.max_peaks]

    def pad(self, arr: np.ndarray) -> np.ndarray:
        """Right-pad/truncate the first axis to ``max_peaks`` with ``pad_value``. [CONCRETE]"""
        arr = np.asarray(arr)
        n = arr.shape[0]
        if n >= self.max_peaks:
            return arr[: self.max_peaks]
        pad_width = [(0, self.max_peaks - n)] + [(0, 0)] * (arr.ndim - 1)
        return np.pad(arr, pad_width, constant_values=self.pad_value)

    def __call__(self, spec: SpectrumRecord, selector) -> dict:
        """Full pipeline: validate min_peaks → trim → normalize → pad → fixed-size array dict. [CONCRETE]

        Returns ``{mz, intensity, valid_mask}`` of length ``max_peaks`` (float32 / bool).
        ``intensity`` is normalized over the *kept* peaks. ``valid_mask`` is True for real peaks.
        """
        if len(spec) < self.min_peaks:
            raise ValueError(f"spectrum has {len(spec)} peaks (< min_peaks={self.min_peaks})")

        idx = self.trim(spec, selector)
        n = int(idx.shape[0])
        mz = np.asarray(spec.mz, dtype=np.float64)[idx]
        intensity = self.normalize_intensity(np.asarray(spec.intensity)[idx])

        valid = np.zeros(self.max_peaks, dtype=bool)
        valid[:n] = True
        return {
            "mz": self.pad(mz).astype(np.float32),
            "intensity": self.pad(intensity).astype(np.float32),
            "valid_mask": valid,
        }


# Convenient presets.
DF_FTICR_DEV = DataFormat(max_peaks=DEV_MAX_PEAKS)     # CPU/MPS development
DF_FTICR_FULL = DataFormat(max_peaks=FULL_MAX_PEAKS)   # scale-up
