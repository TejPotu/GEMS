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
        """Log1p (optional) + relative-abundance normalization. [STUB]

        TODO: ``x = np.log1p(intensity) if log_intensity else intensity``; divide by max (or sum).
        """
        raise NotImplementedError("DataFormat.normalize_intensity is a stub.")

    def trim(self, spec: SpectrumRecord, selector) -> np.ndarray:
        """Apply m/z cutoff then delegate to ``selector`` to choose ≤ max_peaks indices. [STUB]

        TODO: mask peaks with mz > max_mz, then ``selector.select(spec, self.max_peaks)``.
        """
        raise NotImplementedError("DataFormat.trim is a stub (delegates to peak_selection).")

    def pad(self, arr: np.ndarray) -> np.ndarray:
        """Right-pad/truncate the first axis to ``max_peaks`` with ``pad_value``. [STUB]"""
        raise NotImplementedError("DataFormat.pad is a stub.")

    def __call__(self, spec: SpectrumRecord, selector) -> dict:
        """Full pipeline: validate min_peaks → normalize → trim → pad → return tensors dict. [STUB]

        Returns a dict of fixed-length arrays: ``{mz, intensity, mask, ...}`` plus the kept indices,
        suitable for collation into a batch.
        """
        raise NotImplementedError("DataFormat.__call__ is a stub (normalize→trim→pad pipeline).")


# Convenient presets.
DF_FTICR_DEV = DataFormat(max_peaks=DEV_MAX_PEAKS)     # CPU/MPS development
DF_FTICR_FULL = DataFormat(max_peaks=FULL_MAX_PEAKS)   # scale-up
