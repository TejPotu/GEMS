"""FT-ICR peak-list IO — a thin adapter over pyc2mc. [WRAP]

GEMS does not parse ``.pks`` files itself: pyc2mc produced this corpus and its
``pyc2mc.io.peaklist.read_pks`` already handles the (variable) header variants. This module
loads a peak list and projects the pyc2mc ``PeakList`` onto GEMS's canonical record
(plain numpy arrays + ion mode + metadata) ready for tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gems.definitions import IonMode


@dataclass
class SpectrumRecord:
    """Canonical in-memory representation of one FT-ICR spectrum (model-facing).

    Attributes:
        mz: peak m/z values (Da), shape (n_peaks,).
        intensity: peak intensities (the pyc2mc ``.intensity`` = .pks Abundance column).
        sn: signal-to-noise ratios (may be all-NaN if S/N absent).
        resolving_power: per-peak resolving power (often 0 in this corpus).
        frequency: ion-cyclotron frequencies.
        ion_mode: acquisition polarity.
        source_path: originating .pks path.
        metadata: pyc2mc metadata dict (npeaks, mz range, thresholds, time, ...).
    """

    mz: np.ndarray
    intensity: np.ndarray
    sn: np.ndarray
    resolving_power: np.ndarray
    frequency: np.ndarray
    ion_mode: IonMode
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.mz.shape[0])


def load_peaklist(path: str | Path, **kwargs):
    """Read a ``.pks`` file into a pyc2mc ``PeakList``. [WRAP]

    Thin wrapper over ``pyc2mc.io.peaklist.read_pks``. Extra kwargs (e.g. ``skiprows``,
    ``sort_mz``, ``polarity``) are forwarded. Prefer this over calling pyc2mc directly so the
    dependency is centralized and easy to swap.

    Args:
        path: path to a ``.pks`` peak list.
        **kwargs: forwarded to ``read_pks``.

    Returns:
        pyc2mc ``PeakList``.
    """
    from pyc2mc.io.peaklist import read_pks  # local import: pyc2mc is an optional-at-import dep

    return read_pks(str(path), **kwargs)


def peaklist_to_record(peaklist, source_path: str | None = None) -> SpectrumRecord:
    """Project a pyc2mc ``PeakList`` onto a canonical :class:`SpectrumRecord`. [WRAP]

    Pulls ``.mz``, ``.intensity``, ``.SN`` (NaN-filled when ``SN_avail`` is False),
    ``.resolving_power``, ``.frequency``, ``.polarity``, and ``.metadata``.
    """
    mz = np.asarray(peaklist.mz, dtype=np.float64)
    intensity = np.asarray(peaklist.intensity, dtype=np.float64)

    if getattr(peaklist, "SN_avail", False):
        sn = np.asarray(peaklist.SN, dtype=np.float64)
    else:
        sn = np.full_like(mz, np.nan)

    rp = np.asarray(getattr(peaklist, "resolving_power", np.zeros_like(mz)), dtype=np.float64)
    freq = np.asarray(getattr(peaklist, "frequency", np.zeros_like(mz)), dtype=np.float64)

    return SpectrumRecord(
        mz=mz,
        intensity=intensity,
        sn=sn,
        resolving_power=rp,
        frequency=freq,
        ion_mode=IonMode.from_pyc2mc(getattr(peaklist, "polarity", IonMode.UNKNOWN)),
        source_path=source_path or getattr(peaklist, "name", "") or "",
        metadata=dict(getattr(peaklist, "metadata", {}) or {}),
    )


def load_record(path: str | Path, **kwargs) -> SpectrumRecord:
    """Convenience: ``load_peaklist`` + ``peaklist_to_record`` in one call. [WRAP]"""
    pl = load_peaklist(path, **kwargs)
    return peaklist_to_record(pl, source_path=str(path))
