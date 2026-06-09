"""Mass-difference spectra (MDS) — read pre-computed CSVs or (re)generate via pyc2mc. [WRAP + CONCRETE]

Two data sources feed the building-block vocabulary:
  1. the pre-computed ``data/mds_csv/*.csv`` (produced by an earlier pyc2mc run), and
  2. on-the-fly computation via ``pyc2mc.processing.mass_differences_spectrum.MassDifferencesSpectrum``
     (needed for new spectra at inference, or to recompute with different parameters).

Both yield the same 20-column ``md_data`` schema (see ``definitions.MDS_COLUMNS``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gems.definitions import (
    DEFAULT_DELTA_MZ_BOUNDS,
    MANIFEST_COLUMNS,
    MDS_QUALITY_COL,
    MDS_WEIGHT_COL,
)

logger = logging.getLogger(__name__)


def compute_mds(
    peaklist,
    delta_mz_bounds: tuple[float, float] = DEFAULT_DELTA_MZ_BOUNDS,
    bin_size: int = 5,
    weights: str | None = "sum",
    count_pairs: bool = True,
    **kwargs,
):
    """Compute the mass-difference spectrum for a peak list via pyc2mc. [WRAP]

    Args:
        peaklist: a pyc2mc ``PeakList`` (e.g. from ``data.peaklist.load_peaklist``).
        delta_mz_bounds: (min, max) Δm to consider, in Da.
        bin_size: histogram bin width in microDalton.
        weights: ``"sum"`` to weight pairs by intensity product, or ``None`` for equal weights.
        count_pairs: count unique peaks per distribution (enables ``# unique peaks`` / series index).
        **kwargs: forwarded to ``MassDifferencesSpectrum``.

    Returns:
        pyc2mc ``MassDifferencesSpectrum``; its ``.md_data`` is the 20-col DataFrame.
    """
    from pyc2mc.processing.mass_differences_spectrum import MassDifferencesSpectrum

    return MassDifferencesSpectrum(
        peaklist,
        delta_mz_bounds=delta_mz_bounds,
        bin_size=bin_size,
        weights=weights,
        count_pairs=count_pairs,
        **kwargs,
    )


def read_mds_csv(path: str | Path) -> pd.DataFrame:
    """Read one pre-computed mass-difference CSV into a DataFrame. [CONCRETE]"""
    return pd.read_csv(path)


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read ``mds_run_manifest.csv`` (file, n_peaks, n_distributions, seconds, status, error). [CONCRETE]"""
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Manifest %s missing expected columns: %s", path, missing)
    return df


def join_pks_to_mds(
    pks_dir: str | Path,
    mds_dir: str | Path,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join ``.pks`` files to their per-sample ``mds_csv`` by basename. [CONCRETE]

    There are 272 ``.pks`` vs 285 ``mds_csv`` in this corpus; orphans on either side are logged,
    not dropped silently. Returns a DataFrame with columns: ``stem, pks_path, mds_path``
    (paths are None where absent), optionally merged with manifest rows on the .pks filename.
    """
    pks_dir, mds_dir = Path(pks_dir), Path(mds_dir)
    pks = {p.stem: p for p in pks_dir.glob("*.pks")}
    mds = {p.stem: p for p in mds_dir.glob("*.csv")}

    stems = sorted(set(pks) | set(mds))
    rows = []
    for stem in stems:
        p, m = pks.get(stem), mds.get(stem)
        if p is None:
            logger.warning("mds_csv without matching .pks: %s", stem)
        if m is None:
            logger.warning(".pks without matching mds_csv: %s", stem)
        rows.append({"stem": stem,
                     "pks_path": str(p) if p else None,
                     "mds_path": str(m) if m else None})
    out = pd.DataFrame(rows)

    if manifest is not None and "file" in manifest.columns:
        man = manifest.copy()
        man["stem"] = man["file"].astype(str).str.replace(r"\.pks$", "", regex=True)
        out = out.merge(man.drop(columns=["file"]), on="stem", how="left")
    return out


def filter_delta_distributions(
    df: pd.DataFrame,
    min_r_squared: float = 0.9,
    min_occurrences: float = 10.0,
    drop_weird_shape: bool = True,
) -> pd.DataFrame:
    """Quality-filter Δm distributions before vocabulary aggregation. [STUB]

    Intended behavior: keep rows with ``r_squared >= min_r_squared`` and
    ``# occurrences >= min_occurrences``; optionally drop ``weird_shape``/``overlap`` flagged fits.

    TODO: implement once vocabulary-building thresholds are chosen empirically (Phase 0).
    """
    raise NotImplementedError(
        "filter_delta_distributions is a stub. Filter on "
        f"{MDS_QUALITY_COL!r} >= {min_r_squared}, {MDS_WEIGHT_COL!r} >= {min_occurrences}, "
        "and (if drop_weird_shape) the 'weird_shape'/'overlap' boolean columns."
    )
