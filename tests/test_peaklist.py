"""The load-bearing integration test: read a real .pks via pyc2mc into a canonical record."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gems.data.peaklist import load_record
from gems.definitions import IonMode
from tests.conftest import MANIFEST, requires_pyc2mc


@requires_pyc2mc
def test_load_record_basic(sample_pks):
    rec = load_record(sample_pks)
    assert len(rec) > 0
    assert rec.mz.shape == rec.intensity.shape == rec.sn.shape
    assert np.all(np.isfinite(rec.mz))
    assert rec.ion_mode in (IonMode.NEGATIVE, IonMode.POSITIVE)
    # intensity should be the Abundance column (strictly positive), not the all-zero Peak Height.
    assert np.nanmax(rec.intensity) > 0


@requires_pyc2mc
def test_peak_count_matches_manifest(sample_pks):
    """If the file appears in the manifest, the parsed peak count should match n_peaks."""
    if not MANIFEST.exists():
        import pytest
        pytest.skip("manifest absent")
    man = pd.read_csv(MANIFEST)
    from pathlib import Path
    stem = Path(sample_pks).name
    row = man[man["file"] == stem]
    if row.empty:
        import pytest
        pytest.skip(f"{stem} not in manifest")
    rec = load_record(sample_pks)
    assert len(rec) == int(row.iloc[0]["n_peaks"])
