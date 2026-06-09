"""Tests for the mass-difference data layer (read CSV, manifest, join)."""

from __future__ import annotations

from gems.data.mds import join_pks_to_mds, read_manifest, read_mds_csv
from gems.definitions import MDS_DELTA_COL, MDS_WEIGHT_COL
from tests.conftest import MANIFEST, MDS_DIR, PKS_DIR


def test_read_mds_csv_schema(sample_mds_csv):
    df = read_mds_csv(sample_mds_csv)
    assert MDS_DELTA_COL in df.columns
    assert MDS_WEIGHT_COL in df.columns
    assert len(df) > 0


def test_top_delta_is_a_building_block(sample_mds_csv):
    """The most abundant Δm in any complex-mixture spectrum should be a known block (CH2/O/S/H2)."""
    df = read_mds_csv(sample_mds_csv).sort_values(MDS_WEIGHT_COL, ascending=False)
    top = float(df.iloc[0][MDS_DELTA_COL])
    known = [2.01565, 14.01565, 15.99491, 18.01056, 31.97207, 1.00336, 45.98772]
    assert any(abs(top - k) < 5e-3 for k in known), f"top Δm {top} not near a known block"


def test_join_logs_orphans():
    if not (PKS_DIR.exists() and MDS_DIR.exists()):
        import pytest
        pytest.skip("corpus absent")
    manifest = read_manifest(MANIFEST) if MANIFEST.exists() else None
    joined = join_pks_to_mds(PKS_DIR, MDS_DIR, manifest=manifest)
    assert {"stem", "pks_path", "mds_path"} <= set(joined.columns)
    both = joined["pks_path"].notna() & joined["mds_path"].notna()
    assert both.sum() > 0
