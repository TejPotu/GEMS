"""Shared pytest fixtures.

Provides paths to the real corpus and a `pyc2mc` availability flag so tests that need the backend
skip cleanly when it is not installed.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PKS_DIR = REPO_ROOT / "data" / "walking_calibrated_pks"
MDS_DIR = REPO_ROOT / "data" / "mds_csv"
MANIFEST = REPO_ROOT / "data" / "mds_run_manifest.csv"


def _has_pyc2mc() -> bool:
    try:
        import pyc2mc  # noqa: F401
        return True
    except Exception:
        return False


HAS_PYC2MC = _has_pyc2mc()
requires_pyc2mc = pytest.mark.skipif(not HAS_PYC2MC, reason="pyc2mc not installed")


@pytest.fixture(scope="session")
def sample_pks() -> str:
    """Path to one real .pks file, or skip if the corpus is absent."""
    files = sorted(glob.glob(str(PKS_DIR / "*.pks")))
    if not files:
        pytest.skip(f"no .pks files under {PKS_DIR}")
    return files[0]


@pytest.fixture(scope="session")
def sample_mds_csv() -> str:
    files = sorted(glob.glob(str(MDS_DIR / "*.csv")))
    if not files:
        pytest.skip(f"no mds_csv files under {MDS_DIR}")
    return files[0]
