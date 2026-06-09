"""Tests for the ppm chemistry helpers (the dependency-free concrete bits)."""

from __future__ import annotations

from gems.utils.chem import matches_building_block, ppm_error, ppm_window
from tests.conftest import requires_pyc2mc


def test_ppm_window():
    lo, hi = ppm_window(500.0, ppm=1.0)
    assert lo < 500.0 < hi
    assert abs((hi - lo) - 500.0 * 2e-6) < 1e-9


def test_ppm_error_sign():
    assert ppm_error(500.0005, 500.0) > 0
    assert ppm_error(499.9995, 500.0) < 0


def test_matches_building_block():
    # CH2 at 14.01565, within 1 ppm window (~1.4e-5 Da) -> a 1e-6 Da offset matches.
    assert matches_building_block(14.01565 + 1e-6, 14.01565, ppm=1.0)
    # N (14.00307) must NOT match CH2 at 1 ppm.
    assert not matches_building_block(14.00307, 14.01565, ppm=1.0)


@requires_pyc2mc
def test_van_krevelen_delegation():
    from gems.utils.chem import van_krevelen
    oc, hc = van_krevelen("C6 H12 O6")
    assert abs(oc - 1.0) < 1e-9 and abs(hc - 2.0) < 1e-9
