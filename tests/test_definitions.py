"""Sanity checks for the canonical schema/constants."""

from __future__ import annotations

from gems import definitions as d
from tests.conftest import requires_pyc2mc


def test_seed_blocks_present():
    for name in ("H2", "CH2", "O", "S", "CF2", "C13"):
        assert name in d.SEED_BUILDING_BLOCKS


def test_ch2_mass_is_correct():
    assert abs(d.SEED_BUILDING_BLOCKS["CH2"] - 14.01565) < 1e-4


def test_c13_spacing():
    assert abs(d.SEED_BUILDING_BLOCKS["C13"] - d.C13_C12_DELTA) < 1e-9


def test_ion_mode_mapping():
    assert d.IonMode.from_pyc2mc("Polarity.negative") is d.IonMode.NEGATIVE
    assert d.IonMode.from_pyc2mc("Polarity.positive") is d.IonMode.POSITIVE
    assert d.IonMode.from_pyc2mc("something_else") is d.IonMode.UNKNOWN


def test_mds_schema_constants():
    assert d.MDS_DELTA_COL in d.MDS_COLUMNS
    assert d.MDS_WEIGHT_COL in d.MDS_COLUMNS
    assert len(d.MDS_COLUMNS) == 20


@requires_pyc2mc
def test_seed_masses_match_pyc2mc():
    from pyc2mc.core.formula import Formula
    assert abs(Formula.from_string("C H2").exact_mass - d.SEED_BUILDING_BLOCKS["CH2"]) < 1e-6
    assert abs(Formula.from_string("O").exact_mass - d.SEED_BUILDING_BLOCKS["O"]) < 1e-6
