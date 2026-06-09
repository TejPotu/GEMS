"""Chemistry helpers — mostly thin delegations to pyc2mc. [WRAP + small CONCRETE]

ppm math is ours (tiny, no dependency); formula / Kendrick / van Krevelen / DBE are delegated to
``pyc2mc.core.formula.Formula`` and the pyc2mc Kendrick utilities so GEMS never re-derives
chemistry the backend already implements.
"""

from __future__ import annotations

import math

from gems.definitions import DEFAULT_PPM_TOLERANCE


# ---- ppm math (concrete, dependency-free) -------------------------------------------------

def ppm_window(mz: float, ppm: float = DEFAULT_PPM_TOLERANCE) -> tuple[float, float]:
    """Return the (low, high) m/z bounds of a ppm-scaled window around ``mz``. [CONCRETE]"""
    half = mz * ppm * 1e-6
    return mz - half, mz + half


def ppm_error(observed: float, reference: float) -> float:
    """Signed mass error in ppm of ``observed`` relative to ``reference``. [CONCRETE]"""
    return (observed - reference) / reference * 1e6


def matches_building_block(delta_m: float, block_mass: float,
                           ppm: float = DEFAULT_PPM_TOLERANCE) -> bool:
    """True if ``delta_m`` is within a ppm window of ``block_mass``. [CONCRETE]

    The window scales with the block mass (ppm), not a fixed Da tolerance — required at FT-ICR
    resolution where N (14.00307) vs CH2 (14.01565) must stay distinct.
    """
    if block_mass <= 0:
        return False
    return abs(ppm_error(delta_m, block_mass)) <= ppm


# ---- delegations to pyc2mc ----------------------------------------------------------------

def exact_mass(formula_str: str) -> float:
    """Monoisotopic exact mass of a formula string (pyc2mc convention, e.g. 'C H2'). [WRAP]"""
    from pyc2mc.core.formula import Formula
    return float(Formula.from_string(formula_str).exact_mass)


def nominal_mass(formula_str: str) -> int:
    """Nominal (integer) mass of a formula string. [WRAP]"""
    from pyc2mc.core.formula import Formula
    return int(Formula.from_string(formula_str).nominal_mass)


def dbe(formula_str: str) -> float:
    """Double-bond equivalent of a formula. [WRAP]"""
    from pyc2mc.core.formula import Formula
    return float(Formula.from_string(formula_str).dbe_value)


def chem_class(formula_str: str) -> str:
    """Heteroatom class label (e.g. 'CHO', 'CHOS') of a formula. [WRAP]"""
    from pyc2mc.core.formula import Formula
    return Formula.from_string(formula_str).chem_class


def van_krevelen(formula_str: str) -> tuple[float, float]:
    """Return (O/C, H/C) ratios for a formula. [WRAP]

    Computed from pyc2mc's elemental composition; raises if C is absent.
    """
    from pyc2mc.core.formula import Formula
    comp = Formula.from_string(formula_str).elemental_composition
    c = comp.get("C", 0)
    if not c:
        raise ValueError(f"van Krevelen undefined for carbon-free formula {formula_str!r}")
    return comp.get("O", 0) / c, comp.get("H", 0) / c


def kendrick_mass_defect(mz: float, base: str = "C H2") -> float:
    """Kendrick mass defect of an m/z referenced to a base group. [CONCRETE]

    KMD = round(KM) - KM, with KM = mz * nominal(base) / exact(base). For per-peak arrays over a
    whole peak list, prefer pyc2mc's vectorized ``PeakList.get_kendrick_mass_defects(base)``.
    """
    km = mz * nominal_mass(base) / exact_mass(base)
    return round(km) - km
