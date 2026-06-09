"""Canonical constants, schema, and registries for GEMS — the single source of truth.

[CONCRETE] This module is pure data (no heavy logic). It pins down:
  - internal canonical column names used across the package,
  - the pyc2mc-produced ``.pks`` / ``mds_csv`` / manifest schemas (for reference + sanity checks),
  - the ion-mode enum,
  - the seeded building-block Δm vocabulary (formula strings → exact masses, via pyc2mc),
  - physical constants and matching tolerances,
  - heteroatom classes for the class-distribution task,
  - the (initially empty) pretrained-model registry.

Building-block masses are derived from formula strings through ``pyc2mc.core.formula.Formula`` so
they are guaranteed consistent with pyc2mc's own assignment. If pyc2mc is unavailable at import
time, we fall back to a hardcoded mass table so this module stays importable on its own.
"""

from __future__ import annotations

from enum import Enum

# --------------------------------------------------------------------------------------
# Internal canonical column / field names (what GEMS uses everywhere downstream)
# --------------------------------------------------------------------------------------
MZ = "mz"
INTENSITY = "intensity"          # from the .pks "Abundance" column (pyc2mc maps it to .intensity)
PEAK_HEIGHT = "peak_height"
RESOLVING_POWER = "resolving_power"
FREQUENCY = "frequency"
SN = "sn"
ION_MODE = "ion_mode"
SOURCE_PATH = "source_path"
PEAK_ID = "pid"

# --------------------------------------------------------------------------------------
# pyc2mc-produced file schemas (reference only — pyc2mc does the actual parsing)
# --------------------------------------------------------------------------------------
# .pks data-table columns (as they appear in the ASCII files). pyc2mc.io.read_pks handles the
# (variable) header; intensity comes from "Abundance"/"Scaled Abundance" because "Peak Height"
# is frequently 0.000 in this corpus.
PKS_COLUMNS = ["Peak Location", "Peak Height", "Abundance",
               "Resolving Power", "Frequency", "S/N"]
PKS_ABUNDANCE_ALIASES = {"Abundance", "Scaled Abundance"}

# mds_csv columns (exactly MassDifferencesSpectrum.md_data).
MDS_COLUMNS = ["computed_position", "average_position", "gamma", "fwhm", "lmax", "width",
               "# occurrences", "# pairs", "# unique peaks", "hits percentage", "series index",
               "RP", "min", "max", "r_squared", "RMSE", "fit_position", "fit_maximum",
               "overlap", "weird_shape"]
MDS_DELTA_COL = "average_position"   # the Δm value (Da)
MDS_DELTA_COL_ALT = "computed_position"
MDS_WEIGHT_COL = "# occurrences"     # abundance / frequency of that Δm in the spectrum
MDS_QUALITY_COL = "r_squared"        # Lorentzian-fit quality in [0, 1]

# mds_run_manifest.csv columns.
MANIFEST_COLUMNS = ["file", "n_peaks", "n_distributions", "seconds", "status", "error"]


# --------------------------------------------------------------------------------------
# Ion mode
# --------------------------------------------------------------------------------------
class IonMode(str, Enum):
    """Acquisition polarity. Maps from pyc2mc ``Polarity.{negative,positive}``."""
    NEGATIVE = "negative"
    POSITIVE = "positive"
    UNKNOWN = "unknown"

    @classmethod
    def from_pyc2mc(cls, polarity) -> "IonMode":
        """Coerce a pyc2mc Polarity enum (or its name/string) to an IonMode. [CONCRETE]"""
        name = getattr(polarity, "name", str(polarity)).strip().lower()
        if "neg" in name:
            return cls.NEGATIVE
        if "pos" in name:
            return cls.POSITIVE
        return cls.UNKNOWN


# --------------------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------------------
ELECTRON_MASS = 0.000548579909      # Da
PROTON_MASS = 1.007276466879        # Da
C13_C12_DELTA = 1.0033548           # ¹³C–¹²C spacing; abundant Δm — treat as feature OR confounder
KENDRICK_BASE = "C H2"              # default Kendrick base group (pyc2mc formula-string syntax)
KENDRICK_CH2_NOMINAL = 14.0

# --------------------------------------------------------------------------------------
# Seeded building-block Δm vocabulary
# --------------------------------------------------------------------------------------
# Names → pyc2mc formula strings (space-separated element tokens, pyc2mc convention).
# Exact masses are filled at import time from these strings (see SEED_BUILDING_BLOCKS below).
SEED_BLOCK_FORMULAS: dict[str, str] = {
    "H2":   "H2",        # 2.01565  — the DBE / hydrogenation axis
    "CH2":  "C H2",      # 14.01565 — homologous-series spacing
    "O":    "O",         # 15.99491 — oxygenation
    "H2O":  "H2 O",      # 18.01056 — water loss/gain
    "CH2O": "C H2 O",    # 30.01057
    "CO":   "C O",       # 27.99491
    "CO2":  "C O2",      # 43.98983
    "COOH": "C O2 H",    # 44.99765 — carboxyl (note: radical fragment, here as a Δm)
    "S":    "S",         # 31.97207 — sulfur class shift
    "N":    "N",         # 14.00307 (vs CH2 14.01565 — sub-ppm separable at FT-ICR resolution)
    "CHN":  "C H N",     # 27.01090
    "CF2":  "C F2",      # 49.99678 — PFAS probe (a non-CH2 building block)
    "C13":  None,        # isotope spacing 1.0033548; injected from C13_C12_DELTA, flagged include_c13
}

# Hardcoded fallback masses (Da), used only if pyc2mc cannot be imported at module load.
_SEED_FALLBACK_MASSES: dict[str, float] = {
    "H2": 2.0156500, "CH2": 14.0156500, "O": 15.9949146, "H2O": 18.0105646,
    "CH2O": 30.0105646, "CO": 27.9949146, "CO2": 43.9898293, "COOH": 44.9976542,
    "S": 31.9720707, "N": 14.0030740, "CHN": 27.0108990, "CF2": 49.9967660,
    "C13": C13_C12_DELTA,
}


def _compute_seed_masses() -> dict[str, float]:
    """Resolve SEED_BLOCK_FORMULAS to exact masses via pyc2mc, falling back to the table. [CONCRETE]"""
    try:
        from pyc2mc.core.formula import Formula  # local import: keep module import light/optional
    except Exception:
        return dict(_SEED_FALLBACK_MASSES)

    masses: dict[str, float] = {}
    for name, formula_str in SEED_BLOCK_FORMULAS.items():
        if name == "C13" or formula_str is None:
            masses[name] = C13_C12_DELTA
            continue
        try:
            masses[name] = float(Formula.from_string(formula_str).exact_mass)
        except Exception:
            masses[name] = _SEED_FALLBACK_MASSES.get(name, float("nan"))
    return masses


# name → neutral exact mass (Da). Resolved at import.
SEED_BUILDING_BLOCKS: dict[str, float] = _compute_seed_masses()

# --------------------------------------------------------------------------------------
# Matching tolerances and data-format defaults
# --------------------------------------------------------------------------------------
DEFAULT_PPM_TOLERANCE = 1.0         # ppm-scaled window for Δm ↔ building-block matching
DEV_MAX_PEAKS = 256                 # CPU dev default (cap peaks fed into O(n^2) attention)
FULL_MAX_PEAKS = 2048               # scale-up default
DEFAULT_MIN_PEAKS = 32
DEFAULT_MAX_MZ = 1500.0
DEFAULT_DELTA_MZ_BOUNDS = (0.5, 50.0)  # passed to pyc2mc MassDifferencesSpectrum

# --------------------------------------------------------------------------------------
# Heteroatom classes (for the whole-sample class-distribution regression task)
# --------------------------------------------------------------------------------------
HETEROATOM_CLASSES = ["CHO", "CHOS", "CHOS2", "CHN", "CHNO", "CHOP", "CHNOS", "CHOCl"]

# Coarse sample-type labels (for the sample-classification downstream task).
SAMPLE_CLASSES = ["petroleum", "DOM", "PFAS", "lipid", "bio_oil", "other"]

# --------------------------------------------------------------------------------------
# Pretrained-model registry (name → checkpoint path/URL). Empty until models are trained.
# --------------------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, str] = {}

# --------------------------------------------------------------------------------------
# Default corpus locations (relative to repo root).
# --------------------------------------------------------------------------------------
DATA_DIR = "data"
PKS_DIR = "data/walking_calibrated_pks"
MDS_DIR = "data/mds_csv"
MANIFEST_PATH = "data/mds_run_manifest.csv"
PROCESSED_DIR = "data/processed"
