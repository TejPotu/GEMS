"""The building-block Δm vocabulary (``vocab/vocabulary.py``). [CONCRETE seed + STUB aggregation]

The vocabulary maps building-block names → exact Δm masses, plus a corpus-level abundance weight per
block. It is *seeded* with known blocks (CH2, H2, O, S, CF2, …) and can be *grown* by aggregating the
pyc2mc mass-difference distributions across the corpus, optionally auto-naming discovered Δm by
running pyc2mc's ``assign_peaks``. The (abundance-weighted) weights are the novelty the attention bias
exploits, so they are first-class here.

One reserved embedding row past the real blocks holds the ``[masked-edge]`` sentinel: at pre-training
time, edges incident to a masked peak have their Δm-type id + abundance stripped to this sentinel so a
typed edge can never hand the model the masked mass (BUILD_PLAN A2/B1 leakage guard). Code that builds
the edge-bias embedding table should size it to :pyattr:`DeltaVocabulary.n_embeddings`, not ``len``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from gems.definitions import (
    C13_C12_DELTA,
    DEFAULT_PPM_TOLERANCE,
    MASKED_EDGE_TOKEN,
    SEED_BUILDING_BLOCKS,
)


def seed_vocabulary() -> dict[str, float]:
    """Return the seeded building-block name→exact-mass map (from pyc2mc-resolved seeds). [CONCRETE]"""
    return dict(SEED_BUILDING_BLOCKS)


@dataclass
class DeltaVocabulary:
    """A building-block Δm vocabulary with a learnable index map and abundance weights.

    Attributes:
        masses: block name → exact Δm mass (Da).
        weights: block name → corpus abundance weight (0 if seed-only / unobserved).
        ppm_tol: default ppm window for matching an observed Δm to a block.

    Indices ``0..len-1`` address the real building blocks (by ascending mass); index ``len`` is the
    reserved ``[masked-edge]`` sentinel (see :pyattr:`masked_edge_index`). Embedding tables keyed on
    block id must therefore have :pyattr:`n_embeddings` rows.
    """

    masses: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    ppm_tol: float = DEFAULT_PPM_TOLERANCE

    def __post_init__(self):
        # stable index order: by mass ascending, so block_idx is deterministic across runs.
        self._names = sorted(self.masses, key=lambda n: self.masses[n])
        self._index = {n: i for i, n in enumerate(self._names)}

    def __len__(self) -> int:
        return len(self._names)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def masked_edge_index(self) -> int:
        """Reserved block id for the ``[masked-edge]`` leakage-guard sentinel. [CONCRETE]"""
        return len(self._names)

    @property
    def n_embeddings(self) -> int:
        """Rows an edge-bias embedding table needs: real blocks + the masked-edge sentinel. [CONCRETE]"""
        return len(self._names) + 1

    def index_of(self, name: str) -> int:
        """Return the integer index (embedding row) of a block by name. [CONCRETE]

        ``MASKED_EDGE_TOKEN`` resolves to :pyattr:`masked_edge_index`.
        """
        if name == MASKED_EDGE_TOKEN:
            return self.masked_edge_index
        return self._index[name]

    def match(self, delta_m: float, ppm: float | None = None) -> str | None:
        """Return the block name whose mass matches ``delta_m`` within a ppm window, else None. [STUB]

        Intended: ppm window scales with the Δm magnitude (``tol_da = mass * ppm * 1e-6``); on
        multiple matches, pick the nearest. ``include_c13`` is handled by including the "C13" block.

        TODO: implement using ``utils.chem.matches_building_block`` / ``ppm_window``.
        """
        raise NotImplementedError("DeltaVocabulary.match is a stub (ppm-windowed nearest block).")

    # ---- persistence (concrete) ----------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        """Serialize masses + weights + ppm_tol to JSON. [CONCRETE]"""
        payload = {"masses": self.masses, "weights": self.weights, "ppm_tol": self.ppm_tol}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "DeltaVocabulary":
        """Load a vocabulary previously written by :meth:`to_json`. [CONCRETE]"""
        payload = json.loads(Path(path).read_text())
        return cls(masses=payload["masses"],
                   weights=payload.get("weights", {}),
                   ppm_tol=payload.get("ppm_tol", DEFAULT_PPM_TOLERANCE))

    @classmethod
    def from_seeds(cls, include_c13: bool = True) -> "DeltaVocabulary":
        """Build a seed-only vocabulary (no corpus weights yet). [CONCRETE]"""
        masses = dict(SEED_BUILDING_BLOCKS)
        if not include_c13:
            masses.pop("C13", None)
        else:
            masses.setdefault("C13", C13_C12_DELTA)
        return cls(masses=masses, weights={n: 0.0 for n in masses})


def build_vocabulary_from_corpus(
    mds_source,
    manifest=None,
    seed: bool = True,
    min_occurrences: float = 10.0,
    top_k: int = 128,
    include_c13: bool = True,
    auto_name: bool = True,
) -> DeltaVocabulary:
    """Aggregate Δm distributions across the corpus into a weighted vocabulary. [STUB]

    Intended behavior:
      1. Iterate the per-sample MDS (either a directory of pre-computed ``mds_csv`` or pyc2mc
         ``md_data`` frames from :func:`gems.data.mds.compute_mds`).
      2. Quality-filter (``filter_delta_distributions``) and accumulate ``# occurrences`` per Δm
         bin across samples → corpus abundance weights (these weights become the attention bias).
      3. Keep the ``top_k`` most abundant Δm.
      4. If ``auto_name``: run pyc2mc ``assign_peaks`` to label discovered Δm with formulas
         (CH2, O, S, CF2, …); otherwise key blocks by rounded Δm.
      5. If ``seed``: union with :data:`SEED_BUILDING_BLOCKS` (seeds get their corpus weight if seen).

    Args:
        mds_source: a directory path of mds_csv, or an iterable of (name, md_data DataFrame).
        manifest: optional manifest DataFrame for weighting/normalization.

    Returns:
        a :class:`DeltaVocabulary`.
    """
    raise NotImplementedError(
        "build_vocabulary_from_corpus is a stub. Aggregate '# occurrences' per Δm across all "
        "mds_csv (data.mds.read_mds_csv / compute_mds), filter, take top_k, optionally pyc2mc "
        "assign_peaks to name them, then union with seeds."
    )
