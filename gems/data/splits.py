"""Corpus splitting — random, held-out-instrument, and held-out-class. [STUB]

Held-out-instrument splitting is an evaluation-hygiene requirement (PROJECT_IDEA.md): sample-class
labels correlate with the lab/instrument/method, so the model can shortcut to acquisition signatures.
Filename heuristics (date, ESI/APPI, lab code) are parsed to group spectra by acquisition source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_metadata_from_filename(name: str) -> dict:
    """Heuristically parse acquisition metadata from a .pks filename. [STUB]

    e.g. ``2023February20NegESI_fossildom_WholeSample_Sum200_WalkingCal.pks`` →
    ``{date, ion_source: ESI, polarity: neg, sample_hint: fossildom, ...}``.
    """
    raise NotImplementedError("infer_metadata_from_filename is a stub.")


def random_split(items: list, fracs: tuple[float, ...], seed: int = 0) -> list[list]:
    """Random partition of ``items`` into len(fracs) groups. [STUB]"""
    raise NotImplementedError("random_split is a stub.")


def held_out_instrument_split(manifest: pd.DataFrame, val_frac: float = 0.2, seed: int = 0) -> dict:
    """Split so no acquisition source spans train and val. [STUB]"""
    raise NotImplementedError("held_out_instrument_split is a stub (group by inferred instrument).")


def held_out_class_split(labels: np.ndarray, held_out_classes: list, seed: int = 0) -> dict:
    """Hold out entire sample classes for transfer evaluation. [STUB]"""
    raise NotImplementedError("held_out_class_split is a stub.")
