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
    """Random partition of ``items`` into ``len(fracs)`` groups. [CONCRETE]

    Fractions are normalized; each non-empty fraction gets at least one item when possible (so a tiny
    dev corpus still yields a non-empty val split). The last group absorbs any rounding remainder.
    """
    items = list(items)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))

    total = float(sum(fracs)) or 1.0
    counts = [int(np.floor(f / total * len(items))) for f in fracs]
    for j, f in enumerate(fracs):           # guarantee >=1 for non-zero fractions if room allows
        if f > 0 and counts[j] == 0 and sum(counts) < len(items):
            counts[j] = 1
    counts[-1] += len(items) - sum(counts)  # remainder into the last group

    groups, start = [], 0
    for c in counts:
        sel = order[start:start + c]
        groups.append([items[k] for k in sel])
        start += c
    return groups


def held_out_instrument_split(manifest: pd.DataFrame, val_frac: float = 0.2, seed: int = 0) -> dict:
    """Split so no acquisition source spans train and val. [STUB]"""
    raise NotImplementedError("held_out_instrument_split is a stub (group by inferred instrument).")


def held_out_class_split(labels: np.ndarray, held_out_classes: list, seed: int = 0) -> dict:
    """Hold out entire sample classes for transfer evaluation. [STUB]"""
    raise NotImplementedError("held_out_class_split is a stub.")
