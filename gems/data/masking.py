"""Spectrum-denoising corruption — the single shared corrupted view (BUILD_PLAN Part B). [CONCRETE]

The three repair channels (masked-m/z, masked-intensity, replaced-peak) share **one** corrupted view:
different peaks get different corruptions, the encoder runs once, and per-peak heads repair each. This
module builds that view and the per-channel targets, plus the masked-node mask the Δm-graph leakage
guard consumes.

Corruptions (disjoint subsets of a spectrum's valid peaks):
  - **mask-m/z** (default 30%, sampled ∝ intensity): blank the m/z (keep intensity); target = original.
  - **mask-intensity** (separate subset): blank the intensity (keep m/z); target = original.
  - **replace** (default 15%): swap in a plausible-but-wrong m/z (keep intensity); label the peak fake.

Series-span masking (~⅓ of m/z masks on consecutive homologous-series members) needs the Δm graph and
arrives in Pass 2; Pass 1 uses plain intensity-weighted masking.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from gems.definitions import (
    DEFAULT_DEGREE_CAP,
    DEFAULT_MASK_PROB,
    DEFAULT_PPM_TOLERANCE,
    REPLACED_PEAK_FRACTION,
    SERIES_SPAN_FRACTION,
)


class SpectrumDenoisingDataset(Dataset):
    """Wrap a (dformatted) spectrum dataset and emit one corrupted view + per-channel targets. [CONCRETE]

    The base dataset yields fixed-size dicts ``{mz, intensity, valid_mask}``. Each item here adds the
    corrupted ``mz``/``intensity``, the original-value targets, the per-channel masks, the per-peak
    ``replaced_label``, and (for Pass 2's leakage guard) ``masked_nodes`` == ``mz_mask``.

    Args:
        base: an underlying ``Dataset`` of fixed-size spectrum dicts.
        mask_prob: fraction of valid peaks whose m/z is masked (∝ intensity).
        int_mask_prob: fraction (of the remaining valid peaks) whose intensity is masked.
        series_span_fraction: share of m/z masks placed on consecutive series members (Pass 2).
        replaced_fraction: fraction of valid peaks given a plausible-but-wrong m/z.
        vocab: a :class:`~gems.vocab.vocabulary.DeltaVocabulary` for drawing replacement Δm (Pass 2).
        seed: base RNG seed (per-item stream is ``seed + index`` for reproducibility).
    """

    def __init__(self, base: Dataset, mask_prob: float = DEFAULT_MASK_PROB,
                 int_mask_prob: float = 0.15,
                 series_span_fraction: float = SERIES_SPAN_FRACTION,
                 replaced_fraction: float = REPLACED_PEAK_FRACTION, vocab=None, seed: int = 0,
                 ppm_tol: float = DEFAULT_PPM_TOLERANCE, degree_cap: int = DEFAULT_DEGREE_CAP):
        self.base = base
        self.mask_prob = mask_prob
        self.int_mask_prob = int_mask_prob
        self.series_span_fraction = series_span_fraction
        self.replaced_fraction = replaced_fraction
        self.vocab = vocab
        self.seed = seed
        self.ppm_tol = ppm_tol
        self.degree_cap = degree_cap

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        item = self.base[i]
        mz0 = np.asarray(item["mz"], dtype=np.float32)
        int0 = np.asarray(item["intensity"], dtype=np.float32)
        valid = np.asarray(item["valid_mask"], dtype=bool)
        rng = np.random.default_rng(self.seed + i)

        # disjoint corruption subsets over valid peaks
        mz_mask = intensity_weighted_mask(int0, self.mask_prob, valid, None, rng)
        replaced_mask = intensity_weighted_mask(int0, self.replaced_fraction, valid, mz_mask, rng)
        int_mask = intensity_weighted_mask(int0, self.int_mask_prob, valid, mz_mask | replaced_mask, rng)

        # build the single corrupted view
        mz = mz0.copy()
        mz[mz_mask] = 0.0                                   # blank m/z (keep intensity)
        mz = replace_with_implausible_mz(mz, np.where(replaced_mask)[0], self.vocab, rng)
        intensity = int0.copy()
        intensity[int_mask] = 0.0                           # blank intensity (keep m/z)

        replaced_label = replaced_mask.astype(np.float32)

        out = {
            "mz": torch.from_numpy(mz),
            "intensity": torch.from_numpy(intensity),
            "mz_target": torch.from_numpy(mz0),
            "intensity_target": torch.from_numpy(int0),
            "mz_mask": torch.from_numpy(mz_mask),
            "int_mask": torch.from_numpy(int_mask),
            "valid_mask": torch.from_numpy(valid),
            "replaced_label": torch.from_numpy(replaced_label),
            "masked_nodes": torch.from_numpy(mz_mask),       # leakage-guard input
        }
        if self.vocab is not None:
            out.update(self._build_edges(mz0, valid, mz_mask))
        return out

    def _build_edges(self, mz0: np.ndarray, valid: np.ndarray, mz_mask: np.ndarray) -> dict:
        """Δm graph over the *original* selected m/z (valid peaks only), with the leakage guard. [CONCRETE]

        Built on uncorrupted masses so the connectivity reflects the true mixture; the masked peaks'
        incident edges are then stripped to the ``[masked-edge]`` sentinel so a typed edge can't leak
        the masked mass. Padding peaks (invalid) carry sentinel m/z 0 and would form spurious edges,
        so they are excluded by passing them a non-matching mass.
        """
        from gems.vocab.graph import build_delta_graph

        mz = mz0.astype(np.float64).copy()
        mz[~valid] = -1.0e6  # push padding far away so it matches no building block
        edges = build_delta_graph(mz, self.vocab, ppm_tol=self.ppm_tol, degree_cap=self.degree_cap)
        edges = edges.with_masked_edges(mz_mask, self.vocab.masked_edge_index)
        return {
            "edge_src": torch.as_tensor(edges.src, dtype=torch.long),
            "edge_dst": torch.as_tensor(edges.dst, dtype=torch.long),
            "edge_block": torch.as_tensor(edges.block_idx, dtype=torch.long),
            "edge_weight": torch.as_tensor(edges.weight, dtype=torch.float32),
        }


# ---- corruption primitives -----------------------------------------------------------------

def intensity_weighted_mask(intensity: np.ndarray, frac: float, valid_mask=None,
                            exclude_mask=None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a boolean mask over peaks, drawn ∝ intensity. [CONCRETE]

    Chooses ``round(frac * |candidates|)`` peaks from ``valid & ~exclude`` without replacement, with
    probability proportional to intensity (uniform fallback if all candidate intensities are 0).
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    n = intensity.shape[0]
    rng = rng or np.random.default_rng()

    candidates = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).copy()
    if exclude_mask is not None:
        candidates &= ~np.asarray(exclude_mask, dtype=bool)

    cand_idx = np.where(candidates)[0]
    mask = np.zeros(n, dtype=bool)
    k = int(round(frac * cand_idx.size))
    if k <= 0 or cand_idx.size == 0:
        return mask

    w = intensity[cand_idx].copy()
    w[w < 0] = 0.0
    total = w.sum()
    p = (w / total) if total > 0 else None     # None -> uniform
    chosen = rng.choice(cand_idx, size=min(k, cand_idx.size), replace=False, p=p)
    mask[chosen] = True
    return mask


def series_span_mask(mz: np.ndarray, n_spans: int, base: str = "C H2",
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Mask 2–3 *consecutive* members of a homologous (e.g. CH2) series. [STUB — Pass 2]

    Needs the Δm graph / Kendrick series to identify consecutive members; deferred until the graph
    layer lands. Pass 1 uses :func:`intensity_weighted_mask` only.
    """
    raise NotImplementedError("series_span_mask is a stub (needs the Δm graph; Pass 2).")


def replace_with_implausible_mz(mz: np.ndarray, idx: np.ndarray, vocab=None,
                                rng: np.random.Generator | None = None) -> np.ndarray:
    """Shift selected peaks' m/z by a plausible-but-wrong (defect-breaking) Δm. [CONCRETE]

    Pass 1 applies a random non-integer shift in [0.2, 3.0) Da, which breaks defect consistency while
    staying in a believable mass range. Pass 2 can draw from non-vocabulary Δm via ``vocab``.
    """
    mz = np.asarray(mz, dtype=np.float32).copy()
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return mz
    rng = rng or np.random.default_rng()
    shifts = rng.uniform(0.2, 3.0, size=idx.size).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=idx.size).astype(np.float32)
    mz[idx] = np.clip(mz[idx] + signs * shifts, 0.0, None)
    return mz
