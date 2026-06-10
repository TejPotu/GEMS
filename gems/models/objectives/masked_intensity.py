"""Masked-intensity repair — ℒ_int (BUILD_PLAN B2). [CONCRETE]

On a separate intensity-weighted subset, hide each peak's **intensity** (keep its m/z) and reconstruct
it from mass + context. In FT-ICR the intensity profile **along a homologous series** is smooth and
chemically meaningful (relative concentration / ionization), so this channel teaches the series
*envelopes* the m/z head never sees. Low weight (λ_int = 0.2 default). Head: a shallow classifier over
log-intensity bins (intensities are normalized to [0, 1] upstream).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gems.models.objectives.base import PretrainObjective


class MaskedIntensity(PretrainObjective):
    """Reconstruct masked (normalized) intensities via binned classification. [CONCRETE]

    Args:
        dim: encoder output dimension.
        n_bins: number of intensity bins over [0, 1].
    """

    def __init__(self, dim: int, n_bins: int = 64):
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        self.head = nn.Linear(dim, n_bins)

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        emb = encoder_out["peak_emb"]
        mask = batch["int_mask"]  # (B, N) bool
        if mask.sum() == 0:
            return {"intensity": emb.sum() * 0.0}

        sel = emb[mask]                              # (M, dim)
        tgt = batch["intensity_target"][mask].clamp(0.0, 1.0)
        bins = (tgt * self.n_bins).long().clamp_(0, self.n_bins - 1)
        return {"intensity": F.cross_entropy(self.head(sel), bins)}
