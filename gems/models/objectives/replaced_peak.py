"""Replaced-peak detection — ℒ_rpd, Electra-style (BUILD_PLAN B3). [CONCRETE]

On a third subset (default 15% of peaks), each selected peak's m/z is **replaced** with a
plausible-but-wrong value (keeping intensity). A binary head then classifies every *genuine-or-replaced*
peak real/fake (BCE). It forces learning which masses are chemically consistent with the rest of the
mixture — a global, label-free signal — and produces signal on far more peaks than masking alone.

Peaks corrupted by the *other* channels (m/z-masked, intensity-masked) are excluded from this head's
loss: their corruption is a different mechanism, and a blanked m/z is not what "fake" should mean here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gems.models.objectives.base import PretrainObjective


class ReplacedPeakDetection(PretrainObjective):
    """Binary real/fake head over every (non-masked) valid peak. [CONCRETE]

    Args:
        dim: encoder output dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.head = nn.Linear(dim, 1)

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        emb = encoder_out["peak_emb"]
        logits = self.head(emb).squeeze(-1)  # (B, N)
        eval_mask = batch["valid_mask"] & ~batch["mz_mask"] & ~batch["int_mask"]
        if eval_mask.sum() == 0:
            return {"replaced": emb.sum() * 0.0}
        return {"replaced": F.binary_cross_entropy_with_logits(
            logits[eval_mask], batch["replaced_label"][eval_mask])}
