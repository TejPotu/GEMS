"""Replaced-peak detection — ℒ_rpd, Electra-style (BUILD_PLAN B3). [STUB]

On a third subset (default 15% of peaks), **replace** each selected peak's m/z with a
*plausible-but-wrong* value — shift by a non-vocabulary Δm, or by a vocabulary Δm that breaks defect
consistency — keeping intensity. A binary head then classifies **every** peak real/fake (BCE).

Two reasons it earns its place: (a) it forces learning which masses are **chemically consistent with
the rest of the mixture** — a global, label-free signal, not local arithmetic; (b) it produces a
training signal on **100% of peaks** (vs ~30% for masking), which matters a lot on a 272-sample
corpus. Weight λ_rpd = 0.5 default.
"""

from __future__ import annotations

import torch

from gems.models.objectives.base import PretrainObjective


class ReplacedPeakDetection(PretrainObjective):
    """Binary real/fake head over every peak (Electra-style discriminator). [STUB]

    Args:
        dim: encoder output dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # TODO[STUB]: classifier head dim -> 1 (logit per peak).

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "ReplacedPeakDetection.loss is a stub: run the per-peak binary head over ALL peak "
            "embeddings, BCE-with-logits against batch['replaced_label'] (1=replaced), masking "
            "padding, return {'replaced': ...}."
        )
