"""Masked-intensity repair — ℒ_int (BUILD_PLAN B2). [STUB]

On a separate intensity-weighted subset, hide each peak's **intensity** (keep its m/z) and reconstruct
it from mass + context. DreaMS found masking intensity *hurts* — but that is MS2, where intensity is
fragmentation-dependent noise. In FT-ICR the intensity profile **along a homologous series** is smooth
and chemically meaningful (relative concentration / ionization), so this channel teaches the series
*envelopes* the m/z head never sees. Low weight (λ_int = 0.2 default).
"""

from __future__ import annotations

import torch

from gems.models.objectives.base import PretrainObjective


class MaskedIntensity(PretrainObjective):
    """Reconstruct masked log-intensities (homologous-series abundance envelopes). [STUB]

    Args:
        dim: encoder output dimension.
        n_bins: log-intensity bins for the classification head (or set 0 for scalar regression).
    """

    def __init__(self, dim: int, n_bins: int = 64):
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        # TODO[STUB]: shallow FFN head dim -> n_bins (binned CE) or dim -> 1 (regression).

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "MaskedIntensity.loss is a stub: gather intensity-masked peak embeddings, predict the "
            "binned log-intensity (CE) or scalar (MSE), return {'intensity': ...}."
        )
