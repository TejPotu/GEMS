"""Base class for the pre-training repair channels. [CONCRETE interface]

Each channel takes the shared corrupted batch and the single encoder pass and returns a dict of named
losses, so the :class:`~gems.models.objectives.denoising.SpectrumDenoising` composer can sum them into
the one repair objective ``ℒ = ℒ_mz + λ_int·ℒ_int + λ_rpd·ℒ_rpd``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PretrainObjective(nn.Module):
    """Base channel. Subclasses own any prediction heads and a ``loss`` method.

    Contract: ``loss(batch, encoder_out)`` reads the corrupted view + targets from ``batch`` and the
    per-peak / pooled embeddings from ``encoder_out``, and returns ``{name: scalar_tensor}``.
    """

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError
