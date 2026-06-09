"""Self-supervised pretraining objectives. [STUB]

Each objective takes a batch and the encoder output and returns a dict of named losses, so the
LightningModule can multi-task them. ``MaskedPeakModeling`` is the workhorse; the others are added in
Phase 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PretrainObjective(nn.Module):
    """Base class. Subclasses own any prediction heads and a ``loss`` method."""

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class MaskedPeakModeling(PretrainObjective):
    """Reconstruct masked peaks' exact mass defect (and/or m/z) from context. [STUB]

    The workhorse objective: predicting a masked peak's sub-ppm mass defect from its
    homologous-series neighbors forces the model to learn the building-block grammar. Pairs with
    :class:`~gems.data.masking.MaskedPeakDataset`.

    Args:
        dim: encoder output dimension.
        predict: 'defect' (regression), 'mz_bin' (classification into m/z bins), or 'both'.
    """

    def __init__(self, dim: int, predict: str = "defect"):
        super().__init__()
        self.dim = dim
        self.predict = predict
        # TODO[STUB]: prediction head(s) mapping peak embeddings → defect / m/z-bin logits.

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "MaskedPeakModeling.loss is a stub: gather masked-peak embeddings, predict the target "
            "(MSE on mass defect and/or CE on m/z bin), reduce over the target mask."
        )


class MaskedIntensity(PretrainObjective):
    """Reconstruct masked log-intensities (homologous-series abundance envelopes). [STUB]"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError("MaskedIntensity.loss is a stub (MSE on masked log-intensity).")


class ContrastiveInvariance(PretrainObjective):
    """SimCLR-style: pull augmented views of the same spectrum together. [STUB]

    Uses the pooled embedding of two augmented views (subsample/jitter/calibration-drift) with an
    NT-Xent loss. Defends against shortcutting to acquisition signatures.
    """

    def __init__(self, dim: int, proj_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError("ContrastiveInvariance.loss is a stub (NT-Xent over view pairs).")


class ElutionOrdering(PretrainObjective):
    """Predict LC-elution order across fractions (DreaMS retention-order analog). [STUB]

    Only applicable to LC-FT-ICR data where fraction/elution metadata exists.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError("ElutionOrdering.loss is a stub (pairwise elution-order ranking).")


# Registry for config-driven dispatch. [CONCRETE]
OBJECTIVES: dict[str, type[PretrainObjective]] = {
    "masked_peak": MaskedPeakModeling,
    "masked_intensity": MaskedIntensity,
    "contrastive": ContrastiveInvariance,
    "elution_order": ElutionOrdering,
}


def build_objectives(cfg, dim: int) -> list[PretrainObjective]:
    """Instantiate the objectives named in ``cfg.objectives`` (list of names). [CONCRETE]"""
    names = cfg["objectives"] if isinstance(cfg, dict) else getattr(cfg, "objectives", ["masked_peak"])
    objs: list[PretrainObjective] = []
    for name in names:
        if name not in OBJECTIVES:
            raise KeyError(f"Unknown objective {name!r}; choices: {sorted(OBJECTIVES)}")
        objs.append(OBJECTIVES[name](dim))
    return objs
