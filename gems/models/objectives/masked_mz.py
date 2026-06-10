"""Masked-m/z repair — ℒ_mz, the backbone channel (BUILD_PLAN B1). [CONCRETE]

Reconstructing a masked peak's m/z is unsolvable without using the Δm relationships to its neighbors,
so this channel *forces* the model to internalize building-block grammar, homologous series and
Kendrick structure — they fall out latent in it (which is why edge-type/Kendrick/composition are
downstream, not pre-training).

Reconstruction uses **two classification heads** (not regression), so the model can spread probability
when several masses fit a slot:
  - **nominal-mass head** — softmax over integer-Da bins across the corpus m/z range,
  - **mass-defect head** — softmax over fine (~0.1 mDa) defect bins over [0, 1) Da.
``ℒ_mz = ℒ_nominal + ℒ_defect``.

The masked peaks keep their **intensity** as context; intensity is repaired separately. The leakage
guard (Δm-type stripped to the ``[masked-edge]`` sentinel on incident edges) is applied upstream.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gems.definitions import DEFECT_BIN_WIDTH_DA, NOMINAL_MASS_BIN_DA
from gems.models.objectives.base import PretrainObjective


class MaskedMz(PretrainObjective):
    """Repair masked m/z with a nominal-mass head + a mass-defect head. [CONCRETE]

    Args:
        dim: encoder output dimension.
        max_mz: corpus m/z ceiling → number of nominal-mass bins (``max_mz / nominal_bin_da``).
        nominal_bin_da: width of the nominal-mass bins (default 1 Da).
        defect_bin_da: width of the mass-defect bins (default ~0.1 mDa over [0,1) Da).
    """

    def __init__(self, dim: int, max_mz: float = 1500.0,
                 nominal_bin_da: float = NOMINAL_MASS_BIN_DA,
                 defect_bin_da: float = DEFECT_BIN_WIDTH_DA):
        super().__init__()
        self.dim = dim
        self.nominal_bin_da = nominal_bin_da
        self.defect_bin_da = defect_bin_da
        self.n_nominal_bins = int(math.ceil(max_mz / nominal_bin_da))
        self.n_defect_bins = int(math.ceil(1.0 / defect_bin_da))
        self.nominal_head = nn.Linear(dim, self.n_nominal_bins)
        self.defect_head = nn.Linear(dim, self.n_defect_bins)

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        emb = encoder_out["peak_emb"]
        mask = batch["mz_mask"]  # (B, N) bool
        if mask.sum() == 0:
            z = emb.sum() * 0.0
            return {"mz_nominal": z, "mz_defect": z}

        sel = emb[mask]                       # (M, dim)
        tgt_mz = batch["mz_target"][mask]     # (M,)
        floor = torch.floor(tgt_mz)
        nominal_target = (floor / self.nominal_bin_da).long().clamp_(0, self.n_nominal_bins - 1)
        defect_target = ((tgt_mz - floor) / self.defect_bin_da).long().clamp_(0, self.n_defect_bins - 1)

        l_nominal = F.cross_entropy(self.nominal_head(sel), nominal_target)
        l_defect = F.cross_entropy(self.defect_head(sel), defect_target)
        return {"mz_nominal": l_nominal, "mz_defect": l_defect}
