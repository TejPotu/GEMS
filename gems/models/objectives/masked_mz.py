"""Masked-m/z repair — ℒ_mz, the backbone channel (BUILD_PLAN B1). [STUB]

Reconstructing a masked peak's m/z is unsolvable without using the Δm relationships to its neighbors,
so this channel *forces* the model to internalize building-block grammar, homologous series and
Kendrick structure — they fall out latent in it (which is why edge-type/Kendrick/composition are
downstream, not pre-training). Masking is intensity-weighted at 30%, with ~⅓ of masks falling on
2–3 *consecutive* homologous-series members (series-spans) so the model must extrapolate a series and
its defect progression, not just interpolate one edge.

Reconstruction uses **two classification heads** (not regression), so the model can spread probability
when several masses fit a slot:
  - **nominal-mass head** — softmax over integer-Da bins across the corpus m/z range,
  - **mass-defect head** — softmax over fine (~0.1 mDa) defect bins over [0, 1) Da.
``ℒ_mz = ℒ_nominal + ℒ_defect``.

The masked peaks keep their **intensity** as context (masking intensity here hurt in DreaMS; intensity
is repaired separately in the masked-intensity channel). The leakage guard (Δm-type stripped to the
``[masked-edge]`` sentinel on incident edges) is applied upstream when the graph is built.
"""

from __future__ import annotations

import math

import torch

from gems.definitions import DEFECT_BIN_WIDTH_DA, NOMINAL_MASS_BIN_DA
from gems.models.objectives.base import PretrainObjective


class MaskedMz(PretrainObjective):
    """Repair masked m/z with a nominal-mass head + a mass-defect head. [STUB]

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
        self.n_nominal_bins = int(math.ceil(max_mz / nominal_bin_da))
        self.n_defect_bins = int(math.ceil(1.0 / defect_bin_da))
        # TODO[STUB]: nominal_head = Linear(dim, n_nominal_bins); defect_head = Linear(dim, n_defect_bins).

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "MaskedMz.loss is a stub: gather masked-peak embeddings, run the nominal + defect "
            "classification heads, cross-entropy each against the binned targets, "
            "return {'mz_nominal': ..., 'mz_defect': ...}."
        )
