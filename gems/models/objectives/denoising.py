"""Spectrum denoising / repair — the single pre-training objective (BUILD_PLAN Part B). [STUB loss, CONCRETE wiring]

Pre-training is ONE self-supervised objective: corrupt the peak set three independent ways and
reconstruct the original. The three channels share **one** corrupted view (different peaks receive
different corruptions), the encoder runs **once**, and per-peak heads repair each channel:

    ℒ = ℒ_mz + λ_int·ℒ_int + λ_rpd·ℒ_rpd

Conceptually still a denoising autoencoder over the peak set, so the chemistry stays *emergent*, not
hand-labelled. The corrupted view (intensity-weighted masking + series-spans + replaced peaks + the
``[masked-edge]`` leakage guard) is produced upstream by
:class:`~gems.data.masking.SpectrumDenoisingDataset`.
"""

from __future__ import annotations

import torch

from gems.definitions import LAMBDA_INT, LAMBDA_RPD
from gems.models.objectives.base import PretrainObjective
from gems.models.objectives.masked_intensity import MaskedIntensity
from gems.models.objectives.masked_mz import MaskedMz
from gems.models.objectives.replaced_peak import ReplacedPeakDetection


class SpectrumDenoising(PretrainObjective):
    """Compose the three repair channels into the single weighted denoising loss. [STUB loss]

    Args:
        dim: encoder output dimension.
        max_mz: corpus m/z ceiling (sizes the nominal-mass head).
        lambda_int: weight of the masked-intensity channel (λ_int, default 0.2).
        lambda_rpd: weight of the replaced-peak channel (λ_rpd, default 0.5).
    """

    def __init__(self, dim: int, max_mz: float = 1500.0,
                 lambda_int: float = LAMBDA_INT, lambda_rpd: float = LAMBDA_RPD):
        super().__init__()
        self.lambda_int = lambda_int
        self.lambda_rpd = lambda_rpd
        self.masked_mz = MaskedMz(dim, max_mz=max_mz)            # backbone (ℒ_mz)
        self.masked_intensity = MaskedIntensity(dim)            # ℒ_int
        self.replaced_peak = ReplacedPeakDetection(dim)         # ℒ_rpd

    def loss(self, batch: dict, encoder_out: dict) -> dict[str, torch.Tensor]:
        """Run every channel over the shared encoder pass and combine. [STUB]

        Intended (once the channel heads are implemented)::

            losses = {}
            losses |= self.masked_mz.loss(batch, encoder_out)            # mz_nominal, mz_defect
            losses |= self.masked_intensity.loss(batch, encoder_out)     # intensity
            losses |= self.replaced_peak.loss(batch, encoder_out)        # replaced
            total = (losses['mz_nominal'] + losses['mz_defect']
                     + self.lambda_int * losses['intensity']
                     + self.lambda_rpd * losses['replaced'])
            return {**losses, 'total': total}
        """
        raise NotImplementedError(
            "SpectrumDenoising.loss is a stub: sum masked_mz + λ_int·masked_intensity + "
            "λ_rpd·replaced_peak over the single shared encoder pass into {'...': ..., 'total': ...}."
        )


def build_denoising_objective(cfg, dim: int, max_mz: float = 1500.0) -> SpectrumDenoising:
    """Instantiate the locked denoising objective from a pretrain config. [CONCRETE]

    Reads ``lambda_int`` / ``lambda_rpd`` (falling back to the BUILD_PLAN defaults 0.2 / 0.5).
    """
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    return SpectrumDenoising(
        dim,
        max_mz=max_mz,
        lambda_int=float(get("lambda_int", LAMBDA_INT)),
        lambda_rpd=float(get("lambda_rpd", LAMBDA_RPD)),
    )
