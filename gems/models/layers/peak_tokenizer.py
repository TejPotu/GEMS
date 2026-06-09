"""Peak tokenizer — embed each peak into a model-dimension token. [STUB]

LSM1-MS2-style tokenization, natural for FT-ICR where the exact mass defect is sub-ppm: split each
peak's m/z into a **nominal mass** (integer) and an **exact mass defect** (fractional part) and embed
them separately, then add a **log-intensity** embedding. Optionally use the DreaMS Fourier-feature
m/z path instead, and optionally concatenate chemistry aux features (Kendrick mass defect, van
Krevelen O/C & H/C, DBE — computed via pyc2mc).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class PeakTokenizer(nn.Module):
    """Map per-peak (m/z, intensity[, aux]) → token embeddings of size ``dim``. [STUB]

    Args:
        dim: output model dimension.
        max_nominal: vocabulary size for the nominal-mass embedding (``floor(m/z)``).
        use_fourier: if True, encode m/z with :class:`FourierFeatures` instead of nominal+defect.
        aux_features: names of optional chemistry aux features to concatenate (e.g. ``("kmd","dbe")``).
        dropout: embedding dropout.
    """

    def __init__(self, dim: int, max_nominal: int = 2000, use_fourier: bool = False,
                 aux_features: Sequence[str] = (), dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.max_nominal = max_nominal
        self.use_fourier = use_fourier
        self.aux_features = tuple(aux_features)
        # TODO[STUB]: nn.Embedding(max_nominal, dim) for nominal mass; small MLP/FourierFeatures for
        # the exact mass defect; FeedForward for log-intensity; projection to `dim`; aux MLP.

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor,
                aux: torch.Tensor | None = None) -> torch.Tensor:
        """Return token embeddings, shape (batch, n_peaks, dim). [STUB]

        Intended: ``nominal = floor(mz).long().clamp(max=max_nominal-1)``;
        ``defect = mz - nominal``; embed nominal (Embedding), defect (MLP/Fourier), and
        ``log1p(intensity)`` (MLP); sum/concat → project to ``dim``.
        """
        raise NotImplementedError(
            "PeakTokenizer.forward is a stub: nominal-mass embedding + exact-defect embedding + "
            "log-intensity embedding (+ optional Fourier / aux), projected to `dim`."
        )
