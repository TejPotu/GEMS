"""Peak tokenizer — embed each peak into a model-dimension token. [CONCRETE nominal+defect; Fourier deferred]

LSM1-MS2-style tokenization, natural for FT-ICR where the exact mass defect is sub-ppm: split each
peak's m/z into a **nominal mass** (integer part → embedding) and an **exact mass defect**
(fractional part → small MLP), and add a **log-intensity** embedding. The Fourier-feature m/z path
(``use_fourier=True``) is a DreaMS port deferred to a later pass; aux chemistry features are deferred
too (the dev model uses neither).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from gems.models.layers.feed_forward import FeedForward


class PeakTokenizer(nn.Module):
    """Map per-peak (m/z, intensity[, aux]) → token embeddings of size ``dim``. [CONCRETE]

    Args:
        dim: output model dimension.
        max_nominal: vocabulary size for the nominal-mass embedding (``floor(m/z)``).
        use_fourier: if True, encode m/z with Fourier features (DreaMS port — not yet implemented).
        aux_features: names of optional chemistry aux features to concatenate (deferred).
        dropout: embedding dropout.
    """

    def __init__(self, dim: int, max_nominal: int = 2000, use_fourier: bool = False,
                 aux_features: Sequence[str] = (), dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.max_nominal = max_nominal
        self.use_fourier = use_fourier
        self.aux_features = tuple(aux_features)

        if use_fourier:
            raise NotImplementedError(
                "PeakTokenizer Fourier path is deferred (DreaMS port). Use nominal+defect "
                "(use_fourier=False) for now."
            )
        if self.aux_features:
            raise NotImplementedError("PeakTokenizer aux-feature concatenation is deferred.")

        self.nominal_emb = nn.Embedding(max_nominal, dim)
        self.defect_mlp = FeedForward(1, dim, hidden_dims=[dim], dropout=dropout)
        self.intensity_mlp = FeedForward(1, dim, hidden_dims=[dim], dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor,
                aux: torch.Tensor | None = None) -> torch.Tensor:
        """Return token embeddings, shape (batch, n_peaks, dim). [CONCRETE]

        ``nominal = floor(mz)`` (clamped) → embedding; ``defect = mz - floor(mz)`` ∈ [0,1) → MLP;
        ``intensity`` (already normalized upstream) → MLP. The three are summed and LayerNorm'd.
        """
        nominal = torch.floor(mz).clamp_(0, self.max_nominal - 1).long()
        defect = (mz - torch.floor(mz)).unsqueeze(-1).to(intensity.dtype)
        inten = intensity.unsqueeze(-1)

        tok = self.nominal_emb(nominal) + self.defect_mlp(defect) + self.intensity_mlp(inten)
        return self.dropout(self.norm(tok))
