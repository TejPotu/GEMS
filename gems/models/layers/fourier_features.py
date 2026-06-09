"""Fourier-feature encoding of m/z. [PORT from DreaMS]

Maps a scalar m/z to a high-dimensional sinusoidal vector ``concat(cos(2π x B), sin(2π x B))`` with
(optionally learnable) frequencies ``B``. This is the key inductive bias that lets the transformer
reason about *relative* m/z, and it transfers directly from DreaMS (the m/z domain is identical).

TODO[PORT]: copy ``dreams/models/layers/fourier_features.py`` near-verbatim (rename ``dreams`` →
``gems``), keeping the ``voronov_et_al`` (log-spaced), ``lin_float_int``, and ``random`` frequency
strategies. Kept as a stub here so the skeleton stays self-contained.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Sinusoidal high-dimensional encoding of a scalar feature (m/z). [PORT]

    Args:
        num_freqs: number of frequencies (output dim is up to 2*num_freqs with funcs='both').
        x_min, x_max: range used by the log-spaced ('voronov_et_al') frequency init.
        strategy: 'voronov_et_al' | 'lin_float_int' | 'random'.
        learnable: make the frequency matrix a trainable parameter.
        funcs: 'both' | 'cos' | 'sin'.
    """

    def __init__(self, num_freqs: int = 512, x_min: float = 1e-4, x_max: float = 1500.0,
                 strategy: str = "voronov_et_al", learnable: bool = True, funcs: str = "both"):
        super().__init__()
        self.num_freqs = num_freqs
        self.x_min = x_min
        self.x_max = x_max
        self.strategy = strategy
        self.learnable = learnable
        self.funcs = funcs
        # TODO[PORT]: build frequency matrix `b` per `strategy`; register as Parameter if learnable
        # else as a buffer. self.out_dim = (2 if funcs == 'both' else 1) * num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` (..., 1) → (..., out_dim). [PORT]"""
        raise NotImplementedError(
            "FourierFeatures.forward is a stub: port DreaMS's `x = 2π·x@b; concat(cos x, sin x)`."
        )
