"""Configurable feed-forward MLP block. [CONCRETE]

A small reusable MLP (depth, hidden dims, dropout, activation, optional ``act_last``). No
MS-specific logic — mirrors DreaMS's ``feed_forward.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Configurable MLP. [CONCRETE]

    Args:
        in_dim: input dimension.
        out_dim: output dimension.
        hidden_dims: explicit hidden sizes (a sequence), or an int ``depth`` which builds ``depth``
            hidden layers linearly interpolated between ``in_dim`` and ``out_dim``.
        dropout: dropout probability applied after each activation.
        activation: activation module factory (default ``nn.GELU``).
        act_last: apply the activation (and dropout) after the final linear layer.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dims: Sequence[int] | int = (), dropout: float = 0.0,
                 activation=nn.GELU, act_last: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if isinstance(hidden_dims, int):
            # `hidden_dims` interpreted as a depth: interpolate sizes in_dim -> out_dim.
            depth = hidden_dims
            if depth <= 0:
                hidden = []
            else:
                step = (out_dim - in_dim) / (depth + 1)
                hidden = [int(round(in_dim + step * (k + 1))) for k in range(depth)]
        else:
            hidden = list(hidden_dims)

        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for k in range(len(dims) - 1):
            is_last = k == len(dims) - 2
            layers.append(nn.Linear(dims[k], dims[k + 1]))
            if not is_last or act_last:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
