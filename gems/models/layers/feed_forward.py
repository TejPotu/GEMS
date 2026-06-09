"""Configurable feed-forward MLP block. [PORT from DreaMS]

A small reusable MLP (depth, hidden dims, dropout, activation, optional ``act_last``). No
MS-specific logic, so it copies over from DreaMS unchanged.

TODO[PORT]: copy ``dreams/models/layers/feed_forward.py`` near-verbatim.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Configurable MLP. [PORT]

    Args:
        in_dim: input dimension.
        out_dim: output dimension.
        hidden_dims: explicit hidden sizes, or an int depth with 'interpolated' sizing.
        dropout: dropout probability between layers.
        activation: activation module factory (default nn.ReLU).
        act_last: apply the activation after the final linear layer.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dims: Sequence[int] | int = (), dropout: float = 0.0,
                 activation=nn.ReLU, act_last: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # TODO[PORT]: assemble nn.Sequential of Linear/activation/dropout per DreaMS.
        self.net: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FeedForward.forward is a stub: port DreaMS's MLP assembly.")
