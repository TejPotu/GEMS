"""Import every package submodule to prove the skeleton is coherent.

Stubs raise NotImplementedError only when *called*; importing must succeed everywhere.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "gems",
    "gems.definitions",
    "gems.api",
    "gems.cli",
    "gems.data.peaklist",
    "gems.data.mds",
    "gems.data.dformats",
    "gems.data.peak_selection",
    "gems.data.ms_data",
    "gems.data.masking",
    "gems.data.datamodule",
    "gems.data.splits",
    "gems.vocab.building_blocks",
    "gems.vocab.delta_graph",
    "gems.models.layers.fourier_features",
    "gems.models.layers.feed_forward",
    "gems.models.layers.peak_tokenizer",
    "gems.models.layers.attention_bias",
    "gems.models.layers.transformer",
    "gems.models.objectives.pretrain_objectives",
    "gems.models.heads.heads",
    "gems.models.gems.gems",
    "gems.baselines.kmd_vankrevelen_gbm",
    "gems.training.config",
    "gems.training.train",
    "gems.eval.metrics",
    "gems.eval.benchmark",
    "gems.utils.chem",
    "gems.utils.io",
    "gems.utils.logging",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)
