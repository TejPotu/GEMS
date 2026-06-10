"""Tests that the experiment configs compose correctly via OmegaConf."""

from __future__ import annotations

import pytest

from gems.training.config import load_config

EXPERIMENTS = [
    "configs/experiment/gems_pretrain.yaml",
    "configs/experiment/sanity_no_bias.yaml",
    "configs/experiment/sanity_dense_edge_bias.yaml",
]


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_experiment_composes(path):
    cfg = load_config(path)
    # sub-config groups are present after composition
    for group in ("data", "model", "attention", "peak_selection", "pretrain"):
        assert group in cfg, f"missing composed group {group!r} in {path}"
    assert cfg.model.dim > 0
    assert cfg.attention.variant in ("no_bias", "graph", "dense_edge_bias")
    # the single locked pre-training objective
    assert cfg.pretrain.objective == "denoising"


def test_locked_design_is_graph_and_graph_induced():
    cfg = load_config("configs/experiment/gems_pretrain.yaml")
    assert cfg.attention.variant == "graph"
    assert cfg.peak_selection.strategy == "graph_induced"
    assert cfg.model.pooling == "attention"


def test_overrides_apply():
    cfg = load_config("configs/experiment/gems_pretrain.yaml", overrides=["model.dim=32"])
    assert cfg.model.dim == 32
