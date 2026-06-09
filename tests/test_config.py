"""Tests that the experiment configs compose correctly via OmegaConf."""

from __future__ import annotations

import pytest

from gems.training.config import load_config

EXPERIMENTS = [
    "configs/experiment/phase1_baseline.yaml",
    "configs/experiment/phase2_edge_bias.yaml",
    "configs/experiment/phase2_sparse_mask.yaml",
]


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_experiment_composes(path):
    cfg = load_config(path)
    # sub-config groups are present after composition
    for group in ("data", "model", "attention", "peak_selection", "pretrain"):
        assert group in cfg, f"missing composed group {group!r} in {path}"
    assert cfg.model.dim > 0
    assert cfg.attention.variant in ("no_bias", "edge_bias", "sparse_mask")


def test_phase1_is_no_bias():
    cfg = load_config("configs/experiment/phase1_baseline.yaml")
    assert cfg.attention.variant == "no_bias"
    assert cfg.phase == 1


def test_overrides_apply():
    cfg = load_config("configs/experiment/phase1_baseline.yaml", overrides=["model.dim=32"])
    assert cfg.model.dim == 32
