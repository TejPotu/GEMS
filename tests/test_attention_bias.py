"""Tests for the attention-bias strategy registry + the concrete NoBias baseline."""

from __future__ import annotations

import torch

from gems.models.layers.attention_bias import (
    ATTENTION_BIASES,
    EdgeBias,
    NoBias,
    SparseMask,
    build_attention_bias,
)
from gems.vocab.building_blocks import DeltaVocabulary


def test_registry_has_all_variants():
    assert set(ATTENTION_BIASES) == {"no_bias", "edge_bias", "sparse_mask"}


def test_no_bias_is_identity():
    bias = NoBias()
    logits = torch.randn(2, 2, 5, 5)
    out = bias(logits, delta_edges=None, n_peaks=5)
    assert torch.equal(out, logits)


def test_build_dispatch():
    assert isinstance(build_attention_bias({"variant": "no_bias"}), NoBias)
    vocab = DeltaVocabulary.from_seeds()
    eb = build_attention_bias({"variant": "edge_bias", "n_heads": 2}, vocab=vocab)
    assert isinstance(eb, EdgeBias)
    assert eb.bias.shape == (len(vocab), 2)
    assert isinstance(build_attention_bias({"variant": "sparse_mask"}), SparseMask)
