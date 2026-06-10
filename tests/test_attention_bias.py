"""Tests for the attention-bias strategy registry + the concrete NoBias floor."""

from __future__ import annotations

import torch

from gems.models.layers.attention_bias import (
    ATTENTION_BIASES,
    DenseEdgeBias,
    GraphDeltaBias,
    NoBias,
    build_attention_bias,
)
from gems.vocab.vocabulary import DeltaVocabulary


def test_registry_has_all_variants():
    assert set(ATTENTION_BIASES) == {"no_bias", "graph", "dense_edge_bias"}


def test_no_bias_is_identity():
    bias = NoBias()
    logits = torch.randn(2, 2, 5, 5)
    out = bias(logits, delta_edges=None, n_peaks=5)
    assert torch.equal(out, logits)


def test_build_dispatch():
    assert isinstance(build_attention_bias({"variant": "no_bias"}), NoBias)
    vocab = DeltaVocabulary.from_seeds()

    # locked primary: sparse mask + abundance-weighted edge bias; table includes the masked-edge row
    g = build_attention_bias({"variant": "graph", "n_heads": 2}, vocab=vocab)
    assert isinstance(g, GraphDeltaBias)
    assert g.bias.shape == (vocab.n_embeddings, 2)

    # reduced-N sanity check
    d = build_attention_bias({"variant": "dense_edge_bias", "n_heads": 2}, vocab=vocab)
    assert isinstance(d, DenseEdgeBias)
    assert d.bias.shape == (vocab.n_embeddings, 2)
