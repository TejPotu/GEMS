"""Unit tests for the per-spectrum Δm graph + the leakage guard (no pyc2mc needed)."""

from __future__ import annotations

import numpy as np

from gems.definitions import SEED_BUILDING_BLOCKS
from gems.vocab.graph import build_delta_graph
from gems.vocab.vocabulary import DeltaVocabulary


def _vocab() -> DeltaVocabulary:
    return DeltaVocabulary.from_seeds(include_c13=True)


def test_ch2_series_forms_edges():
    v = _vocab()
    ch2 = SEED_BUILDING_BLOCKS["CH2"]
    # three peaks one CH2 apart → a homologous series: edges (0,1) and (1,2) at minimum.
    mz = np.array([300.0, 300.0 + ch2, 300.0 + 2 * ch2])
    edges = build_delta_graph(mz, v, ppm_tol=1.0, degree_cap=32)

    assert edges.src.size >= 2
    ch2_idx = v.index_of("CH2")
    assert (edges.block_idx == ch2_idx).all()
    # every peak participates in the series
    assert edges.node_mask().all()


def test_no_edges_when_delta_matches_nothing():
    v = _vocab()
    mz = np.array([300.0, 300.0 + 0.337])  # not near any building-block Δm
    edges = build_delta_graph(mz, v, ppm_tol=1.0)
    assert edges.src.size == 0


def test_leakage_guard_strips_incident_edges():
    v = _vocab()
    ch2 = SEED_BUILDING_BLOCKS["CH2"]
    mz = np.array([300.0, 300.0 + ch2, 300.0 + 2 * ch2])
    edges = build_delta_graph(mz, v, ppm_tol=1.0)

    masked = np.array([False, True, False])  # mask the middle peak
    guarded = edges.with_masked_edges(masked, v.masked_edge_index)

    incident = masked[guarded.src] | masked[guarded.dst]
    # edges touching the masked peak are stripped to the sentinel with zero weight
    assert (guarded.block_idx[incident] == v.masked_edge_index).all()
    assert (guarded.weight[incident] == 0.0).all()
    # connectivity is preserved (same number of edges, same endpoints)
    assert np.array_equal(guarded.src, edges.src)
    assert np.array_equal(guarded.dst, edges.dst)
