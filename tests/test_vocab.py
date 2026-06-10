"""Tests for the building-block vocabulary (seed construction + persistence)."""

from __future__ import annotations

from gems.vocab.vocabulary import DeltaVocabulary, seed_vocabulary


def test_seed_vocabulary_nonempty():
    masses = seed_vocabulary()
    assert "CH2" in masses and masses["CH2"] > 14.0


def test_from_seeds_indexing_is_deterministic():
    v = DeltaVocabulary.from_seeds(include_c13=True)
    assert len(v) > 0
    # index order is by ascending mass -> the smallest block ("H2") is index 0
    assert v.index_of(v.names[0]) == 0
    assert v.names == sorted(v.names, key=lambda n: v.masses[n])


def test_json_roundtrip(tmp_path):
    v = DeltaVocabulary.from_seeds(include_c13=False)
    p = tmp_path / "vocab.json"
    v.to_json(p)
    v2 = DeltaVocabulary.from_json(p)
    assert v2.masses == v.masses
    assert "C13" not in v2.masses


def test_masked_edge_sentinel_reserves_a_row():
    from gems.definitions import MASKED_EDGE_TOKEN

    v = DeltaVocabulary.from_seeds()
    # the sentinel occupies exactly one reserved row past the real blocks
    assert v.masked_edge_index == len(v)
    assert v.n_embeddings == len(v) + 1
    assert v.index_of(MASKED_EDGE_TOKEN) == v.masked_edge_index
