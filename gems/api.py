"""Public inference API. [STUB]

The user-facing surface (analog of DreaMS's ``api.py``): turn ``.pks`` files into embeddings or
downstream predictions with a pretrained model. Bodies are stubbed until a model is trained.
"""

from __future__ import annotations

from pathlib import Path


def embed_spectrum(pks_path: str | Path, model: str = "gems_dev") -> dict:
    """Embed a single ``.pks`` spectrum → {'peak_emb', 'pooled_emb'}. [STUB]

    Wiring: load_record(pks_path) → dformat → GEMS.from_ckpt(MODEL_REGISTRY[model]).embed(...).
    """
    raise NotImplementedError("embed_spectrum is a stub.")


def embed_corpus(pks_dir: str | Path, model: str, out: str | Path) -> None:
    """Embed every ``.pks`` in a directory and persist the pooled-embedding matrix. [STUB]"""
    raise NotImplementedError("embed_corpus is a stub.")


def predict(pks_path: str | Path, head: str) -> dict:
    """Run a fine-tuned head (e.g. sample classification) on a spectrum. [STUB]"""
    raise NotImplementedError("predict is a stub.")
