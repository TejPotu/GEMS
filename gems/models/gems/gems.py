"""The main GEMS encoder. [STUB]

A ``LightningModule`` that ties the stack together:

    PeakTokenizer → DeltaTransformerEncoder (swappable AttentionBias) → {per-peak, pooled} embeddings

and multi-tasks the self-supervised objectives during pretraining. Mirrors DreaMS's main module.
The Phase 1 ↔ 2 distinction lives entirely in the ``AttentionBias`` chosen by config.
"""

from __future__ import annotations

import pytorch_lightning as pl

from gems.models.layers.attention_bias import build_attention_bias
from gems.models.layers.peak_tokenizer import PeakTokenizer
from gems.models.layers.transformer import DeltaTransformerEncoder
from gems.models.objectives.pretrain_objectives import build_objectives


class GEMS(pl.LightningModule):
    """FT-ICR mass-difference transformer encoder. [STUB]

    Args:
        cfg: composed config (model dims, attention variant, pretrain objectives, optimizer).
        vocab: a :class:`~gems.vocab.building_blocks.DeltaVocabulary` (supplies block count for
            the edge-bias variant); may be None for the ``no_bias`` baseline.
    """

    def __init__(self, cfg, vocab=None):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.save_hyperparameters(ignore=["vocab"])

        m = cfg["model"] if isinstance(cfg, dict) else cfg.model
        dim = m["dim"]

        self.tokenizer = PeakTokenizer(
            dim=dim,
            max_nominal=m.get("max_nominal", 2000),
            use_fourier=m.get("use_fourier", False),
            aux_features=m.get("aux_features", ()),
        )
        attn_bias = build_attention_bias(cfg["attention"] if isinstance(cfg, dict) else cfg.attention, vocab)
        self.encoder = DeltaTransformerEncoder(
            dim=dim,
            n_layers=m["n_layers"],
            n_heads=m["n_heads"],
            ff_mult=m.get("ff_mult", 4),
            dropout=m.get("dropout", 0.0),
            attention_bias=attn_bias,
            pooling=m.get("pooling", "cls"),
        )
        self.objectives = build_objectives(
            cfg["pretrain"] if isinstance(cfg, dict) else cfg.pretrain, dim=dim
        )

    # ---- forward / inference -------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """Encode a batch → {'peak_emb': (B,N,dim), 'pooled_emb': (B,dim)}. [STUB]"""
        raise NotImplementedError(
            "GEMS.forward is a stub: tokenizer(mz,intensity,aux) → encoder(x, delta_edges, mask)."
        )

    def embed(self, batch: dict) -> dict:
        """Inference-only forward (no grad); returns peak + pooled embeddings. [STUB]"""
        raise NotImplementedError("GEMS.embed is a stub.")

    # ---- training ------------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int):
        """Sum every objective's losses, log them, return the total. [STUB]"""
        raise NotImplementedError(
            "GEMS.training_step is a stub: out = self(batch); total = sum over "
            "obj.loss(batch, out) for obj in self.objectives; log and return total."
        )

    def validation_step(self, batch: dict, batch_idx: int):
        raise NotImplementedError("GEMS.validation_step is a stub.")

    def configure_optimizers(self):
        """AdamW + linear warmup / cosine decay (read from cfg). [STUB]"""
        raise NotImplementedError("GEMS.configure_optimizers is a stub.")

    # ---- (de)serialization ---------------------------------------------------------------
    @classmethod
    def from_ckpt(cls, ckpt_path: str, **kw) -> "GEMS":
        """Load a pretrained encoder from a Lightning checkpoint. [STUB]"""
        raise NotImplementedError("GEMS.from_ckpt is a stub.")
