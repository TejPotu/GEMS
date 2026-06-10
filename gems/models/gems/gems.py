"""The main GEMS encoder. [STUB]

A ``LightningModule`` that ties the stack together:

    PeakTokenizer → DeltaTransformerEncoder (GraphDeltaBias) → {per-peak, attention-pooled} embeddings

and trains the single self-supervised denoising objective during pre-training
(``ℒ = ℒ_mz + λ_int·ℒ_int + λ_rpd·ℒ_rpd``). Mirrors DreaMS's main module, but the readout is
**attention-pooling** over the final peak embeddings (no CLS master node — a CLS would need edges to
all peaks, breaking the Δm-graph sparsity).
"""

from __future__ import annotations

import pytorch_lightning as pl

from gems.models.layers.attention_bias import build_attention_bias
from gems.models.layers.peak_tokenizer import PeakTokenizer
from gems.models.layers.transformer import DeltaTransformerEncoder
from gems.models.objectives.denoising import build_denoising_objective


class GEMS(pl.LightningModule):
    """FT-ICR mass-difference transformer encoder. [STUB]

    Args:
        cfg: composed config (model dims, attention variant, denoising objective, optimizer).
        vocab: a :class:`~gems.vocab.vocabulary.DeltaVocabulary` (supplies the block count, incl. the
            masked-edge sentinel, for the ``graph`` / ``dense_edge_bias`` variants); may be None for
            the ``no_bias`` floor.
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
            pooling=m.get("pooling", "attention"),
        )
        data_cfg = (cfg["data"] if isinstance(cfg, dict) else cfg.data)
        max_mz = (data_cfg.get("max_mz", 1500.0) if isinstance(data_cfg, dict) else getattr(data_cfg, "max_mz", 1500.0))
        self.objective = build_denoising_objective(
            cfg["pretrain"] if isinstance(cfg, dict) else cfg.pretrain, dim=dim, max_mz=max_mz
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
        """One encoder pass → denoising loss (ℒ_mz + λ_int·ℒ_int + λ_rpd·ℒ_rpd); log and return total. [STUB]"""
        raise NotImplementedError(
            "GEMS.training_step is a stub: out = self(batch); losses = self.objective.loss(batch, out); "
            "log each; return losses['total']."
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
