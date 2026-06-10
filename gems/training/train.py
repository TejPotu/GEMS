"""Training orchestration — pretraining and fine-tuning entry points. [STUB body, CONCRETE wiring]

Mirrors DreaMS's ``training/train.py``: build the vocab + datamodule + model, configure a Lightning
``Trainer`` (CPU-first; DDP for scale-up), and fit. The function bodies are stubbed but the wiring
(which pieces connect to which) is spelled out so fill-in is mechanical.
"""

from __future__ import annotations

from gems.training.config import load_config


def pretrain(config: str, overrides: list[str] | None = None):
    """Self-supervised pretraining. [STUB]

    Wiring (to implement):
        cfg = load_config(config, overrides)
        vocab = DeltaVocabulary.from_seeds(...) or build_vocabulary_from_corpus(...)   # for graph attention
        dm = PretrainDataModule(cfg)
        model = GEMS(cfg, vocab=vocab)
        trainer = pl.Trainer(accelerator=cfg.device, devices=..., max_steps=cfg.pretrain.trainer.max_steps, ...)
        trainer.fit(model, dm)
    """
    import pytorch_lightning as pl

    from gems.data.datamodule import PretrainDataModule
    from gems.models.gems.gems import GEMS
    from gems.vocab.vocabulary import DeltaVocabulary

    cfg = load_config(config, overrides)

    attention = cfg.get("attention")
    variant = attention.get("variant", "graph")
    # vocab supplies the edge-bias block table; the no_bias floor needs none.
    vocab = None if variant == "no_bias" else DeltaVocabulary.from_seeds(
        include_c13=bool(attention.get("include_c13", True)))

    dm = PretrainDataModule(cfg, vocab=vocab)
    model = GEMS(cfg, vocab=vocab)

    tcfg = cfg.get("pretrain").get("trainer", {})
    trainer = pl.Trainer(
        accelerator=cfg.get("device", "cpu"),
        devices=tcfg.get("devices", 1),
        max_steps=tcfg.get("max_steps", 1000),
        precision=tcfg.get("precision", 32),
        log_every_n_steps=tcfg.get("log_every_n_steps", 10),
    )
    trainer.fit(model, dm)
    return trainer


def finetune(config: str, overrides: list[str] | None = None):
    """Supervised fine-tuning of a frozen-backbone head. [STUB]"""
    cfg = load_config(config, overrides)
    raise NotImplementedError(
        "finetune is a stub. Build FinetuneDataModule → FineTuningHead(backbone_ckpt) → Trainer.fit. "
        f"(Loaded config keys: {list(cfg.keys())})"
    )


def main():
    """fire entry point exposing pretrain/finetune. [CONCRETE wiring]"""
    import fire
    fire.Fire({"pretrain": pretrain, "finetune": finetune})


if __name__ == "__main__":
    main()
