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
        vocab = DeltaVocabulary.from_seeds(...) or build_vocabulary_from_corpus(...)   # for edge_bias
        dm = PretrainDataModule(cfg)
        model = GEMS(cfg, vocab=vocab)
        trainer = pl.Trainer(accelerator=cfg.device, devices=..., max_steps=cfg.pretrain.trainer.max_steps, ...)
        trainer.fit(model, dm)
    """
    cfg = load_config(config, overrides)  # concrete: prove config composition works
    raise NotImplementedError(
        "pretrain is a stub. Build vocab → PretrainDataModule → GEMS → pl.Trainer.fit. "
        f"(Loaded config keys: {list(cfg.keys())})"
    )


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
