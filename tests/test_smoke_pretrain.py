"""End-to-end CPU smoke test for the denoising pretraining step.

Pass 1 (``sanity_no_bias``: plain transformer + top-N peaks) runs end to end and must yield a finite
loss. The locked ``gems_pretrain`` path (sparse Δm-graph attention) stays xfail until Pass 2 fills in
the graph stubs (``build_delta_graph``, ``GraphInduced.select``, ``GraphDeltaBias.forward``).
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_pyc2mc

SMOKE_OVERRIDES = [
    "model.dim=32",
    "data.limit=4",
    "pretrain.batch_size=2",
    "pretrain.trainer.max_steps=1",
]


@requires_pyc2mc
def test_one_denoising_step_no_bias():
    import pytorch_lightning as pl
    import torch

    from gems.data.datamodule import PretrainDataModule
    from gems.models.gems.gems import GEMS
    from gems.training.config import load_config

    cfg = load_config("configs/experiment/sanity_no_bias.yaml", overrides=SMOKE_OVERRIDES)
    dm = PretrainDataModule(cfg, vocab=None)
    model = GEMS(cfg, vocab=None)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, logger=False,
                         enable_checkpointing=False, num_sanity_val_steps=0)
    trainer.fit(model, dm)
    assert trainer.state.finished

    # one explicit forward → denoising loss must be finite
    batch = next(iter(dm.train_dataloader()))
    losses = model.objective.loss(batch, model(batch))
    assert torch.isfinite(losses["total"]), losses
    for key in ("mz_nominal", "mz_defect", "intensity", "replaced", "total"):
        assert key in losses


@requires_pyc2mc
@pytest.mark.xfail(reason="locked graph path: Δm-graph stubs land in Pass 2", strict=False)
def test_one_denoising_step_graph():
    import pytorch_lightning as pl

    from gems.data.datamodule import PretrainDataModule
    from gems.models.gems.gems import GEMS
    from gems.training.config import load_config

    cfg = load_config("configs/experiment/gems_pretrain.yaml", overrides=SMOKE_OVERRIDES)
    vocab = None
    dm = PretrainDataModule(cfg, vocab=vocab)
    model = GEMS(cfg, vocab=vocab)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, logger=False,
                         enable_checkpointing=False, num_sanity_val_steps=0)
    trainer.fit(model, dm)
