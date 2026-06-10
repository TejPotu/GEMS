"""Smoke test for the end-to-end pretraining step.

This is the gate for the Phase-0/1 build order. While the model stubs are unimplemented it is
expected to xfail (raise NotImplementedError); once the build-order stubs are filled in, flip
`@pytest.mark.xfail` off and assert a finite loss from one CPU `training_step`.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_pyc2mc


@requires_pyc2mc
@pytest.mark.xfail(reason="model + datamodule stubs not implemented yet (skeleton)", strict=False)
def test_one_masked_peak_step():
    import pytorch_lightning as pl

    from gems.data.datamodule import PretrainDataModule
    from gems.models.gems.gems import GEMS
    from gems.training.config import load_config

    cfg = load_config(
        "configs/experiment/gems_pretrain.yaml",
        overrides=["model.dim=32", "pretrain.batch_size=1", "pretrain.trainer.max_steps=1"],
    )
    dm = PretrainDataModule(cfg)
    model = GEMS(cfg, vocab=None)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, logger=False,
                         enable_checkpointing=False)
    trainer.fit(model, dm)
