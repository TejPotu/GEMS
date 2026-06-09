"""Lightning DataModules + collation for pretraining and fine-tuning. [STUB]

Mirrors DreaMS's DataModule pattern: ``setup`` builds train/val datasets from an :class:`MSData`
corpus, ``*_dataloader`` wrap them, and ``collate_spectra`` pads a batch and (for Phase 2) attaches
the per-spectrum Δm edges consumed by the attention bias.
"""

from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader


def collate_spectra(batch: list[dict]) -> dict:
    """Collate variable-content spectrum dicts into a padded batch. [STUB]

    Intended: stack ``mz``/``intensity``, build a key-padding mask, and (when a Δm vocabulary is
    attached) attach ``DeltaEdges`` per spectrum for the attention bias.
    """
    raise NotImplementedError("collate_spectra is a stub.")


class PretrainDataModule(pl.LightningDataModule):
    """DataModule for self-supervised pretraining over the FT-ICR corpus. [STUB]

    Args:
        cfg: composed data/pretrain config (paths, batch_size, split, masking, etc.).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        raise NotImplementedError(
            "PretrainDataModule.setup is a stub: build MSData → torch dataset → MaskedPeakDataset, "
            "then split into train/val."
        )

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError("PretrainDataModule.train_dataloader is a stub.")

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError("PretrainDataModule.val_dataloader is a stub.")


class FinetuneDataModule(pl.LightningDataModule):
    """DataModule for supervised fine-tuning (sample classification, class-distribution, …). [STUB]"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        raise NotImplementedError("FinetuneDataModule.setup is a stub.")

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError("FinetuneDataModule.train_dataloader is a stub.")

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError("FinetuneDataModule.val_dataloader is a stub.")
