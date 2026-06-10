"""Lightning DataModules + collation for pretraining and fine-tuning. [CONCRETE pretrain; finetune STUB]

``setup`` builds train/val datasets from an :class:`MSData` corpus, ``*_dataloader`` wrap them, and
``collate_spectra`` stacks the per-spectrum dicts into a padded batch. For Pass 2 the collate also
attaches the per-spectrum Δm edges (with the ``[masked-edge]`` leakage guard) consumed by the
attention bias; in Pass 1 (``no_bias``) there are no edges.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from gems.data.dformats import DataFormat
from gems.data.masking import SpectrumDenoisingDataset
from gems.data.ms_data import MSData
from gems.data.peak_selection import build_peak_selector
from gems.data.splits import random_split


def collate_spectra(batch: list[dict]) -> dict:
    """Collate fixed-size spectrum dicts into a padded batch. [CONCRETE]

    Fixed-size keys are stacked into (B, N) tensors. The ragged per-spectrum Δm edges (the
    ``edge_*`` keys, present only when a vocab is attached) are concatenated into a single batched
    ``delta_edges`` dict with a ``batch_idx`` selector — the form the attention bias consumes.
    """
    fixed = [k for k in batch[0] if not k.startswith("edge_")]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in fixed}

    if "edge_src" in batch[0]:
        batch_idx = torch.cat([
            torch.full((b["edge_src"].numel(),), i, dtype=torch.long) for i, b in enumerate(batch)
        ])
        out["delta_edges"] = {
            "batch_idx": batch_idx,
            "src": torch.cat([b["edge_src"] for b in batch]),
            "dst": torch.cat([b["edge_dst"] for b in batch]),
            "block_idx": torch.cat([b["edge_block"] for b in batch]),
            "weight": torch.cat([b["edge_weight"] for b in batch]),
        }
    return out


class PretrainDataModule(pl.LightningDataModule):
    """DataModule for self-supervised pretraining over the FT-ICR corpus. [CONCRETE]

    Args:
        cfg: composed data/pretrain config (paths, batch_size, split, masking, etc.).
        vocab: optional :class:`DeltaVocabulary`; required by graph-induced selection (Pass 2),
            ignored by the ``top_n`` sanity path (Pass 1).
    """

    def __init__(self, cfg, vocab=None):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.train_ds: Subset | None = None
        self.val_ds: Subset | None = None

    def setup(self, stage: str | None = None) -> None:
        data = self.cfg.get("data")
        pre = self.cfg.get("pretrain")
        seed = int(self.cfg.get("seed", 0))

        selector = build_peak_selector(self.cfg.get("peak_selection"), vocab=self.vocab)
        dformat = DataFormat(
            max_peaks=int(data.get("max_peaks", 256)),
            max_mz=float(data.get("max_mz", 1500.0)),
            log_intensity=bool(data.get("log_intensity", True)),
        )
        corpus = MSData.from_pks_dir(data.get("pks_dir"), dformat=dformat, limit=data.get("limit"))
        if self.vocab is not None:
            corpus.attach_delta_vocab(self.vocab)

        attention = self.cfg.get("attention", {})
        base = corpus.to_torch_dataset(transform=lambda rec: dformat(rec, selector))
        denoise = SpectrumDenoisingDataset(
            base,
            mask_prob=float(pre.get("mask_prob", 0.30)),
            replaced_fraction=float(pre.get("replaced_fraction", 0.15)),
            vocab=self.vocab,
            seed=seed,
            ppm_tol=float(attention.get("ppm_tol", 1.0)) if attention is not None else 1.0,
            degree_cap=int(attention.get("degree_cap", 32)) if attention is not None else 32,
        )

        splits = data.get("splits", {})
        val_frac = float(splits.get("val_frac", 0.2)) if splits is not None else 0.2
        train_idx, val_idx = random_split(list(range(len(denoise))), (1.0 - val_frac, val_frac), seed=seed)
        self.train_ds = Subset(denoise, train_idx)
        self.val_ds = Subset(denoise, val_idx)

    def _loader(self, ds: Subset, shuffle: bool) -> DataLoader:
        batch_size = int(self.cfg.get("pretrain").get("batch_size", 4))
        # num_workers=0: the dformat transform is a closure (not picklable for multiprocessing).
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_spectra, num_workers=0)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)


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
