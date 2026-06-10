"""``MSData`` — the FT-ICR spectrum corpus. [CONCRETE in-memory; HDF5 persistence deferred]

Analog of DreaMS's ``MSData``: decouples messy raw inputs (``.pks`` via pyc2mc) from the tensor
pipeline. Pass 1 keeps the corpus in memory (272 spectra is small); HDF5 materialization via
``utils.io`` is a later scale-up refinement. Build it once from a directory of ``.pks`` files
(``from_pks_dir``), then ``to_torch_dataset(transform)`` for training.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

from torch.utils.data import Dataset

from gems.data.dformats import DataFormat, DF_FTICR_DEV
from gems.data.peaklist import SpectrumRecord, load_record

logger = logging.getLogger(__name__)


class _RecordDataset(Dataset):
    """Torch ``Dataset`` over in-memory records, applying ``transform`` per item."""

    def __init__(self, records: list[SpectrumRecord], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        return self.transform(rec) if self.transform is not None else rec


class MSData:
    """An in-memory corpus of FT-ICR spectra (``SpectrumRecord`` list)."""

    def __init__(self, records: list[SpectrumRecord]):
        self.records = records
        self.vocab = None

    # ---- construction --------------------------------------------------------------------
    @classmethod
    def from_pks_dir(
        cls,
        pks_dir: str | Path,
        out_hdf5: str | Path | None = None,
        dformat: DataFormat = DF_FTICR_DEV,
        limit: int | None = None,
    ) -> "MSData":
        """Build a corpus by reading every ``.pks`` via pyc2mc. [CONCRETE in-memory]

        ``out_hdf5`` is accepted for forward compatibility but persistence is deferred — the corpus
        is held in memory for now.
        """
        files = sorted(glob.glob(str(Path(pks_dir) / "*.pks")))
        if limit is not None:
            files = files[:limit]
        if not files:
            raise FileNotFoundError(f"No .pks files under {pks_dir!r}")

        records = []
        for f in files:
            try:
                records.append(load_record(f))
            except Exception as e:  # skip unreadable spectra rather than aborting the whole corpus
                logger.warning("skipping %s: %s", f, e)
        if not records:
            raise RuntimeError(f"No readable .pks spectra under {pks_dir!r}")
        if out_hdf5 is not None:
            logger.info("MSData: HDF5 persistence deferred; holding %d spectra in memory", len(records))
        return cls(records)

    @classmethod
    def load(cls, hdf5_path: str | Path) -> "MSData":
        """Open an existing HDF5 corpus. [STUB — HDF5 persistence deferred]"""
        raise NotImplementedError("MSData.load is a stub (HDF5 persistence deferred to scale-up).")

    # ---- access --------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> SpectrumRecord:
        return self.records[i]

    def to_torch_dataset(self, transform=None) -> Dataset:
        """Return a torch ``Dataset`` view over the corpus, applying ``transform`` per item. [CONCRETE]"""
        return _RecordDataset(self.records, transform)

    def attach_delta_vocab(self, vocab) -> None:
        """Attach a :class:`~gems.vocab.vocabulary.DeltaVocabulary` for Δm-graph building (Pass 2)."""
        self.vocab = vocab
