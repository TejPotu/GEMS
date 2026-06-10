"""``MSData`` — the HDF5-backed FT-ICR corpus. [STUB]

Analog of DreaMS's ``MSData``: decouples messy raw inputs (``.pks`` via pyc2mc) from the tensor
pipeline by materializing a single ML-friendly HDF5 store, then exposing a torch ``Dataset``.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from gems.data.dformats import DataFormat, DF_FTICR_DEV


class MSData:
    """A corpus of FT-ICR spectra persisted to HDF5.

    Build it once from a directory of ``.pks`` files (``from_pks_dir``), then ``load`` and
    ``to_torch_dataset`` for training. Each item is a dict of canonical arrays (see
    :class:`~gems.data.peaklist.SpectrumRecord`).
    """

    def __init__(self, hdf5_path: str | Path):
        self.hdf5_path = str(hdf5_path)

    # ---- construction --------------------------------------------------------------------
    @classmethod
    def from_pks_dir(
        cls,
        pks_dir: str | Path,
        out_hdf5: str | Path,
        dformat: DataFormat = DF_FTICR_DEV,
        limit: int | None = None,
    ) -> "MSData":
        """Build an HDF5 corpus by reading every ``.pks`` via pyc2mc. [STUB]

        Intended: iterate ``pks_dir/*.pks`` (up to ``limit``), call
        ``data.peaklist.load_record``, apply ``dformat`` (normalize/select/pad), and write one
        HDF5 group/dataset per spectrum (plus filename → metadata for splitting).
        """
        raise NotImplementedError(
            "MSData.from_pks_dir is a stub: loop load_record(.pks) → dformat → write HDF5."
        )

    @classmethod
    def load(cls, hdf5_path: str | Path) -> "MSData":
        """Open an existing HDF5 corpus. [STUB]"""
        raise NotImplementedError("MSData.load is a stub.")

    # ---- access --------------------------------------------------------------------------
    def __len__(self) -> int:
        raise NotImplementedError("MSData.__len__ is a stub.")

    def __getitem__(self, i: int) -> dict:
        raise NotImplementedError("MSData.__getitem__ is a stub (returns one spectrum dict).")

    def to_torch_dataset(self, transform=None) -> Dataset:
        """Return a torch ``Dataset`` view over the corpus, applying ``transform`` per item. [STUB]"""
        raise NotImplementedError("MSData.to_torch_dataset is a stub.")

    def attach_delta_vocab(self, vocab) -> None:
        """Attach a :class:`~gems.vocab.vocabulary.DeltaVocabulary` for Δm-graph building. [STUB]"""
        raise NotImplementedError("MSData.attach_delta_vocab is a stub.")
