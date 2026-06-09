"""IO helpers: HDF5 read/write and path resolution. [STUB + small CONCRETE]"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Best-effort repository root (two levels up from this file). [CONCRETE]"""
    return Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root. [CONCRETE]"""
    p = Path(path)
    return p if p.is_absolute() else (repo_root() / p)


def write_spectrum_group(h5file, key: str, record) -> None:
    """Write one :class:`SpectrumRecord` as an HDF5 group. [STUB]"""
    raise NotImplementedError("write_spectrum_group is a stub.")


def read_spectrum_group(h5file, key: str) -> dict:
    """Read one spectrum group back into a dict of arrays. [STUB]"""
    raise NotImplementedError("read_spectrum_group is a stub.")
