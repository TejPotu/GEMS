"""GEMS — mass-difference-biased transformer encoder for FT-ICR complex-mixture MS1 spectra.

A DreaMS-style self-supervised model re-pointed from MS/MS small molecules to broadband FT-ICR.
Data IO and chemistry are backed by ``pyc2mc``; the model stack mirrors ``pluskal-lab/DreaMS``.

This package is currently a SKELETON: structure and signatures are in place, most logic is stubbed.
See ``README.md`` and ``PROJECT_IDEA.md``.
"""

__version__ = "0.0.1"

# Keep the top-level import light: do NOT eagerly import torch / pyc2mc here so that
# `import gems` stays cheap and side-effect free. Import submodules explicitly, e.g.
#   from gems.data.peaklist import load_peaklist
#   from gems.models.gems.gems import GEMS

__all__ = ["__version__"]
