"""Non-deep baseline: KMD / van Krevelen features + gradient boosting. [STUB]

The bar the encoder must clear. Features are exactly the hand-engineered descriptors petroleomics
already uses (Kendrick mass defect histograms, van Krevelen density grids, abundance statistics),
fed to a gradient-boosting classifier/regressor (sklearn). If GEMS embeddings don't beat this,
the Δm-attention machinery isn't justified.
"""

from __future__ import annotations

import numpy as np


def extract_kmd_vk_features(record, n_kmd_bins: int = 64, vk_grid: int = 20) -> np.ndarray:
    """Hand-engineered feature vector for one spectrum. [STUB]

    Intended features:
      - Kendrick mass-defect histogram (via pyc2mc ``get_kendrick_mass_defects``),
      - van Krevelen (O/C, H/C) density grid for assigned peaks,
      - abundance / m/z summary statistics.
    """
    raise NotImplementedError("extract_kmd_vk_features is a stub.")


def train_gbm_baseline(X: np.ndarray, y: np.ndarray, task: str = "classification"):
    """Fit a gradient-boosting model (sklearn) for a downstream task. [STUB]"""
    raise NotImplementedError("train_gbm_baseline is a stub (sklearn GradientBoosting{Classifier,Regressor}).")


def evaluate_baseline(model, X: np.ndarray, y: np.ndarray, task: str = "classification") -> dict:
    """Evaluate a fitted baseline and return metrics matching the encoder benchmark. [STUB]"""
    raise NotImplementedError("evaluate_baseline is a stub.")
