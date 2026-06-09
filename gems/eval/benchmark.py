"""Downstream benchmark runner — encoder vs hand-engineered baselines. [STUB]

Defining this benchmark is itself a contribution (PROJECT_IDEA.md). Every encoder task is reported
alongside the KMD/van-Krevelen + GBM baseline; if the encoder doesn't beat it, the architecture
isn't earning its keep.
"""

from __future__ import annotations


def run_benchmark(model, datamodule, baselines, tasks):
    """Evaluate ``model`` and ``baselines`` across ``tasks`` and return a comparison table. [STUB]

    Tasks: sample classification, class-distribution regression, formula disambiguation,
    cross-sample similarity, van Krevelen region prediction. Always include the non-deep baseline.
    """
    raise NotImplementedError("run_benchmark is a stub.")
