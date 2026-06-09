"""Logging / experiment-tracker setup. [STUB + small CONCRETE]"""

from __future__ import annotations

import logging


def get_logger(name: str = "gems", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdlib logger. [CONCRETE]"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def build_experiment_logger(cfg):
    """Build a Lightning logger (W&B if configured, else CSV/Tensorboard). [STUB]"""
    raise NotImplementedError("build_experiment_logger is a stub (wandb optional, see [train] extra).")
