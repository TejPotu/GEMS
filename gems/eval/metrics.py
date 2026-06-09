"""Evaluation metrics — torchmetrics wrappers + project-specific measures. [STUB]"""

from __future__ import annotations


def class_distribution_error(pred, true):
    """Error between predicted and true heteroatom-class distributions (e.g. KL / L1). [STUB]"""
    raise NotImplementedError("class_distribution_error is a stub.")


def van_krevelen_region_iou(pred, true):
    """IoU between predicted and true occupied van Krevelen regions. [STUB]"""
    raise NotImplementedError("van_krevelen_region_iou is a stub.")


def embedding_retrieval_metrics(embeddings, labels):
    """Cross-sample similarity retrieval metrics (e.g. top-k accuracy, mAP). [STUB]"""
    raise NotImplementedError("embedding_retrieval_metrics is a stub.")
