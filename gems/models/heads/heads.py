"""Fine-tuning heads — frozen backbone + swappable task head. [STUB]

Mirrors DreaMS's ``heads.py``: :class:`FineTuningHead` loads a pretrained :class:`GEMS` backbone
(frozen by default, optionally unfrozen at ``unfreeze_backbone_at_epoch``), extracts the pooled
embedding, and feeds a task head. The downstream tasks are the project's benchmark.
"""

from __future__ import annotations

import pytorch_lightning as pl

from gems.definitions import HETEROATOM_CLASSES, SAMPLE_CLASSES


class FineTuningHead(pl.LightningModule):
    """Base: load a (frozen) pretrained backbone and attach a task head. [STUB]

    Args:
        backbone_ckpt: path to a pretrained GEMS checkpoint.
        unfreeze_backbone_at_epoch: -1 keeps the backbone frozen; >=0 unfreezes at that epoch.
        use_pooled: use only the pooled (whole-spectrum) embedding vs all peak embeddings.
    """

    def __init__(self, backbone_ckpt: str | None = None,
                 unfreeze_backbone_at_epoch: int = -1, use_pooled: bool = True):
        super().__init__()
        self.backbone_ckpt = backbone_ckpt
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.use_pooled = use_pooled
        # TODO[STUB]: load GEMS.from_ckpt(backbone_ckpt); freeze params; build task head in subclass.

    def forward(self, batch: dict):
        raise NotImplementedError("FineTuningHead.forward is a stub (embed → task head).")

    def configure_optimizers(self):
        raise NotImplementedError("FineTuningHead.configure_optimizers is a stub.")


class SampleClassificationHead(FineTuningHead):
    """Whole-spectrum classification (petroleum / DOM / PFAS / lipid …). [STUB]

    The higher-value downstream task and the cleanest place to beat the KMD/van-Krevelen + GBM
    baseline. PFAS doubles as a vocabulary-discovery probe (CF2 block).
    """

    def __init__(self, n_classes: int = len(SAMPLE_CLASSES), **kw):
        super().__init__(**kw)
        self.n_classes = n_classes


class ClassDistributionRegressionHead(FineTuningHead):
    """Regress a sample's heteroatom-class distribution (CHO/CHOS/…). [STUB]

    Requires integrating across the peak network, so it actually stresses the Δm attention — the
    version that matters (vs trivially-recoverable peak-level class).
    """

    def __init__(self, n_classes: int = len(HETEROATOM_CLASSES), **kw):
        super().__init__(**kw)
        self.n_classes = n_classes


class FormulaDisambiguationHead(FineTuningHead):
    """Rank candidate formulas for high-mass peaks where several fit within tolerance. [STUB]

    Directly useful to the pyc2mc assignment workflow / Analysis Companion.
    """


class CrossSampleSimilarityHead(FineTuningHead):
    """Contrastive/metric head for cross-sample similarity (a 'petroleomics GLEAMS'). [STUB]"""


class VanKrevelenRegionHead(FineTuningHead):
    """Predict van Krevelen (O/C, H/C) region occupancy from the embedding. [STUB]"""


class EdgeTypeProbeHead(FineTuningHead):
    """Δm / edge-type prediction as an *interpretability probe*, not a pre-training objective. [STUB]

    Given two peak embeddings, name the building block linking them (e.g. confirm CF2 is learned).
    As an objective it is trivial when masses are visible (it collapses into masked-m/z), so per
    BUILD_PLAN Part D it lives here, on the pretrained encoder.
    """


class ElutionOrderHead(FineTuningHead):
    """Predict LC-elution order across fractions — LC-FT-ICR only (BUILD_PLAN Part D). [STUB]

    Optional, never part of base pre-training: from two pooled ``z`` predict fraction order. Only
    applicable where fraction/elution metadata exists.
    """
