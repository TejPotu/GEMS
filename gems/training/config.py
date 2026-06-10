"""Config loading / composition / validation via OmegaConf. [CONCRETE load + STUB validate]

Experiment configs compose sub-configs (data/model/attention/peak_selection/pretrain) through a
``defaults:`` list, mirroring a lightweight Hydra-style layout. ``load_config`` resolves that
composition; ``validate_config`` enforces cross-field consistency (e.g. the ``graph`` attention
variant requires a vocab).
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from gems.utils.io import resolve

CONFIGS_DIR = "configs"


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load an experiment config, resolving its ``defaults:`` sub-configs and CLI overrides. [CONCRETE]

    Args:
        path: path to an experiment YAML (e.g. ``configs/experiment/gems_pretrain.yaml``).
        overrides: optional dotlist overrides, e.g. ``["model.dim=32", "pretrain.batch_size=1"]``.

    Returns:
        the composed :class:`DictConfig`.
    """
    path = resolve(path)
    root = path.parent.parent  # .../configs
    cfg = OmegaConf.load(path)

    defaults = cfg.pop("defaults", []) if "defaults" in cfg else []
    composed = OmegaConf.create({})
    for entry in defaults:
        # entry is a mapping like {group: name} → load configs/<group>/<name>.yaml under key <group>
        for group, name in dict(entry).items():
            sub = OmegaConf.load(Path(root) / group / f"{name}.yaml")
            composed = OmegaConf.merge(composed, OmegaConf.create({group: sub}))

    composed = OmegaConf.merge(composed, cfg)  # experiment-level keys win over defaults
    if overrides:
        composed = OmegaConf.merge(composed, OmegaConf.from_dotlist(overrides))
    return composed


def validate_config(cfg: DictConfig) -> None:
    """Check cross-field consistency before a run. [STUB]

    Intended checks: a biased ``attention.variant`` (``graph`` / ``dense_edge_bias``) ⇒ a vocab
    source is configured; ``peak_selection.strategy == 'graph_induced'`` ⇒ a vocab is available;
    ``peak_selection.n_peaks <= model max``; device is one of cpu/mps/cuda. TODO.
    """
    raise NotImplementedError("validate_config is a stub.")
