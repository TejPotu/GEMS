# GEMS — Mass-Difference-Biased Transformer for FT-ICR Complex Mixtures

A DreaMS-style self-supervised transformer encoder, **re-pointed from MS/MS small molecules to
MS1 broadband FT-ICR spectra of complex mixtures** (petroleum, dissolved organic matter, bio-oils).
Each resolved peak is a token; the abundant pairwise mass differences in a spectrum
(Δm: CH₂ = 14.01565, H₂ = 2.01565, O = 15.9949, S = 31.97207, CF₂ = 49.9968 …) act as an
**attention bias**, so peaks linked by an abundant building-block Δm attend to one another. The
model outputs per-peak contextual embeddings plus a pooled whole-spectrum embedding.

> **Honest novelty framing.** The *mechanism* (Graphormer edge-bias attention over pairwise Δm)
> is adapted from [DreaMS](https://github.com/pluskal-lab/DreaMS), not invented here. The defensible
> contribution is the **domain + setting**: FT-ICR / complex mixtures, MS1-only (no fragmentation),
> an abundance-weighted mixture-specific building-block vocabulary, a repository-scale corpus, and a
> new downstream benchmark. See `PROJECT_IDEA.md` for the full spec.

## Status: SKELETON

This repository is a **scaffold**. The architecture, module boundaries, configs, CLI, and training
entry points are in place, but most logic is **stubbed** (`raise NotImplementedError` + TODO, with
docstrings describing intended behavior). The data-IO and chemistry layers are *thin wrappers over
[`pyc2mc`](#data-backex-pyc2mc)*, so they are concrete from day one. Stub conventions used in the code:

| Tag in docstrings | Meaning |
|---|---|
| `[CONCRETE]` | Real logic now (low-ambiguity, e.g. constants, schema) |
| `[WRAP]`     | Thin adapter over `pyc2mc` |
| `[PORT]`     | To be copied near-verbatim from DreaMS (`dreams` → `gems`) |
| `[STUB]`     | Signature + docstring + `NotImplementedError`/TODO |

## Data backend: pyc2mc

The corpus in `data/` was produced by **pyc2mc** (the `.pks` headers read `PyC2MC pks - …`; the
`mds_csv/` schema is exactly `MassDifferencesSpectrum.md_data`). GEMS therefore *depends on
pyc2mc* rather than re-implementing IO/chemistry:

- **Reading `.pks`** → `pyc2mc.io.peaklist.read_pks` → a `PeakList` exposing `.mz`, `.intensity`
  (the Abundance column), `.SN`, `.resolving_power`, `.frequency`, `.polarity`, `.metadata`.
- **Mass-difference spectra** → `pyc2mc.processing.mass_differences_spectrum.MassDifferencesSpectrum`
  → the 20-column `.md_data`; `.assign_peaks(...)` even auto-labels Δm distributions with chemical
  formulas, so the building-block vocabulary is *discovered + named*, not hardcoded.
- **Chemistry** → `pyc2mc.core.formula.Formula` (`exact_mass`, `nominal_mass`, `dbe_value`,
  `chem_class`), `pyc2mc.core.kendrick.KendrickSeries`, KMD via `PeakList.get_kendrick_mass_defects`.

## Data

`data/` (already present):

- `walking_calibrated_pks/` — 272 walking-calibrated FT-ICR `.pks` peak lists (~4.8k–42k peaks each).
- `mds_csv/` — 285 per-sample pairwise Δm-distribution CSVs (the raw material for the vocabulary).
- `mds_run_manifest.csv`, `mds_run.log`, `mds_loop.log` — provenance of the Δm computation run.

## Install

```bash
conda env create -f env.yml -n gems && conda activate gems
pip install -e /Users/tejapotu/School/Projects/Maglab/pyc2mc   # the data/chemistry backend
pip install -e ".[dev]"                                        # this package
```

GPU later: swap in the CUDA torch wheel (`pip install torch --index-url …/whl/cu121`); no other change.

## Repository layout (mirrors `pluskal-lab/DreaMS`)

```
gems/
  definitions.py            # canonical schema / constants / registry
  api.py  cli.py            # inference surface + fire CLI
  data/                     # pyc2mc-backed IO, formats, peak selection, masking, datamodules
  vocab/                    # building-block Δm vocabulary + per-spectrum Δm graph
  models/
    layers/                 # fourier_features, peak_tokenizer, feed_forward, transformer, attention_bias
    objectives/             # masked-peak / intensity / contrastive / elution-order SSL
    heads/                  # frozen-backbone fine-tuning heads
    gems/gems.py    # the main LightningModule
  baselines/                # KMD/van-Krevelen + GBM (the non-deep bar to beat)
  training/  eval/  utils/
```

## Phased plan (`PROJECT_IDEA.md`)

| Phase | Goal | Selected by |
|---|---|---|
| 0 | Data + preprocessing pipeline; Δm vocabulary | `gems build-corpus` / `build-vocab` |
| 1 | Baseline: plain transformer, masked-peak pretraining, **no Δm bias** | `attention: no_bias` |
| 2 | Add mass-difference attention; ablate edge-bias vs sparse-mask, seeded vs learned vocab | `attention: edge_bias` / `sparse_mask` |
| 3 | Scale corpus; add contrastive + LC-ordering objectives | `pretrain: multitask` |
| 4 | Downstream eval suite vs baselines | `gems` finetune + `eval/benchmark.py` |
| 5 | Wire embedding into formula-assignment assist / Analysis Companion | `api.py` |

**Phase 1 → 2 is a config flag** (`configs/attention/*.yaml`) over a swappable `AttentionBias`
strategy — the critical ablation needs no structural change.

## Quickstart / dev smoke test (CPU)

Once the stubs in the [build order](#build-order) are filled in:

```bash
gems build-vocab  --mds_dir data/mds_csv               --out data/processed/vocab.json
gems build-corpus --pks_dir data/walking_calibrated_pks --out data/processed/dev.h5 --limit 8
gems smoke        # read .pks → tokenize → tiny transformer → one masked-peak step on CPU
pytest -q
```

## Build order

Fill in stubs in this order to reach the end-to-end CPU smoke test:
`definitions.py` → `data/peaklist.py` → `utils/chem.py` (ppm helpers) → `data/peak_selection.py`
(TopN) → `data/dformats.py` → `models/layers/peak_tokenizer.py` → `models/layers/attention_bias.py`
(NoBias) → `models/layers/transformer.py` → `models/objectives` (MaskedPeak) + `data/masking.py` →
`data/ms_data.py` + `data/datamodule.py` → `models/gems/gems.py` →
`training/{train,config}.py` + `cli.py smoke` → `tests/test_smoke_pretrain.py`. Phase-2
(`vocab/*`, `EdgeBias`/`SparseMask`) then plugs in via the `attention` config.

## Lineage

DreaMS (Δm attention; Nat. Biotechnol. 2025), LSM1-MS2 (nominal/defect tokenization),
GLEAMS (repository-scale embedding), MIST (formula/neutral-loss featurization).
