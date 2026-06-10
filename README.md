# GEMS — Mass-Difference-Biased Transformer for FT-ICR Complex Mixtures

A DreaMS-style self-supervised transformer encoder, **re-pointed from MS/MS small molecules to
MS1 broadband FT-ICR spectra of complex mixtures** (petroleum, dissolved organic matter, bio-oils).
Each resolved peak is a token; the abundant pairwise mass differences in a spectrum
(Δm: CH₂ = 14.01565, H₂ = 2.01565, O = 15.9949, S = 31.97207, CF₂ = 49.9968 …) act as an
**attention bias**, so peaks linked by an abundant building-block Δm attend to one another. The
model outputs per-peak contextual embeddings plus a pooled whole-spectrum embedding.

> **Honest novelty framing.** The *mechanism* (attention biased by pairwise Δm) is adapted from
> [DreaMS](https://github.com/pluskal-lab/DreaMS), not invented here. The defensible contribution is
> the **domain + setting** plus two FT-ICR-specific commitments: attention that is **sparse over the
> Δm graph *and* abundance-weighted** (each edge carries its Δm type *and* that Δm's per-spectrum
> abundance as the bias), and a pre-training **leakage guard** (edges incident to a masked peak are
> stripped to a `[masked-edge]` sentinel so a typed edge can't reveal the masked mass). The rest:
> MS1-only (no fragmentation), a mixture-specific learnable building-block vocabulary, a
> repository-scale corpus, and a new downstream benchmark. See `BUILD_PLAN.md` for the locked design
> and `PROJECT_IDEA.md` for the original spec.

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
  data/                     # pyc2mc-backed IO, formats, graph-induced peak selection, masking, datamodules
  vocab/                    # vocabulary.py (building-block Δm vocab) + graph.py (per-spectrum Δm graph)
  models/
    layers/                 # fourier_features, peak_tokenizer, feed_forward, transformer, attention_bias
    objectives/             # spectrum-denoising channels: masked_mz, masked_intensity, replaced_peak → denoising
    heads/                  # frozen-backbone downstream heads
    gems/gems.py            # the main LightningModule
  baselines/                # KMD/van-Krevelen + GBM (the non-deep bar to beat)
  training/  eval/  utils/
```

## Locked design (`BUILD_PLAN.md`)

One committed design, not a phased table. The commitments:

- **Graph-induced peak selection** — a peak is a token iff it sits on ≥1 abundant Δm edge (no Top-N stage).
- **Sparse, abundance-weighted Δm-graph attention** — `A_ij = (qᵢ·kⱼ)/√d + b(Δm_type, log_abund)` for
  graph-linked pairs, −∞ otherwise. Mask *and* edge bias come from the same edge set.
- **One pre-training objective: spectrum denoising / repair** — corrupt the peak set three ways over a
  single shared view and reconstruct: `ℒ = ℒ_mz + λ_int·ℒ_int + λ_rpd·ℒ_rpd` (masked m/z with nominal +
  defect classification heads, masked intensity, Electra-style replaced-peak detection).
- **Attention-pool readout** (no CLS master node, which would break Δm-graph sparsity).
- **Leakage guard** — masked peaks' incident edges are stripped to the `[masked-edge]` sentinel.

Everything else — edge-type/Kendrick probes, sample classification, class-distribution regression,
formula disambiguation, similarity, elution-order — is **downstream** on the pretrained encoder.

The locked run is `configs/experiment/gems_pretrain.yaml`. Two config-switchable **sanity checks** (not
milestones) live beside it: `sanity_no_bias.yaml` (plain transformer — the floor the Δm attention must
beat) and `sanity_dense_edge_bias.yaml` (dense edge bias, reduced N — isolates bias-helps from
sparsity-helps). The attention mechanism is a swappable `AttentionBias` strategy
(`configs/attention/{graph,no_bias,dense_edge_bias}.yaml`), so the sanity checks need no structural change.

## Quickstart / dev smoke test (CPU)

Once the stubs in the [build order](#build-order) are filled in:

```bash
gems build-vocab  --mds_dir data/mds_csv               --out data/processed/vocab.json
gems build-corpus --pks_dir data/walking_calibrated_pks --out data/processed/dev.h5 --limit 8
gems smoke        # read .pks → Δm graph → tokenize → tiny transformer → one denoising step on CPU
pytest -q
```

## Build order

Fill in stubs in this order to reach the end-to-end CPU smoke test (BUILD_PLAN Part C):
`vocab/vocabulary.py` (corpus aggregation + `match`) → `vocab/graph.py` (Δm graph + degree cap +
leakage guard) → `data/peak_selection.py` (graph-induced) → `models/layers/fourier_features.py` +
`peak_tokenizer.py` (nominal+defect) → `models/layers/attention_bias.py` (`GraphDeltaBias`) →
`models/layers/transformer.py` (sparse encoder + attention-pool) → `models/objectives/*`
(`masked_mz` nominal+defect, `masked_intensity`, `replaced_peak`, composed by `denoising`) +
`data/masking.py` → `data/ms_data.py` + `data/datamodule.py` → `models/gems/gems.py` →
`training/{train,config}.py` + `cli.py smoke` → `tests/test_smoke_pretrain.py`. **Measure first:** run
`vocab/graph.py` over the corpus and print the degree distribution before fixing `k` / `d` / `L` / batch.
The `no_bias` and `dense_edge_bias` sanity checks then plug in via the `attention` config.

## Lineage

DreaMS (Δm attention; Nat. Biotechnol. 2025), LSM1-MS2 (nominal/defect tokenization),
GLEAMS (repository-scale embedding), MIST (formula/neutral-loss featurization).
