# Project Spec — FT-ICR Encoder for Complex Mixtures (Mass-Difference Attention)

*Source: distilled from the "AI Projects" page in Notion (FT-ICR Encoder spec + fine-tuning addendum). Data lives in `data/` of this folder.*

**One-liner.** A DreaMS-style mass-difference-biased transformer, re-pointed from MS/MS small molecules to **MS1 broadband FT-ICR of complex mixtures**. Input = selected ("special") peaks; the abundant pairwise mass differences (CH₂, H₂, O, …) act as the attention bias so that peaks linked by an abundant building-block Δm attend to each other. Output = per-peak contextual embeddings + a pooled whole-spectrum embedding.

**Novelty framing (be honest).** The *mechanism* (Graphormer edge-bias attention over pairwise Δm) is adapted from DreaMS, not invented here. The defensible novelty is the domain and setting: FT-ICR / complex mixtures, MS1-only (no fragmentation), abundance-weighted mixture-specific building-block vocabulary, repository-scale NHMFL corpus, and a new downstream benchmark. Frame as **adaptation + new domain + new benchmark**, not a new mechanism.

## Core idea

In a complex-mixture FT-ICR spectrum, the chemistry lives in the recurring spacings between peaks: CH₂ (14.01565), H₂ (2.01565, the DBE axis), O (15.9949), CH₂O, COOH, etc. A histogram of all pairwise Δm is sharply peaked at these building blocks — the mixture's "grammar," exactly what Kendrick / van Krevelen analysis exploits by hand. The model lets peaks connected by an *abundant* building-block Δm attend to one another, so representations encode compositional structure rather than a bag of masses.

## Input representation

- **Peaks as tokens.** Each token = one resolved peak. Per-peak features: log-intensity + m/z split into **nominal mass and exact mass defect as separate embeddings** (LSM1-MS2 tokenization; even more natural here since FT-ICR mass defect is sub-ppm and homologous series fall on clean KMD lines). Optional auxiliary features: Kendrick mass defect, and (if pyc2mc assignment is run) van Krevelen O/C, H/C, DBE.
- **Peak selection ("special peaks").** Cannot feed 10⁴–10⁵ peaks into O(n²) attention. Selection is a first-class, tunable preprocessing stage to ablate: top-N by abundance, series-anchoring peaks, or assigned-formula peaks after noise removal.
- **Raw-peak vs formula-aware — decide up front.** Build the **raw-peak self-supervised** variant first (more novel, scalable, free of assignment bias). Treat formula assignment as a *downstream task*, not an input, to avoid circularity (model just relearning the assignment rules).

## Attention mechanism (test both)

1. **Edge-bias (Graphormer-style).** For pair (i,j): compute Δm, map to a learned bias added to the QKᵀ logit. Mapping keys on whether Δm matches a building block within ppm tolerance, *which* block, and **how globally abundant that Δm is in this spectrum** (the novel abundance weighting). Seed the vocabulary (CH₂, H₂, O, CH₂O, COOH, H₂O, CHN…) but make it learnable so mixture-specific blocks can emerge.
2. **Sparse / masked attention.** Restrict attention to pairs whose Δm matches some building block. This is simultaneously the inductive bias and the tractability fix — turns dense O(n²) attention into a sparse mass-difference graph. Likely the **primary path** at FT-ICR scale.

**Gotchas.** ¹³C spacing (1.00336) is itself an abundant Δm and will appear as an edge — decide feature vs confounder. Δm matching needs a **ppm-scaled window**, not a fixed-Da tolerance. Output pooling via CLS token or attention-pooling.

## Pretraining (self-supervised; labels scarce)

- **Masked peak modeling** — mask m/z and/or intensity, reconstruct from context. Predicting a masked peak's exact mass defect from homologous-series neighbors forces learning the building-block grammar (workhorse task).
- **Masked intensity reconstruction** — homologous series have characteristic abundance envelopes.
- **Contrastive invariance** — augment (subsample peaks, jitter intensities, simulate calibration drift), pull augmentations together (SimCLR-style) for cross-instrument robustness.
- **LC-elution ordering** — if using lab LC-FT-ICR data, predict elution order across fractions (analog of DreaMS retention-order objective).

## Data

NHMFL FT-ICR archives (petroleum, DOM/NOM, bio-oils) are the enabling asset — few groups have a pretraining corpus at this scale. Pipeline: calibration → peak pick → noise threshold → (optional) pyc2mc assignment → Δm-statistics / vocabulary extraction. Pretrain across sample types; hold out sample classes for transfer eval.

**Local data in this folder (`data/`):**
- `walking_calibrated_pks/` — 285 walking-calibrated `.pks` peak lists (raw FT-ICR spectra; the tokenizable input).
- `mds_csv/` — 285 per-sample pairwise mass-difference CSVs (the Δm statistics for vocabulary extraction).
- `mds_run_manifest.csv`, `mds_loop.log`, `mds_run.log` — provenance for the Δm computation run.

## Downstream evaluation (defining the benchmark is itself a contribution)

- Sample classification (oil source/maturity, biodegradation, DOM provenance).
- **Formula-assignment disambiguation** — use learned context to rank candidate formulas for high-mass peaks where several fit within tolerance (directly useful to auto_analysis).
- van Krevelen region prediction.
- Cross-sample similarity (a "petroleomics GLEAMS").
- Novelty flagging (unusual series → feeds Analysis Companion).
- **Always ablate vs a strong non-deep baseline:** KMD / van Krevelen features + gradient boosting. If the encoder doesn't beat that, the architecture isn't earning its keep.

## Fine-tuning tasks — class & sample-level supervision

Two label sources, doing different jobs.

**1. Heteroatom class (CHO, CHOS, CHOS₂, CHN, CHOP…) — useful, with a caveat.**
Near-free supervision: every assigned peak already carries a class → millions of labels, no new annotation. *Caveat:* peak-level class is almost trivially recoverable from exact mass defect alone (S, N, P shift the defect in known directions). A peak-level class head may pass *without* the Δm-attention contributing anything → weak validator. Make it earn its keep two ways:
- **Auxiliary pretraining head** — multi-task alongside masked-peak modeling; cheap representation regularizer. Fine to keep "easy."
- **Whole-sample heteroatom-class-distribution regression** — predict the sample's CHO/CHOS/CHOS₂… class fingerprint. This *requires* integrating across the peak network, so it actually stresses the Δm attention. This is the version that matters.

**2. Sample classification (petroleum / DOM / PFAS / lipid…) — the higher-value task.**
The whole-spectrum downstream task the encoder is built for; strong headline benchmark. Forces the pooled embedding to capture mixture-level structure (class distributions, KMD-series patterns, abundance envelopes) — nothing a single peak encodes. Cleanest place to visibly beat the KMD/van Krevelen + gradient-boosting baseline (or learn that you don't).
- **PFAS as a vocabulary-discovery probe:** PFAS is compositionally distinctive via the CF₂ homologous series (Δm ≈ 49.9968, a *non-CH₂* building block). If the learnable Δm vocabulary auto-discovers high-weight CF₂ in PFAS samples, that's a clean interpretability win — direct evidence the model finds domain-specific blocks rather than just CH₂/H₂/O.

**Design flags for sample-class labels.** Labels are coarse and few, and correlated with which instrument/lab/method produced them → risk the model shortcuts to **acquisition signatures, not chemistry**. Defend with the contrastive calibration-drift augmentation *and* an explicit **held-out-instrument eval**. Few classes saturate fast → treat high accuracy skeptically; pair with the harder class-distribution regression as the real stress test.

## Phased plan

| Phase | Goal | Deliverable |
|---|---|---|
| 0 | Data + preprocessing pipeline; peak-selection strategy; pairwise-Δm statistics; seeded building-block vocabulary | Clean tokenized corpus + Δm vocab |
| 1 | Baseline: plain transformer, nominal/defect tokenization, masked-peak pretraining, **no Δm bias** | Performance floor |
| 2 | Add mass-difference attention; ablate edge-bias vs sparse-masking, seeded vs learned vocab | **Thesis-worthy claim** |
| 3 | Scale corpus; add contrastive + LC-ordering objectives | Robust pretrained encoder |
| 4 | Downstream eval suite vs baselines and hand-engineered features | Benchmark + results |
| 5 | Wire embedding into formula-assignment assist / Analysis Companion | Integrated tool |

## Honest risk register

- **Scale is the central engineering risk** — sparse Δm-graph attention likely mandatory, not optional.
- **MS1-only is a bet** — with no fragmentation, all structural signal comes from the mass-difference network. Plausible, untested at this scale.
- **Novelty is incremental over DreaMS at the mechanism level** — lean on domain + abundance-weighted vocabulary + MS1 + benchmark.
- **Negative-result risk** — Δm attention may not beat a well-tokenized plain transformer; the Phase 1→2 ablation is the test. Do not skip Phase 1.

## Lineage / see also

Builds on the *Encoding the Mass Spectrum* survey: DreaMS (mass-difference attention; Nat Biotechnol 2025), LSM1-MS2 (nominal/defect tokenization; ChemRxiv 2024), GLEAMS (repository-scale embedding + clustering; Nat Methods 2022), MIST (formula+neutral-loss featurization; Nat Mach Intell 2023).
