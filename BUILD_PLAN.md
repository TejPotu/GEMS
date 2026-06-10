# GEMS — Concrete Build Plan (model + training objectives)

A single committed design, replacing the phased table. Decisions that were "ablate A vs B" are now **locked** to one choice; the alternative is kept only as a config-switchable sanity check, not a milestone. Numbers marked *(default)* are starting points to calibrate empirically; numbers without it are firm design choices.

---

## 0. Locked decisions (the commitments)

1. **Peak selection is graph-induced.** A peak is a node iff it sits on ≥1 abundant-Δm edge. No standalone Top-N stage.
2. **Attention is sparse over the Δm graph** — message passing only along Δm edges. Dense edge-bias is *not* a primary path (kept as a reduced-N sanity check only).
3. **Edges are abundance-weighted.** Each edge carries its Δm type + that Δm's per-spectrum abundance, fed in as the attention bias. This is the novelty over DreaMS.
4. **Whole-spectrum vector = attention-pooling** over final peak embeddings. No precursor/CLS master node (a CLS would need edges to all peaks, breaking sparsity).
5. **Masked-peak prediction = two heads: nominal mass + mass defect.** Not a single coarse mass-bin head (FT-ICR sub-mDa defect carries the chemistry).
6. **One pre-training objective: spectrum denoising / repair.** Corrupt the peak set three independent ways — mask m/z, mask intensity, swap in plausible-but-wrong masses — and reconstruct the original. Masked-m/z reconstruction is the backbone (it forces using the Δm relationships to neighbors, so building-block grammar, homologous series and Kendrick structure all fall out *latent in it*); masked-intensity adds homologous-series envelope structure; replaced-peak detection adds global chemical-consistency pressure on every peak. Edges incident to a masked peak have their Δm-type **stripped to a sentinel**, so a typed edge can't hand the model the masked mass (the FT-ICR leakage fix over DreaMS).
7. **Everything else is downstream / fine-tuning, not pre-training.** Edge-type (Δm) prediction, Kendrick-series detection, heteroatom-class/composition, sample classification, similarity, PFAS flagging — all run on the *pretrained* encoder. Elution-order stays an optional head for LC-FT-ICR fractions only.

---

## Part A — Model architecture

### A1. Input representation  (`data/peaklist.py`, `models/layers/peak_tokenizer.py`)

Per node (peak), features:
- **m/z → mass-tolerant Fourier features**, split so low frequencies encode the **nominal (integer) mass** and high frequencies encode the **exact mass defect**; ppm-scaled, sized to FT-ICR accuracy (`m_min ≈ 1e-4`). Pushed through `FFN_F`.
- **log-intensity** (normalized abundance) through a shallow `FFN_P`, concatenated with the Fourier branch.
- Optional aux scalars (concat): **Kendrick mass defect**, S/N, resolving power — cheap, from pyc2mc.
- Output: a **d-dim peak token**, `d = 256` *(default; reduced from DreaMS's 1024 because N is 100–500× larger)*.

No positional encoding — peaks are a set; order comes only from the Δm graph.

### A2. Δm graph construction  (`vocab/vocabulary.py`, `vocab/graph.py`)

- **Vocabulary:** seed the known building blocks (CH₂ 14.01565, H₂ 2.01565, O 15.9949, H₂O 18.0106, CH₂O 30.0106, CO 27.9949, COOH 44.9977 (−H), CHN, S 31.97207, **CF₂ 49.9968**, …) and make the embedding table **learnable** so mixture-specific blocks can emerge. Names come from `pyc2mc … assign_peaks` over `mds_csv` (discovered + named, not hardcoded).
- **Edge rule:** connect peaks i,j when `|m_i − m_j|` matches a vocabulary Δm within a **ppm-scaled tolerance window** (not fixed Da).
- **Edge features:** `(Δm-type id, log per-spectrum abundance of that Δm, direction sign)`.
- **Degree cap:** keep top-`k` edges per node by Δm abundance, `k = 32` *(default — set after measuring the real degree distribution; this is the knob that guarantees linear cost)*.
- **¹³C edges (1.00336):** included but tagged with their own Δm-type embedding so the model can learn to use or down-weight them; ablatable via a config toggle.
- **Pretraining leakage guard:** at training time, edges incident to a **masked** peak keep their connectivity (the node still aggregates from neighbors) but have their `Δm-type id` + abundance replaced by a sentinel `[masked-edge]` embedding — the model learns a masked peak is *related* to its neighbors, never *which* building block links them (which would reveal the masked mass). Stricter config variant: rebuild connectivity from unmasked peaks only.

### A3. Encoder  (`models/layers/attention_bias.py`, `transformer.py`)

- **Sparse-attention transformer**, `L = 6` layers *(default)*, `H = 8` heads, hidden `d = 256`, pre-norm, GELU, residuals.
- Attention is **masked to graph edges**; for an allowed pair the logit is
  `A_ij = (qᵢ·kⱼ)/√d + b(Δm_type_ij, log_abund_ij)`,
  where `b(·)` is a small learned map from the edge embedding — i.e. **sparse mask *and* abundance-weighted edge bias together** (the edge set already encodes "which Δm").
- Implementation: ragged batching via **PyTorch Geometric** (or a block-sparse attention kernel); **bf16 + gradient checkpointing** to fit 10k–30k nodes.

### A4. Readout  (`models/layers/…`, used by heads)

- **Attention-pooling** over the final per-peak embeddings → `z ∈ ℝ^d`, the whole-spectrum embedding (the GEMS analog of DreaMS's precursor embedding).
- The encoder also exposes the **per-peak embeddings** `s₁…s_N`, which feed the four pre-training heads (nominal-mass, mass-defect, masked-intensity, replaced-peak detector) and downstream peak-level tasks (formula disambiguation, DeepSets-style fingerprint/class heads).

---

## Part B — Training objective (spectrum denoising / repair)

Pre-training is fully self-supervised on the 272-sample corpus and uses **a single denoising / repair objective**: corrupt the peak set three independent ways, then reconstruct the original. The loss is the sum of the three repair channels:

> **ℒ = ℒ_mz + λ_int·ℒ_int + λ_rpd·ℒ_rpd**   *(masked m/z + masked intensity + replaced-peak detection)*

with starting weights `λ_int = 0.2`, `λ_rpd = 0.5` *(default — calibrate against the linear-probe emergence curve)*. The three channels share **one** corrupted view: different peaks receive different corruptions, the encoder runs once, and per-peak heads repair each. Masking is intensity-weighted at **30%** *(default)*. This is conceptually still one objective — a denoising autoencoder over the peak set — so the chemistry stays *emergent*, not hand-labelled.

### B1. Masked m/z — ℒ_mz  (`models/objectives/masked_mz.py`)  — the backbone

Reconstructing a masked m/z is unsolvable without using the Δm relationships to a peak's neighbors, so the model is *forced* to internalize building-block grammar, homologous series and Kendrick structure — they're latent in this one channel. Edge-type/Δm prediction, Kendrick detection, composition, classification, similarity and PFAS are therefore **downstream**, not pre-training (Part D).

1. Mask **30%** *(default)* of nodes per spectrum, sampled **proportionally to intensity**. About **⅓ of the masks are series-spans** — 2–3 *consecutive* members of a homologous series rather than isolated peaks. Interpolating one edge is easy; extrapolating a series and its defect progression is where compositional reasoning forms.
2. Hide the masked peaks' **m/z only** (set sentinel) — **keep intensity as context** (masking intensity *within this channel* hurt in DreaMS's ablation; intensity is handled separately in B2).
3. **Leakage guard (the FT-ICR fix over DreaMS):** strip the `Δm-type id` + abundance on every edge incident to a masked peak to a sentinel `[masked-edge]` embedding (connectivity stays, so the node still aggregates from neighbors). Otherwise a typed edge `(CH₂, neighbor j)` hands the model `m_j ± 14.01565` and the defect head collapses to arithmetic.
4. Reconstruct each masked m/z with **two classification heads**:
   - **Nominal-mass head:** softmax over integer-Da bins across the corpus m/z range → cross-entropy.
   - **Mass-defect head:** softmax over fine defect bins, e.g. **0.1 mDa** over [0,1) Da *(default; ~10k classes — tune to instrument accuracy)* → cross-entropy.
   - `ℒ_mz = ℒ_nominal + ℒ_defect`. Classification (not regression) so the model can spread probability when several masses fit a slot.

### B2. Masked intensity — ℒ_int  (`models/objectives/masked_intensity.py`)

On a separate intensity-weighted subset, hide each peak's **intensity** (keep m/z) and reconstruct it from mass + context. DreaMS found masking intensity *hurts* — but that's MS2, where intensity is fragmentation-dependent noise. In FT-ICR the intensity profile **along a homologous series** is smooth and chemically meaningful (relative concentration / ionization), so this channel teaches series envelopes the m/z head never sees. Head: a shallow FFN on the per-peak embedding → **binned classification** over log-intensity bins (or simple regression). Low weight (`λ_int = 0.2` default).

### B3. Replaced-peak detection — ℒ_rpd  (`models/objectives/replaced_peak.py`)  — Electra-style

On a third subset (default **15%** of peaks), **replace** each selected peak's m/z with a *plausible-but-wrong* value — shift by a non-vocabulary Δm, or by a vocabulary Δm that breaks defect consistency — keeping intensity. A binary head classifies **every** peak real/fake (BCE). Two reasons it earns its place: (a) it forces learning what masses are **chemically consistent with the rest of the mixture** — a global, label-free signal, not local arithmetic; (b) it produces a training signal on **100% of peaks** (vs ~30% for masking), which matters a lot on a 272-sample corpus. Weight `λ_rpd = 0.5` default.

### B4. Optimizer & schedule  (`training/train.py`)

- **AdamW**, lr `3e-4` *(default)* with warmup + cosine decay, weight decay `0.01`.
- **bf16** mixed precision, **gradient checkpointing**, gradient accumulation for a large effective batch over ragged graphs.
- **Linear-probe callback** every N steps: freeze encoder, fit logistic regression from `z` to heteroatom-class / van-Krevelen-region labels → the "structure emerges" curve (DreaMS Fig. 3c analog), diagnostic only.

---

## Part C — Build order (single sequence, not phases)

Fill stubs in this order to reach an end-to-end CPU smoke test, then scale:

1. `data/peaklist.py` — pyc2mc `.pks` reader → `PeakList`. **[WRAP]**
2. `vocab/vocabulary.py` — seed building blocks + load named Δm from `mds_csv`. **[CONCRETE]**
3. `vocab/graph.py` — per-spectrum Δm graph: ppm tolerance, edge features, **degree cap**. **[CONCRETE]**
4. `data/peak_selection.py` — **graph-induced** selection (node = on an abundant edge). **[CONCRETE]**
5. `models/layers/fourier_features.py` + `peak_tokenizer.py` — nominal+defect tokenization. **[PORT/CONCRETE]**
6. `models/layers/attention_bias.py` — sparse edge mask + abundance/Δm-type bias. **[CONCRETE]**
7. `models/layers/transformer.py` — sparse pre-norm encoder + **attention-pool readout**. **[PORT]**
8. `models/objectives/` — `masked_mz.py` (nominal+defect), `masked_intensity.py`, `replaced_peak.py`, composed by `denoising.py` into the single repair loss. **[CONCRETE]**
9. `models/gems/gems.py` — LightningModule wiring encoder + readout + the three repair heads (`ℒ = ℒ_mz + λ_int·ℒ_int + λ_rpd·ℒ_rpd`). **[CONCRETE]**
10. `training/train.py` + `cli.py smoke` — AdamW/bf16/checkpointing + linear-probe callback. **[CONCRETE]**
11. `tests/test_smoke_pretrain.py` — read `.pks` → graph → encoder → one masked-peak step on CPU.
12. `baselines/` — KMD / van-Krevelen features + gradient boosting (the bar to beat).
13. `eval/benchmark.py` — downstream + the validation suite below.

**Measure first, then set:** before step 6, run `vocab/graph.py` over the corpus and print the **degree distribution** — it fixes `k`, the feasible `d`/`L`, and the batch size.

---

## Part D — Downstream / analysis (on the *pretrained* encoder, not pre-training)

These were candidate pre-training objectives; they're better as fine-tuning tasks or post-hoc analyses, because the denoising objective already teaches the chemistry they'd target. Each takes the frozen or fine-tuned encoder and adds a small head (`models/heads/`).

- **Edge-type / Δm prediction & Kendrick-series detection** — given two peaks, name the building block connecting them / assign series membership. (As an *objective* it's trivial when masses are visible — it collapses into B1 — so it lives here as an interpretability probe, e.g. confirming CF₂ is learned.)
- **Sample classification** — petroleum / DOM / PFAS / lipid from the pooled `z`. Headline benchmark; guard with held-out-instrument eval.
- **Whole-sample heteroatom-class-distribution regression** — predict the CHO/CHOS/CHOS₂… fingerprint; stresses the Δm attention more than peak-level class (which is near-trivially recoverable from defect alone).
- **Formula-assignment disambiguation** — rank candidate formulas for high-mass peaks using learned context; feeds auto_analysis.
- **Cross-sample similarity** — a "petroleomics GLEAMS" over `z`.
- **Composition / DBE auxiliary head** — the DreaMS divergence: a formula-aware head trained with pyc2mc labels, if you want explicit composition awareness. Semi-supervised, separable from the base model.
- **Elution-order** — only for LC-FT-ICR fractions; predict fraction order from two `z`. Optional head, never part of base pre-training.

---

## Validation (what proves it works)

- **Emergence:** linear-probe curve for heteroatom class / van-Krevelen properties rising as ℒ falls.
- **Attention sanity:** attention concentrates on peaks sitting on real homologous-series edges (CH₂, O, CF₂), not isolated noise.
- **Organization:** PCA/UMAP of `z` clusters samples by **chemistry** (DOM / petroleum / PFAS), and a **held-out-instrument** test confirms it isn't shortcutting to acquisition signature.
- **PFAS probe:** the learnable vocab auto-surfaces high-weight **CF₂ (49.9968)** in PFAS samples — the interpretability win.
- **Always vs the non-deep bar:** KMD/van-Krevelen + gradient boosting. If GEMS doesn't beat it, the architecture isn't earning its keep.

---

## Knobs to calibrate empirically (not guesses to lock)

`k` (degree cap) · `d` (hidden) · `L` (layers) · defect-bin width · masking ratio (30%) · series-span fraction (⅓) · channel weights `λ_int` (0.2) / `λ_rpd` (0.5) · replaced-peak fraction (15%) · ppm tolerance window. Set these from the degree distribution + the CPU smoke run, not up front.
