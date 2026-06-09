# Notebooks

Exploratory notebooks (placeholders — create as `.ipynb` when you start the corresponding work):

- **01_explore_pks** — load `.pks` via `gems.data.peaklist.load_record`; plot m/z vs intensity,
  peak-count distribution, ion-mode split across the corpus.
- **02_delta_m_vocabulary** — aggregate `mds_csv` `# occurrences` across samples; visualize the
  building-block Δm histogram; sanity-check seeds (CH₂, O, S, CF₂) against pyc2mc `assign_peaks`.
- **03_embedding_inspection** — once a model is trained: UMAP of pooled embeddings, colored by
  sample type / instrument; check the held-out-instrument generalization.

Keep heavy outputs out of git (see `.gitignore`).
