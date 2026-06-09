# Data

The raw FT-ICR corpus (~965 MB) is **not committed to git** (see `.gitignore`). This file documents
the expected local layout so the pipeline and configs resolve correctly.

```
data/
├── walking_calibrated_pks/   # 272 walking-calibrated FT-ICR .pks peak lists (the tokenizable input)
├── mds_csv/                  # 285 per-sample pairwise mass-difference (Δm) distribution CSVs
├── mds_run_manifest.csv      # provenance: file, n_peaks, n_distributions, seconds, status, error
├── mds_run.log               # MDS computation run log
├── mds_loop.log              # MDS loop execution log
└── processed/                # (generated) HDF5 corpus + vocab.json + splits — also gitignored
```

## Formats

- **`.pks`** — ASCII peak lists produced by pyc2mc. Header (peak count, ion mode, thresholds,
  center of mass) then columns: `Peak Location` (m/z), `Peak Height`, `Abundance`,
  `Resolving Power`, `Frequency`, `S/N`. Read via `gems.data.peaklist.load_record`
  (wraps `pyc2mc.io.peaklist.read_pks`). Intensity = the `Abundance` column.
- **`mds_csv/`** — 20-column Δm-distribution statistics (exactly `MassDifferencesSpectrum.md_data`):
  `average_position` = the Δm value, `# occurrences` = its abundance, `r_squared` = fit quality, …
  Read via `gems.data.mds.read_mds_csv`, or regenerate from a `.pks` with `gems.data.mds.compute_mds`.

## Obtaining the data

The corpus is NHMFL FT-ICR archive data (petroleum / DOM / bio-oils). It is not redistributed here.
Place the files under the paths above (the defaults referenced by `configs/data/*.yaml`), or point the
configs at your own location.
