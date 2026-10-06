# NSD Replication Experiments

## Purpose

This package replicates the GLMsingle single-trial fMRI analyses from Prince et al. (2022) on the Natural Scenes Dataset (NSD) using boldtailor, a Python implementation of the GLMsingle workflow.

## Prerequisites

- `uv sync --group dev`
- Connectome Workbench `wb_command` on `PATH`
- Each subject's native-surface `lh.nsdgeneral.mgz` and `rh.nsdgeneral.mgz` under `freesurfer_dir/subjNN/label/`
- NSD ppdata layer-B2 time series resampled to fsLR 32k under `ppdata_dir` (`subjNN/func1pt8mm/timeseries/`); `onset_offset` is calibrated by the pilot (Task 12)
- The released betas converted to CIFTI under `released_dir` (`derivatives/betas-fsLR`)

## Tests

Run tests with:
```
uv run pytest experiments/nsd_replication -W error
```

## Commands

Run each stage with `uv run python -m experiments.nsd_replication.run <fit|metrics|features|figures> --config <toml> --source <ppdata|released|comparison>`.
Required order: ppdata fit and metrics, released fit and metrics, `metrics --source comparison`, and only then `--discard-betas-after-metrics` for ppdata.
With `released_dir` configured, discarding ppdata betas is refused until every comparison output exists: `alignment.tsv`, the six comparison tables (`r1`, `r1_median`, `r3b`, `r4_t0.0`, `r4_t0.3`, `r6`) and `r1_difference_{b1,b2,b4}.npy`.
The comparison writes `alignment.tsv` first and stops before the other tables if any session is below `alignment_floor`.
It computes R1, R3B, R4 and R6 over ppdata and released b1, b2 and b4 with one shared composite, under `metrics/comparison/<subject>/`.

- **Onset offset (pilot):** `uv run python -m experiments.nsd_replication.calibrate_offset --config <toml> --subject <sub-NN> --session <ses-...>` fits candidate offsets, writes `calibration/<subject>_<session>_onset_offset.tsv` and prints the best offset to set as `onset_offset` in the config.
- **Levels:** ppdata `fit` and `metrics` default to every level, b1, b2, b2-lib20, b3, b4 and the LSS levels `lss-assume` and `lss-fit`; pass `--levels` to restrict them. Fits are written level by level, so a failure keeps the levels already finished.
- **Metrics:** each table is written as soon as it is computed, and existing tables are skipped. Every composite-based table records its level set in a `composite_levels` column, and every table records its session window in a `sessions` column. A rerun with a different level set or session window raises instead of reusing or overwriting; pass `--recompute` to rebuild existing tables. Group RSA tables are written under the subject set, `metrics/<source>/group_<sub-..._sub-...>/rsa_<level>.tsv`, so runs over different subjects (pilot, primary, extension) never share them. The ppdata metrics stage also writes R2 (`r2_<level>.tsv` and the ROI summary `r2_<level>_roi.tsv`) for b2 and b2-lib20 when their fits exist. R6 runs on `n_jobs` workers.
- **Features:** `run features --config <toml>` (ppdata only) runs the denoising gate (real and circularly shifted null events, seeded by subject, session and run), the task-modulator comparison (b4 and b4-taskonly) and the RT correlation, and writes `features_{gate,rt,hrf}.tsv`. Finished sessions are skipped unless `--recompute`.
- **Pilot variability:** `uv run python -m experiments.nsd_replication.pilot_report --config <toml> --source ppdata [--levels ...]` prints and writes `metrics/<source>/<subject>/variability.tsv` (leave-one-session-out and split-half SD of the median ROI reliability). Run it before betas are discarded.

## Verified API Notes

**ROI resampling:** `wb_command -metric-resample <in> <source sphere> <target sphere> ADAP_BARY_AREA <out> -area-metrics <source area> <target area>`, with inputs read from the released betas' JSON sidecars, then thresholded at 0.5.

## Figures

`uv run python -m experiments.nsd_replication.run figures --config <toml>` renders PNGs from whichever metric TSVs exist: per-subject figures under `<output_dir>/figures/<subject>/` and group RSA and parity figures under `<output_dir>/figures/group/`. Missing inputs are skipped with a message. Run configs: `configs/primary.toml` (sub-01 to sub-04) and `configs/extension.toml` (sub-05, sub-06, sub-08).
