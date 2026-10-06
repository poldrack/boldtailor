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
With `released_dir` configured, discarding ppdata betas is refused until the comparison outputs (`alignment.tsv`, `r1.tsv`) exist.
The comparison writes `alignment.tsv` first and stops before the other tables if any session is below `alignment_floor`.

## Verified API Notes

**ROI resampling:** `wb_command -metric-resample <in> <source sphere> <target sphere> ADAP_BARY_AREA <out> -area-metrics <source area> <target area>`, with inputs read from the released betas' JSON sidecars, then thresholded at 0.5.

## Figures

`uv run python -m experiments.nsd_replication.run figures --config <toml>` renders PNGs from whichever metric TSVs exist: per-subject figures under `<output_dir>/figures/<subject>/` and group RSA and parity figures under `<output_dir>/figures/group/`. Missing inputs are skipped with a message. Run configs: `configs/primary.toml` (sub-01 to sub-04) and `configs/extension.toml` (sub-05, sub-06, sub-08).
