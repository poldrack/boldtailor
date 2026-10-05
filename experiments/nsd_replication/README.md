# NSD Replication Experiments

## Purpose

This package replicates the GLMsingle single-trial fMRI analyses from Prince et al. (2022) on the Natural Scenes Dataset (NSD) using boldtailor, a Python implementation of the GLMsingle workflow.

## Prerequisites

- `uv sync --group dev`
- Connectome Workbench `wb_command` on `PATH`
- Each subject's native-surface `lh.nsdgeneral.mgz` and `rh.nsdgeneral.mgz` under `freesurfer_dir/subjNN/label/`
- The released betas converted to CIFTI under `released_dir` (`derivatives/betas-fsLR`)

## Tests

Run tests with:
```
uv run pytest experiments/nsd_replication -W error
```

## Verified API Notes

**ROI resampling:** `wb_command -metric-resample <in> <source sphere> <target sphere> ADAP_BARY_AREA <out> -area-metrics <source area> <target area>`, with inputs read from the released betas' JSON sidecars, then thresholded at 0.5.
