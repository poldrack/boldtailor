# Empirical HRF experiments

Scope: sub-07, sessions 10–19, NSD-preprocessed layerB2 CIFTIs. Motion comes from matching NSD-preprocessed TSVs; six parameters are expanded to 24, and the model generates 80-second cosine drifts. No scans are censored. See `plan.md` for confirmed decisions and `reml_model.md` for the direct REML assumptions.

The implementation is experimental and does not change the production package API. Each feature uses one HRF for binary trial type, common RT, and separate missing/incorrect junk regressors. Held-out scores refit amplitudes and measure HRF transfer, not fully held-out BOLD prediction.

## Tests

```sh
uv run python -m pytest tests experiments/hrf_estimation -q
```

Tests were committed before implementation. Tests cover nuisance construction, anatomical identities, task/junk partitioning, independent convolution checks, library CV agreement, dense REML likelihoods, continuous off-grid recovery, inner regularization selection, score definitions, stability, and verified output restart.

## Reproduce the initial pilot

From this worktree's repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run python -m experiments.hrf_estimation.pilot \
  --bids-root /Volumes/extdata1/NSD/BIDS \
  --mask experiments/hrf_estimation/pilot_inputs/sub-07_space-fsLR_den-32k_desc-fusiform_mask.npz \
  --output experiments/hrf_estimation/outputs/reproduced-pilot \
  --methods canonical librarycv reml
```

The default is session 11, four runs, 256 anatomically sampled vertices, and one complementary pair of 50/50 splits. Use `--session`, `--runs`, `--partitions`, and `--full-roi` to select a larger within-session experiment. `--library-samples 512` gives the default 533-kernel library including canonical and empirical kernels. Continuous estimators are named `continuouslbfgsb` and `continuouspowell`; they run nested regularization selection and are substantially more expensive. The direct Python API accepts explicit optimizer budgets and lambda grids.

The ROI file contains both all 1,750 fusiform vertices and the seeded 256-vertex pilot subset. Its sidecar records anatomical sources, hashes, the mapping rule, and the CIFTI's original registration provenance. Public TemplateFlow spheres were downloaded into `pilot_inputs`; source data were not changed.

## Outputs and restart

Open [results.ipynb](results.ipynb) for the executed session-11/session-12 VTC
comparison, HRF curves and stability, and the separate continuous-optimizer
smoke test. Figures and tables are embedded; standalone PNGs are in
`reports/figures/`. Rerunning the notebook reads the saved `outputs/` artifacts
and verifies their hashes without refitting models or loading raw NSD data.

Each comparison writes total/partial/incremental task R² in `scores.tsv`, run-level SSE/rank tables, cortical partial-task-R² maps, HRF curves/parameters, diagnostic JSON, and a manifest. Unprocessed cortical vertices are NaN. `COMPLETE.json` is written last and records output hashes. Repeating an identical completed run verifies those hashes before reuse; incomplete/corrupted outputs are recomputed. Changed data/settings/scientific code require a new output directory. Restart currently operates on a whole session comparison, not individual feature chunks.

`input_inventory.tsv` verifies all 120 matched runs for sessions 10–19. Pilot output directories distinguish the initial canonical/library check, the REML extension, the reduced four-vertex continuous smoke check, and the expanded session-11/session-12 VTC checks. Reduced-library continuous smoke results are not a fair final ranking against the 533-kernel library methods.

## Remaining full-study work

The complete ten-session, 20-split comparison, extended noise/RT/library-coverage simulations, and between-session reporting are not yet finished. The current runner is serial within a session and has no cluster submission layer. Select the compute scope before launching full-ROI nested continuous optimization. All scientific routines remain under this experiment directory; every `__init__.py` is untouched.

## Full-VTC continuous and sensitivity stage

See [full_vtc_plan.md](full_vtc_plan.md) for the matched 16-job empirical queue
and 288-case factorial noise/RT simulation design. The empirical stage uses
sessions 11–12, the entire 1,750-vertex anatomical fusiform ROI, both RT
formulations, full and parametric-only library comparisons, and L-BFGS-B
with nested regularization selection. The simulation scores are a
separate recovery analysis, not empirical VTC benchmarks.

Launch metadata are in `outputs/lbfgsb-launch.json`; progress logs are
`outputs/full-vtc-lbfgsb.log` and `outputs/noise-rt-lbfgsb.log`.
Only a verified `COMPLETE.json` denotes a finished restart unit. A partial
directory is not a completed result. Current long-running jobs use the fixed
`bcebcfb` checkout at `/tmp/boldtailor-hrf-lbfgsb-run`, with outputs written
into this checkout. Preserve that execution worktree while the queues run.

The existing results notebook describes the earlier completed pilots; it does
not yet include this running stage. The authorized REML extensions are documented in
[reml_model.md](reml_model.md), implemented as `remlar1`, `remlcomponents`,
`remlshared`, and `remlcombined`. They retain `reml` as the original baseline.
Powell remains available for historical reproducibility but is excluded from
both active queues at the user’s request.


The completed [REML extension validation](reml_validation.md) records all 100
passing tests and the four-vertex/two-session real-data check. Full-ROI REML
variant comparisons (primary RT-amplitude model) and the matching 288-case
synthetic checks run from c79a2a5 in `/tmp/boldtailor-hrf-reml-run`. Their exact
commands and PIDs are in `outputs/reml-launch.json`; logs are
`outputs/full-vtc-reml.log` and `outputs/noise-rt-reml.log`. These are separate
from the ongoing L-BFGS-B queues and from the earlier completed notebook.
