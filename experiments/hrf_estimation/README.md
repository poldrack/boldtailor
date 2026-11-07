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

Each comparison writes total/partial/incremental task R² in `scores.tsv`, run-level SSE/rank tables, cortical partial-task-R² maps, HRF curves/parameters, diagnostic JSON, and a manifest. Unprocessed cortical vertices are NaN. `COMPLETE.json` is written last and records output hashes. Repeating an identical completed run verifies those hashes before reuse; incomplete/corrupted outputs are recomputed. Changed data/settings/scientific code require a new output directory. Restart currently operates on a whole session comparison, not individual feature chunks.

`input_inventory.tsv` verifies all 120 matched runs for sessions 10–19. Pilot output directories distinguish the initial canonical/library check, the REML extension, the reduced four-vertex continuous smoke check, and the expanded session-11/session-12 VTC checks. Reduced-library continuous smoke results are not a fair final ranking against the 533-kernel library methods.

## Remaining full-study work

The complete ten-session, 20-split comparison, extended noise/RT/library-coverage simulations, and between-session reporting are not yet finished. The current runner is serial within a session and has no cluster submission layer. Select the compute scope before launching full-ROI nested continuous optimization. All scientific routines remain under this experiment directory; every `__init__.py` is untouched.
