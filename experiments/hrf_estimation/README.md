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

Report sources are Jupytext percent-format Python files:

- [session_comparison.py](session_comparison.py): complete ten-session methods and REML variants in sessions 10/11.
- [regularized_ar1_results.py](regularized_ar1_results.py): pooled, shrinkage, and surface AR(1) pilot.
- [all_methods_results.py](all_methods_results.py): broader historical coverage.
- [autodiff_results.py](autodiff_results.py): autodiff benchmark.
- [results.py](results.py): initial VTC and REML validation results.

Only `.py` inputs are tracked; local `.ipynb` files retain outputs and are ignored.
All five reports read committed tables/curve summaries and can be rerun without
raw NSD data or local fit directories. The oldest report uses a compact verified
snapshot under `reports/legacy_notebook`. Regenerated figure files go to ignored
`outputs/notebook-figures`.

### Notebook workflow

After cloning, install dependencies and enable the tracked Git safeguard:

```sh
uv sync --group dev
uv run git config core.hooksPath .githooks
```

Create a local notebook from a Python source:

```sh
uv run jupytext --to ipynb experiments/hrf_estimation/session_comparison.py
```

In Jupyter with the Jupytext integration, saving a paired notebook updates its
Python source automatically. Other editors may require explicit synchronization:

```sh
uv run python scripts/notebook_sync.py
```

The pre-commit hook runs the same synchronization over all local HRF notebooks,
including ignored files. It updates whichever representation is unchanged since
HEAD, preserves unaffected local outputs, and refuses conflicting edits to both
representations. If both versions changed, explicitly choose/merge the intended
inputs before retrying; the hook does not choose by modification time.

The hook never stages files. Review and `git add` the generated `.py` files,
then retry the commit. It rejects staged `.ipynb` files even when force-added.
CI also rejects tracked notebooks and executes all five reports from Python
sources. Notebook metadata and output bundles are excluded from Python exports.
A fresh checkout therefore does not need committed rendered outputs for testing.

Do not push old pre-migration backup/experiment branches with `git push --all`:
they may retain the old rendered notebook history. Normal pushes of the cleaned
`main` branch contain only the text sources and report data.

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


The primary REML comparison has been expanded to **sessions 10–19**, all 120
runs and all 1,750 VTC vertices. Sessions 11/12 continue in their existing
queue; 10 and 13–19 run in a separate one-worker queue using identical frozen
code and settings. See `outputs/reml-sessions10to19.log` and
`outputs/reml-sessions10to19-launch.json`. The preflight inventory is
`reports/reml_sessions10to19_inventory.tsv`. This stage retains one
complementary 6/6 split pair per session, not the planned twenty-split study.
