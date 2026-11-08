# Full-VTC continuous comparison and sensitivity execution

User authorization: proceed with full-VTC continuous optimization and extended noise/RT checks; investigate REML stability improvements.

Scope: sessions 11 and 12, all 1750 anatomical fusiform vertices, all 12 runs, the existing seed 20261107 and one complementary 6/6 partition. This is a matched extension of the completed pilots, not yet the ten-session repeated-split study.

Primary: canonical, full-library CV/REML, parametric-only library CV/REML, continuous L-BFGS-B and Powell; retain identical training/test IDs and ordinary unwhitened transfer scoring. Continuous tuning retains all training vertices and leave-one-run-out inner CV, lambda grid 0/.001/.01/.1/1, four starts, 500 iterations and 5000 evaluations per start. RT amplitude is primary; RT duration is a separately labelled full-ROI sensitivity. No outer-test-dependent tuning.

Execution steps:
1. RED/commit/GREEN: cached design and nuisance-projected objective; compare with explicit original least squares including duration RT, collinear nuisance, and noisy data. Cache library initialization across vertices; parallelize independent feature optimizations with bounded workers.
2. RED/commit/GREEN: reproducible simulation generation with known HRFs, condition-correlated RT, motion/drift and junk, independent or AR(1) noise, amplitude/duration ground truth; retain trial and signal seeds.
3. Launch restartable method/session/RT jobs with progress logs and immutable settings. Include full and parametric libraries in the same manifest-based comparison system. Run simulation checks separately and report bias, transfer, convergence, and timing without asserting a winning method.
4. Update results notebook when completed; retain output data in the current checkout. Report pending jobs honestly.

REML follow-up proposals (not silently added to baseline): candidate-independent training-only temporal covariance; separate task/RT/junk variance components with fixed scaling; weak HRF-shape shrinkage tuned only inside training folds. Pooling across an evaluated split/session is excluded from stability assessments. Diagnose near-tied candidate objectives and zero-variance cases before interpreting instability.

Ruling: retain sessions 11–12 for this stage because these are the completed full-ROI pilots and provide matched baselines; the cost is limited between-session generalizability. Ruling: retain the original OLS transfer endpoint for all comparisons; noise simulations probe misspecification rather than silently changing the estimator.

## Progress and current data availability

- Cached event sampling, batched initial-library scoring, and bounded process parallelism implemented. Tests compare amplitude/duration designs and noisy profile losses against explicit original least squares.
- Independent review identified projection-roundoff rank loss and bypassed timing validation. Added failing tests first and fixed both; near-degenerate nuisance overlap falls back to the original full-design least-squares solve.
- Full-VTC runner defines 20 restart units: 2 sessions × 2 RT formulations × (full-library baselines, parametric CV, parametric REML, L-BFGS-B, Powell). All use identical seeded complementary splits. Restarts are per completed method/session/formulation directory, not partial optimizer checkpoints.
- Synthetic sensitivity grid: 4 true shapes (canonical, fast, off-grid, empirical) × 2 generating RT models × 2 fitted RT models × AR coefficients 0/.4/.7 × SNR .5/2 × 3 noise seeds = 288 cases. Each case compares 7 methods using training runs 1–2 and held-out runs 3–4, with the full 533-candidate library and unchanged continuous budgets/tuning. Fixed synthetic NSD-like timing includes 24 motion terms, 80 s drift terms, junk trials, and condition-correlated RT. These are simulations, not reconstructed NSD events. Three seeds provide an initial sensitivity screen, not precise Monte Carlo uncertainty.
- Ruling: synthetic timing permits independent checks while the external data drive is unavailable; cost: simulation conclusions cannot capture all properties of the actual NSD events/noise.
- Data blocker: `/Volumes/extdata1/NSD` is absent (only Macintosh HD and Recovery mounted). Asked user to mount drive or supply new path. No full-VTC optimization has been launched.

## Commands

Run from the repository root with the existing uv environment. Preserve output directories for reproducibility.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 UV_CACHE_DIR=/tmp/boldtailor-uv-cache uv run --no-sync python -m experiments.hrf_estimation.full_vtc --bids-root /Volumes/extdata1/NSD/BIDS --output experiments/hrf_estimation/outputs/full-vtc-continuous --workers 4
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 UV_CACHE_DIR=/tmp/boldtailor-uv-cache uv run --no-sync python -m experiments.hrf_estimation.sensitivity_runner --output experiments/hrf_estimation/outputs/noise-rt-sensitivity
```

`--job 0..19` or `--case 0..287` runs a single restart unit. Keep the code/settings fixed during a run; manifests reject reuse after scientific code changes. The output hierarchy is under the main checkout, so notebooks will find it locally.

Update: user reports the NSD drive mounted again. Both sessions discovered with 12 matched runs; proceed with empirical jobs after final code commit. Independent review found no remaining important issues after the two cached-objective fixes. Simulations use four synthetic runs and a single 2/2 split; their scores are not directly comparable with twelve-run empirical 6/6-transfer scores.

## Subsequent user amendments

- User requested dropping Powell. Active queues now use L-BFGS-B only (16 empirical jobs, 6 methods per sensitivity case), frozen at bcebcfb in `/tmp/boldtailor-hrf-lbfgsb-run`. Outputs/logs are `outputs/full-vtc-lbfgsb*` and `outputs/noise-rt-lbfgsb*`; launcher metadata are `outputs/lbfgsb-launch.json`. The earlier Powell-containing queues were stopped; their completed outputs are preserved under the original paths. Do not merge results across differing manifests.
- User authorized all three REML stability extensions. Added separately named AR-noise, separate-component, shared-task, and combined estimators; see reml_model.md. Added exact covariance/gradient/scalar-reduction tests and projection-corrected AR calibration. The fixed original-baseline/L-BFGS-B queues continue independently while these extensions are validated.
- Real validation uses four deterministically spaced anatomical vertices [19694,21262,50665,52617], all 12 runs in sessions 11 and 12, the full 533-candidate library and the identical seeded complementary 6/6 splits. This tiny pilot checks execution and metadata, not a definitive stability ranking. Outputs are `outputs/reml-validation-session11` and `outputs/reml-validation-session12`.

Review outcome: projection-roundoff covariance and inactive-junk boundary defects fixed with RED/GREEN tests. Full experiment suite: 100 tests pass. Reviewer found no blocker for the configured real validation. Minor (deferred): at extremely large variance bounds, activation of the numerical Q floor makes the analytic gradient inconsistent with the floored objective; the reproduced case is outside the default normalized 1e6 bound for these twelve-run fits. Do not enlarge that bound without floor-activation handling and tests.

Completed final validation: `outputs/reml-final-session11` and `outputs/reml-final-session12`, frozen c79a2a5 checkout, all 120 vertex/evaluation records status ok and all selected extended fits converged. See reml_validation.md and reports/reml_validation_*.tsv. The pre-fix `reml-validation-session*` attempts remain incomplete and excluded. Launched full 1750-vertex primary-amplitude REML variant comparisons for sessions 11/12 and a separate 288-case synthetic REML-variant sweep; exact processes/commands are in outputs/reml-launch.json. These queues retain the baseline and test the three extensions individually and combined.

## Expansion to all sessions 10–19

User authorized the REML analysis across sessions 10–19. Input preflight verified all 120 matched runs, 226 samples/motion rows per run, 80 s nuisance construction, event categories/timing, and identical identities for all 1,750 mask vertices. Inventory: `reports/reml_sessions10to19_inventory.tsv`.

Keep the scientific specification and c79a2a5 execution checkout unchanged: RT amplitude, all 12 runs per session, original REML plus AR1/components/shared/combined, all-run fits and one complementary 6/6 partition. The existing queue owns sessions 11/12; the new one-worker queue owns sessions 10,13,14,15,16,17,18,19. Both write distinct subdirectories under `outputs/full-vtc-reml`. The original 20-split full-study proposal is not silently enabled by this session expansion.

New queue: `outputs/reml-sessions10to19.log`; exact command/PID/settings: `outputs/reml-sessions10to19-launch.json`. Existing sessions: `outputs/full-vtc-reml.log` and `outputs/reml-launch.json`. This launches computation; it is not a completed ten-session result.

## Requested notebook refresh after completion (2026-11-07)

User request: “once the earlier analyses are done then revise the results notebook to reflect these new results rather than the older ones.”

Pending: replace the older pilot comparisons in `results.ipynb` with the completed full-VTC L-BFGS-B/library comparisons, expanded REML results for sessions 10–19, and noise/RT sensitivity results. Keep the separate `autodiff_results.ipynb` benchmark and link it for context. The earlier pilot results should no longer be presented as the current results. Preserve a copy of the current notebook before replacement, including existing user edits.

Before refreshing, verify all expected outputs and completion markers: 16 full-VTC comparison units in `outputs/full-vtc-lbfgsb`, 288 cases in each of `outputs/noise-rt-lbfgsb` and `outputs/noise-rt-reml`, and all ten session outputs in `outputs/full-vtc-reml` (sessions 10–19, original REML plus four variants). Validate the expected methods, vertices, evaluations, provenance and failed-fit diagnostics; process exit alone is not evidence of successful completion. Use matched valid vertices and pooled SSE for aggregate R², retain within-session split interpretation, and distinguish RT modes and library scopes. Execute the refreshed notebook and verify its tables against saved outputs.

At request time, all five queues were still running. This is a recorded follow-up, not an installed automatic notebook refresh or scheduler.

## Autodiff replacement launch (2026-11-07, 15:44 local)

User authorized stopping the finite-difference full-VTC queue and restarting unfinished continuous fits with autodiff. The old queue and its descendants were terminated; REML queues were left running. All artifact hashes for the three completed library units were verified before reuse. No completed result was overwritten.

The replacement queue uses frozen commit `eed72b76e66943efe5ed9f2672b4b4490123c959` in `/tmp/boldtailor-hrf-autodiff-run`, pinned JAX/jaxlib 0.12.0.1 in an isolated uv overlay, CPU float64, four workers, and `gradient="autodiff"` for every continuous fit including inner regularization tuning. All other settings, split definitions, initialization, and optimizer budgets are preserved. Nine autodiff/tuning tests passed in that frozen environment before launch; worker startup confirmed the pinned JAX environment.

Use `outputs/full-vtc-lbfgsb` for completed jobs 0–2 only (session 11 amplitude full baseline, parametric CV, parametric REML). Use `outputs/full-vtc-autodiff` for remaining jobs 3–15, including all four continuous units and remaining library units. The incomplete finite-difference continuous directory remains historical and must not be included as a completed result. These two roots together define the 16-unit comparison for the pending notebook refresh. Noise/RT sensitivity outputs and expanded REML output locations remain unchanged.

Launch metadata, exact command, script hash, source revision, and environment overrides: `outputs/autodiff-launch.json`. Execution log: `outputs/full-vtc-autodiff.log`. The launcher is saved at `outputs/full-vtc-autodiff-launch.py`. Nested tuning runtime has not yet been measured; the eight-vertex speedup is not a guaranteed full-workflow speedup.

## Persistent cache and worker memory guard (2026-11-07)

User authorized substantially loosening worker memory recycling and trying disk caching. Added opt-in `compilation_cache_dir` and `worker_memory_growth_bytes` settings. The configured limit is 8 GiB above each worker's baseline (four workers; host physical RAM verified as 128 GiB), replacing loky's approximately 300 MB default. This is a growth guard, not a hard RSS cap. Its private loky override is isolated in `autodiff_runtime.py` and fails explicitly if the expected hook disappears. JAX persistent caching includes all compilation durations and entry sizes.

Validation: all 113 experiment tests passed with pinned JAX. A 128-vertex, all-12-run session-11 fixed-lambda benchmark took 21.282 s with an empty cache and 19.814 s with a populated cache in fresh processes. The latter logged 12 persistent-cache hits (including objective compilation); objective values, function counts and fit statuses matched exactly. Neither run logged worker-replacement warnings. Sampled worker RSS was 0.8–1.0 GB. This short check does not establish memory stability across all nested folds. Raw results and logs: `outputs/cache-runtime-benchmark*`.

To avoid discarding the active queue's completed tuning work, a one-shot supervisor is waiting for the current session-11 amplitude continuous comparison's COMPLETE marker, verifies all artifact hashes and its 5,250 score rows, then stops only the old queue process group and launches jobs 4–15 from frozen commit `24315f8` in `/tmp/boldtailor-hrf-cached-run`. It stops with a recorded error if the original queue exits early or cannot be safely terminated. Status: `outputs/cache-handoff-state.json`; supervisor log: `outputs/cache-handoff.log`; metadata: `outputs/cached-autodiff-launch.json`. It does not automatically revise the notebook.

Notebook inputs after handoff: original `full-vtc-lbfgsb` jobs 0–2, `full-vtc-autodiff` job 3, and `full-vtc-autodiff-cached` jobs 4–15. New queue log: `outputs/full-vtc-autodiff-cached.log`; persistent cache: `/tmp/boldtailor-jax-compilation-cache`. Current work and REML queues remain unchanged until that verified boundary.

## Stop and AR(1) scalar-solver correction (2026-11-08)

User requested killing the ongoing jobs and fixing the unnecessary AR(1) solver overhead. Terminated the cached full-VTC queue and both REML queues, including worker descendants; verified no analysis launchers remained. Preserved all existing results and both user-edited notebooks. No full queue was restarted.

Implemented a vectorized scalar restricted-likelihood solver for the AR(1)-only variant, preserving its noise model and normalized variance bounds. See `reml_scalar_validation.md` and `reports/reml_scalar_validation.json`: full session-11 validation finished in 38 seconds with identical HRF winners at all 1,750 vertices and likelihood differences below 5.5e-12. Other expanded variants still use the generic solver. Future restarts must use fresh provenance directories because implementation fingerprints have changed.

## Complex-model scope narrowed (2026-11-08)

User does not need further multi-session expansion of the complex REML models. Summarized only the completed full-VTC sessions 10 and 11 in `reml_completed_sessions.md`; incomplete sessions 12/13 were excluded. Do not resume the old expanded-model queues merely to finish sessions 10–19. All queues remain stopped.
