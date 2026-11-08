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

## Model scope decision (2026-11-08)

User explicitly dropped separate-components, shared-effects, and combined REML models. Retain original REML and the AR(1)-only extension for ongoing comparisons. Preserve completed complex-model results as historical sensitivity analyses, but do not launch or resume additional fits of those variants.

## AR(1)-only completion launch (2026-11-08)

User authorized finishing AR(1) for sessions 14–19 with the new scalar solver. Verified 12 input runs per session. Launched two independent queues (14/16/18 and 15/17/19), each using one BLAS thread, from frozen commit `6fea6cf` in `/tmp/boldtailor-hrf-ar1-scalar-run`. Output root: `outputs/full-vtc-ar1-scalar`; launch provenance: `outputs/ar1-scalar-launch.json`; logs: `outputs/ar1-scalar-even.log` and `outputs/ar1-scalar-odd.log`. Preserve existing sessions 10–13. Each new session retains 1,750 VTC vertices, 533 HRFs, RT amplitude, NSD 24-motion nuisance terms, 80-second cosine HPF, seed 20261107, all-run plus both complementary 6/6 transfer fits, and refitted held-out amplitudes. No complex variants or continuous-optimization queues were restarted.

### AR(1) completion verified (2026-11-08)

Both scalar queues finished successfully. Verified completion hashes and all 5,250 score rows (1,750 vertices × three evaluations) for each session 14–19. Total: 18 new fits and 31,500 score records. There are 365 zero-variance records and 31,135 `ok` records; zero variance is a fitted boundary outcome, not an execution failure. No `not_converged` records were reported. Together with saved sessions 10–13, AR(1) now covers all requested sessions 10–19. Inventory: `reports/ar1_sessions14to19_completion.tsv`.

## All-method comparison notebook (2026-11-08)

Created `all_methods_results.ipynb` as a separate executed notebook, preserving both existing notebooks. It includes canonical, full/parametric library CV and REML, continuous L-BFGS-B, AR(1), historical complex REML variants, and a separately labeled earlier-pilot/Powell section. The current snapshot contains 43 complete model/session/RT combinations with explicit coverage for sessions 10–19. Only complete three-fit model units are compared; individual completed methods from stopped multi-method queues are recovered with artifact/run-set checks. Matched vertices are chosen within each session and RT mode; SSE is pooled before forming scores. Missing models are not treated as zero, and synthetic results are not mixed with empirical sessions. Portable summaries, compressed vertex-level scores and provenance are under `reports/all_methods_*`.

## Complete baseline coverage for AR(1) comparison (2026-11-08)

User requested canonical, full-library CV, parametric-library CV, and original REML for all sessions 10–19. Retain the primary RT-amplitude model and existing within-session 6/6 split/80-second HPF/24-motion settings. Reuse all completed session-11/12 comparisons and original REML sessions 10/13. Run the 30 missing method/session units in four single-thread session queues: 10/13, 14/15, 16/17, 18/19. Frozen source `d9e340a`, checkout `/tmp/boldtailor-hrf-baselines-all-sessions`. Output root `outputs/full-vtc-all-session-baselines`; exact launch metadata `outputs/all-session-baselines-launch.json`. Refresh `all_methods_results.ipynb` after verification. Complex variants and continuous optimization remain stopped.

### Baseline completion and focused notebook (2026-11-08)

All four baseline queues completed. Verified all 16 new output directories, artifact hashes, 157,500 score records (30 new method/session combinations × 3 evaluations × 1,750 vertices), and exact input fingerprints/splits against AR(1). Canonical, full-library CV, parametric-library CV, original REML and AR(1) now cover sessions 10–19. Inventory: `reports/all_session_baselines_completion.tsv`.

User requested a focused notebook: first only these five methods with complete ten-session coverage, then a separate comparison of the five REML variants complete in both sessions 10 and 11. Created and executed `session_comparison.ipynb`; excluded all other partial-coverage results and pilots. Section 1 uses a fixed 1,518-vertex mask across all sessions/methods/directions (1,467 for the converged-only sensitivity). Section 2 uses its own per-session matched masks. Portable data/provenance: `reports/ten_session_core_*`, `reports/two_session_reml_comparison.tsv`, `reports/focused_comparison_*`. Retained the earlier notebooks and their snapshots unchanged. All seven focused/existing comparison-notebook checks and all 1,671 repository tests passed.

## Non-REML and across-session HRF shape comparisons (2026-11-08)

Expanded `session_comparison.ipynb` with HRF curves and between-model agreement, ten-session split-half agreement, per-session mean HRF curves, and all 45 distinct cross-session comparisons for each of canonical, full-library CV, parametric CV, original REML and AR(1). Verified all 150 source HRF fit hashes. Use a fixed 1,487-vertex mask with finite nonconstant HRFs for every model/session/evaluation; normalize each curve by its positive peak. Correlations compare the same vertex, not ROI averages. Across-session comparisons use all-run HRFs, while split-half comparisons use six-run fits; interpret their different training sizes explicitly. Canonical perfect stability is by construction. Saved portable shape tables, mean curves, and provenance as `reports/core_hrf_*`. No model refitting or partial-coverage analyses were added.

## Empirical HRF selection report (2026-11-08)

Added session-wise empirical versus parametric HRF selection counts for full-library and parametric-only CV, using all-run fits and the existing 1,487-vertex shape mask. Validated all 20 source fit hashes and library fingerprints. Added all 45 session-pair comparisons of family agreement, Cohen's kappa, empirical Jaccard overlap, conditional exact empirical-ID agreement, and exact candidate agreement for both methods. The notebook includes figures and tables; `reports/empirical_selection_report.md` is a standalone report, with portable TSVs and source provenance. `build_empirical_report.py` rebuilds these summaries without refitting models.

## Regularized AR(1) pilot (2026-11-08)

User approved comparing pooled, empirical-Bayes shrinkage, and surface-smoothed AR(1), then requested a new results notebook. Pilot: sessions 11 and 16, existing seed-20261107 complementary 6/6 splits, 256 fixed vertices sampled with seed 20261108 from the common 1,487-vertex shape mask. Noise estimation uses all 1,750 anatomical VTC vertices in training runs only. Full 533-candidate HRF library, original binary condition/RT-amplitude/junk/motion24/cosine80s model, no censoring.

Shrinkage: atanh-transformed, residual-projection-bias-corrected run coefficients; uncertainty of the training-run mean, with run variances moderated toward their median using four prior degrees of freedom. Moment estimate of nonnegative between-vertex variance and shrinkage toward the pooled session-training estimate. This treats run variation as uncertainty and is an approximate empirical-Bayes model. Surface method: mean corrected training-run coefficients, area-weighted 5-mm FWHM Gaussian smoothing of mesh-edge geodesics (truncated at three sigma) on subject-specific fsLR midthickness meshes; pool only VTC neighbors and keep hemispheres separate. These anatomical surfaces come from fMRIPrep, but motion confounds remain NSD-preprocessed.

Keep the pooled coefficient exact; bin spatial estimates at width 0.02 for grouped scalar REML, with a finer-bin sensitivity planned. Coefficients remain bounded to [-0.8,0.8] as in the pooled baseline. Evaluate common-mask transfer R², split-half HRF shape stability, and held-out fixed-reference-design residual ACF before/after GLS refitting with the training-only coefficients (lags 1–10; descriptive diagnostics, not a formal whiteness test). No held-out tuning. A separate notebook will show the pilot without modifying the complete-ten-session comparison.

Pilot completed: all 256 selected vertices valid across five methods, two sessions, and both transfer directions. Pooled HRF selections reproduce all prior choices exactly. Both spatial methods improve held-out reference-residual whitening (median absolute lag-1 ~0.104→0.071–0.073 in session 11; ~0.121→0.092–0.094 in session 16), but predictive differences from pooled AR(1) are <0.05 percentage points and session-16 HRF stability remains ~0.913–0.915. EB shrinkage is mild (median vertex weights ~0.94–0.96). Finer-bin sensitivity (0.01 vs 0.02 in session16 transfer01) preserves 98–99% of choices and changes partial R² by <0.004 percentage points. Separate notebook: `regularized_ar1_results.ipynb`; portable artifacts: `reports/regularized_ar1`.
