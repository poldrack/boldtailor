# Empirical HRF Estimation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. Execution started after the user requested continuation; amendments in plan.md take precedence.

**Goal:** Compare library CV, experimental library REML, and regularized continuous HRF estimation for sub-07, sessions 10–19, accounting for trial type, RT, motion, and drift.

**Architecture:** Keep experimental methods and the comparison harness under `experiments/hrf_estimation`. Share event construction, nuisance handling, convolution, splits, and scoring across methods. Reuse package primitives where their semantics match; do not change the production evaluation API to implement this experiment.

**Tech Stack:** Existing Python environment managed by uv; NumPy, SciPy, pandas, nibabel, Nilearn, pytest, and matplotlib. TOML configuration via Python's `tomllib`.

**Spec:** [plan.md](plan.md), including the confirmed Q&A dated 2026-11-07.

## Global constraints

- Write clean, modular code; prefer short functions.
- Use uv for package management and `uv run` for local commands.
- Every `__init__.py` must remain completely empty.
- Write tests, demonstrate RED, and **commit failing tests before implementation**. Then implement, demonstrate GREEN, refactor, and commit implementation. Never weaken tests to obtain a pass.
- Keep REML and continuous optimization experimental. Preserve source data and the neighboring prototype repository.
- Freeze only the HRF for held-out evaluation; refit amplitudes on held-out runs.
- Split runs within sessions; one HRF per location shared by task regressors; binary trial type only.
- Report both complete-model and nuisance-adjusted task R².
- Include 24 motion regressors and cosine drifts; no temporal censoring. Missing and incorrect responses get separate junk regressors.
- Start with ventral temporal cortex (VTC). Parameters below marked “proposed” are implementation choices, not additional user decisions.

## Evidence from initial inspection

- All ten requested sessions have 12 `desc-layerB2` CIFTIs (120 files total). This is a file count, not a completed integrity audit.
- The inspected session-11 run-01 CIFTI has 226 samples, a 1.3333334 s sampling interval, and 59,412 cortical vertices. Its BrainModelAxis contains left and right cortex only.
- The matching raw NIfTI has 188 volumes at 1.6 s. The matching fMRIPrep confounds table has 188 rows, including all 24 motion columns and cosine columns. These cannot be attached directly to the CIFTI rows.
- The user subsequently selected NSD-preprocessed motion tables and model-generated 80-second cosine drifts, superseding fMRIPrep confounds.
- `src/boldtailor/hrf_selection.py:select_hrfs` selects by leave-one-run-out prediction with amplitudes trained on other runs. `evaluate_hrf_split` freezes both HRF and amplitudes, so it is not the requested outer evaluator.
- `src/boldtailor/hrf_library.py` provides a default library containing canonical SPM, 512 timing-space double-gamma candidates, and 20 GLMsingle kernels. Existing timing bounds include peaks from 2.5 to 8.5 s. Do not assume this proves sufficient coverage of the observed fast responses.
- `src/boldtailor/_hrf_design.py` uses peak-normalized event responses. This convention matters particularly for any variable-duration RT sensitivity analysis.
- `../boldtailor_testing/src/boldtailor_testing/reml.py` estimates random single-trial amplitude variance and noise variance, with optional noise models. The inspected `analysis/hrf_offset0` selection script compares library CV and single-trial in-sample RSS, not a ready-made REML HRF selector with the requested task model.
- Existing replication code provides CIFTI loading and ROI mapping patterns, but uses polynomial nuisance terms and has historical timing offsets. Do not inherit those analysis settings.

## Analysis design

### Inputs, timing, and VTC pilot

Discover runs by subject/session/task/run entities, not sorted-file pairing. Require matching events, confounds, timing metadata, and cortical vertex identities. Inventory all runs before fitting. Use acquisition seconds throughout and retain event durations from the TSV (3 s in the inspected rows).

**Updated nuisance policy:** use matched NSD-preprocessed motion tables directly on the CIFTI grid. Require equal row counts and validate finite values; do not interpolate fMRIPrep confounds. Construct the 24-parameter model from six supplied translations/rotations: originals, backward differences (first row zero), original squares, and difference squares. Generate a discrete cosine basis separately for each run with cutoff 80 s (0.0125 Hz), excluding its constant column; include one explicit intercept. Use floor(2 * n_scans * TR / 80) cosine terms, with samples cos(pi * (i + 0.5) * k / n_scans) and k starting at 1. Reject nonfinite motion rather than imputing or censoring scans. Do not add polynomials, global signal, CompCor, or learned nuisance components.

**Proposed pilot:** bilateral anatomical fusiform cortex as an operational VTC subset, using `freesurfer-NSD/subj07/label/{lh,rh}.aparc.annot`. Verify source surfaces and the exact registration used to generate the target CIFTIs before resampling categorical labels. Map labels using hemisphere and surface vertex IDs, never array position alone. Broader VTC is a later configurable mask. Do not select the pilot vertices by their observed R².

Begin with session 11, four runs, and up to 256 vertices balanced across hemispheres, sampled deterministically with seed 20261107. Then use all 12 runs in sessions 11 and 12 and the full pilot ROI before extending to sessions 10–19. Full-cortex execution remains a separate scaling decision after resource measurement.

### Common task model

Proposed mutually exclusive trial partition:

1. Missing: missing response/invalid RT (nonfinite or nonpositive), or an explicit missing-data indicator.
2. Incorrect: a response with valid RT and `is_correct == 0`, excluding the missing category.
3. Valid: a response with valid RT and `is_correct == 1`.

Verify the events JSON definitions, particularly `missing_data` and `is_correct` versus `is_correct_session`, before applying this mapping. Unexpected codes or unknown correctness fail validation rather than becoming valid trials. Missing takes precedence over incorrect to avoid duplicate nuisance events.

Use valid trials for two condition regressors (`trial_type` 0 and 1) and one common RT-amplitude regressor. Use uncentered RT in seconds and no serial orthogonalization. Missing and incorrect events each form one separate convolved junk regressor and do not also enter the valid task columns. All five event columns share the location's HRF. Empty junk categories produce absent/zero columns with explicit metadata, not failed runs. Retain scans containing junk events.

The supplied Mumford paper distinguishes amplitude modulation from a common RT-duration regressor (Fig. 2 and Discussion). **Primary analysis follows the requested amplitude modulation.** Proposed sensitivity: replace that RT column with a unit-height, RT-duration boxcar convolved with the same HRF; retain the constant-duration condition columns. Do not peak-normalize each variable-duration event separately, which would remove the intended duration-dependent scaling. Label this analysis separately; do not choose the RT formulation using outer test results.

### Split and scoring definitions

Proposed full analysis: 20 seeded 6/6 run splits within each 12-run session, with identical stored splits for every method. Generate 10 distinct unordered partitions and evaluate both directions. The four-run smoke pilot uses two complementary 2/2 fits. Store run IDs, not only array indices. Require at least two training runs for inner CV. Unexpected run omissions require a revised manifest; never silently change split balance.

For each outer training half, estimate the HRF without seeing outer test signals. Fit task and nuisance coefficients independently within each evaluation run using the frozen HRF and ordinary least squares. The same evaluator serves every method. Also fit each method on all runs of a session and report descriptive in-sample scores.

Let C contain intercept, motion, and cosine terms; J(h) the two junk columns; X(h) the three valid-task columns. For a frozen h:

- `SSE_full`: residual sum of squares after fitting `[C, J(h), X(h)]` per run.
- `SSE_nuisance`: residual sum of squares after fitting `[C, J(h)]` per run.
- `SST`: sum of squares around each run's mean.
- `r2_full = 1 - sum(SSE_full) / sum(SST)`.
- `r2_task_partial = 1 - sum(SSE_full) / sum(SSE_nuisance)`.
- Also record `delta_r2_task = (sum(SSE_nuisance) - sum(SSE_full)) / sum(SST)` to distinguish partial R² from incremental total variance explained.

Pool sums of squares before taking ratios. Return NaN and a reason when a denominator is numerically zero; do not silently replace NaNs by zero. Record ranks, residual degrees of freedom, and failure masks. Because J depends on h, also retain `SSE_nuisance` and the C-only SSE so changes in junk fit remain visible across methods.

Call the held-out metric **HRF-transfer R² with refitted amplitudes**. It is not fully out-of-sample prediction of BOLD; test-run amplitudes are fitted in sample. Do not present overlapping split estimates or cortical vertices as independent replicates for significance tests.

### Estimators and fair comparisons

**Canonical baseline:** fixed canonical SPM HRF, evaluated by the identical scoring procedure.

**Library CV:** preserve the package's inner leave-one-run-out predictive selection principle. Train on outer-training runs only. Build an experimental adapter if the package TaskModel cannot represent the exact valid/junk partition and run-profiled junk columns. Verify agreement with `select_hrfs` on the subset of models both support; never approximate the specified design merely to reuse an API.

**Library REML:** the user selected direct REML with random task effects. The implemented initial model uses common HRF-independent fixed nuisance C and five HRF-dependent random task/junk coefficients, independent across runs, with one shared random-effect variance and one shared noise variance across training runs. No separate single-trial deviation component is included in this initial task-level comparison. See [reml_model.md](reml_model.md) for the complete covariance, profiled objective, boundary rules, and dense-reference validation. Outer evaluation still uses the common refitted-amplitude OLS scorer.

**Continuous double-gamma optimization:** use variable projection: solve run-specific linear coefficients exactly at each HRF parameter vector, then optimize a regularized training SSE. This shares a single HRF across conditions, following the shared-shape principle in the supplied Pedregosa paper, but is not claimed to reproduce its rank-1 basis model.

Proposed starting parameter box: response delay [2, 10] s; undershoot delay [8, 24] s; response dispersion [0.3, 2.5]; undershoot dispersion [0.5, 4]; response/undershoot ratio [2, 20]; fixed onset 0 s; fixed support 36 s. Use scaled/log coordinates for positive parameters. Reject nonfinite/degenerate kernels and undershoot-before-response solutions. Record realized peak, width, trough, depth, and boundary proximity; gamma delay is not peak time.

Proposed penalty: squared distance in normalized parameter coordinates from the best training-library double-gamma initialization, with scales equal to the parameter box widths. Normalize training SSE by C-only residual energy. Select lambda from {0, 0.001, 0.01, 0.1, 1} using inner run splits and the HRF-transfer scorer; recompute the reference initialization inside each inner-training fold. Refit on the outer training half after selection. This is stabilization around a training-derived reference, not a physiological prior.

Compare multistart L-BFGS-B and bounded Powell using identical starts (canonical plus up to three distinct training-library double-gamma shapes), stopping budgets, and objective. Pilot budget: 500 iterations and 5,000 objective evaluations per start; record actual counts and status. Never use an empirical GLMsingle kernel as though it had SPM parameters. Keep the best finite objective including initialization; flag failed optimization explicitly.

Use one identical library for both library methods. Start with the existing 533-candidate default. Report continuous comparisons both against the full library and its parametric subset, since the full library includes empirical shapes outside the double-gamma family. Freeze library construction and bounds before the full comparison. Proposed sensitivity checks: 128/512/2048 timing candidates and expanded bounds when pilot boundary diagnostics indicate inadequate coverage. Report the coverage mismatch with the continuous box explicitly; do not attribute all gains to optimization alone.

Use OLS/independent-noise fitting for the initial shared baseline. Simulate temporally correlated errors and inspect residual autocorrelation. Any AR sensitivity must specify the same training-only noise-estimation policy across methods and must retain unwhitened BOLD R² for comparability. No inferential standard errors are justified by the initial independent-noise assumption.

### Stability, simulation, and reporting

Within-session stability compares HRFs estimated independently on complementary run halves. Between-session stability compares all-run estimates at matched vertices across the ten sessions (45 pairs). Evaluate normalized curves on a common 0.1 s grid over 0–36 s, zero-padding shorter-support kernels. Report curve correlation, curve RMSE, absolute peak-time difference, FWHM difference, and undershoot differences where defined. Candidate-ID agreement is supplementary for identical libraries only. Report all anatomically selected vertices and failure fractions; any signal-strength stratification must be explicitly descriptive or defined from independent/training data.

Simulation fixtures should include known fast double-gamma kernels, canonical kernels, off-grid kernels, non-double-gamma shapes, RT-correlated trial types, amplitude- and duration-driven RT effects, junk trials, motion/drift, zero signal, and correlated noise. Separate optimizer correctness from statistical recovery: use exact/noiseless cases for strict numerical assertions and seeded Monte Carlo summaries for bias and stability. Do not require a method to win by construction.

Report paired method differences per session, descriptive distributions across sessions and splits, convergence/failure rates, boundary hits, runtime, and memory. This is one participant; conclusions concern this dataset and experiment. Select a preferred method based primarily on transfer R², with HRF stability and computational cost visible alongside it.

## Review focus

1. Mismatched NSD motion/CIFTI row counts: actionable failure, never row padding (Task 1).
2. Overlapping missing/incorrect codes and absent categories: exclusive trial partition and stable scoring (Task 2).
3. Outer-test leakage through tuning, eligibility, or initialization: perturbing held-out signals must not alter the trained HRF (Tasks 3–6).
4. Candidate-dependent REML fixed effects/scaling: dense reference likelihood and explicit comparable model required (Task 5).
5. Constant signals, collinear designs, bad optimizer exits, and reordered CIFTI vertices: explicit flags and correct mapping (Tasks 1, 3, 6, 7).

## Implementation tasks

All paths below are relative to `experiments/hrf_estimation/`. Use function-based tests with pytest fixtures in `conftest.py`. Do not add dependencies or scaffold modules during planning.

For **each coding task**, follow this exact sequence: write the stated tests; run `uv run python -m pytest experiments/hrf_estimation/test_<name>.py -q` and verify failure for the missing behavior; stage and commit only that task's failing tests/fixtures; implement; rerun that command to GREEN; refactor and rerun affected tests; commit implementation. Unrelated existing changes must remain untouched.

### Task 1: Input manifest, temporal alignment, and anatomical pilot

**Files:** `inputs.py`, `timing.py`, `roi.py`, `test_inputs.py`, `test_timing.py`, `test_roi.py`, `conftest.py`.
**Interfaces:** `discover_runs(root: Path, subject: str, sessions: tuple[int, ...]) -> tuple[RunSpec, ...]`; `load_run(spec: RunSpec, vertices: ndarray, cutoff: float = 80.0) -> RunData`; `build_nuisance(motion: DataFrame, frame_times: ndarray, cutoff: float = 80.0) -> DataFrame`; `vtc_indices(brain_axis, labels_by_hemisphere) -> ndarray`.
Define frozen RunSpec/RunData records in `inputs.py` carrying run identity, source paths, signals, events, frame times, confounds, and brain-axis identity.

- [ ] Verify matched NSD motion/CIFTI rows and document source-to-fsLR registration before real-data fitting.
- [ ] Tests assert entity matching despite shuffled paths, rejection of missing/mismatched runs, 24 expanded motion columns plus 80-second cosines/intercept, exact derivative boundary handling, and rejection of row-count mismatches.
- [ ] Tests assert a known hemisphere/vertex permutation maps correctly, empty ROI fails, derived first-row motion differences are zero, and no scan is censored.
- [ ] Complete the RED/commit/GREEN/refactor cycle separately for each test module; validate the real-data manifest read-only.

### Task 2: Shared event design

**Files:** `design.py`, `test_design.py`.
**Interfaces:** `partition_events(events: DataFrame) -> DataFrame`; `build_design(run: RunData, kernel: HrfCandidate, rt_mode: str) -> DesignParts`. Define DesignParts with `task`, `junk`, `nuisance`, column names, and rank diagnostics in `design.py`.

- [ ] Tests assert every event belongs to exactly one category; missing takes precedence; valid task columns have no junk contributions; both trial types and one common RT column are present.
- [ ] Tests compare convolution with an independently constructed small event example; assert RT enters once, no orthogonalization, shared HRF, and absent junk columns are harmless.
- [ ] Test that RT-duration sensitivity preserves increased integrated neural input for longer durations instead of normalizing each duration away.
- [ ] Complete RED/commit/GREEN/refactor before fitting any data.

### Task 3: Splits and frozen-HRF transfer evaluation

**Files:** `splits.py`, `scoring.py`, `estimators.py` (result record only), `test_splits.py`, `test_scoring.py`.
**Interfaces:** `make_splits(run_ids: tuple[str, ...], seed: int, n_partitions: int) -> tuple[Split, ...]`; `score_hrfs(runs: tuple[RunData, ...], estimate: HrfEstimate, rt_mode: str) -> ScoreResult`. Define Split in `splits.py`, ScoreResult in `scoring.py`, and HrfEstimate in `estimators.py`: feature IDs, kernels, parameters if defined, status, training run IDs, and diagnostics. Task 4 extends this module with estimator dispatch.

- [ ] Tests assert 6/6 disjoint within-session halves, reproducible complementary directions, and identical stored splits across methods.
- [ ] Test known held-out amplitudes different from training amplitudes: refitting recovers an exact synthetic fit, proving amplitudes were not frozen.
- [ ] Assert the three R² formulas against direct least-squares residuals, pooled-SSE weighting for unequal run energies, NaN on constant signals, and explicit rank diagnostics.
- [ ] Complete RED/commit/GREEN/refactor; do not call the production frozen-amplitude evaluator for this endpoint.

### Task 4: Canonical and library-CV estimators

**Files:** `estimators.py`, `library_cv.py`, `test_library_cv.py`.
**Interfaces:** `fit_hrfs(runs: tuple[RunData, ...], method: str, settings: dict) -> HrfEstimate`; `fit_library_cv(runs, library: HrfLibrary, settings: dict) -> HrfEstimate`. Settings must serialize all seeds and model choices.

- [ ] Test recovery of a planted on-grid kernel in an identifiable noiseless fixture, deterministic ties, and invalid-feature handling.
- [ ] Assert agreement with package CV for a compatible task model; assert the custom valid/junk design is used in the full experimental case.
- [ ] Assert changing outer-test signals has no effect on chosen training HRFs or eligibility; test training-set provenance.
- [ ] Complete RED/commit/GREEN/refactor and score the canonical/library smoke pilot only after timing/ROI prerequisites pass.

### Task 5: Experimental REML

**Files:** `reml_model.md`, `reml.py`, `test_reml.py`.
**Interface:** `fit_library_reml(runs, library: HrfLibrary, settings: dict) -> HrfEstimate` using the model in `reml_model.md`.

- [ ] Implement the user-selected direct REML model documented in `reml_model.md`; compare its objective against a dense restricted-covariance reference.
- [ ] Write dense small-matrix reference tests for the complete chosen objective, variance-zero boundary, scale convention, candidate-dependent terms, and multi-run parameter sharing; require equality with the optimized computation.
- [ ] Test planted-HRF simulations, zero-signal failure flags, and outer-test perturbation invariance.
- [ ] Commit failing tests before adapting prototype formulas; then GREEN/refactor. Record which prototype components were reused and which assumptions changed.

### Task 6: Continuous optimization and training-only tuning

**Files:** `continuous.py`, `tuning.py`, `test_continuous.py`, `test_tuning.py`.
**Interfaces:** `fit_continuous(runs, settings: dict) -> HrfEstimate`; `choose_regularization(runs, settings: dict) -> float`. Use Task 3 splits/scorer and Task 4 initialization.

- [ ] Test the profiled objective against explicit joint least squares at fixed parameters; compare finite-difference gradients where gradients are supplied.
- [ ] Test an identifiable off-grid noiseless shape, parameter constraints, non-worsening relative to initialization, deterministic multistart behavior, and explicit failed/boundary solutions.
- [ ] Assert lambda selection and initialization use inner-training data only; altering outer-test data cannot change either.
- [ ] Complete RED/commit/GREEN/refactor, then compare L-BFGS-B/Powell convergence, cost, and recovery on the same simulations.

### Task 7: Reproducible runner, stability, and artifacts

**Files:** `run.py`, `outputs.py`, `stability.py`, `report.py`, `configs/pilot.toml`, `configs/full.toml`, `test_outputs.py`, `test_stability.py`, `test_run.py`.
**Interfaces:** `run_experiment(config_path: Path) -> Path`; `write_estimate(estimate, brain_axis, destination: Path) -> None`; `compare_hrfs(left: HrfEstimate, right: HrfEstimate) -> DataFrame`.

- [ ] Tests assert end-to-end synthetic outputs preserve cortical vertex IDs, invalid/unprocessed vertices remain NaN, interrupted writes cannot masquerade as complete, and resumed results match fresh results.
- [ ] Test stability on identical and deliberately shifted curves, undefined trough handling, and session/feature alignment.
- [ ] Complete RED/commit/GREEN/refactor; run `uv run python -m pytest experiments/hrf_estimation -q` and affected package tests. The default pytest testpaths do not include these experiment tests.
- [ ] Execute in stages: synthetic validation; four-run/256-vertex smoke pilot; full VTC in sessions 11–12; then sessions 10–19 after timing, convergence, and resource checks.

## Outputs and execution readiness

Write under `experiments/hrf_estimation/outputs/<analysis-id>/`, using names such as `sub-07_ses-nsd11_space-fsLR_den-32k_desc-cvsplit01_stat-r2task.dscalar.nii`. Store method, split, ROI, RT formulation, kernel normalization, and exact metric definitions in JSON sidecars. Use TSV summaries and NPZ arrays for curves/parameters and diagnostics, with explicit feature IDs. Do not imply formal BIDS compliance for custom entities.

Save a manifest, source hashes, code revision and dirty-state record, uv lock hash, library fingerprint, masks, configuration, split IDs, timing transformation, optimizer logs, and run-level sums of squares. Write outputs atomically and make cache keys depend on all scientific settings. Proposed local default: one worker, 256-feature chunks, bounded candidate batches; measure runtime/RSS and estimate the ten-session cost before larger execution. No full-cortex runtime commitment is made here.

Implementation may begin with the input/design/scoring tasks after the plan is accepted. Real-data fitting requires verified temporal alignment and anatomical mapping. The direct REML objective in Task 5 has now been specified and verified against a dense reference. These are concrete prerequisites, not grounds to substitute easier scientific requirements.

## Local references

- [Mumford et al. (2024)](references/Mumford_2024_NHB.pdf): RT-amplitude versus RT-duration models, Fig. 2, Results, and Discussion.
- [Pedregosa et al. (2015)](references/pedregosa2015.pdf): shared HRF/rank-1 formulation and optimization sections; motivation rather than an exact implementation recipe.
- [Package HRF selection](../../src/boldtailor/hrf_selection.py), [kernel construction](../../src/boldtailor/_hrf_design.py), and [HRF libraries](../../src/boldtailor/hrf_library.py).
- Neighboring prototype: `../boldtailor_testing/src/boldtailor_testing/reml.py` relative to the repository root; inspected read-only.

## Execution status (2026-11-07)

Implemented and tested: entity-matched NSD inputs, 24 motion terms, 80-second cosine drifts, exclusive trial categories, shared-HRF designs, within-session splits, all three estimator families (both continuous optimizers), training-only lambda tuning, frozen-HRF transfer scoring, waveform stability, cortical map export, and verified whole-analysis restart with input/code/settings identities. Independent review led to additional anatomical identity, rank-deficiency, boundary, zero-variance, and RT-metadata checks.

Executed: session-11 four-run/256-vertex canonical/library/REML pilot; a four-vertex continuous-optimizer smoke check with a smaller initialization library. Full-VTC 12-run pilots for sessions 11 and 12 are the next scaling check. The full ten-session/20-split experiment and comprehensive Monte Carlo sensitivity suite are not yet complete. The current runner handles one session at a time; a cluster scheduler/configuration layer is not implemented.

Implementation files differ slightly from the initial decomposition: `pilot.py` contains the comparison runner and atomic exports; `provenance.py` owns content hashes and verified restart; `tuning.py` owns the common lambda choice across the supplied feature subset. No production package API has been changed. Cosine generation replaces the obsolete interpolation task. The initial anatomical pilot is bilateral fusiform cortex (1,750 mapped vertices); categorical label mapping uses nearest vertices on registered spheres, with hashes and original CIFTI registration provenance retained.
