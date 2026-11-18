# API reference

[User guide](user-guide.md) · [Practical guide](practical-guide.md) · [Developer guide](development.md) · [Code-review guide](code-review.md)

Functions and classes can be imported from their defining modules below. The
top-level `boldtailor` package also re-exports the primary entry points:
`AnalysisData`, `HrfCandidate`, `HrfLibrary`, `ModelSpec`, `RemlFit`,
`RemlGroup`, `RemlSelection`, `TaskModel`, `fit`, `fit_reml`,
`fit_single_trials`, `from_arrays`, `select_denoising`,
`select_random_effects_design`, `task_delta_r2`, and `with_denoising`. Modules
beginning with `_` are private implementation details.

Returned numerical arrays are read-only NumPy arrays (`writeable=False`); call
`.copy()` before modifying them. Result dataclasses are keyword-only on direct
construction and compare by object identity (`eq=False`) whenever they hold
arrays or tables.

## Data and model specification

### `boldtailor.data`

`from_arrays(signals, events, *, tr=None, frame_times=None, confounds=None,
sources=None, provenance_metadata=None)` returns an `AnalysisData`.

| Argument | Meaning |
| --- | --- |
| `signals` | One time × feature array, or a sequence of arrays |
| `events` | One event DataFrame per run, with `onset` and `duration` in seconds |
| `tr` / `frame_times` | A shared TR, or one explicit frame-time array per run; supply one |
| `confounds` | Optional DataFrame per run, with one row per time point |
| `sources` | Optional sequence of `RunSources`, one per run |
| `provenance_metadata` | Optional descriptive metadata for the provenance record |

`AnalysisData` exposes `signals`, `events`, `confounds`, `frame_times`, `n_runs`,
`n_features`, `timing_source`, and `provenance`. Signal and frame-time arrays
are owned and read-only; `events` and `confounds` return copies.

- `run_labels_for(data, run_labels)` returns validated run labels as a tuple of
  unique `[A-Za-z0-9_-]+` strings, defaulting to `("run-01", "run-02", ...)`.
- `subset_runs(data, run_indices)` returns an `AnalysisData` containing the
  selected zero-based runs.

### `boldtailor.model`

`ModelSpec(contrasts, ...)` requires `contrasts`, mapping contrast names to
string expressions or column-weight dicts.

| Option | Default | Purpose |
| --- | --- | --- |
| `confounds` | `()` | Names of confound columns to include |
| `hrf_model` | `"glover"` | HRF passed to Nilearn (`"glover"`, `"spm"`, derivative basis, callable, or `None`) |
| `drift_model` | `"cosine"` | Drift basis (`"cosine"`, `"polynomial"`, or `None`) |
| `high_pass` | `0.01` | Cosine high-pass cutoff in Hz |
| `drift_order` | `1` | Polynomial drift order |
| `oversampling` | `50` | Temporal oversampling factor (`OVERSAMPLING`) |
| `min_onset` | `-24.0` | Earliest modeled onset relative to the first frame, in seconds (`MIN_ONSET`) |
| `noise_model` | `"ar1"` | `"ar1"` or `"ols"` |
| `task_model` | `None` | `TaskModel` shared with HRF selection; requires `hrf_model` `"spm"` or `"glover"` and raw per-trial events |

`contrast_names` preserves the order of `contrasts`. At least one nonzero t
contrast is required; numeric vectors and F contrasts are rejected.

Additional exports in `boldtailor.model`:

| Name | Purpose |
| --- | --- |
| `Modulator(column, missing="error", kind="numeric", levels=None, reference=None)` | Trial modulator specification (`missing` is `"error"` or `"indicator"`; `kind` is `"numeric"` or `"categorical"`) |
| `TaskModel(modulators=())` | Shared task-regressor specification with `regressor_names`, `profiled_names`, `fingerprint`, `is_subset_of(other)`, and `to_dict()` |
| `level_name(value)` | Canonical text for a categorical value (`1`, `"1"`, `1.0` -> `"1"`), `None` for missing (`"n/a"`, `""`, NaN), or `ValueError` for booleans/infinities |
| `DIAGNOSTIC_NOISE_MODEL` | `"ols"`, used for nested full/nuisance $\Delta R^2$ fits |
| `HRF_NORMALIZATION` | `"peak_one_event_response"` |
| `contrast_metadata(contrasts)` / `contrasts_from_metadata(metadata)` | Convert contrast mappings to and from JSON-serializable records |
| `model_identity(model, *, hrf_model="keep")` | Return `ModelIdentity(activity, fingerprint, warnings)` |
| `nuisance_model_settings(model)` | Nuisance-only model record (`events=False`, `noise_model="ols"`) |

A `Modulator` with `kind="categorical"` uses reference coding: `task` is the
reference-level response (first sorted level by default), and each non-reference
level adds a 0/1 indicator column `<column>[<level>]`. A categorical modulator
created without `levels` has `resolved=False` until `with_levels(values)` is
called; `TaskModel` requires resolved modulators. With `missing="indicator"`,
missing trials receive zero modulation and runs with missing values gain a
`missing_<column>` regressor (`indicator_name`).

## Conventional fits

| Function | Returns |
| --- | --- |
| `boldtailor.design.compile_designs(data, model)` | Tuple of `CompiledDesign(matrix, excluded_event_count, min_onset_cutoff)` per run |
| `boldtailor.design.compile_nuisance_designs(data, model)` | Corresponding per-run designs without task events |
| `boldtailor.design.kernel_task_columns(frame_times, events, kernel, model, run)` | Nilearn task columns for one unit-peak kernel, with unit-peak event scaling and no drift/constant columns |
| `boldtailor.fit.fit(data, model, *, hrf_selection=None, feature_signature=None)` | `AnalysisResult` (or `HrfAnalysisResult` when `hrf_selection` is given) |
| `boldtailor.fit.task_delta_r2(data, model, full_result)` | `TaskDeltaR2Result`, from nested OLS fits |

With `hrf_selection`, `fit()` replaces `model.hrf_model` with the selected HRF
at each feature and checks `feature_signature`, `oversampling`, `min_onset`, and
`TaskModel` compatibility (`selection.task_model.is_subset_of(model.task_model)`).

`AnalysisResult` (`boldtailor.results`) provides:

| Member | Contents |
| --- | --- |
| `effect(name)`, `variance(name)` | Contrast estimate and variance |
| `stat(name)`, `z_score(name)`, `one_sided_p_value(name)` | t statistic, z score, and one-sided uncorrected p-value |
| `contrast_names` | Tuple of contrast names |
| `run_r2`, `r2` | Per-run and pooled full-model $R^2$ |
| `design_matrices`, `design_provenance` | Fitted run DataFrames and design metadata |
| `provenance` | `ProvenanceRecord` |

Across multiple runs, contrasts are combined as equal-weight fixed effects with
summed degrees of freedom. A feature that is constant in any run has NaN
contrast statistics and NaN `run_r2` for that run; pooled `r2` is NaN only if
the feature is constant in every run.

`HrfAnalysisResult` (`boldtailor.hrf_glm_results`) exposes the same contrast and
$R^2$ interface, plus `hrf_indices`, `hrf_selection`, `selection_provenance`,
`group_design_provenance` (keyed by `(run_index, hrf_id)`),
`rebuild_group_design(run_index, hrf_id)` (`KeyError` for unfitted pairs), and
`materialize_group_designs()`. Undefined HRFs (`-1`) yield NaN contrast and $R^2$
values.

`TaskDeltaR2Result` (`boldtailor.results`) exposes `full_r2`, `nuisance_r2`,
`raw_delta_r2`, zero-clipped `delta_r2`, `negative_voxel_count`, `raw_min`,
`nuisance_design_matrices`, and `provenance`. Result builders in
`boldtailor.results` are `make_result`, `make_task_delta_r2_result`,
`contrast_result`, `mask_contrast`, and `owned_contrasts`.

## Prepared designs

`boldtailor.prepared.PreparedDesignAnalysis.from_arrays(signals, design_matrices,
*, tr=None, frame_times=None, column_roles=..., sources=None, run_metadata=None,
provenance_metadata=None)` validates caller-supplied per-run DataFrames.
`column_roles` maps every column to `"task"`, `"nuisance"`, `"intercept"`, or
`"other"`.

From `boldtailor.prepared_fit`:

```text
fit_prepared(prepared, *, contrasts, noise_model="ar1", model_metadata=None) -> AnalysisResult
task_delta_r2_prepared(prepared, full_result) -> TaskDeltaR2Result
```

`task_delta_r2_prepared` requires a complete `task`/`nuisance`/`intercept`
partition (no `"other"` columns) and complete source metadata, reading contrast
and noise settings from `full_result.provenance`.

## Single-trial fits

From `boldtailor.single_trial`:

```text
fit_single_trials(data, *, ridge_alpha=0.0, run_labels=None, hrf_model="spm",
                  ridge_fraction=None) -> SingleTrialResult
fit_selected_hrfs(data, *, hrf_selection, ridge_alpha=0.0, run_labels=None,
                  feature_signature=None, ridge_fraction=None) -> SingleTrialResult
```

`hrf_model` accepts `"spm"` or an `HrfCandidate`. Positive `ridge_alpha` and
`ridge_fraction` (a float in `(0, 1]` or a per-feature array with NaNs for
excluded features) are mutually exclusive. All supplied confounds plus an added
run intercept are included unpenalized.

`SingleTrialResult` (`boldtailor.single_trial_results`) exposes:
- `run_betas`: tuple of `(trials, features)` arrays.
- `trial_table`: DataFrame with `trial_index`, `trial_id`, `run_label`,
  `event_index`, original event columns, and `run_index`.
- `design`: `SharedTrialDesign` (`design.matrices` tuple of DataFrames) or
  `SelectedTrialDesign` (`design.hrf_indices`, `design.design_fingerprint`,
  `design.selection_provenance`, and `design.rebuild_matrix(run_index, hrf_id)`
  returning a read-only array with trial columns followed by nuisance columns).
- `materialize_designs()`: dict mapping `(run_index, hrf_id)` to read-only
  design arrays (`hrf_id=0` for `SharedTrialDesign`).
- `run_full_r2`, `run_nuisance_r2`, `full_r2`, `nuisance_r2`, `delta_r2`,
  `diagnostics`, `ridge_alpha`, `ridge_fraction`, `run_ridge_alphas`, and
  `provenance`.

## Per-grayordinate fractional ridge

From `boldtailor.fractional_ridge`:

```text
score_fraction_candidates(data, predictors, *, fractions, library=None,
                          run_labels=None, feature_signature=None,
                          encoding_mode="within_run", hrf_method="cv",
                          reml_options=None, task_model=TaskModel()) -> CandidateScores
select_ridge_fractions(scores, *, feature_mask=None) -> FractionSelection
select_ridge_fractions(candidate_r2, fractions, *, feature_mask=None) -> FractionSelection
```

- `CandidateScores` (`boldtailor.ridge_results`) holds
  `regularization="fractional_ridge"`, descending `grid`, `cv_r2`
  (`(candidates, features)`), `fold_sse`, `fold_sst`, `fold_hrf_indices`,
  `trial_masks`, `run_labels`, and `provenance`. Validation targets are fixed
  OLS betas under each fold's training-selected HRF.
- `FractionSelection` holds `fractions`, per-feature `ridge_fraction`,
  `selected_r2`, `fraction_indices`, `scoring_mask`, and `at_boundary`. Ties
  within `1e-12` choose the largest fraction; ineligible features have NaN
  `ridge_fraction` and index `-1`.

## Encoding-guided ridge selection and trial encoding

From `boldtailor.ridge_selection`:

```text
score_ridge_candidates(data, predictors, *, alphas, library=None,
                       run_labels=None, feature_signature=None,
                       encoding_mode="within_run", hrf_method="cv",
                       reml_options=None, task_model=TaskModel()) -> CandidateScores
select_ridge_penalty(scores, *, percentile=90.0, feature_mask=None) -> RidgeSelection
select_ridge_penalty(candidate_r2, alphas, *, percentile=90.0, feature_mask=None) -> RidgeSelection
```

- `CandidateScores` has `regularization="normalized_ridge"` and ascending
  `grid`. Validation targets use the candidate `ridge_alpha`.
- `RidgeSelection` holds scalar `ridge_alpha`, sorted `alphas`,
  `objective_scores`, `percentile`, `scoring_mask`, and `at_boundary`. Ties
  within `1e-12` choose the smallest alpha.
- `boldtailor.ridge_results` also exports `paired_scores(scores, grid, *, kind,
  grid_name)` and `scoring_mask(scores, feature_mask)`.

Both candidate scorers require at least two runs with `library=None` (canonical
SPM) or `hrf_method="reml"`, or at least three runs with an `HrfLibrary` under
`hrf_method="cv"`. Both scorers emit lifecycle events
(`encoding_guided_ridge_cv_*`, `encoding_guided_fractional_ridge_cv_*`) and
record software environment metadata in provenance.

From `boldtailor.trial_encoding`:

```text
trial_predictors(events_tables, task_model, labels=None) -> list[pd.DataFrame]
EncodingDesign.prepare(predictors, *, train_runs, test_runs,
                       encoding_mode="within_run", n_trials=None) -> EncodingDesign
encoding_design.score(beta_runs) -> TrialEncodingResult
evaluate_trial_encoding(beta_runs, predictors, *, train_runs, test_runs,
                        encoding_mode="within_run") -> TrialEncodingResult
validate_encoding_mode(encoding_mode) -> str
encoding_metadata(encoding_mode) -> EncodingMetadata
```

- `trial_predictors` converts `TaskModel` modulators into numeric per-run
  DataFrames aligned with event rows (nonfinite or nonpositive RTs become NaN;
  categorical modulators become 0/1 indicators with NaN where missing).
- `EncodingDesign` validates the train/test split and prepares the predictor
  design once so `.score(beta_runs)` can evaluate multiple beta sets without
  recomputing predictor rank or centering.
- `TrialEncodingResult` holds `coefficients` (`(1 + p, features)`, row 0 is
  `task`), `predictor_means`, `predictor_names`, `train_runs`, `test_runs`,
  `trial_masks`, `predictions`, `run_sse`, `run_sst`, `r2`, `encoding_mode`,
  `train_run_predictor_means`, `train_run_intercepts`, and `scoring_offsets`.

## Scalar REML

From `boldtailor.reml`:

```text
RemlGroup(signals, fixed_design)
fit_reml(groups, random_design, *, rho=0.0) -> RemlFit
select_random_effects_design(groups, candidates, *, rho=0.0) -> RemlSelection
```

- `RemlGroup` holds one run's `signals` (`(scans, features)`) and `fixed_design`
  (`(scans, fixed_columns)`). `random_design` is a sequence of per-group random
  design matrices (`(scans, random_columns)` or `None`). `rho` is a scalar or a
  per-feature vector with `|rho| < 1`.
- `RemlFit` holds per-feature read-only `ratio` ($\lambda = \tau^2 / \sigma^2$),
  `noise_variance` (marginal $\sigma^2$), `effect_variance` ($\tau^2$),
  `criterion` ($-2$ restricted log-likelihood up to a constant), `status`
  (`"ok"`, `"variance_zero"`, `"no_residual_signal"`,
  `"numerically_unresolved_residual"`, or `"unidentifiable"`), `at_upper`,
  `unrefined`, `rho`, and integer `restricted_dof`.
- `RemlSelection` holds `winner` (candidate index, or `-1` when no candidate has
  status `"ok"`), `fit` (`RemlFit` for the winner, or the $\lambda=0$ / NaN fit
  when `winner == -1`), `gap` (runner-up minus winning criterion, NaN if fewer
  than two `"ok"` candidates), `n_candidates`, `any_at_upper`, and
  `any_unrefined`.

Because each distinct `rho` value requires a separate fixed-design projection
and candidate SVD per group, bin per-feature `rho` values before calling REML;
see [REML conventions](development.md#reml-conventions).

From `boldtailor.ar1`:

```text
whiten(values, rho) -> np.ndarray
prepare_reference(design) -> RhoReference
estimate_rho(signals, reference, *, rho_range=(-0.2, 0.95)) -> RhoEstimate
```

- `whiten` applies stationary AR(1) whitening normalized so $W R(\rho) W^\top = I$.
- `prepare_reference` caches the orthonormal basis, column-conditioning
  amplification, and expected lag-1 ratio curve for a reference design.
- `estimate_rho` computes bias-corrected lag-1 autocorrelation per group and
  feature (`values`), uncorrected ratios (`raw`), and the median across all
  finite estimates clipped to `rho_range` (`pooled`, `0.0` if none are finite,
  with `pooled_at_bound` marking a boundary hit).

## HRF libraries, selection, and evaluation

From `boldtailor.hrf_library`:

| Entry point | Purpose |
| --- | --- |
| `default_hrf_library(n_samples=512, *, seed=0)` | Default library: canonical SPM (ID 0), `n_samples` timing-space Sobol candidates, and 20 GLMsingle empirical HRFs; cached per `(n_samples, seed)` |
| `timing_hrf_library(n_samples=512, *, seed=0, bounds=None, onset=0.0, duration=36.0)` | Sample realized waveform timing (`REALIZED_NAMES`) over `TIMING_BOUNDS` (with log-uniform `LOG_SCALED_TIMING`) and refine to SPM parameters |
| `sobol_hrf_library(n_samples=512, *, seed=0)` | Sample continuous double-gamma parameters (`PARAMETER_NAMES`) with scrambled Sobol; canonical SPM at ID 0 |
| `expanded_hrf_library()` | 649-candidate discrete parameter grid |
| `glmsingle_hrf_library()` / `glmsingle_hrf_curves()` | Canonical SPM plus the 20 empirical GLMsingle HRFs / the `(20, 501)` unit-peak curves at 0.1 s (`GLMSINGLE_HRF_COUNT = 20`) |
| `realized_timing(parameters)` / `spm_parameters_from_realized(timing)` | Measure realized timing on a combined curve / invert realized timing to SPM double-gamma parameters |
| `timing_parameters(parameters)` / `spm_parameters(timing)` | Exact closed-form conversion between SPM parameters and single-lobe timing (`TIMING_NAMES`) |
| `HrfLibrary.from_parameters(parameters, origin=None, *, include_glmsingle=False)` | Build a library from 7-element parameter tuples, adding canonical SPM at ID 0 and optional GLMsingle curves |
| `HrfLibrary.from_table(table, origin=None)` | Reconstruct a library from its `parameter_table` |
| `HrfCandidate(id, kind, parameters)` | Single candidate (`kind` in `("spm", "double_gamma", "glmsingle")`), with `candidate.kernel(tr, oversampling=50)` returning a unit-peak array |

`HrfLibrary` exposes `candidates`, `parameter_table`, `timing_table`,
`parameter_bounds`, `informative_parameters`, `candidate_bound_flags(margin=0.02)`,
`curves`, `times`, `origin`, and `fingerprint`.

From `boldtailor.hrf_selection` and `boldtailor.hrf_options`:

```text
RemlHrfOptions(rho=0.0, reference_hrf_id=0)
select_hrfs(data, *, library=None, run_labels=None, feature_signature=None,
            candidate_batch_size=32, task_model=TaskModel(),
            method="cv", reml_options=None) -> HrfSelectionResult
evaluate_hrf_split(data, *, library=None, train_runs, test_runs,
                   run_labels=None, feature_signature=None,
                   task_model=TaskModel(), candidate_batch_size=32,
                   method="cv", reml_options=None) -> HrfEvaluationResult
```

- `HrfSelectionResult` (`boldtailor.hrf_results`) exposes `hrf_indices`,
  `method` (`"cv"` or `"reml"`), `cv_r2`, `canonical_cv_r2`, `delta_cv_r2` (all
  three `None` when `method="reml"`), `reml` (`RemlHrfDiagnostics` when
  `method="reml"`, else `None`), `library`, `task_model`, `eligibility`,
  `run_labels`, `feature_signature`, `parameter_bound_flags`,
  `at_parameter_bound`, `parameter_bound_table()`, and `provenance`.
- `RemlHrfDiagnostics` holds `selection` (`RemlSelection`), `candidate_ids`,
  `canonical_fit` (`RemlFit` or `None`), `delta_criterion` (canonical minus
  winning criterion), and `reference_hrf_id`.
- `HrfEvaluationResult` exposes `training_selection`, `training_amplitudes`
  (`(n_regressors, n_features)` in `amplitude_names` order), `amplitude_names`,
  `test_r2`, `canonical_test_r2`, `delta_test_r2`, `train_runs`, `test_runs`,
  and `provenance`.

## Task-guided denoising

From `boldtailor.denoising` and `boldtailor.denoising_results`:

**Attribution.** The denoising procedure follows GLMsingle's GLMdenoise
stage (Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M.J.,
Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. *eLife*, 11, e77599.
[doi:10.7554/eLife.77599](https://doi.org/10.7554/eLife.77599)), with the
deviations listed in the [user guide](user-guide.md#task-guided-denoising).
It is an independent reimplementation;
GLMsingle (Copyright (c) 2021, Kendrick Kay) is distributed under the BSD
3-Clause License, reproduced in `src/boldtailor/_resources/GLMsingle-LICENSE.txt`.
GLMsingle's authors have not reviewed or endorsed Boldtailor.

```text
select_denoising(data, *, task_model=TaskModel(), library=None,
                 counts=(0, 1, ..., 10), pool_r2_threshold="auto",
                 pcstop=1.05, significance_gate=True, gate_alpha=0.05,
                 gate_binomial_alpha=0.05, gate_noise_model="ar1",
                 feature_signature=None, run_labels=None) -> DenoisingResult
with_denoising(data, result, *, feature_signature=None) -> AnalysisData
```

- `select_denoising` requires at least three runs. `pool_r2_threshold="auto"`
  fits a two-component Gaussian mixture (`MixtureThreshold`) to finite ON-OFF
  $R^2$ values; non-convergence emits a `RuntimeWarning` and records
  `converged=False`, while a degenerate fit or missing 0.5 posterior crossing
  raises `ValueError`.
- `DenoisingResult` holds:
  - Settings: `counts`, `pool_r2_threshold`, `pcstop`, `run_labels`,
    `feature_signature`, `source_identity`, and `provenance`.
  - `significance_gate` (`SignificanceGate`), with properties `n_components`
    (gated count) and `pcstop_count` (pre-gate `pcstop` count) delegating to it.
  - `pool_masks` (`PoolMasks` with `pool`, `scoring`, `threshold`, `mixture`,
    `fallback`, `pool_size`, `scoring_size`), with properties `noise_pool`,
    `scoring_mask`, `noise_pool_threshold`, `noise_pool_mixture`, and
    `scoring_fallback` delegating to it.
  - `onoff_r2`, `scored`, `initial_selection`, `selection_cv_r2`, and
    `initial_hrf_indices`.
  - `run_components` (tuple of `(scans, n_components)` arrays),
    `component_names`, and `pca_diagnostics` (`PcaDiagnostics` with
    `run_labels`, `pool_size`, `ranks`, `singular_values`, `rank_tolerances`,
    `retained_columns`, and `table()`).
  - `folds` (tuple of `DenoisingFold`), `candidate_scores`, `fold_scores`,
    `perf`, and `curve`.
- `with_denoising` verifies run, signal, timing, confound, and event identity,
  appends `denoise_pc_000`, ... (`component_names(count)`, prefix
  `COMPONENT_PREFIX`), and annotates each run's confound `SourceRef` with
  `denoising_augmentation`.

## Source records and saving

From `boldtailor.provenance`:
- `SourceRef(role, uri=None, media_type=None, byte_size=None, modified_at=None,
  sha256=None, annotations=None)`: `role` is `"signal"`, `"events"`, or
  `"confounds"`. Complete source identity requires a relative POSIX `uri`,
  nonnegative `byte_size`, and UTC `modified_at` (`Z` or `+00:00`). Optional
  `sha256` is 64 lowercase hex characters.
- `RunSources(signal, events, confounds=None)`: per-run source bundle.
- `ProvenanceRecord`: holds `schema` (`SCHEMA_ID = "boldtailor.provenance/1"`),
  `execution_id`, `sources`, `activities`, `events`, `warnings`,
  `data_fingerprint`, `analysis_fingerprint`, `latest_activity` (owned copy of
  the most recent activity dict, or `{}`), `activity_records()`, `to_dict()`,
  `from_dict()`, and `canonical_json()`.
- Helpers: `analysis_fingerprint(data_fingerprint, model)`,
  `identity_activity(record)`, `extend_provenance(...)`,
  `validate_relative_path(text, *, name="path")`, and `validate_uri(uri)`.

From `boldtailor.bids_provenance`:
- `project_bids_provenance(record, ...)` returns a dict mapping relative paths
  to UTF-8 bytes (`dataset_description.json` at `STABLE_BIDS_VERSION = "1.11.1"`,
  `logs/boldtailor_provenance.json`, and optional draft BEP028 provenance files
  when `export_bids_prov=True`).

From `boldtailor.publication`:
- `Artifact(path, payload)` and `publish_artifact_set(destination, artifacts, *,
  source_paths=(), overwrite=False, lock_timeout=30.0, keep_existing=(),
  manifest_path=None)` stage and atomically replace files under a sibling
  `<destination>.boldtailor` lock directory (`is_control_directory(path)`),
  rolling back overwritten and manifest-removed files on `PublicationError`.

```python
from io import BytesIO
import numpy as np

from boldtailor.bids_provenance import project_bids_provenance
from boldtailor.publication import Artifact, publish_artifact_set

buffer = BytesIO()
np.save(buffer, result.effect("face_gt_house"))
metadata_files = project_bids_provenance(result.provenance, export_bids_prov=False)
artifacts = [Artifact("face_gt_house.npy", buffer.getvalue())]
artifacts.extend(Artifact(path, content) for path, content in metadata_files.items())
saved = publish_artifact_set("example-results", artifacts)
```

From `boldtailor.logging`:
- `bind_context(*, execution_id=..., data_fingerprint=...,
  analysis_fingerprint=..., run_index=..., inherit=True)`
- `emit_event(event, *, stage, level=logging.INFO, error=None, ...)`
- `append_event_history(history, event)` (retains the latest 8 events)
- `normalization(stage)` (context manager for input-normalization lifecycles)

## Imaging, diagnostics, and parallel helpers

| Function | Purpose |
| --- | --- |
| `boldtailor.cifti.scalar_artifact(path, brain, values, names)` | Build a float32 `.dscalar.nii` `Artifact` on a `BrainModelAxis` |
| `boldtailor.cifti.read_scalar(path, brain, names=None)` | Read a dense scalar CIFTI after verifying its grayordinate axis and map names |
| `boldtailor.cifti.cortical_values(values, brain)` | Extract `left` and `right` cortical surface arrays (missing vertices NaN) |
| `boldtailor.cifti.spatial_signature(brain, indices)` | SHA-256 spatial signature for `feature_signature` |
| `boldtailor.diagnostics.one_sample_t(run_betas)` | Pooled trial-beta `mean_beta`, `t`, `p_uncorrected`, `n_trials`, and `df` (`ONE_SAMPLE_T_NAMES`) |
| `boldtailor.diagnostics.pearson_correlation(cross, beta_ss, rt_ss, counts)` | Pearson $r$ clipped to $[-1, 1]$; NaN below 3 trials or zero variance |
| `boldtailor.diagnostics.correlate_rt(beta_runs, rt_runs, *, run_numbers)` | Within-run centered beta/RT Pearson $r$ and counts (`per_run`, `all`, `odd`, `even`); pooled sets require $\ge 3$ trials and $\ge 2$ pooled residual degrees of freedom |
| `boldtailor.diagnostics.even_run_points(beta_runs, rt_runs, run_numbers, vertex)` | Centered `(rt, beta)` arrays from even runs at one feature |
| `boldtailor.reliability.curve_correlations(library, ids_a, ids_b)` | Full-curve Pearson $r$ for `a_b`, `a_canonical`, and `b_canonical` (`CORRELATION_NAMES`) |
| `boldtailor.reliability.compare_hrfs(library, hrf_indices, sessions)` | Multi-session curve agreement, matched canonical baselines, parameter SDs, and mean peak times (`SUMMARY_NAMES`) |
| `boldtailor.parallel.map_blocks(function, blocks, *, args=(), n_jobs=1)` | Evaluate `function` over feature blocks in bounded `loky` worker batches (1 inner thread per worker) |

Lower-level helpers exported for example and validation scripts include
`expand_events`, `task_columns`, `hrf_model`, and `event_response_scales`
(`boldtailor.design`); `prepare_runs` and `subset_runs`
(`boldtailor.hrf_selection`); `fraction_grid`, `regularization`, and
`NORM_BASIS` (`boldtailor.fractional_ridge`); and `r_squared`, `validate_alpha`,
and `compile_trial_run` (`boldtailor.single_trial`).

## Session workflow and command line

`boldtailor.workflow` implements `boldtailor run` (`boldtailor.cli`):

| Name | Module | Purpose |
| --- | --- | --- |
| `WorkflowSettings` | `boldtailor.workflow.settings` | Validated session workflow settings (`to_dict()`, `from_dict()`, `build_library()`, `selection_options()`) |
| `run_workflow(settings)` / `describe_inputs(settings)` | `boldtailor.workflow.run` | Run enabled stages and publish `WorkflowResult`, or return the dry-run plan and `problems` |
| `detect_task_model`, `task_model_notes`, `observed_levels` | `boldtailor.task_detection` | Infer or validate `TaskModel` modulators from per-run event tables |
| `load_session`, `WorkflowRun`, `InputError`, `selection_task_model`, `validate_glm_events`, `glm_model`, `make_blocks`, `block_signals`, `load_block`, `run_summary`, `min_runs_per_half` | `boldtailor.workflow.inputs` | Session loading, confound/event validation, non-constant block partitioning, and run-count rules |
| `discover_runs`, `load_runs`, `load_inputs`, `select_confounds`, `run_sources`, `source_ref`, `odd_even_parity`, `reaction_times`, `bids_label`, `input_paths` | `boldtailor.workflow.files` | BIDS + fMRIPrep file discovery, confound selection, and source-reference construction |
| `select_hrfs`, `fit_glms`, `fit_beta_series` | `boldtailor.workflow.analysis` | Blockwise HRF selection, conventional GLM fitting, and fixed-penalty beta-series fitting |
| `selection_maps`, `all_selection_maps`, `selection_names`, `selection_method`, `selection_values` | `boldtailor.workflow.hrf_maps` | Assemble grayordinate maps for CV and REML HRF selections and split evaluations |
| `fit_beta_models`, `fit_cv_beta_series`, `BetaModel` | `boldtailor.workflow.beta_series` | Fit canonical and selected-HRF trial models, inner ridge CV, and odd/even outer encoding evaluations |
| `GlmFit`, `BetaFit`, `OuterEvaluation`, `TuningResult`, `CvBetaResult` | `boldtailor.workflow.records` | Frozen dataclasses holding assembled session-wide workflow results |
| `tuning_provenance`, `link_final_provenance` | `boldtailor.workflow.ridge_provenance` | Provenance records for ridge tuning and linked final refits |
| `save_workflow`, `workflow_artifacts`, `check_output`, `metadata_name`, `report_name` | `boldtailor.workflow.outputs` | Assemble stage artifacts and publish derivatives and the HTML report |
| `glm_artifacts`, `hrf_artifacts`, `beta_model_artifacts`, `activation_artifacts`, `figure_artifacts`, `input_artifacts` | `boldtailor.workflow.export_cifti` | Build stage CIFTI, NPZ, TSV, and JSON artifacts |
| `figure_bundle` | `boldtailor.workflow.figures` | Render report and surface figures from fitted workflow outputs |
| `tuned_models`, `scope_selection`, `predictor_table` | `boldtailor.workflow.tables` | Build tuning and predictor summary tables |
| `metadata` | `boldtailor.workflow.manifest` | Build the `<stem>_desc-boldtailor_metadata.json` manifest |
| `json_artifact`, `table_artifact`, `npz_artifact`, `figure_artifact`, `dataset_description`, `parameter_artifact`, `scalar_map` | `boldtailor.workflow.artifacts` | Construct in-memory `Artifact` payloads |
| `selected_hrfs`, `encoding_scores`, `ridge_boundary`, `input_tables`, `reml_selection_counts` | `boldtailor.workflow.summaries` | Build report summary tables |
| `design_figure`, `library_figure`, `glm_comparison`, `parameter_agreement`, `curve_agreement`, `activation_histogram`, `rt_check_figure`, `fraction_selection_figure` | `boldtailor.workflow.plots` | Matplotlib report figures |
| `find_surface_meshes`, `surface_figure` | `boldtailor.workflow.surfaces` | Discover fsLR 32k meshes and render 4-view cortical surface figures |
| `render_report`, `ReportInputs` | `boldtailor.workflow.report` | Render the self-contained HTML session report |
| `build_parser`, `sessions_from_args`, `settings_from_args`, `main` | `boldtailor.cli` | CLI parser and entry point |

## Example workflows

Located under `examples/` in the repository checkout (not installed with the
wheel):

| Module / function | Purpose |
| --- | --- |
| `examples.bids_workflow.nsd_settings` | Build `WorkflowSettings` for the NSD example notebooks |
| `examples.bids_workflow.session_hrf.estimate_sessions` | Select and cache HRFs across multiple NSD sessions |
| `examples.bids_workflow.multisession_workflow.ensure_session_outputs` | Reuse or compute session workflow outputs for multi-session analyses |
