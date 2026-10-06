# API reference

[User guide](user-guide.md) · [Developer guide](development.md)

Import from the modules shown below, for example
`from boldtailor.single_trial import fit_single_trials`. The top-level package
does not re-export these names. Modules beginning with `_` are implementation
details. Returned numerical arrays are read-only; use `.copy()` to edit them.
See the [ownership migration note](ownership-migration.md) for the ordinary-NumPy
contract and its limits. The [result migration guide](result-migration.md)
lists changed candidate-score and single-trial access paths.

## Data and model specification

### `boldtailor.data`

`from_arrays(signals, events, *, tr=None, frame_times=None, confounds=None,
sources=None, provenance_metadata=None)` returns an `AnalysisData`.

| Argument | Meaning |
| --- | --- |
| `signals` | One time × feature array, or a sequence of arrays |
| `events` | One event DataFrame per run, with `onset` and `duration` in seconds |
| `tr` / `frame_times` | A shared TR, or one explicit frame-time array per run; choose one |
| `confounds` | Optional DataFrame per run, with one row per time point |
| `sources` | Optional sequence of `RunSources`, one per run |
| `provenance_metadata` | Optional descriptive metadata for the analysis record |

`AnalysisData` exposes `signals`, `events`, `confounds`, `frame_times`, `n_runs`,
`n_features`, `timing_source`, and `provenance`. Signal arrays and frame times
are owned by the result; table access returns copies.

`run_labels_for(data, run_labels)` returns validated run labels as a tuple:
one unique `[A-Za-z0-9_-]+` string per run, defaulting to `run-01`, `run-02`, ...
when `run_labels` is `None`; anything else raises `ValueError`.

### `boldtailor.model.ModelSpec`

`contrasts` is required and maps contrast names to expressions or regressor-weight
mappings. The other arguments have these defaults:

| Option | Default | Purpose |
| --- | --- | --- |
| `confounds` | `()` | Names of confound columns to include |
| `hrf_model` | `"glover"` | HRF passed to Nilearn; also accepts a callable or `None` |
| `drift_model` | `"cosine"` | Drift basis, or `None` |
| `high_pass` | `0.01` | Cosine high-pass cutoff in Hz |
| `drift_order` | `1` | Polynomial drift order when using that basis |
| `oversampling` | `50` | Temporal oversampling for HRF convolution |
| `min_onset` | `-24.0` | Earliest modeled onset relative to the first frame, in seconds |
| `noise_model` | `"ar1"` | `"ar1"` or `"ols"` |
| `task_model` | `None` | `TaskModel` shared with HRF selection; requires `hrf_model` `"spm"` or `"glover"` and raw per-trial events |

`contrast_names` preserves the order of the contrast mapping. A model needs at
least one nonzero t contrast. Numeric contrast vectors and F contrasts are not
accepted by this interface.

`boldtailor.model` also exports the defaults and identity helpers the fitting
functions use:

| Name | Use |
| --- | --- |
| `OVERSAMPLING`, `MIN_ONSET` | The `oversampling` (`50`) and `min_onset` (`-24.0`) defaults above |
| `DIAGNOSTIC_NOISE_MODEL` | `"ols"`: the noise model for the full/nuisance R² diagnostic fits |
| `contrast_metadata(contrasts)` | JSON form of a contrast mapping: `{"kind": "expression", "value": ...}` or `{"kind": "weights", "weights": {...}}` per name |
| `contrasts_from_metadata(metadata)` | Invert `contrast_metadata` back to expressions and weight maps |
| `model_identity(model, *, hrf_model="keep")` | A `ModelIdentity` with the settings that identify a model; `hrf_model` overrides how the HRF is recorded (for example `{"kind": "selected"}`) |
| `ModelIdentity(activity, fingerprint, warnings)` | The recorded model settings, the settings used for the analysis fingerprint (`None`, with a reproducibility warning, for a non-importable HRF callable), and those warnings |
| `nuisance_model_settings(model)` | The nuisance-only model record: `events=False`, the model's confounds and drift settings, and `noise_model=DIAGNOSTIC_NOISE_MODEL` |

## Conventional fits

| Function | Returns |
| --- | --- |
| `boldtailor.design.compile_designs(data, model)` | Tuple of `CompiledDesign` objects, one per run |
| `boldtailor.design.compile_nuisance_designs(data, model)` | Corresponding designs without task events |
| `boldtailor.fit.fit(data, model)` | `AnalysisResult` |
| `boldtailor.fit.task_delta_r2(data, model, full_result)` | `TaskDeltaR2Result`, using nested OLS fits |

`fit()` also accepts the keyword arguments `hrf_selection=None` and
`feature_signature=None`. With an `HrfSelectionResult`, it returns an
`HrfAnalysisResult` and replaces `model.hrf_model` with the selected HRF at each
feature. The supplied signature must match the selection's signature; it may
be omitted if selection also omitted it. `feature_signature` without a
selection is an error.

With `model.task_model` set, the selection's task model must equal it or be a
subset of it with identical modulator settings (`TaskModel.is_subset_of`), and
the model's `oversampling` and `min_onset` must equal the selection's recorded
values. The selected-GLM provenance records both `task_model_fingerprint` and
`selection_task_model_fingerprint`. With `task_model=None`, the selection must
use the default task-only model.

`CompiledDesign` contains `matrix`, `excluded_event_count`, and
`min_onset_cutoff`. Events earlier than the cutoff are excluded with a warning.

`boldtailor.design.kernel_task_columns(frame_times, events, kernel, model, run)`
returns the Nilearn task columns for one peak-one kernel, in Nilearn's order,
with each event's amplitude scaled so its response peaks at one and no drift or
constant column; compilation errors are re-raised as `ValueError` naming `run`.

`AnalysisResult` provides:

| Member | Contents |
| --- | --- |
| `effect(name)`, `variance(name)` | Contrast estimate and its variance |
| `stat(name)`, `z_score(name)`, `one_sided_p_value(name)` | t statistic, z score, and directional uncorrected p-value |
| `contrast_names` | Available contrast names |
| `run_r2`, `r2` | Per-run and pooled full-model R² |
| `design_matrices`, `design_provenance` | Fitted run designs and their records |
| `provenance` | Analysis record |

A feature whose signal is constant in a run has NaN for every contrast statistic and for that run's `run_r2`. Pooled `r2` is NaN only when the feature is constant in every run. Constant features cannot support inference.

For several runs, contrasts are combined as equal-weight fixed effects. This is not precision-weighted: each run contributes equally regardless of its length or noise level, and degrees of freedom are summed. For runs with very different noise, compare against nilearn's `compute_fixed_effects` with precision weighting.

`boldtailor.hrf_glm_results.HrfAnalysisResult` has the same contrast methods,
`contrast_names`, `run_r2`, `r2`, and `provenance`. Instead of `design_matrices`
and `design_provenance`, it exposes `group_design(run, hrf_id)`, which
recompiles one fitted design on request (`KeyError` for an unfitted pair), and
`group_design_provenance`, keyed by `(zero_based_run_index, hrf_id)`.
The latter records excluded-event counts, onset cutoffs, design ranks, and
residual degrees of freedom. Additional members are `hrf_indices`,
`hrf_selection`, and `selection_provenance`. Designs are returned as copies.
Undefined HRF assignments produce NaN contrast/R² values.

`TaskDeltaR2Result` provides `full_r2`, `nuisance_r2`, `raw_delta_r2`,
`delta_r2`, `negative_voxel_count`, `raw_min`, `nuisance_design_matrices`, and
`provenance`. See the [user guide](user-guide.md#measuring-task-related-variance)
for source-identity requirements, NaN handling, and the difference from ridge R².
For an `HrfAnalysisResult`, the comparison automatically uses its HRF assignment
and retains NaNs at undefined or constant features. It verifies source metadata,
effective model settings, spatial assignment, and compiled design identity.

`boldtailor.results` holds the builders the fitting functions use:

| Function | Use |
| --- | --- |
| `make_result(contrasts, designs, design_provenance, run_r2, r2, provenance)` | Assemble an `AnalysisResult` |
| `make_task_delta_r2_result(*, full_r2, nuisance_r2, nuisance_designs, provenance, allow_undefined=False)` | Validate matching 1-D R² arrays and finite numeric nuisance designs, then build a `TaskDeltaR2Result` with raw and zero-clipped ΔR², the negative count, and the raw minimum; `allow_undefined` permits NaN entries |
| `contrast_result(contrast)` | Copy a Nilearn contrast's effect, variance, statistic, z score, and p-value |
| `mask_contrast(result, undefined)` | A copy of a contrast result with every statistic NaN where `undefined` is True |
| `owned_contrasts(contrasts)` | A read-only name-to-contrast mapping |

## Prepared designs

`boldtailor.prepared.PreparedDesignAnalysis.from_arrays(signals, design_matrices,
*, tr=None, frame_times=None, column_roles=..., sources=None, run_metadata=None,
provenance_metadata=None)` takes one labeled design per run.

`column_roles` is required and labels every column as `task`, `nuisance`,
`intercept`, or `other`. The object exposes the supplied data, designs, roles,
timing, and run metadata, plus `run_design_fingerprints`, `design_fingerprint`,
`n_runs`, `n_features`, and `provenance`.

From `boldtailor.prepared_fit`:

```text
fit_prepared(prepared, *, contrasts, noise_model="ar1", model_metadata=None)
task_delta_r2_prepared(prepared, full_result)
```

These return the same result types as their event-based counterparts. Neither
function adds regressors or transforms the design. The R² comparison requires
a complete task/nuisance/intercept partition and complete source metadata. It
reads the contrasts, noise model, and model metadata from `full_result`'s
provenance, so `full_result` must come from `fit_prepared`. Passing
`contrasts=`, `noise_model=`, or `model_metadata=` again is deprecated (a
`DeprecationWarning`), and supplied values must match the stored model.

## Single-trial fits

From `boldtailor.single_trial`:

```text
fit_single_trials(data, *, ridge_alpha=0.0, run_labels=None, hrf_model="spm",
                  ridge_fraction=None)
fit_selected_hrfs(data, *, hrf_selection, ridge_alpha=0.0, run_labels=None,
                  feature_signature=None, ridge_fraction=None)
```

`fit_single_trials` accepts `"spm"` or an `HrfCandidate` as `hrf_model`, its
fixed HRF. `fit_selected_hrfs` takes an `HrfSelectionResult` as
`hrf_selection`, the same keyword `fit()` uses. The earlier keywords `hrf=` and
`selection=` still work for one release with a `DeprecationWarning`. The ridge
penalty must be finite and nonnegative. All supplied confounds and a run intercept are included.

Both fitting functions return `SingleTrialResult`, which exposes:

- `run_betas`: tuple of trials × features arrays.
- `trial_table`: `trial_index`, `trial_id`, `run_label`, `event_index`, the
  original event metadata, then `run_index`. Both fitting functions build it
  with the same helper, so the columns are identical.
- `run_full_r2`, `run_nuisance_r2`: per-run diagnostic arrays.
- `full_r2`, `nuisance_r2`, `delta_r2`: pooled diagnostics.
- `diagnostics`, `ridge_alpha`, and `provenance`.

The `design` field contains a `SharedTrialDesign` for a fixed HRF (SPM or a
custom candidate), or a `SelectedTrialDesign` for feature-specific HRFs.
`design.matrices` returns a tuple of DataFrames for a shared design. A selected
design does not retain its matrices: `design.matrix(run_index, hrf_id)` rebuilds
one fitted design on request (`KeyError` for an unfitted pair). The selected
container also exposes `design.hrf_indices`, `design.design_fingerprint`, and
`design.selection_provenance`.
Matrices include trial columns followed by nuisance columns; a selected design's
`matrix()` returns an unlabelled read-only NumPy array rather than a DataFrame. See the
[result migration guide](result-migration.md) for the previous access paths.

## Per-grayordinate fractional ridge

From `boldtailor.fractional_ridge`:

```text
score_fraction_candidates(data, predictors, *, fractions, library=None,
                          run_labels=None, feature_signature=None,
                      encoding_mode="within_run")
select_ridge_fractions(scores, *, feature_mask=None)
select_ridge_fractions(candidate_r2, fractions, *, feature_mask=None)
```

Pass the `CandidateScores` returned by `score_fraction_candidates`, or a
candidate-by-feature array together with its fraction grid.

`CandidateScores` has `regularization="fractional_ridge"`, descending `grid`, `cv_r2`,
`fold_sse`, `fold_sst`, `fold_hrf_indices`, `trial_masks`, `run_labels`, and
`provenance`. Array dimensions match the normalized-ridge scores below, with
fraction replacing alpha. Predictor requirements and nested HRF selection
are the same; targets are fixed OLS betas under the training-selected HRF.
Targets do not depend on the fraction grid.

`FractionSelection` has the candidate `fractions`, per-feature
`ridge_fraction`, `selected_r2`, `fraction_indices`, `scoring_mask`, and
`at_boundary` (boolean per feature: the winner is the largest or smallest
fraction). A winner at the smallest fraction (the shrinkage end) means the
grid should be extended; a winner at fraction 1.0 means no regularization was
preferred.
Every candidate must have a finite score for an eligible feature. Ties within
`1e-12` choose the largest fraction; excluded features have NaN values and
index -1. An entirely invalid block returns undefined maps.

Single-trial fit functions accept `ridge_fraction` as a scalar in `(0,1]` or
a feature array (NaNs explicitly exclude features). Positive `ridge_alpha`
and `ridge_fraction` are mutually exclusive. Fractional results add read-only
`ridge_fraction` and `run_ridge_alphas` arrays, and set `ridge_alpha=None`.
The norm ratio uses raw trial coefficients after nuisance projection;
the returned beta arrays use native signal units. See the
[fractional-ridge guide](user-guide.md#fractional-ridge-at-each-grayordinate).

## Encoding-guided ridge selection

From `boldtailor.ridge_selection`:

```text
score_ridge_candidates(data, predictors, *, alphas, library=None,
                      run_labels=None, feature_signature=None,
                      encoding_mode="within_run")
select_ridge_penalty(scores, *, percentile=90.0, feature_mask=None)
select_ridge_penalty(candidate_r2, alphas, *, percentile=90.0, feature_mask=None)
```

Pass the `CandidateScores` returned by `score_ridge_candidates`, or a
candidate-by-feature array together with its alpha grid. Scores of the other
regularization kind are rejected.

`predictors` contains one numeric DataFrame per run, aligned positionally with
event rows, with matching named columns. Do not include `task`: an intercept is
added. Nonfinite predictor rows are excluded from encoding only. The scorer
requires at least two runs (`library=None`, canonical SPM), or three with an
`HrfLibrary`. Supplied libraries trigger fresh HRF selection within each
inner-training set. Supply only outer-training runs when nesting the call.

`CandidateScores` contains `regularization="normalized_ridge"`, ascending
`grid`, `cv_r2` (alpha × feature),
`fold_sse` and `fold_sst` (validation run × alpha × feature),
`fold_hrf_indices` (validation run × feature), `trial_masks`, `run_labels`,
and `provenance`. Validation targets use the candidate's own regularization.
Scores pool SSE and within-run SST; they are penalty-selection statistics.

`select_ridge_penalty` selects once across all supplied features. Its optional
`feature_mask` is a matching boolean vector. Features must have finite scores
at every alpha; an empty common mask raises `ValueError`. Percentiles use linear
interpolation. Ties within `1e-12` choose the smaller alpha, including zero.
`RidgeSelection` exposes `ridge_alpha`, sorted `alphas`, `objective_scores`,
`percentile`, `scoring_mask`, and `at_boundary` (the chosen alpha is an end of
the grid). A winner at the largest alpha (the shrinkage end) means the grid
should be extended; a winner at alpha 0 means no regularization was
preferred. Merge spatial blocks before this call.

Both selectors use two helpers from `boldtailor.ridge_results`:
`paired_scores(scores, grid, *, kind, grid_name)` returns the score matrix and
grid from a `CandidateScores` of regularization `kind` (passing a grid as well
raises `TypeError`; another kind raises `ValueError`) or from an array plus its
grid (required, else `TypeError`). `scoring_mask(scores, feature_mask)` marks
features finite at every candidate, intersected with an optional matching
boolean `feature_mask`.

From `boldtailor.trial_encoding`:

```text
evaluate_trial_encoding(beta_runs, predictors, *, train_runs, test_runs,
                        encoding_mode="within_run")
```

The default `encoding_mode="within_run"` fits shared OLS slopes with a separate
intercept for each training run. Both predictors and betas are centered within
each training run over complete predictor rows. Validation residuals are centered
within each test run and feature for scoring only. SSE and within-run SST are
pooled before division; negative R² is retained. `encoding_mode="absolute"`
reproduces the original shared-intercept fit and uncentered residual loss. Both
candidate scorers accept the same mode and default.
`validate_encoding_mode(encoding_mode)` returns the mode or raises `ValueError`
for anything other than `"within_run"` or `"absolute"`.
`encoding_metadata(encoding_mode)` returns the dict that describes the
objective in cross-validation provenance and exported artifacts
(`encoding_mode`, `encoding_objective_version=2`, `score`, `encoding_model`,
`predictor_transform`, `validation_intercept`, `prediction_reference`).

Split indices are zero-based. `TrialEncodingResult` contains `coefficients`
(shape `(1+p, features)`), pooled training `predictor_means`, `predictor_names`,
`train_runs`, `test_runs`, `trial_masks`, `predictions` (test runs in requested
order, original trial rows), `run_sse`, `run_sst`, and pooled `r2`. Its additional
required fields are:

- `encoding_mode`: the selected objective.
- `train_run_predictor_means`: `(training runs, p)`, in `train_runs` order.
- `train_run_intercepts`: `(training runs, features)`, intercepts in raw predictor
  coordinates; absolute mode repeats its shared intercept.
- `scoring_offsets`: `(test runs, features)`, the mean residual removed for
  within-run scoring, or zero in absolute mode; invalid entries are NaN.

The first coefficient, named `task`, is the pooled training beta mean. In
within-run mode this is a reference level, not a common fitted run intercept.
Predictions equal `coefficients[0] + (X - predictor_means) @ coefficients[1:]`
and never include scoring offsets. Within-run predictions depend only on training
outcomes and test predictors; test beta validity affects scores, not predictions.
Numerical arrays are owned, ordinary NumPy arrays marked read-only. This protects
against accidental writes; it does not prevent deliberate flag changes. Use
`.copy()` before editing. Every result dataclass is keyword-only: direct
construction must name each field (positional construction raises
`TypeError`), and each class copies its arrays into read-only storage.

Within-run fitting requires at least one complete row per training run, positive
residual degrees of freedom (`total complete rows > training runs + p`), and full
rank after within-run predictor centering. A one-row training run supplies only
an intercept; a predictor varying exclusively between runs is unidentifiable.
Each test run needs at least two complete predictor rows. Nonfinite betas on
included rows invalidate the affected feature fit/score; constant test targets
have undefined R². Excluded trial predictions remain NaN.

## HRF libraries, selection, and evaluation

From `boldtailor.hrf_library`:

| Entry point | Use |
| --- | --- |
| `default_hrf_library(n_samples=512, *, seed=0)` | The library selection uses when `library` is omitted: canonical SPM, `n_samples` timing-space candidates, then the 20 GLMsingle HRFs; cached per `(n_samples, seed)` |
| `glmsingle_hrf_library()` / `glmsingle_hrf_curves()` | Canonical SPM plus the 20 GLMsingle (NSD) empirical HRFs; the `(20, 501)` unit-peak curves at 0.1 s |
| `sobol_hrf_library(n_samples=512, *, seed=0)` | Sample continuous parameters with scrambled Sobol; add canonical SPM as ID 0 |
| `expanded_hrf_library()` | Return the original 649-candidate grid library |
| `timing_hrf_library(n_samples=512, *, seed=0, bounds=None, onset=0.0, duration=36.0)` | Sample realized HRF timing (peak time, response FWHM, trough time, undershoot FWHM at half the trough depth, trough depth; `REALIZED_NAMES`) with scrambled Sobol, convert each point to SPM gamma parameters by numerical refinement, skip infeasible points and keep drawing until `n_samples` are accepted (`origin["rejected"]`), add canonical SPM as ID 0; `bounds` overrides `TIMING_BOUNDS` (peak 2.5-8.5 s, response FWHM 2-6.5 s, trough 8-19 s, undershoot FWHM 4-10 s, depth 0.01-0.4; depth is log-uniform per `LOG_SCALED_TIMING`, recorded in `origin["scales"]`) |
| `realized_timing(parameters)` | Peak time, response FWHM, trough time, undershoot FWHM, and trough depth measured on the combined curve (0.01 s grid, interpolated crossings) |
| `spm_parameters_from_realized(timing)` | SPM gamma parameters whose curve realizes the requested peak time, FWHMs, trough time, and depth (onset carried through); raises `ValueError` for targets no double gamma can realize |
| `timing_parameters(parameters)` / `spm_parameters(timing)` | Exact closed-form conversion between the SPM gamma parameters and the gamma-lobe timing `TIMING_NAMES` (lobe peak time and SD, undershoot-lobe peak time and SD, lobe depth, onset, duration) |
| `HrfLibrary.timing_table` | One row per candidate with both parameterizations |
| `HrfLibrary.origin` | Read-only mapping describing how the library was built (`kind`, such as `"default"`, `"timing_sobol"`, `"sobol"`, `"expanded_grid"`, `"glmsingle"`, or `"explicit"`, plus the generator's settings); not part of equality or `fingerprint` |
| `HrfLibrary.from_parameters(parameters, origin=None, *, include_glmsingle=False)` | Build a library from seven-value parameter rows, adding canonical SPM as ID 0 and, optionally, the 20 GLMsingle kernels after them |
| `HrfLibrary.from_table(table, origin=None)` | Rebuild a library exactly from its saved `parameter_table` (double-gamma rows plus the GLMsingle block if present) |
| `HrfCandidate(id, kind, parameters)` | Describe one kernel; `kind` is `"spm"`, `"double_gamma"`, or `"glmsingle"` (for `"glmsingle"`, `parameters` holds the one-based GLMsingle index) |
| `candidate.kernel(tr, oversampling=50)` | Sample a read-only kernel scaled to a peak of one at TR/oversampling; canonical SPM is Nilearn's `spm_hrf` divided by its maximum |

Parameter order is `response_delay`, `undershoot_delay`, `response_dispersion`,
`undershoot_dispersion`, `response_undershoot_ratio`, `onset_delay`, `duration`.
SPM candidates require the exact canonical parameter tuple
`(6, 16, 1, 1, 6, 0, 32)`.

The module constants are `CANONICAL_PARAMETERS` (that tuple), `OVERSAMPLING`
(`50`, the default `kernel` oversampling), `GLMSINGLE_HRF_COUNT` (`20`), and
`PARAMETER_NAMES` (the order above).

`HrfLibrary` exposes `candidates`, `parameter_table`, `parameter_bounds`
(`low`/`high` of the six sampled parameters over custom candidates),
`informative_parameters` (names among the six with at least three distinct
custom values and nonzero width), `candidate_bound_flags(margin=0.02)`
(per-candidate `(n_candidates, 6, 2)` edge flags), `curves`, `times`, and
`fingerprint`. Every curve peaks at one. Every event's predicted response is scaled to a peak of one (kernels are
also stored at unit peak). A beta is therefore the peak BOLD response to that
presentation in signal units, independent of TR, oversampling, and event
duration, and comparable across grayordinates with different selected HRFs.
This matches GLMsingle's convention. Nilearn derivative and FIR bases are passed to Nilearn unchanged (sum-to-one for the canonical bases); user-supplied kernels are used exactly as given. Plain `hrf_model="spm"` or
`"glover"` in `ModelSpec` follows the same convention, so canonical and
selected-HRF betas share a scale; provenance records
`hrf_normalization="peak_one_event_response"` (Nilearn derivative bases record
`"nilearn_sum_one"`; FIR and user-supplied kernels record no normalization). Table rows and curve rows follow stable candidate IDs. The
table's `peak_time` is the full-curve maximum on the 0.1-second export grid.
Custom rows are sorted deterministically; duplicate rows are rejected.

For Sobol sampling, `n_samples` is the number of custom HRFs and must be a
positive integer power of two; `seed` must be a nonnegative integer. The
default returns 513 candidates. Samples span response delay [3, 6],
undershoot delay [10, 16], response dispersion [0.5, 1.5], undershoot
dispersion [0.5, 2.5], response/undershoot ratio [2, 8], and onset delay
[0, 2], with duration fixed at 36 seconds. These bounds match the grid
library, but sampling is continuous. Coverage is balanced in normalized
parameter coordinates, not in HRF waveform distance. Keep the exact table
and curves with results; the seed is useful for regeneration, while the
saved library records precisely which candidates were used.

From `boldtailor.hrf_selection`:

```text
select_hrfs(data, *, library=None, run_labels=None, feature_signature=None,
            candidate_batch_size=32, task_model=TaskModel())
evaluate_hrf_split(data, *, library=None, train_runs, test_runs,
                   run_labels=None, feature_signature=None,
                   task_model=TaskModel(), candidate_batch_size=32)
```

`select_hrf` is a deprecated alias of `select_hrfs` for one release.
`candidate_batch_size` bounds memory in both functions and never changes the
result.

`select_hrfs` returns `HrfSelectionResult`: `hrf_indices`, `cv_r2`,
`canonical_cv_r2`, `delta_cv_r2`, `library`, `eligibility`, `run_labels`,
`feature_signature`, `provenance`, `parameter_bound_flags`, and
`at_parameter_bound`. `parameter_bound_flags` is a read-only boolean array of
shape `(n_features, 6, 2)` over `PARAMETER_NAMES[:6]` and `[low, high]`: True
where the selected custom kernel's parameter lies within 2 % of the library
box width of that edge; all False for IDs 0 and -1 and for GLMsingle kernels.
`at_parameter_bound` is the
per-feature `any()` over `library.informative_parameters` only, so a constant or
two-level grid parameter (for `expanded_hrf_library()`, `undershoot_delay`) does
not flag every pick. `parameter_bound_table()` returns a DataFrame with
columns `parameter`, `edge`, and `fraction_flagged` (over features with ID > 0;
NaN when there are none). The default library (`TIMING_BOUNDS`) spans peak
times of 2.5–8.5 s; the Sobol and expanded grid box implies roughly 1.5–7.5 s,
so late-peaking responses saturate there. Eligibility records are checked lazily;
an unchecked candidate is not an excluded candidate.

`evaluate_hrf_split` uses disjoint, zero-based run indices. Its
`HrfEvaluationResult` contains `training_selection`, `training_amplitudes`,
`test_r2`, `canonical_test_r2`, `delta_test_r2`, `train_runs`, `test_runs`, and
`provenance`. It fixes the HRF and one amplitude per task regressor before predicting the test runs.
Selection requires two runs; independent evaluation requires two training runs
and at least one test run. Undefined HRF indices are `-1`.

Both functions score the task model's regressors;
`HrfSelectionResult.task_model` records it.
`HrfEvaluationResult.training_amplitudes` has one row per task regressor, in
`amplitude_names` order. `HrfEvaluationResult.training_amplitudes` is now
two-dimensional, `(n_regressors, n_features)`; under the default task model
index row 0 where earlier releases returned a one-dimensional array. Selection provenance records `task_model`,
`task_model_fingerprint`, `task_regressors`, `profiled_regressors`,
`min_onset`, and `oversampling`.

From `boldtailor.model`: `Modulator(column, missing="error", kind="numeric",
levels=None, reference=None)` and `TaskModel(modulators=())`, with `regressor_names`, `profiled_names`,
`fingerprint`, and `to_dict()`.

A `Modulator` with `kind="categorical"` is reference-coded: `levels` (at least
two distinct values, canonicalised with `level_name` and sorted numerically
when all are numeric, else lexically) and `reference` (default: the first
level). It contributes one 0/1 regressor `<column>[<level>]` per non-reference
level; `task` is the reference-level response. `kind="numeric"` takes neither
`levels` nor `reference`. `level_name(value)` returns the canonical text of one
value (`1`, `"1"`, `1.0` give `"1"`) or `None` for a missing value (`n/a`, `""`,
NaN) and raises `ValueError` for infinities and booleans.

`missing="error"` (the default) rejects any trial whose modulator value is
missing (nonfinite for `kind="numeric"`, `level_name(...) is None` for
categorical). `missing="indicator"` instead gives those trials a zero
amplitude in the modulator's regressors and, in each run that has missing
values, adds a 0/1 regressor named by `indicator_name` (`missing_<column>`);
`TaskModel.profiled_names` lists these indicators. A numeric modulator with no
observed value in a run is an error under either policy. `column` must be a
nonempty string other than `task`, `constant`, `onset`, or `duration`, and must
not start with `missing_`. A categorical modulator created without `levels` is
unresolved (`resolved` is `False`, while it is always `True` for numeric);
`with_levels(values)` returns a copy with the levels set, and `TaskModel`
rejects unresolved modulators.

## Task-guided denoising

From `boldtailor.denoising` (opt-in; see the
[user guide](user-guide.md#task-guided-denoising)):

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

`select_denoising` needs at least three runs. It is anatomy-agnostic: there
is no mask parameter and every input feature is a candidate. `counts` must
contain 0, be nonnegative integers (not Booleans), and is sorted and
deduplicated; the default is 0 through 10. `pcstop` must be a finite number
of at least 1. `pool_r2_threshold` is `"auto"` or a finite number. `"auto"`
applies GLMsingle's `findtailthreshold` rule once, to the finite ON-OFF R²
values of all features. It fits a two-component
`sklearn.mixture.GaussianMixture` (`tol=1e-10`, `reg_covar=0`, 3
initialisations) to at most 1,000,000 values. It evaluates posteriors on 500
points over GLMsingle's `robustrange`, widened to include both means. The
threshold is the rightmost point where the posterior of the component
dominating the right end is at most 0.5. Deviations from GLMsingle:
`random_state=0` and a seeded subsample make the result reproducible;
convergence warnings are suppressed but recorded (`converged`, `n_iter`);
`robustrange` uses all finite values (GLMsingle uses the subsample); and no
crossing raises (GLMsingle warns). Fewer than two distinct values, a failed
fit, or no crossing raise an error that names the runs and suggests a fixed
`pool_r2_threshold`; there is no fallback. A number is applied as a fixed
threshold. ON-OFF R² is a fraction, so a GLMsingle `brainR2` value (percent)
must be divided by 100. GLMsingle's `bright` pool criterion is not applied,
and the best-100 fallback considers only features with a defined HRF.
`library=None` uses
`default_hrf_library()`. Every input, and every run's design (errors name
runs by label), is validated before any fitting.

The procedure follows GLMsingle's GLMdenoise stage (see the
[user guide](user-guide.md#task-guided-denoising) for the steps and the
deviations: baseline confounds instead of polynomials, time-series scoring
because repeated conditions are not assumed, and a fixed scoring target).
HRFs come from one `select_hrfs` call on all runs and are frozen. The pool
statistic is GLMsingle's ON-OFF R² (one amplitude-1 task regressor with the
library's canonical HRF, a coefficient shared across runs, each run's
confounds plus intercept as nuisance). The pool is the features below the
threshold; the scoring features are those above it with a defined HRF, or
the best 100 when none passes. Run-wise PCs come from that single pool.
Counts are scored by leave-one-run-out held-out task prediction against a
target fixed across counts; each feature's SSE and SST are pooled across
folds and the median over scored features is the count's performance. The
count is chosen by GLMsingle's `pcstop` rule. The selection activity is named
`denoising_selection`.

`significance_gate` (a bool, default `True`) applies an F-test gate that is a
Boldtailor addition, not part of GLMsingle. If pcstop chose `k* > 0`, each
feature pcstop scored gets an in-sample F-test on all runs comparing the
reduced model (shared task regressors with frozen HRFs, plus each run's
baseline confounds, intercept, and missing-value indicators) with the full
model (plus each run's first `k*` PCs). `df1 = rank(full) - rank(reduced)`
and `df2 = scans - rank(full)` use the existing rank tolerances, and
`p = scipy.stats.f.sf(F, df1, df2)`. With `m` of `n` tested features at
`p < gate_alpha`, `k*` is kept when
`scipy.stats.binomtest(m, n, gate_alpha, alternative="greater").pvalue <
gate_binomial_alpha`; otherwise the count is 0. With no testable feature the
count is 0. `gate_alpha` and `gate_binomial_alpha` must be finite numbers
strictly between 0 and 1. Features whose stacked design is rank deficient,
has `df2 <= 0`, or gains no PC columns (`df1 = 0`), and scoring features with
a numerically zero target in any run (not scored by pcstop), are excluded
from `n` (with none left, `binomial_p` is NaN). `significance_gate=False`
returns the pcstop count. `gate_noise_model` is `"ar1"` (default) or
`"ols"`. With `"ar1"` (a Boldtailor addition following Nilearn's first-level
`noise_model="ar1"`), both fits are prewhitened: per run and feature the
lag-1 coefficient is Nilearn's Yule-Walker estimate from the full model's OLS
residuals in that run, truncated to 1/100 bins as in Nilearn's `run_glm`,
and that run's BOLD and all design columns are whitened with
`nilearn.glm.ARModel` before the stacked refits (same `df1`, `df2`). With
`"ols"` the F-tests are unwhitened and anti-conservative under
autocorrelated noise. AR(1) may under-whiten higher-order autocorrelation,
and the binomial test treats features as independent. The F-test measures
variance explained by the PCs, not task-prediction benefit (see the user
guide). The selection activity records the gate under `significance_gate`.

`with_denoising` checks content identity before it changes anything: run count
and order, features, rows, time grids, signals, baseline confounds, and events.
`feature_signature` must equal `result.feature_signature`. Re-application and
column-name collisions raise. It returns new `AnalysisData` in which each run's
confounds are the original columns followed by `denoise_pc_000`, ...
(`component_names(count)`, prefix `COMPONENT_PREFIX`). Signals, events, and
timing are unchanged. Each run's confounds `SourceRef` (signal when there are
no confounds) gains a `denoising_augmentation` annotation, which holds
`selection_execution_id`, `component_fingerprints`, and `n_components`. The
other source fields are unchanged, so downstream analysis ids distinguish
augmentations. Provenance gains a `denoising_augmentation` activity.

From `boldtailor.denoising_results`:

- `DenoisingResult` holds:
  - the choice: `n_components` (after the gate), `pcstop_count` (pcstop's
    choice), `counts`, `pool_r2_threshold` (the setting), `pcstop`
  - `significance_gate` (`SignificanceGate`)
  - the threshold: `noise_pool_threshold` and `noise_pool_mixture`
    (`MixtureThreshold` with `threshold`, `means`, `sds`, `weights`, `tail`,
    `n_values`, `n_fitted`, `converged`, `n_iter`, `to_dict()`; `None` for a
    fixed threshold)
  - the full-data statistic and masks: `onoff_r2`, `noise_pool`,
    `scoring_mask`, `scoring_fallback`, and `scored`
  - `initial_selection` (`HrfSelectionResult`, the frozen HRFs)
  - `run_components`, one `(scans, n_components)` array per run
  - `components` (`PcaDiagnostics`)
  - `folds` (`DenoisingFold` per held-out run)
  - `run_labels`, `feature_signature`, `source_identity`, and `provenance`
  - the properties `candidate_scores` (`count`, `eligible`, `perf`, `curve`,
    `reason`), `perf` and `curve` arrays, `fold_scores` (`validation_run`,
    `count`, `eligible`, `reason`, `median_r2`, `n_scored`, `n_zero_target`),
    `component_names`, `selection_cv_r2`, and `initial_hrf_indices`

  Arrays are read-only and tables are copies. Scores are selection
  statistics, not independent performance estimates.
- `SignificanceGate` holds `enabled`, `alpha`, `binomial_alpha`,
  `noise_model` (`"ar1"` or `"ols"`), `pcstop_count`, `n_components`, `decision` (`"kept"`, `"rejected"`,
  `"skipped_zero_count"`, `"disabled"`), per-feature `f_statistic`,
  `p_value`, `df1`, `df2` (NaN where not tested), `ar_coefficients` (runs x
  features, the binned lag-1 coefficients used; NaN where not tested or
  with `"ols"`), the `tested` and
  `excluded` masks, `exclusions` (`(hrf_index, n_features, reason)`), `m`,
  `n`, `binomial_p` (NaN when the binomial test was not run), and the
  property `n_excluded`.
- `DenoisingFold` holds `validation_run`, `training_runs`, `zero_target`, and
  `target_energy` (NaN for unscored features).
- `PcaDiagnostics` holds `run_labels`, `pool_size`, `ranks`,
  `singular_values`, `rank_tolerances`, and `retained_columns`. `table()`
  returns one row per run.

## Source records and saving

From `boldtailor.provenance`:

- `SourceRef(role, uri=None, media_type=None, byte_size=None, modified_at=None,
  sha256=None, annotations=...)` describes a source. A complete identity needs
  its relative URI, byte size, and UTC modification time ending in `Z` or
  `+00:00` (for example, `2026-09-27T12:00:00Z`); media type is optional, and
  `sha256`, when given, is 64 lowercase hexadecimal characters.
- `RunSources(signal, events, confounds=None)` groups source references for one
  run. Include a confound reference when supplying a confound table.
- `ProvenanceRecord` holds sources, activities, lifecycle events, warnings,
  execution ID, and available source/analysis fingerprints.
- All three types support `to_dict()` and `from_dict()`;
  `ProvenanceRecord.canonical_json()` serializes a record deterministically.
- `ProvenanceRecord.metadata_fingerprint` is the SHA-256 of the run sources
  when every source has a complete identity, else `None`;
  `ProvenanceRecord.analysis_fingerprint` is the stored analysis fingerprint,
  or `None`. `SCHEMA_ID` (`"boldtailor.provenance/1"`) is the default `schema`.
- `analysis_fingerprint(metadata_fingerprint, model)` returns the SHA-256 of
  the source fingerprint together with the model settings, or `None` when
  `metadata_fingerprint` is `None`.
- `identity_activity(record)` returns the record's last activity without its
  environment-only `software` key, for embedding in other identities.
- `extend_provenance(record, *, execution_id, activity, events, warnings,
  analysis_id)` returns a copy with the new execution ID, `activity` appended,
  `events` replacing the event history, `warnings` appended, and
  `analysis_fingerprint` set to `analysis_id`.
- `validate_relative_path(text, *, name="path")` returns `text` if it is a
  non-empty dataset-relative POSIX path of `[A-Za-z0-9+_.-]` components (no
  leading `/`, empty, `.`, or `..` components); otherwise it raises
  `ValueError` mentioning `name`.

`boldtailor.bids_provenance.project_bids_provenance(record, ...)` returns a
mapping of relative filenames to bytes. Options include `dataset_name`,
`label`, `code_url`, `container`, `source_datasets`, `dataset_links`,
`derivative_sidecars`, and `export_bids_prov`. It produces dataset metadata and
the analysis record, with optional draft BIDS provenance files.
`STABLE_BIDS_VERSION` (`"1.11.1"`) is the `BIDSVersion` written to
`dataset_description.json`. The draft files pin `BEP028_DRAFT_IDENTIFIER`
(`"BEP028"`) at specification commit `BEP028_DRAFT_SNAPSHOT` and implement only
`BEP028_SUPPORTED_SUBSET` (Activities, Files, Environments, Software, the
provenance label table, and file GeneratedBy and Sources relationships).

`boldtailor.publication.Artifact(path, payload)` describes a file to
save. `publish_artifact_set(destination, artifacts, *, source_paths=(),
overwrite=False, lock_timeout=30.0, keep_existing=())` returns the saved
paths. Paths listed in `keep_existing` (shared files such as a
`dataset_description.json`) are written only if absent once the lock is held;
an existing file there is kept and left out of the returned paths. Files are replaced individually under a writer lock; publication is
not an atomic snapshot for concurrent readers. Existing files are protected by
default; `source_paths` protects inputs from accidental replacement. Promotion
failures raise `PublicationError`; invalid arguments and preflight collisions
can raise `ValueError`, `TypeError`, or `FileExistsError` before publication starts.
If rollback fails, `PublicationError.recovery_directory` identifies the retained
transaction directory and `rollback_errors` contains the recovery exceptions.
The original operation error remains the exception's cause. See the
[publication migration](publication-migration.md) for recovery and the removed
`retain_incomplete` argument.
`is_control_directory(path)` is True for a path ending in `.boldtailor`, the
`<destination>.boldtailor` sibling that holds the writer lock and staging files.

For example, after the README's fit, save a contrast array with its records:

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

Logging uses the standard Python logger named `boldtailor`. Configure handlers
in your application. Fit and comparison attempts emit start and completion or
failure events, with completion delayed until the result exists. Start records
omit analysis identity; completion includes it when available. New failure
records contain `error_code` instead of exception text; callers still receive
the original exception. See the [lifecycle migration](lifecycle-migration.md)
and [developer guide](development.md) for event names and identity rules.

`boldtailor.logging` provides the helpers behind those records:

| Function | Use |
| --- | --- |
| `bind_context(*, execution_id=None, data_id=None, analysis_id=None, run_index=None, inherit=True)` | Context manager adding the given identity fields to every event emitted inside it; `inherit=False` starts from an empty context |
| `emit_event(event, *, stage, level=logging.INFO, error=None, execution_id=None, data_id=None, analysis_id=None, run_index=None)` | Log one compact sorted-key JSON record (`timestamp`, `sequence`, `level`, `event`, `stage`, bound and explicit identity fields, and `error_code` when `error` is given) and return it as a read-only mapping |
| `append_event_history(history, event)` | Return `history` plus the event's standard fields (those listed for `emit_event`), keeping the last 8 entries |

## Imaging, diagnostics, and parallel helpers

These modules operate on fitted arrays and CIFTI axes; none fits a model.

| Function | Use |
| --- | --- |
| `boldtailor.cifti.scalar_artifact(path, brain, values, names)` | A float32 `.dscalar.nii` `Artifact` with one named map per row on a `BrainModelAxis` |
| `boldtailor.cifti.read_scalar(path, brain, names=None)` | Load saved maps after checking the grayordinate axis and, optionally, map names |
| `boldtailor.cifti.cortical_values(values, brain)` | Scatter a grayordinate vector onto `left`/`right` cortical vertices; absent vertices are NaN |
| `boldtailor.cifti.spatial_signature(brain, indices)` | SHA-256 feature identity of the ordered axis and selected grayordinates, for `feature_signature` |
| `boldtailor.diagnostics.one_sample_t(run_betas)` | Pooled trial-beta mean, t, uncorrected two-sided p, count, and df (`ONE_SAMPLE_T_NAMES`), assuming independent trials |
| `boldtailor.diagnostics.pearson_correlation(cross, beta_ss, rt_ss, counts)` | Pearson r from centered cross-products and sums of squares, clipped to [-1, 1]; NaN below 3 trials or with zero variance |
| `boldtailor.diagnostics.correlate_rt(beta_runs, rt_runs, *, run_numbers)` | Within-run-centered beta/RT Pearson r and counts per run and for all, odd, and even runs |
| `boldtailor.diagnostics.even_run_points(beta_runs, rt_runs, run_numbers, vertex)` | The matched, centered (RT, beta) points from even runs at one feature |
| `boldtailor.reliability.curve_correlations(library, ids_a, ids_b)` | Full-curve Pearson r for a/b, a/canonical, and b/canonical selections (`CORRELATION_NAMES`) |
| `boldtailor.reliability.compare_hrfs(library, hrf_indices, sessions)` | Pairwise session curve agreement, matched canonical baselines, parameter SDs, and mean peak times |
| `boldtailor.parallel.map_blocks(function, blocks, *, args=(), n_jobs=1)` | Ordered `(block, result)` pairs; several workers run in bounded loky batches with one inner thread |

For `map_blocks` workers: result objects that hold rebuild closures (`SingleTrialResult` with a `SelectedTrialDesign`, `HrfAnalysisResult`) cannot be serialized with the standard `pickle` module; loky's cloudpickle can move them between processes, but workers should prefer returning arrays or dicts to keep transfers small.
`library_indices`, `finite_mean`, and `validate_n_jobs` are the
corresponding validation helpers.
The `summary` rows returned by `compare_hrfs` follow
`boldtailor.reliability.SUMMARY_NAMES`: `mean_between_session_r`,
`mean_matched_canonical_r`, `mean_between_minus_canonical_r`, `valid_sessions`,
and `valid_pairs`.

Lower-level helpers used by the examples are also importable from public
modules: `expand_events`, `task_columns`, `hrf_model`, and
`event_response_scales` (`boldtailor.design`),
`prepare_runs` and `subset_runs` (`boldtailor.hrf_selection`),
`fraction_grid`, `regularization`, and `NORM_BASIS`
(`boldtailor.fractional_ridge`), `r_squared`, `validate_alpha`, and
`compile_trial_run` (`boldtailor.single_trial`), and
`HRF_NORMALIZATION` (`boldtailor.model`).

## Session workflow and command line

`boldtailor.workflow` is the installed implementation behind `boldtailor run`:
it discovers one subject/session/task, runs the enabled stages, and publishes a
BIDS derivative with an HTML report. `boldtailor.cli` is the command-line
front end.

| Name | Module | Use |
| --- | --- | --- |
| `WorkflowSettings` | `boldtailor.workflow.settings` | Frozen, validated settings (paths, HRF library, ridge mode, stages, `existing_results`); `to_dict()` is the JSON form |
| `run_workflow(settings)` | `boldtailor.workflow.run` | Run all enabled stages and publish; returns a `WorkflowResult` (`settings`, `paths`, `report_path`, `skipped`, `task_model`, `library_fingerprint`) |
| `describe_inputs(settings)` | `boldtailor.workflow.run` | Resolved plan (runs, task model, notes) and the `problems` a run would hit (`existing_results` or `input`), without fitting; backs `--dry-run` |
| `load_session(settings, *, hrf_only=False)` | `boldtailor.workflow.inputs` | Load, trim, and validate the session's CIFTI runs and events; `hrf_only=True` skips the two-odd/two-even-run requirement of the reliability stages |
| `detect_task_model(events_tables, modulators=None, labels=None)` | `boldtailor.workflow.inputs` | `response_time`, a categorical `trial_type` (two or more levels), or explicit modulators (`()` for task-only) to a `TaskModel`; `labels` name runs in errors |
| `task_model_notes(events_tables, modulators=None)` | `boldtailor.workflow.inputs` | Why detection left out a column every run has (a `trial_type` with fewer than two levels) |
| `InputError` | `boldtailor.workflow.inputs` | `ValueError` subclass for missing or malformed inputs; the CLI maps it to exit code 2 |
| `WorkflowRun` | `boldtailor.workflow.inputs` | One loaded run: `inputs`, `image`, `events`, `confounds`, `frame_times`, `label`, `number`, and `retained_frames` |
| `observed_levels(tables, column, labels=None)` | `boldtailor.workflow.inputs` | Canonical non-missing level names of `column` across every events table |
| `selection_task_model(task_model, include_rt=True)` | `boldtailor.workflow.inputs` | The GLM task model, or the same model without `response_time` for HRF selection |
| `validate_glm_events(events, task_model)` | `boldtailor.workflow.inputs` | Require an observed positive RT and every categorical level, if modeled |
| `glm_model(runs, task_model)` | `boldtailor.workflow.inputs` | The OLS `ModelSpec` (SPM HRF, no drift, the runs' confounds) with one contrast per task regressor |
| `make_blocks(runs, *, block_size=4096, max_grayordinates=None)` | `boldtailor.workflow.inputs` | Feature-index blocks that skip constant signals; exports restore those positions as NaN |
| `block_signals(runs, indices)` / `load_block(runs, root, indices, task_model)` | `boldtailor.workflow.inputs` | Trimmed signals for one feature block / the block as `AnalysisData` with raw trial rows and sources |
| `run_summary(runs, task_model)` | `boldtailor.workflow.inputs` | One summary row per run (the `boldtailor_runs.tsv` table) |
| `discover_runs(settings)` | `boldtailor.workflow.files` | The session's `RunInputs` (events, CIFTI in the settings' space, confounds and BOLD sidecars); every file must exist and CIFTI and events runs must match |
| `load_runs(inputs)` / `load_inputs(inputs)` | `boldtailor.workflow.files` | `RawRun`s in BIDS run order with one shared `BrainModelAxis` (returned with it) / one run's image, raw events, selected confounds, and sidecar-corrected frame times |
| `select_confounds(table, metadata)` | `boldtailor.workflow.files` | The fMRIPrep nuisance set: 24 motion terms (`MOTION`), the top six retained combined-mask aCompCor components, cosines, and non-steady-state spikes |
| `run_sources(run, root, indices)` / `source_ref(path, root, role, **annotations)` | `boldtailor.workflow.files` | `RunSources` for one run's BOLD (with grayordinate indices), events, and confounds / one file's `SourceRef` relative to `root` |
| `odd_even_parity(runs)`, `reaction_times(runs, *, missing_ok=False)`, `bids_label(value, entity)`, `input_paths(run)` | `boldtailor.workflow.files` | Odd/even BIDS run positions; per-run `response_time` arrays (with `missing_ok`, absent or non-numeric entries are NaN); BIDS label check; a run's five input paths |
| `select_hrfs(runs, root, blocks, library, *, task_model, n_jobs=1, splits=True)` | `boldtailor.workflow.analysis` | Blockwise HRF selection on all runs and, with `splits`, odd/even split evaluations; results keyed by block indices |
| `selection_maps(selections, n_features)` | `boldtailor.workflow.analysis` | Grayordinate maps of selected IDs and CV R² for `all`, `odd`, `even`, plus `odd_to_even`/`even_to_odd` test R² |
| `fit_glms(runs, root, blocks, model, *, selections=None, n_jobs=1)` | `boldtailor.workflow.analysis` | Blockwise `fit` and `task_delta_r2` with SPM or the selected HRFs: effect, variance, t, and z maps per task regressor, R² rows, designs, and provenance |
| `fit_beta_series(runs, root, blocks, *, task_model, selections=None, ridge_alpha=0.0, ridge_fraction=None, n_jobs=1)` | `boldtailor.workflow.analysis` | Blockwise single-trial fits at a fixed penalty or fractions, with R² rows, designs, provenance, and RT correlations when RT is modeled |
| `json_artifact`, `table_artifact`, `npz_artifact`, `figure_artifact` | `boldtailor.workflow.artifacts` | In-memory `Artifact`s: strict indented JSON, TSV with `n/a`, compressed NPZ, 130-dpi PNG |
| `dataset_description(name)` | `boldtailor.workflow.artifacts` | The derivative's BIDS `dataset_description.json` artifact |
| `parameter_artifact(brain, library, ids, path)` / `scalar_map(stem, space_entity, brain, descriptor, statistic, values, names)` | `boldtailor.workflow.artifacts` | Dense scalar map of the selected HRFs' parameters and peak time (undefined features NaN) / a `<space>_desc-<descriptor>_stat-<statistic>` dense scalar map |
| `design_figure`, `library_figure`, `glm_comparison`, `parameter_agreement`, `curve_agreement`, `activation_histogram`, `rt_check_figure`, `fraction_selection_figure` | `boldtailor.workflow.plots` | Report figures (`glm_comparison`, `parameter_agreement`, and `curve_agreement` return `(summary DataFrame, figure)`) from fitted arrays and tables; nothing is refit, shown, or saved |
| `find_surface_meshes(fmriprep_dir, subject, *, paths=None)` | `boldtailor.workflow.surfaces` | The subject's fsLR 32k midthickness meshes, or explicit `left`/`right` paths; `None` if a hemisphere is missing |
| `surface_figure(maps, brain, meshes, *, statistic, title=None)` | `boldtailor.workflow.surfaces` | Maps in four cortical views (`VIEWS`) with a shared, unthresholded color scale |
| `fit_beta_models(runs, root, blocks, settings, library, selections, task_model)` | `boldtailor.workflow.beta_series` | Canonical and selected-HRF trial models (OLS plus the configured ridge mode) over feature blocks |
| `BetaModel` | `boldtailor.workflow.beta_series` | One beta model: name, HRF, estimator, fit, and (tuned ridge) tuning and outer-split evaluation |
| `save_workflow(...)` | `boldtailor.workflow.outputs` | Publish maps, designs, tables, provenance, figures, settings file, and report together |
| `render_report(settings, *, runs, task_model, ...)` | `boldtailor.workflow.report` | The self-contained HTML report with embedded PNG figures (bytes) and the equivalent command line |
| `selected_hrfs`, `encoding_scores`, `ridge_boundary`, `input_tables` | `boldtailor.workflow.summaries` | Report tables built from in-memory results |
| `build_parser`, `settings_from_args`, `main` | `boldtailor.cli` | Argument parser, args to `WorkflowSettings`, and the `boldtailor` entry point (returns the exit code) |

See the [user guide](user-guide.md#running-the-full-workflow) for flags, outputs,
and exit codes.

## Example workflows

These functions live under `examples/`, rather than the installed core package.
Run them from the repository checkout:

| Module / function | Use |
| --- | --- |
| `examples.NSD.nsd_settings` | Workflow-notebook settings from defaults, overrides, and data paths |
| `examples.NSD.session_hrf.estimate_sessions` | Select and cache HRFs separately for several sessions |
| `examples.NSD.multisession_workflow.ensure_session_outputs` | Reuse complete session results or fit missing ones with the workflow notebook |

See [running the NSD notebooks](../examples/NSD/README.md#run-the-notebooks).
The retired NSD scripts are replaced by [`boldtailor run`](#session-workflow-and-command-line).
