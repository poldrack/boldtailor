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
Matrices include trial columns followed by nuisance columns. See the
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
| `sobol_hrf_library(n_samples=512, *, seed=0)` | Sample continuous parameters with scrambled Sobol; add canonical SPM as ID 0 |
| `expanded_hrf_library()` | Return the original 649-candidate grid library |
| `timing_hrf_library(n_samples=512, *, seed=0, bounds=None, onset=0.0, duration=36.0)` | Sample lobe timing (response peak/SD, undershoot peak/SD, depth) with scrambled Sobol, convert to SPM gamma parameters, add canonical SPM as ID 0; `bounds` overrides `TIMING_BOUNDS` |
| `timing_parameters(parameters)` / `spm_parameters(timing)` | Exact conversion between the seven SPM gamma parameters and `TIMING_NAMES` (lobe peak time and SD, undershoot peak time and SD, undershoot depth, onset, duration) |
| `HrfLibrary.timing_table` | One row per candidate with both parameterizations |
| `HrfLibrary.from_parameters(parameters)` | Build a library from seven-value parameter rows, adding canonical SPM as ID 0 |
| `HrfCandidate(id, kind, parameters)` | Describe one kernel; `kind` is `"spm"` or `"double_gamma"` |
| `candidate.kernel(tr, oversampling=50)` | Sample a read-only kernel scaled to a peak of one at TR/oversampling; canonical SPM is Nilearn's `spm_hrf` divided by its maximum |

Parameter order is `response_delay`, `undershoot_delay`, `response_dispersion`,
`undershoot_dispersion`, `response_undershoot_ratio`, `onset_delay`, `duration`.
SPM candidates require the exact canonical parameter tuple
`(6, 16, 1, 1, 6, 0, 32)`.

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
select_hrfs(data, *, library, run_labels=None, feature_signature=None,
            candidate_batch_size=32, task_model=TaskModel())
evaluate_hrf_split(data, *, library, train_runs, test_runs,
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
box width of that edge; all False for IDs 0 and -1. `at_parameter_bound` is the
per-feature `any()` over `library.informative_parameters` only, so a constant or
two-level grid parameter (for `expanded_hrf_library()`, `undershoot_delay`) does
not flag every pick. `parameter_bound_table()` returns a DataFrame with
columns `parameter`, `edge`, and `fraction_flagged` (over features with ID > 0;
NaN when there are none). The default box implies peak times of roughly
1.5–7.5 s, so late-peaking responses saturate there. Eligibility records are checked lazily;
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

From `boldtailor.model`: `Modulator(column, center=True, missing="error")` and
`TaskModel(modulators=())`, with `regressor_names`, `profiled_names`,
`fingerprint`, and `to_dict()`.

## Source records and saving

From `boldtailor.provenance`:

- `SourceRef(role, uri=None, media_type=None, byte_size=None, modified_at=None,
  annotations=...)` describes a source. A complete identity needs its relative
  URI, byte size, and UTC modification time with a trailing `Z` (for example,
  `2026-09-27T12:00:00Z`); media type is optional.
- `RunSources(signal, events, confounds=None)` groups source references for one
  run. Include a confound reference when supplying a confound table.
- `ProvenanceRecord` holds sources, activities, lifecycle events, warnings,
  execution ID, and available source/analysis fingerprints.
- All three types support `to_dict()` and `from_dict()`;
  `ProvenanceRecord.canonical_json()` serializes a record deterministically.

`boldtailor.bids_provenance.project_bids_provenance(record, ...)` returns a
mapping of relative filenames to bytes. Options include `dataset_name`,
`label`, `code_url`, `container`, `source_datasets`, `dataset_links`,
`derivative_sidecars`, and `export_bids_prov`. It produces dataset metadata and
the analysis record, with optional draft BIDS provenance files.

`boldtailor.publication.Artifact(path, payload)` describes a file to
save. `publish_artifact_set(destination, artifacts, *, source_paths=(),
overwrite=False, lock_timeout=30.0)` returns the saved
paths. Files are replaced individually under a writer lock; publication is
not an atomic snapshot for concurrent readers. Existing files are protected by
default; `source_paths` protects inputs from accidental replacement. Promotion
failures raise `PublicationError`; invalid arguments and preflight collisions
can raise `ValueError`, `TypeError`, or `FileExistsError` before publication starts.
If rollback fails, `PublicationError.recovery_directory` identifies the retained
transaction directory and `rollback_errors` contains the recovery exceptions.
The original operation error remains the exception's cause. See the
[publication migration](publication-migration.md) for recovery and the removed
`retain_incomplete` argument.

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

## Imaging, diagnostics, and parallel helpers

These modules operate on fitted arrays and CIFTI axes; none fits a model.

| Function | Use |
| --- | --- |
| `boldtailor.cifti.scalar_artifact(path, brain, values, names)` | A float32 `.dscalar.nii` `Artifact` with one named map per row on a `BrainModelAxis` |
| `boldtailor.cifti.read_scalar(path, brain, names=None)` | Load saved maps after checking the grayordinate axis and, optionally, map names |
| `boldtailor.cifti.cortical_values(values, brain)` | Scatter a grayordinate vector onto `left`/`right` cortical vertices; absent vertices are NaN |
| `boldtailor.cifti.spatial_signature(brain, indices)` | SHA-256 feature identity of the ordered axis and selected grayordinates, for `feature_signature` |
| `boldtailor.diagnostics.one_sample_t(run_betas)` | Pooled trial-beta mean, t, uncorrected two-sided p, count, and df (`ONE_SAMPLE_T_NAMES`), assuming independent trials |
| `boldtailor.diagnostics.correlate_rt(beta_runs, rt_runs, *, run_numbers)` | Within-run-centered beta/RT Pearson r and counts per run and for all, odd, and even runs |
| `boldtailor.diagnostics.even_run_points(beta_runs, rt_runs, run_numbers, vertex)` | The matched, centered (RT, beta) points from even runs at one feature |
| `boldtailor.reliability.curve_correlations(library, ids_a, ids_b)` | Full-curve Pearson r for a/b, a/canonical, and b/canonical selections (`CORRELATION_NAMES`) |
| `boldtailor.reliability.compare_hrfs(library, hrf_indices, sessions)` | Pairwise session curve agreement, matched canonical baselines, parameter SDs, and mean peak times |
| `boldtailor.parallel.map_blocks(function, blocks, *, args=(), n_jobs=1)` | Ordered `(block, result)` pairs; several workers run in bounded loky batches with one inner thread |

For `map_blocks` workers: result objects that hold rebuild closures (`SingleTrialResult` with a `SelectedTrialDesign`, `HrfAnalysisResult`) cannot be serialized with the standard `pickle` module; loky's cloudpickle can move them between processes, but workers should prefer returning arrays or dicts to keep transfers small.
`library_indices`, `finite_mean`, and `validate_n_jobs` are the
corresponding validation helpers.

Lower-level helpers used by the examples are also importable from public
modules: `expand_events`, `task_columns`, `hrf_model`, and
`event_response_scales` (`boldtailor.design`),
`prepare_runs` and `subset_runs` (`boldtailor.hrf_selection`),
`fraction_grid`, `regularization`, and `NORM_BASIS`
(`boldtailor.fractional_ridge`), `r_squared`, `validate_alpha`, and
`compile_trial_run` (`boldtailor.single_trial`), and
`HRF_NORMALIZATION` (`boldtailor.model`).

## Example workflows

These functions live under `examples/`, rather than the installed core package.
Run them from the repository checkout:

| Module / function | Use |
| --- | --- |
| `examples.NSD.settings.resolve_settings` | Workflow-notebook settings from defaults, overrides, and data paths |
| `examples.NSD.workflow_inputs.load_session`, `make_blocks`, `load_block` | Load, trim, and block an NSD session's CIFTI runs |
| `examples.NSD.workflow_analysis.select_hrfs`, `fit_glms`, `fit_beta_series` | All/odd/even HRF selection, matched GLMs, and beta series over feature blocks |
| `examples.NSD.ridge_workflow.fit_cv_beta_series` | Tune per-grayordinate fractions (`fractions=...`) or one global alpha (`alphas=...`), evaluate both odd/even outer splits, and fit final all-run betas |
| `examples.NSD.workflow_outputs.save_workflow` | Publish the notebook's maps, designs, tables, and provenance together |
| `examples.NSD.session_hrf.estimate_sessions` | Select and cache HRFs separately for several sessions |
| `examples.NSD.multisession_workflow.ensure_session_outputs` | Reuse complete session results or fit missing ones with the workflow notebook |
| `examples.NSD.nsd_cifti.run_analysis` | Script: conventional stimulus-plus-RT CIFTI analysis |
| `examples.NSD.nsd_single_trial.run_single_trial_analysis` | Script: canonical or selected-HRF CIFTI beta series and RT diagnostics |
| `examples.NSD.rt_diagnostics.select_vertices`, `scatter_artifact` | Script: select cortical vertices by odd-run RT correlation; even-run RT scatterplots |
| `examples.stop_signal_demo.discover_run_inputs`, `common_brain_mask`, `make_masker`, `load_run` | Load aligned NIfTI runs using an intersected mask |
| `examples.stop_signal_demo.whole_brain_image`, `result_artifacts` | Reconstruct and prepare NIfTI outputs |

The scripts accept BIDS, fMRIPrep, and output roots (the BIDS root from
`--bids-root` or `NSD_BIDS_ROOT`) plus `subject`, `session`, and `block_size`;
the single-trial runner also accepts `ridge_alpha`, `hrf_library`
(`"canonical"` or `"expanded"`), and `n_jobs`. See
[running the NSD notebooks](../examples/NSD/README.md#run-the-notebooks) and
[command-line scripts](../examples/NSD/README.md#command-line-scripts).
