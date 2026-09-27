# API reference

[User guide](user-guide.md) · [Developer guide](development.md)

Import from the modules shown below, for example
`from boldtailor.single_trial import fit_single_trials`. The top-level package
does not re-export these names. Modules beginning with `_` are implementation
details. Returned numerical arrays are read-only; use `.copy()` to edit them.

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

`boldtailor.hrf_glm_results.HrfAnalysisResult` has the same contrast methods,
`contrast_names`, `run_r2`, `r2`, and `provenance`. Instead of `design_matrices`
and `design_provenance`, it exposes `group_designs` and
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
task_delta_r2_prepared(prepared, full_result, *, contrasts,
                      noise_model="ar1", model_metadata=None)
```

These return the same result types as their event-based counterparts. Neither
function adds regressors or transforms the design. The R² comparison requires
a complete task/nuisance/intercept partition, complete source metadata, and the
same fit settings as `full_result`.

## Single-trial fits

From `boldtailor.single_trial`:

```text
fit_single_trials(data, *, ridge_alpha=0.0, run_labels=None, hrf="spm")
fit_selected_hrfs(data, *, selection, ridge_alpha=0.0, run_labels=None,
                  feature_signature=None)
```

`fit_single_trials` accepts `"spm"` or an `HrfCandidate` for its fixed HRF.
`fit_selected_hrfs` takes an `HrfSelectionResult`. The ridge penalty must be
finite and nonnegative. All supplied confounds and a run intercept are included.

`SingleTrialResult` and `HrfSingleTrialResult` both expose:

- `run_betas`: tuple of trials × features arrays.
- `trial_table`: original event metadata plus `trial_id`, `event_index`,
  `run_label`, `run_index`, and `trial_index`.
- `run_full_r2`, `run_nuisance_r2`: per-run diagnostic arrays.
- `full_r2`, `nuisance_r2`, `delta_r2`: pooled diagnostics.
- `diagnostics`, `ridge_alpha`, and `provenance`.

The fixed-HRF result also has `design_matrices`. The selected-HRF result
instead has `group_designs`, keyed by `(run_index, hrf_id)`, as well as
`hrf_indices` and `selection_provenance`. These grouped matrices include the
trial columns followed by the nuisance columns.

## HRF libraries, selection, and evaluation

From `boldtailor.hrf_library`:

| Entry point | Use |
| --- | --- |
| `sobol_hrf_library(n_samples=512, *, seed=0)` | Sample continuous parameters with scrambled Sobol; add canonical SPM as ID 0 |
| `expanded_hrf_library()` | Return the original 649-candidate grid library |
| `HrfLibrary.from_parameters(parameters)` | Build a library from seven-value parameter rows, adding canonical SPM as ID 0 |
| `HrfCandidate(id, kind, parameters)` | Describe one kernel; `kind` is `"spm"` or `"double_gamma"` |
| `candidate.kernel(tr, oversampling=50)` | Sample a discrete-sum-normalized kernel at TR/oversampling |

Parameter order is `response_delay`, `undershoot_delay`, `response_dispersion`,
`undershoot_dispersion`, `response_undershoot_ratio`, `onset_delay`, `duration`.
SPM candidates require the exact canonical parameter tuple
`(6, 16, 1, 1, 6, 0, 32)`.

`HrfLibrary` exposes `candidates`, `parameter_table`, `curves`, `times`, and
`fingerprint`. Table rows and curve rows follow stable candidate IDs. The
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
select_hrf(data, *, library, run_labels=None, feature_signature=None,
           candidate_batch_size=32)
evaluate_hrf_split(data, *, library, train_runs, test_runs,
                   run_labels=None, feature_signature=None)
```

`select_hrf` returns `HrfSelectionResult`: `hrf_indices`, `cv_r2`,
`canonical_cv_r2`, `delta_cv_r2`, `library`, `eligibility`, `run_labels`,
`feature_signature`, and `provenance`. Eligibility records are checked lazily;
an unchecked candidate is not an excluded candidate.

`evaluate_hrf_split` uses disjoint, zero-based run indices. Its
`HrfEvaluationResult` contains `training_selection`, `training_amplitudes`,
`test_r2`, `canonical_test_r2`, `delta_test_r2`, `train_runs`, `test_runs`, and
`provenance`. It fixes HRF and mean amplitude before predicting the test runs.
Selection requires two runs; independent evaluation requires two training runs
and at least one test run. Undefined HRF indices are `-1`.

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
overwrite=False, retain_incomplete=False, lock_timeout=30.0)` returns the saved
paths. Existing files are protected by default; `source_paths` protects inputs
from accidental replacement. Publication failures raise `PublicationError`.

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
in your application. Contributor-facing identity, logging, and publication
details are in the [developer guide](development.md).

## Example workflows

These functions live under `examples/`, rather than the installed core package.
Run them from the repository checkout:

| Module / function | Use |
| --- | --- |
| `examples.NSD.nsd_cifti.run_analysis` | Conventional stimulus-plus-RT CIFTI analysis |
| `examples.NSD.nsd_single_trial.run_single_trial_analysis` | Canonical or selected-HRF CIFTI beta series and diagnostics |
| `examples.NSD.rt_diagnostics.correlate_rt` | Within-run-centered beta/RT correlations and counts, pooled by run partition |
| `examples.NSD.rt_diagnostics.select_vertices` | Select cortical vertices by absolute odd-run correlation |
| `examples.NSD.rt_diagnostics.scatter_artifact` | Create even-run RT scatterplots for the selected vertices |
| `examples.stop_signal_demo.discover_run_inputs`, `common_brain_mask`, `make_masker`, `load_run` | Load aligned NIfTI runs using an intersected mask |
| `examples.stop_signal_demo.whole_brain_image`, `result_artifacts` | Reconstruct and prepare NIfTI outputs |

The NSD runners accept BIDS, fMRIPrep, and output roots plus `subject`,
`session`, and `block_size`. The single-trial runner also accepts `ridge_alpha`,
`hrf_library` (`"canonical"` or `"expanded"`), and `n_jobs`.
See [NSD commands and options](../examples/NSD/README.md#run-an-analysis).
