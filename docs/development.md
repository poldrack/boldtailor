# Developer guide

For running an analysis, start with the [user guide](user-guide.md).
This page describes the implementation conventions used when extending Boldtailor.

## Environment and checks

Use Python 3.12 or later and uv:

```bash
uv sync --group dev
uv run pytest -q -W error
uv run black --check src tests examples/NSD examples/stop_signal_demo.py
uv run git diff --check
```

`uv run pytest` runs the package suite in `tests/`. Example and notebook tests are
opt-in: `uv run pytest examples/NSD` and
`uv run pytest --run-notebooks tests/test_stop_signal_demo.py examples/NSD`.
Tests use synthetic arrays and small generated imaging fixtures; the example
data on the external NSD volume are not required for the test suite.

CI runs the full suite with warnings treated as errors and checks Python
formatting on a clean Python 3.12 runner. It also builds and tests the installed
wheel in an isolated environment, without the checkout's editable installation
or notebook development dependencies:

```bash
uv build --wheel
uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl python tests/check_installed_package.py
```

The smoke check verifies an installed-package import, absence of `ipykernel`,
and a synthetic OLS fit against NumPy least squares. Update the wheel filename
when changing the package version. Notebook execution uses `ipykernel` from the
development dependency group, not the library's runtime requirements.

Software versions are resolved when provenance is created. An uninstalled
Boldtailor checkout reports its own version as `unknown`; missing dependency
metadata still raises an error rather than inventing a version.

Keep functions short and separate numerical work from file I/O. All
`__init__.py` files stay empty. For behavioral changes, write and run failing
pytest tests, commit the tests, then implement the change. Use numerical
oracles that exercise the scientific behavior rather than tests that duplicate
implementation details. Repository-specific rules are in [AGENTS.md](../AGENTS.md).

## Code organization

| Area | Modules |
| --- | --- |
| Owned arrays, event tables, timing, and source records | `data`, `_arrays`, `provenance` |
| Event-based conventional GLMs | `model`, `design`, `fit`, `_conventional`, `results` |
| Fixed designs supplied by callers | `prepared`, `prepared_fit` |
| Shared conventional/prepared diagnostics | `_fit_diagnostics`, `model.contrast_metadata` |
| Encoding and penalty selection | `trial_encoding`, `ridge_selection`, `fractional_ridge`, `_ridge_cv`, `_fractional_ridge`, `ridge_results` |
| Trial design construction and OLS/ridge estimation | `single_trial`, `_single_trial_design`, `_single_trial_fit`, `single_trial_results` |
| Candidate HRFs, selection, evaluation, and grouped fits | `hrf_library`, `hrf_selection`, `hrf_results`, `_hrf_design`, `_hrf_cv`, `_selected_hrf_fit` |
| Conventional GLMs using voxelwise HRFs | `_hrf_assignment`, `_hrf_glm_design`, `_hrf_glm`, `hrf_glm_results` |
| Records, log events, and file publication | `provenance`, `logging`, `bids_provenance`, `publication` |
| Dataset discovery and image reconstruction | `examples/NSD`, `examples/stop_signal_demo.py` |

The core numerical functions accept arrays and tables and return results in
memory. Example workflows own dataset discovery, image loading, spatial axes,
and image reconstruction. The prepared-design API is the entry point for
externally compiled designs; it does not perform BIDS parsing or transformations.

Keep NSD and imaging-format adapters in the examples, outside the installed
package. The NSD notebooks retain scientific settings and fitting calls;
`examples/NSD/workflow_plots.py` and `session_hrf_plots.py` return presentation
figures from fitted results. `notebook_paths.py` resolves explicit configuration
and environment paths without file writes. See the [NSD setup instructions](../examples/NSD/README.md#full-workflow-notebook).
These helpers use development dependencies; they add no runtime dependencies
or public imaging API to Boldtailor.

Public numeric arrays are owned, ordinary NumPy arrays marked read-only through
`_arrays.readonly_array`. Float values use float64, indices use int64, and masks
use bool. Construction copies inputs; accessing an array does not copy it again.
These flags prevent accidental writes. Deliberately re-enabling writes can
invalidate the owning analysis and its provenance assumptions. Copy an array
before editing it. Mutable table/dictionary accessors return defensive copies,
including nested event metadata. Private containers are not a public mutation
interface.

Prepared fitting reads `PreparedDesignAnalysis._design_matrices` and
`_column_roles` internally. These are already owned at construction; fitting
and diagnostics must not mutate them. Public accessors keep returning copies,
and result construction still makes its own design copies. This avoids copying
all runs' tables repeatedly inside each run's diagnostics.

## Scientific result schemas

`ridge_results.CandidateScores` owns the common cross-validation arrays and
labels their rows with `grid` and an explicit `regularization` kind. Grid
validation and sorting stay in the scoring functions; result construction
does not reorder scores.

`single_trial_results.SingleTrialResult` owns trial estimates, diagnostics,
and penalty metadata. Its `design` is a `SharedTrialDesign` for one HRF across
features or a `SelectedTrialDesign` for grouped HRF assignments. Each design
container owns its matrices (a selected design rebuilds them on request
instead of retaining one per `(run, hrf_id)`), so the result does not duplicate
that copying logic. `SingleTrialResult` with a `SelectedTrialDesign`, and
`HrfAnalysisResult`, hold rebuild closures and cannot be pickled; parallel
workers must return arrays or dicts (as the NSD examples do). Common numerical
fields stay directly on the result. See
[result migration](result-migration.md) for API changes.

## Numerical conventions

Conventional fits use Nilearn's OLS/AR(1) estimator and t contrasts. Each run
has independent coefficients. Contrasts are combined through Nilearn contrast
addition and equal-run averaging. Full-model R² uses predictions on the original
signal scale and pools within-run sums of squares.

`fit(..., hrf_selection=...)` dispatches to grouped conventional fitting.
Each HRF group uses the same numerical contrast engine as a common-HRF fit.
Custom-HRF task columns are compiled separately from nuisances so Nilearn's
callable-name suffix cannot rename a semantic contrast or collide with a
confound. Canonical groups use the existing SPM design path. Group results
are restored to input feature order; undefined HRFs remain NaN.

The grouped result exposes a design for each `(run_index, hrf_id)`, with owned
arrays and copied DataFrames. Its identity includes effective model settings,
the HRF library/assignment, and labeled design values plus frame times. The
ignored fixed `hrf_model` setting does not affect this identity. The grouped
delta-R² path checks the parent identity, refits nested OLS models, and retains
NaNs for undefined selections or constant signals. Common-HRF/prepared delta
paths still require finite pooled R².

Selected-HRF completion logs include the analysis identity. Start records use
execution/source IDs because the identity is established during design
compilation.

Conventional and prepared fits share rank warnings, dimension validation,
and nested-OLS thresholds in `_fit_diagnostics.py`. Their contrast provenance
uses `model.contrast_metadata`. Fit-specific scientific operations remain
visible in the entry points; grouped-HRF validation retains its NaN contract.

Task-versus-nuisance diagnostics use nested OLS fits. Small negative differences
from floating-point roundoff are clipped in `TaskDeltaR2Result.delta_r2`;
`raw_delta_r2` retains the original values. Single-trial ridge comparisons use
the actual penalized fit and preserve signed differences.
All ΔR² entry points record one shared activity schema built by
`delta_r2_activity`, and compute R² pairs through `nested_ols_delta`; the
result constructor rejects differences below `-NESTED_OLS_TOLERANCE`.

Single-trial ridge projects task columns and signals off the nuisance span,
normalizes projected trial columns to unit L2 norm, applies a fixed penalty,
then restores coefficient units. Nuisance coefficients are unpenalized.
Trial identifiability and positive residual degrees of freedom are required.

The trial solvers separate preparation from candidate evaluation. For a run,
`x` has shape `(scans, trials)`, nuisance columns have shape `(scans, confounds)`,
and signals have shape `(scans, features)`. `_single_trial_fit` removes the
nuisance span and stores the normalized design's SVD in `ProjectedTrialDesign`.
`PreparedTrialBetas.betas_at(alpha)` uses stored signal coordinates to return
fresh `(trials, features)` coefficients. Alpha zero uses the same decomposition
for OLS. Full fits recover nuisance coefficients and sums of squares separately.

Fractional ridge uses `PreparedFractionBetas` in `_fractional_ridge`. Its rank
check uses normalized columns, but its decomposition for shrinkage uses raw
projected columns. This distinction preserves the requested ratio of raw trial
coefficient norms to their OLS norms. `solve` returns betas and feature-specific
implied alphas; `betas_at` returns only betas. Neither solver retains candidate
outputs. The older `trial_beta_path` and `fraction_beta_path` iterators remain
thin adapters for validation scripts.

In `_ridge_cv`, `RunBetaPath` groups features by their training-selected HRF and
restores their original column positions. Each fold prepares each run once,
then explicitly evaluates a requested candidate. Fractional CV evaluates the
validation run at fraction one once and retains that fixed OLS target;
normalized ridge CV evaluates its validation target at each candidate alpha.
Only one candidate's beta arrays are assembled at a time. HRF selection and
encoding training still use only the training runs.

Shared numerical references live in `tests/oracles.py`: independent augmented
least squares and root finding provide checks against the production SVD.
Reusable synthetic datasets are pytest fixtures in `tests/conftest.py`.

HRF ID 0 dispatches to the exact Nilearn SPM kernel with 32-second support.
`expanded_hrf_library()` adds 648 double-gamma candidates with 36-second support.
The NSD notebooks instead default to `sobol_hrf_library(n_samples=512, seed=0)`,
which adds 512 sampled custom HRFs over the same parameter ranges.
Candidate order is deterministic. Exported curves use a 0.1-second grid;
convolution uses TR/50 and the supplied acquisition times. Kernels have discrete
sum one. Peak-normalizing them would change the interpretation of trial betas.

HRF selection pools leave-one-run-out prediction errors for a shared mean
stimulus response. Training amplitudes are held fixed on the omitted run.
RT and stimulus identity are not selection targets. Structural eligibility
uses timing and confounds, with lazy validation of proposed winners. A candidate
cannot drop trials or change the observation set to obtain a better score.

The independent split evaluation selects inside training runs, then freezes
the HRF and mean amplitude for test prediction. Both halves of the NSD reliability
comparison use the same timing/confound eligibility context. Production HRFs
selected over all runs are shared by the final OLS and ridge fits.

Nilearn 0.14 divides the OLS/AR(1) dispersion by `n - columns` (`regression.py:196`), inconsistent with its `df_residuals = n - rank`; Boldtailor avoids it by fitting rank-deficient designs in a full-rank basis (`_conventional._full_rank_basis`).

## Source identity and provenance

Threat model: cooperating writers on one host, no adversary, no power-loss
guarantee. Records are scientific metadata, not a privacy or security boundary.

`SourceRef` holds a role, a dataset-relative or `bids:<dataset>:<path>` URI,
media type, byte size, UTC `modified_at` (`Z` or `+00:00`), optional `sha256`,
and JSON annotations; `RunSources` groups signal, events, and confounds. One
rule, `provenance.validate_relative_path`, governs URIs and BIDS projection
paths: POSIX, no traversal or empty parts, components `[A-Za-z0-9+_.-]+`.
Annotation, metadata, and model-key text is stored as given, even if it looks
like a path. `metadata_fingerprint` hashes source metadata (not file contents)
and is absent for incomplete sources. Each activity records `software`, which
analysis identities exclude. Execution IDs are fresh UUIDs per attempt.

## Logging

The library emits JSON messages through the standard `boldtailor` logger and
does not install handlers. Applications choose the destination and log level.
`bind_context()` carries execution, source, analysis, and run IDs;
`emit_event()` creates a lifecycle record; `append_event_history()` retains
the most recent eight events in provenance.

Fit and comparison entry points use the private `fit_operation()` context
manager. Their statistical steps remain explicit in each caller. The context
owns execution IDs and events; provenance and result construction both occur
inside its failure boundary. A pending completion is included in final
provenance and emitted only after result construction succeeds. Start records
omit `analysis_id`; completion includes it when metadata identity is available.
Each nested fit or normalization starts a fresh ID scope and restores its
caller’s context on exit. Ordinary `bind_context()` remains additive unless
called with `inherit=False`.

Failure records export an `error_code`: `invalid_input`, `numerical_failure`,
`io_failure`, or `operation_failed`. Exceptions are re-raised unchanged for
callers. The logging boundary never exports exception text or inspects the
environment; the prepared-fit sanitizer has been removed. Event names, IDs,
and annotations are caller-supplied metadata, so this is not a blanket privacy
guarantee. Existing saved records retain their old fields. See the
[lifecycle migration](lifecycle-migration.md) for event prefixes and category rules.

Input normalization constructs one final provenance record. Extension creates
a validated record directly, avoiding a parent-record serialization round trip.
Source identity formulas, scientific activities, and quality warnings are
preserved; source fingerprints describe metadata, not BOLD contents.

## BIDS metadata projection

`project_bids_provenance()` returns relative paths mapped to bytes without I/O:
BIDS 1.11.1 metadata plus an optional (`export_bids_prov`) draft subset pinned
to `BEP028@02172700aac8d1bdd67b45191f43533f426848dc`. The logs equal the record
(`logs/boldtailor_provenance.json` is its canonical JSON); no keys are dropped.
Each Activity's `Command` names the entry point (`boldtailor.fit`,
`boldtailor.select_hrf`, ...); the last carries `StartedAtTime`/`EndedAtTime`
from its lifecycle events; `Environments[0]` has `Python`, `Platform`, and a
`Software` list taken from the last activity's `software`.

## Writing result files

Workflows build `Artifact` objects in memory and pass them to
`publish_artifact_set()`. Under a `FileLock`, it checks targets, stages files in
`<destination>.boldtailor/stage-<uuid>/`, moves overwritten files to
`backup-<uuid>/`, `os.replace`s each file into place, and fsyncs files (not
directories). Failure restores the backups. Writers serialize; readers can see a
partial set. Payloads are not parsed. Ancestor symlinks such as macOS `/tmp` are
resolved; a symlinked destination or a symlink inside it is refused, as are
unrequested overwrites and outputs overlapping `source_paths`.
`<destination>.boldtailor/` is a sibling, not part of the dataset (so no
`.bidsignore`); it holds `lock` and `failures.jsonl`, whose fixed-field records
omit exception text. If rollback fails, `PublicationError.recovery_directory`
names the retained `backup-<uuid>/`; `rollback_errors` lists the failures.

## NSD exports and parallel execution

The NSD runners load feature blocks, preserve CIFTI axes, and reconstruct full
output arrays in the parent process. `n_jobs>1` uses bounded batches of joblib
loky processes. Each worker has all runs for its assigned feature block and
limits numerical-library threads to one. Worker failure prevents publication;
workers do not write derivatives. Library/timing/design caches are process-local.
Full output beta arrays and the assembled artifact set still consume memory.

For optimized HRFs, each run's NPZ stores `hrf_<id>` trial matrices, shared
`nuisance`, `frame_times`, `trial_columns`, `nuisance_columns`, `hrf_ids`, and
`library_fingerprint`. Matrices are float64; labels use string arrays. Load with
`allow_pickle=False` and assemble a full design as
`np.column_stack([saved[f"hrf_{h}"], saved["nuisance"]])`.

These archives cover all-run production fits. An HRF used only by an independent
RT diagnostic may need reconstruction from the saved library, event timing,
frame times, and nuisance matrix. The installed core has no general loader
that reconstructs an
`HrfSelectionResult` from image files. The session-reliability example can
reuse compatible saved selection maps and provenance through its dedicated
cache/import helpers; it does not provide a general selection-object loader.
Standalone all-run response-delay and time-to-peak files in
the development NSD dataset were created separately; automatic exports contain
those quantities in the multi-map HRF parameter images.

## Provenance digests

`SourceRef` accepts an optional caller-supplied `sha256` (64 lowercase hex
characters). It enters the metadata fingerprint and is emitted as the BEP028
`Digest` of the file entity. The earlier ban on a `digest` key was removed on
2026-10-02: a content digest of imaging data is not identifying, and BEP028
defines `Digest`.

## Design history and validation

Current behavior is described by the [user guide](user-guide.md) and
[API reference](api.md). The `docs/superpowers` tree contains implementation
plans, design decisions, and historical validation records; some older plans
describe features that have since shipped.

- [NSD session results and benchmarks](validation/nsd-session.md)
- [Expanded HRF validation](superpowers/validation/2026-09-26-hrf-selection.md)
- [Parallel execution validation](superpowers/validation/2026-09-26-nsd-parallel.md)
- [Odd/even HRF validation](superpowers/validation/2026-09-27-split-hrf.md)
