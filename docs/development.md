# Developer guide

For running an analysis, start with the [user guide](user-guide.md).
This page describes the implementation conventions used when extending Boldtailor.

## Environment and checks

Use Python 3.12 or later and uv:

```bash
uv sync --group dev
uv run pytest -q -W error
uv run black --check src tests examples/NSD
uv run git diff --check
```

`uv run pytest` runs the package suite in `tests/`. Example and notebook tests are
opt-in: `uv run pytest examples/NSD`, `uv run pytest examples/validation`, and
`uv run pytest --run-notebooks examples/NSD`.
Tests use synthetic arrays and small generated imaging fixtures; the example
data on the external NSD volume are not required for the test suite.
Notebooks are tracked as jupytext py:percent scripts and paired with gitignored
`.ipynb` files by `jupytext.toml`; notebook tests read the `.py` sources. After
editing an `.ipynb` outside a jupytext-enabled Jupyter server, run
`uv run jupytext --sync <notebook>` so the change reaches the `.py`.

CI splits the suites. On every push and pull request it runs the package
suite in `tests/` with warnings treated as errors, uploads its line coverage
to Codecov (the README badge), and checks Python formatting on a clean
Python 3.12 runner. A separate `examples` job, run only on manual dispatch and
on schedule, runs `examples/NSD` and `examples/validation` with
`--run-notebooks` and warnings as errors. The push job also builds and tests
the installed wheel in an isolated environment, without the checkout's
editable installation or notebook development dependencies:

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
| Owned arrays, event tables, timing, and source records | `data`, `_arrays`, `_scalars`, `provenance` |
| Event-based conventional GLMs | `model`, `design`, `_task_design`, `fit`, `_conventional`, `results` |
| Fixed designs supplied by callers | `prepared`, `prepared_fit` |
| Shared conventional/prepared diagnostics | `_fit_diagnostics`, `model.contrast_metadata` |
| Encoding and penalty selection | `trial_encoding`, `ridge_selection`, `fractional_ridge`, `_ridge_cv`, `_fractional_ridge`, `ridge_results` |
| Trial design construction and OLS/ridge estimation | `single_trial`, `_single_trial_design`, `_single_trial_fit`, `single_trial_results` |
| Candidate HRFs, selection, evaluation, and grouped fits | `hrf_library`, `hrf_selection`, `hrf_results`, `_hrf_design`, `_hrf_cv`, `_selected_hrf_fit` |
| Conventional GLMs using voxelwise HRFs | `_hrf_assignment`, `_hrf_glm_design`, `_hrf_glm`, `hrf_glm_results` |
| Task-guided (GLMdenoise-style) denoising | `denoising`, `denoising_results`, `_denoising_pool`, `_denoising_cv`, `_denoising_gate`, `_denoising_identity`, `_mixture_threshold` |
| Records, log events, and file publication | `provenance`, `logging`, `bids_provenance`, `publication`, `_software` |
| Shared constants and deprecation shims | `_constants`, `_deprecation` |
| CIFTI scalar I/O, descriptive diagnostics, HRF agreement, process batches | `cifti`, `diagnostics`, `reliability`, `parallel` |
| Session workflow and `boldtailor run` | `workflow` (`settings`, `inputs`, `analysis`, `beta_series`, `run`, `outputs`, `artifacts`, `files`, `summaries`, `plots`, `surfaces`, `report`), `cli` |
| Notebook-only dataset helpers | `examples/NSD` |

The core numerical functions accept arrays and tables and return results in
memory. Example workflows own dataset discovery, image loading, spatial axes,
and image reconstruction. The prepared-design API is the entry point for
externally compiled designs; it does not perform BIDS parsing or transformations.

The installed `boldtailor.workflow` package and the `boldtailor run` command
(`boldtailor.cli`) own NSD-style CIFTI discovery, loading, fitting stages,
output naming, plotting, and the HTML report. Generic CIFTI dense-scalar I/O
lives in `boldtailor.cifti` (nibabel is a runtime dependency). Package modules
never import from `examples`. The remaining NSD example modules
(`nsd_settings.py`, `session_hrf*.py`, `multisession_*.py`) and the notebooks
import only public modules; plotting uses matplotlib, a runtime dependency.
Shared NSD test fixtures live in `examples/NSD/conftest.py`; each notebook has
one `@pytest.mark.notebook` kernel smoke test. See the
[NSD setup instructions](../examples/NSD/README.md#full-workflow-notebook).

`tests/workflow` runs in the default suite (`uv run pytest -q`) using a small
synthetic BIDS dataset. `examples/NSD` tests are opt-in
(`uv run pytest -q examples/NSD`), and the notebook smoke tests additionally
need `--run-notebooks`.

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
that copying logic. Result objects that hold rebuild closures (`SingleTrialResult` with a `SelectedTrialDesign`, `HrfAnalysisResult`) cannot be serialized with the standard `pickle` module; loky's cloudpickle can move them between processes, but workers should prefer returning arrays or dicts to keep transfers small. The NSD examples return arrays and dicts. Common numerical
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
outputs. The older `trial_beta_path` and `fraction_beta_path` iterators now
live in `tests/oracles.py` as adapters for tests and validation scripts.

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

HRF ID 0 is the Nilearn SPM kernel (32-second support) divided by its maximum.
`expanded_hrf_library()` adds 648 double-gamma candidates with 36-second support.
`default_hrf_library()`, used by `boldtailor run` and the workflow and
multisession notebooks, holds canonical SPM, 512 timing-space Sobol candidates
(`timing_hrf_library`), and GLMsingle's 20 empirical HRFs (533 in total). The
session-reliability notebook uses `sobol_hrf_library(n_samples=512, seed=0)`,
512 Sobol samples over the original grid's gamma-parameter ranges.
Candidate order is deterministic. Exported curves use a 0.1-second grid;
convolution uses TR/50 and the supplied acquisition times. Kernels are stored
at unit peak and each event's response is scaled to peak one
(`HRF_NORMALIZATION = "peak_one_event_response"`), so a trial beta is the peak
BOLD response to that event.

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
like a path: callers must not put secrets or environment details in
annotations or metadata; they are written verbatim. `metadata_fingerprint`
hashes source metadata (not file contents) and is absent for incomplete sources. Each activity records `software`, which
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
`boldtailor.select_hrfs`, ...); the last carries `StartedAtTime`/`EndedAtTime`
from its lifecycle events; `Environments[0]` has `Python`, `Platform`, and a
`Software` list taken from the last activity's `software`.

## Writing result files

Workflows build `Artifact` objects in memory and pass them to
`publish_artifact_set()`. Under a `FileLock`, it checks targets, stages files in
`<destination>.boldtailor/stage-<uuid>/`, moves overwritten files to
`backup-<uuid>/`, `os.replace`s each file into place, and fsyncs files (not
directories). Failure restores the backups. Writers serialize; readers can see a
partial set. Payloads are not parsed. The destination is resolved first (so a
linked destination or macOS `/tmp` works); symlinks inside it are refused, as
are unrequested overwrites and outputs overlapping `source_paths`.
`<destination>.boldtailor/` is a sibling, not part of the dataset (so no
`.bidsignore`); it holds `lock` and `failures.jsonl`, whose fixed-field records
omit exception text. If rollback fails, `PublicationError.recovery_directory`
names the retained `backup-<uuid>/` (or `stage-<uuid>/` when nothing was backed
up); `rollback_errors` lists the failures.

## NSD exports and parallel execution

The session workflow (`boldtailor.workflow`) loads feature blocks, preserves CIFTI axes, and
reconstruct full output arrays in the parent process. `n_jobs>1` uses
`boldtailor.parallel.map_blocks`: bounded batches of joblib loky processes,
each with all runs for its assigned feature block and one numerical-library
thread. Workers return arrays and dictionaries (see the serialization note
above), never write derivatives, and
a worker failure prevents publication. Library/timing/design caches are
process-local. Full output beta arrays and the assembled artifact set still
consume memory.

Each `<stem>_desc-<GLM descriptor>_designs.npz` stores, per run label and HRF ID, the
float64 design (`<run>_hrf-<id>`), its column names (`_columns`), and frame
times (`_frame_times`). Load with `allow_pickle=False`. The installed core has
no general loader that reconstructs an `HrfSelectionResult` from image files.
The session-reliability example reuses compatible saved selection maps and
provenance through its dedicated cache/import helpers.

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

## Dependency notes

- `uv.lock` pins nilearn 0.14.0, a yanked release. `pyproject.toml` allows
  `nilearn>=0.14.0,<0.15`.
- `_hrf_design` imports Nilearn's private `_sample_condition` to build the
  oversampled boxcar that the peak-normalized event scales are computed on.
  `tests/oracles.py` imports it as well to build independent references.
- Decision: stay on `<0.15` until the fast convolution path (the private
  `_sample_condition` contract and `compute_regressor` numerics) is re-verified
  against the next Nilearn release by the Nilearn oracle tests.

## Branch inventory

Branches merged into `main` and not checked out in a worktree were deleted with
`git branch -d`. `fix/dynamic-contrast-artifacts` is merged but kept because
its worktree (`.worktrees/dynamic-contrast-artifacts`) still exists. The
remaining branches, with purposes taken from their last commit, are:

| Branch | Purpose (last commit) |
| --- | --- |
| `fix/dynamic-contrast-artifacts` | Merged; BIDS-safe contrast labels (kept for its worktree) |
| `feature/pre-fitlins-remediation` | Remediation work before FitLins integration (privacy coverage tests) |
| `fix/red-team-remediation` | Red-team remediation (shared-checkout integration verification) |
| `safety/prepared-notebook-source-access` | Safety copy: prepared notebook source-access documentation |
| `safety/task-delta-r2-task4-d86b117` | Safety copy: task delta R-squared notebook diagnostics |
| `safety/task-delta-r2-task4-fd2db4c` | Safety copy: task delta R-squared display |
| `safety/task3-pre-clipped-regression` | Safety copy: serialized task delta R-squared map |
| `safety/task4-before-missing-term-review` | Safety copy: prepared design-matrix fits |
| `task3-review-safety-dbc5b14` | Safety copy: whole-brain notebook task report |
| `task5-review-safety-daa401f` | Safety copy: prepared fit provenance |

The `safety/` and `task*-safety` branches are review snapshots; delete them once
the maintainer confirms they are no longer needed.
