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

`uv run pytest` collects both `tests/` and `examples/NSD`, including the CIFTI
workflows and real-process parallel tests. The explicit command
`uv run pytest tests examples/NSD -q -W error` runs the same suite.
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
| Trial design construction and OLS/ridge estimation | `single_trial`, `_single_trial_design`, `_single_trial_fit`, `single_trial_results` |
| Candidate HRFs, selection, evaluation, and grouped fits | `hrf_library`, `hrf_selection`, `hrf_results`, `_hrf_design`, `_hrf_cv`, `_selected_hrf_fit` |
| Conventional GLMs using voxelwise HRFs | `_hrf_assignment`, `_hrf_glm_design`, `_hrf_glm`, `hrf_glm_results` |
| Records, log events, and file publication | `provenance`, `logging`, `bids_provenance`, `publication` |
| Dataset discovery and image reconstruction | `examples/NSD`, `examples/stop_signal_demo.py` |

The core numerical functions accept arrays and tables and return results in
memory. Example workflows own dataset discovery, image loading, spatial axes,
and image reconstruction. The prepared-design API is the entry point for
externally compiled designs; it does not perform BIDS parsing or transformations.

Public numeric arrays are owned, ordinary NumPy arrays marked read-only through
`_arrays.readonly_array`. Float values use float64, indices use int64, and masks
use bool. Construction copies inputs; accessing an array does not copy it again.
These flags prevent accidental writes. Deliberately re-enabling writes can
invalidate the owning analysis and its provenance assumptions. Copy an array
before editing it. Mutable table/dictionary accessors return defensive copies,
including nested event metadata. Private containers are not a public mutation
interface.

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

Task-versus-nuisance diagnostics use nested OLS fits. Small negative differences
from floating-point roundoff are clipped in `TaskDeltaR2Result.delta_r2`;
`raw_delta_r2` retains the original values. Single-trial ridge comparisons use
the actual penalized fit and preserve signed differences.

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
The default custom grid adds 648 double-gamma candidates with 36-second support.
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

## Source identity and provenance

`SourceRef` accepts dataset-relative POSIX URIs or BIDS URIs, an optional media
type, byte size, UTC modification time, and annotations. `RunSources` groups
signal, events, and optional confound references for a run. Supply complete
references for all actual sources; array construction does not inspect files.

For example, a caller that has loaded these files can attach their metadata:

```python
from pathlib import Path
from datetime import datetime, timezone
from boldtailor.provenance import RunSources, SourceRef

def source_ref(path, root, role):
    path, root = Path(path), Path(root)
    info = path.stat()
    return SourceRef(
        role=role,
        uri=path.relative_to(root).as_posix(),
        byte_size=info.st_size,
        modified_at=datetime.fromtimestamp(info.st_mtime, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    )

# root and the two paths refer to the files used to construct the arrays.
sources = (RunSources(
    signal=source_ref(signal_path, root, "signal"),
    events=source_ref(events_path, root, "events"),
),)
```

`metadata_fingerprint` hashes canonical source metadata, not file contents.
It is unavailable for missing or incomplete source descriptions. Anonymous
arrays still work, with reduced provenance quality recorded in the result.
Conventional fits combine source identity with model settings for an analysis
fingerprint; prepared and trial workflows also record labeled design identities.
Importable custom HRFs can be identified; local functions and lambdas may leave
a deterministic conventional model identity unavailable.

Execution IDs are fresh UUIDs for individual attempts. They are distinct from
source and analysis fingerprints. `analysis_fingerprint()` and
`extend_provenance()` in `boldtailor.provenance` support this bookkeeping when
adding a workflow. `ProvenanceRecord.from_dict()` reconstructs records while
retaining supported extension fields.

For spatially varying HRFs, selection provenance identifies the library and
feature-to-HRF assignment. Grouped-fit provenance identifies each used design.
The NSD feature signature hashes the ordered CIFTI BrainModel axis and feature
indices. Feature count alone cannot establish spatial correspondence.

## Logging

The library emits JSON messages through the standard `boldtailor` logger and
does not install handlers. Applications choose the destination and log level.
`bind_context()` carries execution, source, analysis, and run IDs;
`emit_event()` creates a lifecycle record; `append_event_history()` retains
the most recent eight events in provenance.

Records contain lifecycle metadata and correlation IDs rather than arrays,
coefficients, or table contents. Metadata validation and error sanitization
exclude absolute paths and sensitive execution details such as command lines,
environment variables, usernames, and hostnames. Callers still need to ensure
that relative source names and annotations are suitable for sharing.

## BIDS metadata projection

`project_bids_provenance()` converts a record into a mapping of relative paths
to file bytes. The implementation currently writes stable BIDS 1.11.1 metadata.
Optional draft provenance output is pinned to
`BEP028@02172700aac8d1bdd67b45191f43533f426848dc`.

The draft subset covers Activities, Files, Environments, Software, the
provenance label table, and file `GeneratedBy`/`Sources` relationships.
It does not claim conformance to a final BIDS provenance standard.
`export_bids_prov=False` omits those draft files while retaining dataset
metadata, canonical provenance, and event logs. Projection performs no I/O.

## Writing result files

An imaging workflow first creates its images, tables, sidecars, and provenance
files in memory as `Artifact` objects, then passes them to
`publish_artifact_set()`. Keeping the files in one publication operation lets
the writer restore the previous output set if a later write fails. New adapters
should use this path rather than writing their final files piecemeal.

The publisher validates paths, protects supplied source paths, locks the output
directory, stages and fsyncs files, and rolls back failed promotion. Existing
files are protected unless `overwrite=True`. Preflight known output collisions
before expensive fitting, and still let the publisher perform its own checks.

On failure, the normal diagnostic is
`.boldtailor/publication_failures.jsonl`. If rollback cannot restore an
original file, its last recoverable copy is retained under
`.boldtailor/failed/<execution-id>/recovery/`. This recovery copy is kept
regardless of `retain_incomplete`. Setting `retain_incomplete=True` also keeps
the failed new artifact set and marks retained canonical provenance as failed
and unpublished. See `tests/test_publication.py` for recovery and path-safety cases.

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
frame times, and nuisance matrix. A saved-map-to-selection loader has not yet
been implemented. Standalone all-run response-delay and time-to-peak files in
the development NSD dataset were created separately; automatic exports contain
those quantities in the multi-map HRF parameter images.

## Design history and validation

Current behavior is described by the [user guide](user-guide.md) and
[API reference](api.md). The `docs/superpowers` tree contains implementation
plans, design decisions, and historical validation records; some older plans
describe features that have since shipped.

- [NSD session results and benchmarks](validation/nsd-session.md)
- [Expanded HRF validation](superpowers/validation/2026-09-26-hrf-selection.md)
- [Parallel execution validation](superpowers/validation/2026-09-26-nsd-parallel.md)
- [Odd/even HRF validation](superpowers/validation/2026-09-27-split-hrf.md)
