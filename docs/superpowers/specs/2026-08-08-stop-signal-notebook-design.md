# Stop-Signal Real-Data Notebook Design

## Purpose

Create an executable notebook that demonstrates the Boldtailor features implemented
so far with real fMRIPrep data. The example uses two stop-signal sessions from the
local BIDS dataset at `/Users/poldrack/data_unsynced/rdoc_fmri` while remaining
portable to another dataset location through configuration.

The notebook is documentation and an integration example. It does not introduce a
general public BIDS-loading API.

## Deliverables

- `examples/stop_signal_demo.ipynb`: the narrative, executable example.
- `examples/stop_signal_demo.py`: example-specific BIDS, ROI, source-metadata, and
  serialization helpers.
- Pytest coverage for the helper and a smoke test that executes the notebook against
  a generated two-session fixture.
- Updated uv development dependencies needed to execute notebooks in tests.

No `__init__.py` file will be added or modified.

## Configuration

The first code cell contains the user-facing configuration:

- Dataset root, read from `BOLDTAILOR_BIDS_ROOT` when set and otherwise defaulting to
  `/Users/poldrack/data_unsynced/rdoc_fmri`.
- Subject `sub-s4`, task `stopSignal`, and default real-data sessions `ses-06` and
  `ses-08`.
- Optional `BOLDTAILOR_SESSIONS` override containing exactly two non-empty,
  comma-separated session labels; surrounding whitespace is ignored. The synthetic
  smoke test selects its `ses-02` and `ses-04` fixture through this override.
- fMRIPrep derivative directory and MNI152NLin2009cAsym resolution-2 space.
- The illustrative ROI definition.
- `PERSIST_DERIVATIVES`, false by default.
- `OVERWRITE_EXISTING`, false by default.

Temporary publication is the default. If `PERSIST_DERIVATIVES` is true, the
destination is `<dataset>/derivatives/boldtailor`. Persistent publication still
refuses collisions unless `OVERWRITE_EXISTING` is also explicitly true.

## Architecture

The helper module has four bounded responsibilities:

1. Discover exactly one events file, preprocessed BOLD image, brain mask, and
   confounds file for each configured session.
2. Map the illustrative MNI ROI into voxel coordinates and extract its bounding box
   over time without materializing a whole multi-gigabyte image in memory.
3. Construct Boldtailor source references from dataset-relative URIs, sizes,
   timestamps, and media types.
4. Serialize design matrices and compact result summaries into immutable publication
   artifacts.

The notebook keeps the substantive Boldtailor API calls visible. It performs
ingestion, model construction, fitting, contrast access, provenance projection, and
transactional publication directly rather than hiding those steps in the helper.

The helper remains under `examples/`; it is not installed as a supported public API.

## Data Flow

For each of the configured sessions, defaulting to real-data sessions 06 and 08, the
notebook discovers matching `run-01` inputs:

- Raw BIDS `events.tsv`.
- fMRIPrep MNI preprocessed BOLD image.
- Matching MNI brain mask.
- fMRIPrep confounds TSV.

The helper validates uniqueness and entity agreement. It then extracts an
illustrative right inferior frontal ROI. The notebook labels this as a computational
demonstration rather than a confirmatory anatomical or inferential choice.

Only the four observed trial conditions are modeled: `go_success`, `go_failure`,
`stop_success`, and `stop_failure`. Rows for non-trial phases remain in the source
file but do not become modeled conditions. Motion parameters and framewise
displacement are selected from the fMRIPrep confounds. The helper handles the initial
undefined framewise-displacement value explicitly and rejects other non-finite or
length-mismatched confounds.

Each session becomes one Boldtailor run. This demonstrates different scan counts,
run-specific events, confounds, frame times, and design matrices. The model uses
AR(1) noise and named semantic contrasts, including `stop_success - stop_failure`.

## Demonstrated Features

The notebook demonstrates:

- Array ingestion with two runs and explicit source metadata.
- Run-specific design construction and design provenance.
- AR(1) model fitting.
- Named contrasts.
- Contrast effects, variance, t statistics, z scores, and one-sided p-values.
- Run-level and combined R-squared values.
- Model and run diagnostics, warnings, execution identifiers, and analysis
  fingerprints.
- Structured logging and the recorded provenance event history.
- In-memory reconstruction and visualization of an ROI z-statistic map.
- Stable BIDS derivative metadata and the pinned draft BIDS provenance projection.
- Transactional, collision-aware publication.

## Published Artifacts

The projected provenance artifacts are published without modification, including the
derivative dataset description, structured log, internal provenance record, and
pinned draft BIDS provenance files.

The example adds compact report artifacts:

- One TSV design matrix per session.
- A TSV contrast summary containing aggregate ROI statistics rather than full arrays.
- A JSON file recording the notebook configuration needed to interpret the example.

The publication call receives all discovered inputs as protected source paths, so an
output destination cannot overlap or replace source data. Publication uses
`overwrite=False` unless the user explicitly enables the overwrite configuration.

## Failure Behavior

The example fails with specific errors when:

- An expected input is missing or discovery returns multiple matches.
- File entities do not agree across events, BOLD, mask, and confounds.
- BOLD scan count and confound row count differ.
- Modeled event timing exceeds the acquisition.
- Required confounds are missing or contain unexpected non-finite values.
- The ROI does not overlap the run mask.
- A publication target collides and overwrite is disabled.
- A persistent destination is outside the dataset's
  `derivatives/boldtailor` directory.

The example never silently truncates events, confounds, or BOLD data.

## Testing Strategy

Implementation follows a committed RED-GREEN-Refactor sequence. Failing tests are
committed before helper or notebook implementation.

Pytest fixtures generate a small, synthetic two-session BIDS/fMRIPrep-like dataset.
Tests cover:

- Exact discovery and entity matching.
- Missing and duplicate input failures.
- Bounded ROI extraction, voxel ordering, and mask intersection.
- Confound selection and validation.
- Source-reference metadata.
- Deterministic report serialization.
- Temporary and persistent output selection.
- Source/destination separation and overwrite behavior.
- Notebook execution against the generated fixture.

`nbclient` and `ipykernel` are added to the uv development dependency group. The
smoke test executes every notebook cell in a temporary working directory with the
fixture supplied through `BOLDTAILOR_BIDS_ROOT` and a noninteractive plotting
backend.

The notebook-output state is deliberately not tested. The initial version may be
committed without outputs, but users may execute it and commit outputs for GitHub
sharing. The notebook avoids displaying absolute source paths, raw participant-level
tables, large arrays, or excessively verbose logs by default.

## Verification

Completion requires:

- The focused helper and notebook tests pass.
- The full pytest suite passes normally and with warnings treated as errors.
- Black, lockfile, repository-contract, and empty-initializer checks pass.
- The notebook executes successfully against the synthetic fixture.
- A separate execution against default real sessions 06 and 08 completes
  successfully and publishes only to a temporary directory.
- The repository contains no executed real-data copy or generated derivative output.

## Out of Scope

- A general BIDS or fMRIPrep loader in the installed Boldtailor package.
- Whole-brain analysis or scientific validation of the illustrative ROI.
- Group-level modeling.
- Automatic publication into the real dataset.
- A supported spatial-derivative writer API.
