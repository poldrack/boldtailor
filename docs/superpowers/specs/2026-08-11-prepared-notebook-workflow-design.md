# Prepared-Design Stop-Signal Notebook Design

**Date:** 2026-08-11

## Goal

Make the real-data stop-signal notebook demonstrate Boldtailor's intended
prepared-design workflow without fitting the five-session whole-brain dataset
twice. The notebook remains an end-to-end BIDS example, but its estimation
boundary becomes the same fixed-design array/DataFrame boundary intended for
FitLins interoperability.

## Intended Usage Boundary

The notebook continues to discover BIDS inputs, load and mask fMRIPrep images,
normalize events and confounds, and build an `AnalysisData` object. It uses the
public `boldtailor.design` API to compile fixed, labeled full and nuisance-only
design matrices.

After design compilation, the notebook constructs a
`PreparedDesignAnalysis` and performs its sole whole-brain fit with
`fit_prepared()`. Task-attributable variance is computed with
`task_delta_r2_prepared()`. The event-level `fit()` and `task_delta_r2()` entry
points are not invoked by the notebook.

This makes the boundary explicit:

- In the notebook, Boldtailor compiles designs from events and confounds for a
  self-contained demonstration.
- In FitLins, PyBIDS/FitLins supplies the already compiled matrices and enters
  at `PreparedDesignAnalysis`.
- Boldtailor receives arrays, DataFrames, timing, semantic roles, source
  descriptors, and curated metadata. It receives no image, BIDS layout,
  PyBIDS, Nipype, or FitLins object.

## Data Flow

1. Discover the configured stop-signal runs and load the common-mask signals.
2. Normalize signals, events, confounds, timing, and source descriptors with
   `data.from_arrays()`.
3. Define the conventional model, semantic t contrasts, and the structural
   role-classification policy.
4. Compile full designs with `design.compile_designs()` and nuisance-only
   designs with `design.compile_nuisance_designs()`.
5. Resolve and validate an explicit column-role mapping for each run against
   the concrete columns produced by both compilers.
6. Construct `PreparedDesignAnalysis` from the normalized signals, full design
   matrices, frame times, sources, column roles, curated run metadata, and
   bounded provenance metadata.
7. Fit once with `fit_prepared()` using the model's contrasts and inferential
   noise model.
8. Compute task delta R-squared with `task_delta_r2_prepared()`. Both its full
   and nuisance diagnostic fits use OLS even when contrast inference uses
   AR(1).
9. Reuse the prepared results for design displays, whole-brain maps,
   provenance summaries, and derivative publication.

## Column Roles

The role-classification policy is declared before compilation, but a concrete
mapping cannot be materialized until the compiler has produced the labeled
columns. Roles are then resolved by comparing the public full and nuisance-only
compiled designs for each run:

- `constant` is `intercept`.
- Every other full-design column also present in the nuisance-only design is
  `nuisance`; this includes selected confounds and drift regressors.
- Every full-design column absent from the nuisance-only design is `task`.

The notebook validates before fitting that every nuisance-only column exists
in the corresponding full design, every full column receives exactly one
role, each run has at least one task column and an intercept or nuisance
column, and no column receives the incomplete `other` role. Failures identify
the affected session/run.

This structural rule avoids naming guesses and makes the prepared nuisance
model identical to the public nuisance compiler's model. The roles interpret
an already compiled matrix; they do not control design construction.

## Provenance and Display

Prepared inputs reuse the normalized `AnalysisData` signal arrays, frame
times, and `RunSources`. Curated run metadata contains subject, session, task,
and run labels. Prepared provenance metadata identifies the public design
compiler and fixed-design boundary without recording absolute paths, signal
values, or design values.

The notebook displays compact summaries only:

- design shapes, columns, and role counts;
- stable data/design/analysis fingerprints;
- diagnostic and inferential noise models;
- lifecycle event summaries;
- contrast, per-run R-squared, aggregate R-squared, and task delta R-squared
  images using the shared slice coordinates.

Existing publication safety, relative-path provenance, image counts, common
mask behavior, and no-raw-signal-display guarantees remain unchanged.

## Stored Outputs

Existing stored outputs are cleared because they describe the earlier
two-session execution while the source defaults to five sessions. Tests must
not require an output-free notebook. A future execution with the real dataset
may be committed with outputs for documentation and GitHub sharing.

## Testing

Tests are written and committed before notebook changes. They must fail
because the current notebook still uses the event-level fitting path.

The tests require:

- public imports and calls for full/nuisance design compilation,
  `PreparedDesignAnalysis`, `fit_prepared()`, and
  `task_delta_r2_prepared()`;
- absence of calls to event-level `fit()` and `task_delta_r2()`;
- complete structural column roles and curated run metadata;
- execution from both the repository root and notebook directory;
- numerical parity between notebook prepared results and an independently
  computed event-level fit on the bounded test fixture;
- prepared fingerprints, lifecycle context, diagnostic/inferential model
  disclosure, privacy, plots, and derivative publication;
- tolerance of either empty or populated notebook outputs.

The real-data notebook does not perform a second event-level whole-brain fit;
parity is established only in the small automated fixture.

## Non-Goals

This change does not add FitLins code, a standalone BIDS adapter, F contrasts,
image reconstruction APIs, HRF optimization, GLMdenoise, fractional ridge, or
a GLMsingle orchestration recipe. It does not change Boldtailor's numerical or
prepared-design library APIs.
