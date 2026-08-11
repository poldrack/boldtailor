# Boldtailor–FitLins Interoperability Design

**Date:** 2026-08-11

**Status:** Approved design

**Scope:** Conventional first-level integration and the architecture boundary for later adaptive methods

## Context

Boldtailor has completed its array-based conventional GLM foundation, including
multi-run estimation, immutable results, structured logging, canonical
provenance, transactional publication, whole-brain examples, R-squared maps,
and task delta R-squared diagnostics. Its existing architecture planned to add
BIDS Stats Models compilation, BIDS discovery, spatial adapters, and derivative
writing as independent later phases.

FitLins already owns much of the corresponding application-level workflow. Its
current GitHub `dev` branch is usable and is expected to be released. It uses
PyBIDS to execute BIDS Stats Models, constructs first-level designs, schedules
Nilearn or AFNI estimators through Nipype, reconstructs NIfTI and CIFTI outputs,
runs higher-level models, and creates derivatives and reports. FitLins also has
an explicit first-level estimator abstraction intended to allow estimators to
be exchanged.

Building a second end-to-end BIDS application in Boldtailor would duplicate
FitLins and PyBIDS. Boldtailor will instead be a reusable first-level modeling
engine that FitLins can invoke. This preserves Boldtailor's standalone library
API while keeping BIDS application orchestration in FitLins.

## Decisions

1. Boldtailor is a reusable, representation-neutral first-level modeling
   engine with optional BIDS adapters.
2. FitLins remains the BIDS application, hierarchical workflow, reporting, and
   derivative-organization layer.
3. PyBIDS owns BIDS indexing, Stats Models validation and graph execution,
   transformations, grouping, and initial design construction.
4. FitLins may make non-breaking changes to expose a stable first-level backend
   boundary.
5. The dependency direction is FitLins to Boldtailor. Boldtailor must not
   depend on FitLins, Nipype, or FitLins internal types.
6. The first integration supports conventional first-level estimation only.
   Voxelwise HRF selection, GLMdenoise, and fractional ridge remain separate
   later components.
7. BIDS Stats Models JSON remains a future direct Boldtailor input, but parsing,
   transformations, and design construction will be delegated to PyBIDS rather
   than reimplemented.

## Ownership Boundaries

### PyBIDS

PyBIDS owns:

- BIDS indexing and metadata inheritance;
- BIDS Stats Models schema handling and graph execution;
- variable transformations, grouping, and model-node selection; and
- construction of the model information used to create labeled designs and
  semantic contrasts.

Boldtailor will not implement a parallel transformations engine.

### FitLins

FitLins owns:

- BIDS and derivative discovery;
- selection of BOLD images, masks, events, and confounds;
- application workflow scheduling and resource management;
- spatial masking and NIfTI or CIFTI reconstruction;
- higher-level model execution;
- derivative paths, dataset-level metadata, and reports; and
- linkage between application provenance and estimator provenance.

### Boldtailor

Boldtailor owns:

- validation and immutable ownership of prepared numerical inputs;
- OLS and AR(1) first-level estimation;
- semantic contrast resolution and fixed-effects combination;
- conventional inferential maps and model diagnostics;
- OLS task delta R-squared when design-column roles are sufficiently specified,
  independent of the inferential OLS or AR(1) setting;
- canonical estimator provenance and structured lifecycle logging; and
- later voxelwise HRF, GLMdenoise, and fractional-ridge operations.

### NiBabel and Nilearn

NiBabel and Nilearn remain the established image and numerical primitives.
Their runtime objects do not cross Boldtailor's public boundary.

## Prepared-Design API

Boldtailor will add an advanced immutable input type named
`PreparedDesignAnalysis` and a `fit_prepared()` operation:

```python
prepared = PreparedDesignAnalysis.from_arrays(
    signals=signals,
    design_matrices=design_matrices,
    frame_times=frame_times,
    sources=sources,
    run_metadata=run_metadata,
    column_roles=column_roles,
)

result = fit_prepared(
    prepared,
    contrasts=contrasts,
    noise_model="ar1",
    model_metadata=model_metadata,
)
```

The type represents a fixed, already-compiled design. Its name deliberately
distinguishes it from the raw event-level `AnalysisData` required by later HRF
optimization.

### Required inputs

For each run, the prepared input contains:

- a finite numeric `time x features` signal array;
- a finite labeled `time x regressors` design matrix;
- frame times;
- a mapping from every design column to its known role;
- curated run metadata, including subject, session, task, and run identities
  when available; and
- source descriptors suitable for canonical provenance.

Column roles are `task`, `nuisance`, `intercept`, or `other`. Unknown roles do
not prevent conventional estimation, but diagnostics that require a complete
task-versus-nuisance partition fail explicitly instead of guessing. Session
and fold identities are retained so the same normalized metadata can support
later cross-validation components.

The API accepts one or multiple runs. Run designs may contain different
columns and column orders. Signals must have compatible feature counts, and
each design must have the same number of rows as its corresponding signal
array.

Contrasts remain semantic expressions or mappings from regressor names to
weights. Positional contrast vectors are not accepted. Every nonzero referenced
regressor must exist and be estimable in every contributing run.

### Behavior

`fit_prepared()` skips event convolution, drift construction, and confound
selection because those operations are already represented in the supplied
design. It reuses the existing Boldtailor fitting, contrast, fixed-effects,
R-squared, result, logging, and provenance operations.

The operation performs no file I/O. Public inputs and results satisfy the same
owned-storage, defensive-copy, and immutability guarantees as the existing
array API. It accepts no NiBabel images, Nilearn maskers, PyBIDS objects,
Nipype interfaces, or FitLins objects.

The first compatibility target is the currently supported semantic t-contrast
surface. FitLins model features outside the supported intersection, including
F contrasts until Boldtailor implements them, are rejected during capability
validation before image data are fitted. The prepared protocol can add F
contrast support later without changing its ownership boundary.

## FitLins Integration

FitLins will retain its existing estimator names, defaults, and behavior. It
will add `boldtailor` as an optional conventional first-level estimator.
Boldtailor will be an optional FitLins dependency and will not be imported when
another estimator is selected.

For each conventional run:

1. PyBIDS executes the relevant Stats Models node and transformations.
2. FitLins constructs the labeled design matrix through its existing design
   path.
3. FitLins loads and masks the BOLD image through its spatial path.
4. The adapter constructs a `PreparedDesignAnalysis` with column roles and
   BIDS entities.
5. `fit_prepared()` returns an `AnalysisResult`.
6. The adapter reconstructs NIfTI or CIFTI maps.
7. FitLins exposes those maps through its existing estimator output collections
   and continues derivative writing, reporting, and higher-level execution.

The adapter returns effect, variance, statistic, z-score, and p-value maps when
the requested Boldtailor contrast supports them. It may expose Boldtailor model
diagnostics as additional optional model maps. Existing consumers continue to
receive their established output fields.

Selecting `boldtailor` without the optional dependency produces an immediate,
installation-oriented error. Unsupported model features produce an explicit
compatibility error before fitting. Existing Nilearn and AFNI estimators are
not reinterpreted or silently redirected.

## Future First-Level Backend Boundary

The existing per-run estimator slot is sufficient for conventional prepared
designs but not for voxelwise HRF selection, which requires multiple runs or
sessions and unconvolved event information. FitLins will therefore evolve a
non-breaking first-level backend boundary capable of receiving a collection of
matching runs before final design construction.

This boundary must not imply a single generic adaptive algorithm. It hosts
separate, explicitly typed modes:

- conventional prepared-design estimation;
- voxelwise HRF optimization using fixed nuisance regressors;
- GLMdenoise component estimation; and
- the later GLMsingle recipe that explicitly orchestrates HRF selection,
  GLMdenoise, and fractional ridge.

Each mode declares the inputs it requires and outputs it produces. Reports and
downstream workflow nodes inspect declared capabilities rather than assuming
that every mode yields every statistical map.

## Separation of Adaptive Components

### Voxelwise HRF optimization

HRF optimization receives raw event timing or pre-convolution task variables,
a candidate HRF library, fixed confound and drift designs, and session-based
fold identities. It generates candidate task designs, evaluates held-out
predictions, selects an HRF independently for each feature, and refits selected
feature groups.

It does not discover or add GLMdenoise regressors. Its outputs include
point-estimate contrast maps, selected-HRF indices, candidate and cross-
validated scores, fold diagnostics, and labeled in-sample R-squared. It does
not claim post-selection z-scores or p-values.

### GLMdenoise

GLMdenoise begins with a baseline fit, constructs a noise pool, derives
candidate nuisance components, and selects the number of components using its
own cross-validation process. It returns learned nuisance regressors and
selection diagnostics. This is a distinct component and provenance activity
from HRF optimization.

### Fractional ridge and GLMsingle

Fractional-ridge single-trial estimation consumes selected HRFs and selected
nuisance regressors. The GLMsingle recipe is a small orchestration operation
over the HRF, GLMdenoise, and ridge components. It does not collapse their APIs
or implementations into a generic adaptive stage engine.

## Provenance and Privacy

Boldtailor records:

- design dimensions, ordered column names, roles, and deterministic design
  identity;
- semantic contrasts and temporal-noise configuration;
- curated source and run identities supplied by FitLins;
- numerical backend and software versions; and
- lifecycle events and bounded aggregate diagnostics.

Prepared numerical design values determine the fitted analysis, so their
canonical values contribute to a deterministic design digest. This is distinct
from source-file provenance: Boldtailor continues not to open source files or
calculate source-content digests. The design digest is stored as identity
metadata; design values are never written to structured logs.

FitLins records dataset discovery, the Stats Models document and node, PyBIDS
transformations and grouping, masking and spatial reconstruction, workflow
execution, and final derivative paths. It links the Boldtailor analysis
fingerprint and canonical provenance artifact into its derivative metadata.
Neither layer independently recreates the other's provenance record.

Later HRF provenance separately records candidate-library identity, folds,
selection metric, selected-HRF output, and refit configuration. Later
GLMdenoise provenance separately records the noise-pool rule, candidate
components, selection process, and retained components.

Existing privacy restrictions remain in force. Logs and provenance exclude
signals, design values, events, confounds, estimates, statistics, absolute
paths, environment variables, command lines, host information, and unstable
object representations.

## Validation and Errors

Prepared inputs fail before estimation when they contain:

- mismatched run counts, scan counts, or feature counts;
- empty, nonnumeric, or nonfinite arrays;
- duplicate, empty, or unlabeled design columns;
- missing or invalid column-role mappings (the explicit `other` role remains
  valid);
- positional, missing, or unestimable contrast references; or
- unsupported temporal-noise or contrast types.

FitLins performs capability validation before loading or fitting image data
where possible. Errors identify the unsupported Stats Models node, contrast,
or estimator option without leaking private paths or data values.

## Testing Strategy

All implementation follows repository-local RED-GREEN-Refactor requirements:
tests are written and committed before production changes, and tests are not
weakened to accommodate incomplete behavior.

### Boldtailor tests

Tests cover:

- prepared input validation, owned storage, and immutability;
- semantic contrasts over run-specific design columns and column order;
- OLS and AR(1) parity with the existing Boldtailor event-level path;
- numerical parity with Nilearn for supported contrast outputs;
- fixed-effects aggregation across runs;
- R-squared and nested OLS task delta R-squared with explicit column roles;
- deterministic design fingerprints and privacy-safe logging; and
- proof that `fit_prepared()` performs no file I/O.

### FitLins tests

Tests cover:

- additive CLI selection with unchanged existing defaults;
- missing optional-dependency and unsupported-capability errors;
- exact translation of designs, roles, contrasts, entities, and sources;
- NIfTI and CIFTI spatial reconstruction;
- compatibility with existing report and higher-level consumers;
- linked Boldtailor provenance in derivative metadata; and
- unchanged Nilearn and AFNI behavior.

### Cross-project fixtures

Small generated BIDS datasets and identical Stats Models documents run through
FitLins's Nilearn and Boldtailor estimators. Tolerances are documented per
output type. The private stop-signal dataset may be used for local smoke tests
but is not a package or CI dependency.

## Non-goals for the First Integration

The first integration does not include:

- voxelwise HRF selection;
- GLMdenoise or learned nuisance regressors;
- fractional-ridge or single-trial estimation;
- a general plugin framework;
- a second BIDS workflow scheduler;
- higher-level estimation in Boldtailor; or
- independent reimplementation of BIDS Stats Models transformations.

## Revised Development Sequence

1. **Completed foundation:** array conventional GLM, provenance and publication,
   whole-brain examples, R-squared, and task delta R-squared.
2. **Prepared-design estimation:** implement `PreparedDesignAnalysis` and
   `fit_prepared()` with roles, deterministic identity, and parity tests.
3. **FitLins conventional integration:** add the optional estimator and
   cross-project fixtures through non-breaking FitLins changes.
4. **Thin standalone BIDS convenience layer:** implement PyBIDS-backed
   `model_from_json()` and `from_bids()` plus representation-level NIfTI and
   CIFTI reconstruction. Do not duplicate FitLins orchestration, hierarchical
   estimation, or reporting.
5. **Voxelwise HRF optimization:** implement and test the independent HRF
   component, then expose its separate FitLins mode.
6. **GLMdenoise:** implement and test noise-pool, component-generation, and
   component-selection operations independently.
7. **Fractional ridge and GLMsingle:** implement single-trial ridge estimation,
   then compose the independent operations through the explicit recipe.

## Consequences

This design expands Boldtailor's deliberately small public surface by one
advanced immutable type and one fitting operation. That cost is justified by a
stable, representation-neutral estimator boundary that avoids dependencies on
workflow internals.

FitLins integration moves ahead of standalone BIDS convenience APIs. Real
application requirements can therefore shape those adapters without turning
Boldtailor into another BIDS-App. Later adaptive work has access to raw events,
fixed nuisance designs, and fold identities without forcing HRF optimization
through a completed-design interface or conflating it with GLMdenoise.
