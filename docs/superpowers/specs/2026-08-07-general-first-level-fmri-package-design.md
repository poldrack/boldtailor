# General First-Level fMRI Package Design

**Status:** Approved design

**Date:** 2026-08-07

## Context

The existing GLMsingle Python package began as a close port of MATLAB code. Its
legacy implementation mixed data loading, design construction, estimation,
diagnostics, and persistence in large functions. The current GLMsingle 2.0 work
has repaired many correctness problems and established valuable tests, numerical
oracles, and modular components. It has also clarified that the desired project
is broader than a GLMsingle revision.

The new project will be a general first-level fMRI modeling package. It will
support conventional condition-level models, single-trial models, voxelwise HRF
selection, alternative denoising approaches, and the GLMsingle workflow. It will
be centered on BIDS data and model specifications while retaining an array API.

This scope justifies a new package and repository identity. GLMsingle will be a
bundled analysis recipe, not the organizing abstraction of the package.

## Goals

- Analyze one subject/task/space unit containing one or more runs and sessions.
- Load preprocessed BIDS Derivatives and associate them with raw BIDS event
  files and derivative confound tables.
- Accept equivalent run-wise arrays without requiring BIDS.
- Support volumetric NIfTI and CIFTI grayordinate inputs as first-class use
  cases through a representation-neutral numerical core.
- Accept models through both a small Python API and BIDS Stats Models JSON.
- Use Nilearn as the required conventional GLM engine.
- Produce conventional contrast-effect maps, single-trial estimates, voxelwise
  HRF selections, and explicit quality-control maps.
- Include the scientific operations of GLMsingle without retaining its
  monolithic API or internal architecture.
- Keep the public architecture small, explicit, and maintainable.

## Non-goals for the first release

- Raw-data preprocessing such as motion correction, registration, distortion
  correction, or surface reconstruction.
- Dataset-wide participant scheduling or workflow execution.
- Group-level statistical inference.
- Resampled or selective inference after adaptive HRF selection.
- A plugin registry, entry-point ecosystem, generic stage graph, or support for
  multiple conventional GLM backends.
- Mixed NIfTI and CIFTI runs within one analysis.
- A custom HDF5 or Zarr persistence format.

## Approaches considered

### Selected: a new Nilearn-centered package

The package owns BIDS discovery, analysis and model semantics, adaptive modeling
workflows, typed results, provenance, and derivative output. Nilearn owns
conventional design construction, OLS/AR fitting, and contrasts. Selected code,
tests, and numerical fixtures may be ported from GLMsingle when they meet the
new boundaries.

### Rejected: generalize the current GLMsingle repository

This would accelerate early development but preserve GLMsingle terminology,
compatibility constraints, persistence history, and legacy numerical kernels as
architectural commitments. A general first-level package would continue to feel
like an expanded GLMsingle.

### Rejected: a backend-agnostic plugin framework

A multi-backend framework would introduce speculative interfaces for noise
models, contrasts, results, and failures before a second backend exists. The
first release will use Nilearn directly and add abstractions only when multiple
real implementations demonstrate a stable common boundary.

## Repository transition

This design document is the final decision record in the GLMsingle repository.
After human review, development moves to a new repository before any package
scaffolding, implementation planning, or source code is created. The approved
design is copied into the new repository as its first substantive commit.

The existing GLMsingle repository remains a reference implementation and a
source of independently validated tests, fixtures, and scientific behavior. New
development must not share a source tree with the compatibility package. Code is
ported selectively with attribution and new-package tests, rather than copied
wholesale.

## Public architecture

The public model is deliberately limited to three important immutable types:

- `AnalysisData`: compatible runs, events, confounds, timing, provenance, and
  optional spatial reconstruction metadata.
- `ModelSpec`: the full design, HRF, nuisance, temporal-noise, estimation, and
  contrast specification.
- `AnalysisResult`: estimates, contrast effects, diagnostics, adaptive-model
  outputs, and provenance.

Users perform four conceptual operations:

```python
data = from_bids(...)          # or from_arrays(...)
model = model_from_json(...)   # or ModelSpec(...)
result = fit(data, model)
write_derivatives(result, ...)
```

Loading and writing are explicit. `fit()` performs no implicit file I/O and has
no hidden fitted state. Run containers, compiled designs, spatial helpers,
Nilearn regression objects, and PyBIDS objects remain internal.

There are no public base classes, factories, registries, or backend interfaces.
Built-in scientific choices use small frozen configuration values. Narrowly
specified callables are accepted only where genuine customization is already a
requirement, such as a custom HRF function or denoising function.

## Dependency boundaries

### PyBIDS

PyBIDS owns BIDS indexing, entity parsing, scoped raw/derivative queries,
metadata inheritance, file associations, and BIDS Stats Models parsing. A thin
package adapter issues precise PyBIDS queries, validates that the selected files
form one coherent analysis, and returns `AnalysisData` without leaking PyBIDS
types into the scientific core.

The adapter requires unambiguous BOLD, event, and confound matches and validates
run ordering, TRs, scan counts, spaces, feature layouts, and source provenance.
Unsupported Stats Models nodes or transformations raise explicit errors rather
than being ignored.

### Nilearn

Nilearn is a required dependency and the sole conventional GLM backend in the
first release. Standard design construction uses
`make_first_level_design_matrix`. Array-level fitting uses `run_glm`, and
contrast computation uses Nilearn's contrast functions.

The core uses the array-level API because it accepts `time x features` data.
Nilearn's image-oriented `FirstLevelModel` is a numerical validation oracle and
may support convenience examples, but it is not the internal architecture.

### NiBabel

NiBabel owns NIfTI and CIFTI file I/O. The package preserves spatial metadata
and reconstructs results but does not implement either file format.

## Representation-neutral data

All scientific code receives a sequence of arrays with shape
`time x features`. An internal spatial record preserves the information needed
to reconstruct maps:

- NIfTI inputs preserve the mask, spatial shape, affine, and header geometry.
- CIFTI inputs preserve the `SeriesAxis` and `BrainModelAxis`.
- Array inputs may omit spatial metadata and receive array results only.

CIFTI runs must have identical grayordinate order, structures, density, and
volumetric geometry. Their `SeriesAxis` timing must agree with BIDS metadata.
NIfTI runs must have compatible shapes and affines after masking. An analysis
cannot mix NIfTI and CIFTI runs.

## Model specification

The Python API and supported first-level BIDS Stats Models nodes compile into
the same `ModelSpec`. It contains:

- event transformations and condition definitions;
- fixed, FIR, custom, or cross-validated HRF configuration;
- drift and explicitly selected confound terms;
- optional denoising configuration;
- OLS or AR(N) temporal-noise configuration;
- conventional or GLMsingle estimation configuration; and
- named t or F contrasts with semantic regressor references.

Compiled run designs are internal and retain semantic column identities.
Contrast definitions never rely on undocumented positional ordering.

## Conventional scientific flow

For a conventional model, `fit()`:

1. validates the analysis and model;
2. compiles one design matrix per run with Nilearn;
3. adds selected confounds, drift terms, and any generated nuisance regressors;
4. fits each run with Nilearn using OLS or AR(N);
5. computes named contrasts; and
6. combines corresponding run/session contrast effects with fixed effects.

The result retains contrast effects, variances, available Nilearn statistics,
design provenance, and residual diagnostics. Conventional inferential outputs
are inexpensive to retain, but group inference is outside the package scope.

## Cross-validated voxelwise HRFs

A conventional `ModelSpec` may use a cross-validated HRF library. This is a
concrete HRF configuration, not a generic workflow graph.

For leave-one-session-out selection, `fit()`:

1. constructs folds from BIDS session entities or explicit array metadata;
2. evaluates every candidate HRF using training-only fits and held-out
   predictions;
3. accumulates held-out residual and total sums of squares across folds;
4. selects the best candidate independently for every feature;
5. groups features by the selected HRF;
6. refits all sessions for each feature group with the corresponding design;
7. computes named contrast effects for each group; and
8. reassembles whole-brain or whole-grayordinate outputs.

The primary quality metric is:

```text
CV R2 = 1 - sum(held-out SSE) / sum(held-out SST)
```

Sums are combined before computing R-squared; session R-squared values are not
naively averaged. Negative values are retained. The result also includes the
selected-HRF map, candidate-wise CV scores, fold diagnostics, and clearly
labeled in-sample R-squared.

The full-data contrast maps are point estimates intended for downstream
analysis. The adaptive workflow does not promise post-selection p-values or
z-maps and does not propagate HRF-selection uncertainty.

## GLMsingle recipe

GLMsingle is a bundled model recipe that reuses the same internal primitives:

1. fit a baseline design;
2. select voxelwise HRFs with the shared HRF-library implementation;
3. estimate GLMdenoise-style nuisance regressors;
4. refit with the selected HRFs and nuisance regressors; and
5. perform fractional-ridge single-trial estimation.

The recipe returns trial estimates and typed HRF-selection, denoising, and
regularization diagnostics. Fractional-ridge results do not fabricate classical
standard errors or statistical maps.

The implementation is a small orchestration function over shared operations,
not a separate framework or a general stage engine.

## Outputs

`AnalysisResult` always contains representation-neutral arrays. Explicit output
adapters reconstruct spatial artifacts when the input contains spatial metadata.

### NIfTI

- One 3D effect image per named contrast.
- Separate 3D cross-validated and in-sample R-squared maps.
- A selected-HRF index map.
- Optional candidate-score and beta-series 4D images.
- A TSV mapping beta-series volumes to trials.

### CIFTI

- Contrast effects and R-squared maps as `dscalar.nii`.
- Candidate HRF scores as multi-map `dscalar.nii`.
- Selected HRFs as `dlabel.nii` with library-member labels.
- Trial estimates as `dscalar.nii` with a `ScalarAxis` identifying trials.
- The same trial TSV used for NIfTI outputs.

A trial axis is not time, so trial estimates are not written as
`dtseries.nii`. Every CIFTI output preserves the input `BrainModelAxis`
exactly.

Both spatial formats include run design matrices as TSV and JSON sidecars with
units, model configuration, cross-validation scheme, source files, software
versions, and provenance. The derivative includes `dataset_description.json`
with `GeneratedBy` metadata.

The first release does not introduce a custom result container. Array users
receive arrays and may save them independently.

## Errors and publication safety

Validation happens before expensive fitting and again at each boundary. Errors
name the subject/session/run, source path, model term, or contrast that failed.
Missing required confounds, inconsistent timing or geometry, non-finite data,
empty feature groups, failed HRF candidates, and unestimable contrasts are
explicit failures.

Rank deficiency is recorded and warned about; a requested contrast that cannot
be estimated is an error. Negative cross-validated R-squared is ordinary output.
There is no silent fallback between noise models, HRFs, denoisers, or estimation
methods.

Writing checks all collisions before publication, writes through temporary
files, and refuses replacement unless `overwrite=True`. A failed write cannot
leave a partially replaced output file.

## Testing strategy

Development follows strict RED-GREEN-Refactor cycles. Every behavior begins as
a failing pytest test committed before its implementation. Tests use functions
and fixtures, and every package `__init__.py` remains empty.

### Pure numerical tests

Small synthetic arrays with known coefficients, contrasts, HRFs, and
predictions verify design compilation, R-squared, fixed effects, and trial
estimates.

### Nilearn parity tests

Conventional results are compared directly with Nilearn design matrices,
`run_glm`, contrast functions, and `FirstLevelModel` reference outputs.

### Cross-validation tests

Synthetic multi-session data verify voxelwise HRF recovery, training/held-out
separation, aggregate SSE/SST R-squared, negative R-squared preservation, and
full-data contrast refitting. Leakage tests alter held-out data and confirm that
it cannot affect the corresponding fold's training decisions.

### Representation tests

Equivalent array, NIfTI, and CIFTI inputs must produce equivalent numerical
results. NIfTI round trips preserve shape, affine, and header geometry. CIFTI
round trips preserve the `BrainModelAxis`, grayordinate order, density, scalar
names, and labels exactly.

### Workflow and failure tests

Minimal generated BIDS fixtures cover PyBIDS discovery across sessions, raw
events, derivative confounds, ambiguity, inconsistent metadata, collisions, and
interrupted writes. GLMsingle components are checked against trusted synthetic
results from the existing package.

The fast suite is offline. Larger real-data validation and 91k-grayordinate
memory/performance checks are separate integration tests.

## Implementation sequence

The project is delivered in independently useful phases:

1. array-based conventional Nilearn GLM;
2. Python and BIDS Stats Models compilation;
3. PyBIDS loading plus NIfTI and CIFTI adapters and writers;
4. cross-validated voxelwise HRF selection; and
5. GLMsingle denoising and fractional-ridge recipe.

Each phase is fully tested before the next begins. Because the phases are too
large for one implementation plan, each receives its own focused specification
and plan. After this design is reviewed and copied to the new repository, the
first planning session covers only phase 1.

## References

- [BIDS Stats Models](https://bids-standard.github.io/stats-models/)
- [PyBIDS `BIDSLayout`](https://bids-standard.github.io/pybids/generated/bids.layout.BIDSLayout.html)
- [Nilearn first-level GLM API](https://nilearn.github.io/stable/glm/first_level_model.html)
- [NiBabel CIFTI-2 axes](https://nipy.org/nibabel/reference/nibabel.cifti2.html)
- [fMRIPrep CIFTI outputs](https://fmriprep.org/en/25.1.1/outputs.html)
