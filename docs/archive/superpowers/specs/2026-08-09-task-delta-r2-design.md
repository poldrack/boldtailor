# Task-Attributable Delta R-Squared Design

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

**Amended:** 2026-08-11. Task delta R-squared is an ordinary least-squares
diagnostic for both the full and nuisance-only designs, independent of the
noise model used for coefficient inference. This replaces the earlier
separately estimated AR(1) comparison.

## Goal

Add an aggregate whole-brain display and derivative showing how much additional
variance the complete task model explains relative to a model containing only
the selected confounds, drift regressors, and constant.

The comparison is descriptive. It does not provide inferential significance or
multiple-comparison correction.

## Statistical Definition

The existing inferential model remains unchanged. For the descriptive
variance-partition diagnostic, the full and nuisance-only designs are each fit
with ordinary least squares to the same signals. For each run, the
nuisance-only design is built from the same:

- frame times;
- selected confounds;
- drift model, high-pass setting, and drift order;
- constant term.

The nuisance-only design contains no event-derived task regressors. It is fit to
the same immutable whole-brain signals as the complete model.

Aggregate full and nuisance R-squared are computed from residual and centered
total sums of squares pooled across runs. Both use OLS regardless of whether
the corresponding inferential model uses OLS or AR(1). The raw comparison is:

```text
raw_delta_r2 = complete_model_r2 - nuisance_only_r2
```

Nested OLS guarantees that the full-model SSE cannot exceed the nuisance-only
SSE in exact arithmetic. The displayed and published comparison retains a
zero-floor as a numerical-roundoff guard:

```text
delta_r2 = maximum(raw_delta_r2, 0)
```

Provenance records the raw minimum and the number of values affected by the
roundoff guard. A materially negative raw difference is an implementation or
numerical-stability failure, not an expected model outcome.

## Package Interface

Add:

```python
fit.task_delta_r2(
    data: AnalysisData,
    model: ModelSpec,
    full_result: AnalysisResult,
) -> TaskDeltaR2Result
```

`TaskDeltaR2Result` is immutable and exposes:

- `full_r2`;
- `nuisance_r2`;
- `raw_delta_r2`;
- `delta_r2`;
- `negative_voxel_count`;
- `raw_min`;
- nuisance-only design matrices; and
- structured provenance.

All returned arrays are owned, finite, immutable float64 arrays. Design-matrix
accessors return defensive copies.

The supplied complete-model result establishes parent identity and dimensions;
it is not used as the diagnostic full-model R-squared when the inferential fit
uses AR(1). The operation refits the already compiled full and nuisance designs
with OLS. It validates that the supplied result belongs to the data and model
by comparing analysis fingerprints and feature dimensions. The comparison
requires fingerprintable data and model provenance; it fails clearly when a
reliable parent analysis identity is unavailable.

## Nuisance Design and Fit

The design layer gains a focused nuisance-design compiler. It calls Nilearn's
design-matrix builder without events while preserving the complete model's
confound and drift options. It does not infer task columns by string matching
and does not mutate or filter the complete design matrix.

The fit layer reuses the existing rank, residual-degrees-of-freedom,
prediction, sums-of-squares, and aggregate-R-squared behavior while forcing
`noise_model="ols"` for both sides of this diagnostic. No feature chunking,
resampling, truncation, smoothing, detrending, temporal filtering, or
standardization is introduced.

## Logging and Provenance

The operation emits structured started, completed, and failed events. Its
provenance identifies the complete-model analysis as its parent and records:

- the nuisance-only design columns for every run;
- confound, drift, high-pass, and drift-order settings;
- diagnostic noise model `ols` and the separate inferential noise-model
  setting;
- the absence of event regressors;
- the raw-difference definition;
- the zero-floor policy `numerical_roundoff_guard` and tolerance `1e-12`;
- raw minimum, numerical-floor count, mean delta, and maximum delta; and
- the comparison execution and analysis identifiers.

No absolute BIDS root or protected input path is added to shareable metadata.

## Notebook Display

The complete-model aggregate R-squared display remains. Immediately after it,
the notebook adds an aggregate map titled:

```text
Task-attributable delta R-squared (OLS diagnostic)
```

The plot uses the common whole-brain masker, a sequential colormap, `vmin=0`,
`threshold=None`, a colorbar, and a non-symmetric colorbar. A compact displayed
summary includes:

- mean clipped delta R-squared;
- maximum clipped delta R-squared;
- raw minimum delta R-squared; and
- numerical-floor count.

The existing disclosure continues to state that maps are descriptive and
unthresholded. It is extended to identify delta R-squared as an OLS descriptive
variance-partition diagnostic rather than inferential evidence. AR(1), when
selected, continues to govern contrast inference rather than this diagnostic.

## Derivative Publication

Publish one additional aggregate NIfTI:

```text
images/sub-<subject>_task-<task>_space-<space>_res-<resolution>_desc-taskDelta_stat-r2_statmap.nii.gz
```

The payload contains the clipped aggregate delta R-squared values reconstructed
through the already fitted common-mask `NiftiMasker`. It shares the common mask
shape and affine.

The image set increases from 10 to 11. The new image participates in the same
deterministic gzip serialization and image manifest, including exact relative
path, `application/gzip` media type, byte size, and SHA-256 digest.

Shareable configuration and provenance include the comparison definition,
nuisance-model settings, numerical-floor policy, and diagnostics. Temporary
publication remains the default; optional persistent publication retains all
existing path, overwrite, symlink, source-protection, and transaction safety.

## Validation and Errors

Reject:

- a full result whose analysis fingerprint does not match the supplied data
  and complete model;
- a result or signal set with inconsistent run or feature dimensions;
- nonfinite fit outputs;
- nuisance designs with duplicate columns or nonfinite values;
- nuisance fits without positive residual degrees of freedom; and
- a raw OLS difference below `-1e-12`; and
- comparison requests without reliable parent analysis identity.

Failures emit a structured failed event before propagating a clear exception.

## Testing and Delivery

Development follows committed RED-GREEN-Refactor cycles. Tests precede every
production behavior change.

Coverage includes:

- nuisance designs retain confounds, drift terms, and constant while excluding
  every task regressor;
- nested OLS fits yield full-model aggregate R-squared greater than or equal to
  nuisance-only aggregate R-squared within a strict floating-point tolerance;
- an AR(1) inferential model still uses OLS for both diagnostic fits;
- the numerical floor equals `maximum(raw_delta_r2, 0)`, diagnostics match the
  raw values, and a difference below `-1e-12` is rejected;
- mismatched full results and invalid dimensions are rejected;
- result arrays are owned immutable float64 values and design accessors are
  defensive;
- structured logs and provenance contain parent identity, model settings,
  design columns, definition, clipping policy, and diagnostics;
- the notebook makes exactly three contrast z-map calls, one complete-model
  aggregate R-squared call, and one delta-R-squared call;
- notebook summaries and disclosures describe OLS, the numerical floor, and
  descriptive use;
- exactly 11 deterministic image artifacts round-trip with correct geometry,
  values, metadata, sizes, and hashes;
- both supported synthetic notebook working directories pass; and
- the real `ses-06`/`ses-08` execution succeeds without creating a persistent
  derivative.

The existing assumption of at least 32 GiB RAM remains. No feature chunking is
added.

## Out of Scope

- Per-session delta-R-squared maps.
- Publishing a separate nuisance-only R-squared image.
- Clipping or changing the existing complete-model R-squared image.
- Inferential tests or thresholds for delta R-squared.
- Generalizing `ModelSpec` to support arbitrary contrast-free or event-free
  analyses.
