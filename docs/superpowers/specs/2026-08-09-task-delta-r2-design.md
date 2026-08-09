# Task-Attributable Delta R-Squared Design

## Goal

Add an aggregate whole-brain display and derivative showing how much additional
variance the complete task model explains relative to a model containing only
the selected confounds, drift regressors, and constant.

The comparison is descriptive. It does not provide inferential significance or
multiple-comparison correction.

## Statistical Definition

The existing complete model remains unchanged. For each run, a nuisance-only
design is built from the same:

- frame times;
- selected confounds;
- drift model, high-pass setting, and drift order;
- constant term; and
- AR(1) noise-model setting.

The nuisance-only design contains no event-derived task regressors. It is fit to
the same immutable whole-brain signals as the complete model.

Aggregate nuisance R-squared is computed from residual and centered total sums
of squares pooled across runs, matching the existing aggregate complete-model
R-squared calculation. The raw comparison is:

```text
raw_delta_r2 = complete_model_r2 - nuisance_only_r2
```

The displayed and published comparison is clipped at zero:

```text
delta_r2 = maximum(raw_delta_r2, 0)
```

Separate AR(1) estimation can produce negative raw differences. Clipping is an
explicit presentation policy rather than a claim that these values did not
occur. Provenance records the raw minimum and the number of clipped voxels.

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

The operation reuses the supplied complete-model result instead of refitting
the complete model. It validates that the result belongs to the supplied data
and model by comparing analysis fingerprints and feature dimensions. The
comparison requires fingerprintable data and model provenance; it fails
clearly when a reliable parent analysis identity is unavailable.

## Nuisance Design and Fit

The design layer gains a focused nuisance-design compiler. It calls Nilearn's
design-matrix builder without events while preserving the complete model's
confound and drift options. It does not infer task columns by string matching
and does not mutate or filter the complete design matrix.

The fit layer reuses the existing rank, residual-degrees-of-freedom,
prediction, sums-of-squares, and aggregate-R-squared behavior. No feature
chunking, resampling, truncation, smoothing, detrending, temporal filtering, or
standardization is introduced.

## Logging and Provenance

The operation emits structured started, completed, and failed events. Its
provenance identifies the complete-model analysis as its parent and records:

- the nuisance-only design columns for every run;
- confound, drift, high-pass, drift-order, and AR(1) settings;
- the absence of event regressors;
- the raw-difference definition;
- the `clip_below_zero` presentation policy;
- raw minimum, clipped-voxel count, mean clipped delta, and maximum clipped
  delta; and
- the comparison execution and analysis identifiers.

No absolute BIDS root or protected input path is added to shareable metadata.

## Notebook Display

The complete-model aggregate R-squared display remains. Immediately after it,
the notebook adds an aggregate map titled:

```text
Task-attributable delta R-squared (clipped at zero)
```

The plot uses the common whole-brain masker, a sequential colormap, `vmin=0`,
`threshold=None`, a colorbar, and a non-symmetric colorbar. A compact displayed
summary includes:

- mean clipped delta R-squared;
- maximum clipped delta R-squared;
- raw minimum delta R-squared; and
- clipped-voxel count.

The existing disclosure continues to state that maps are descriptive and
unthresholded. It is extended to identify delta R-squared as descriptive
variance accounting rather than inferential evidence.

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
nuisance-model settings, clipping policy, and diagnostics. Temporary
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
- comparison requests without reliable parent analysis identity.

Failures emit a structured failed event before propagating a clear exception.

## Testing and Delivery

Development follows committed RED-GREEN-Refactor cycles. Tests precede every
production behavior change.

Coverage includes:

- nuisance designs retain confounds, drift terms, and constant while excluding
  every task regressor;
- synthetic task signal yields higher complete-model than nuisance-only
  aggregate R-squared;
- clipping equals `maximum(raw_delta_r2, 0)` and diagnostics match the raw
  values;
- mismatched full results and invalid dimensions are rejected;
- result arrays are owned immutable float64 values and design accessors are
  defensive;
- structured logs and provenance contain parent identity, model settings,
  design columns, definition, clipping policy, and diagnostics;
- the notebook makes exactly three contrast z-map calls, one complete-model
  aggregate R-squared call, and one delta-R-squared call;
- notebook summaries and disclosures describe clipping and descriptive use;
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
