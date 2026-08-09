# Whole-Brain Stop-Signal Notebook Design

## Purpose

Replace the stop-signal notebook's illustrative single-ROI workflow with a
two-session whole-brain analysis limited to the intersection of the two
fMRIPrep brain masks. The notebook remains an executable demonstration of the
existing array API, run-specific designs, AR(1) fitting, contrasts, diagnostics,
logging, provenance, and derivative publication.

## Scope

The change covers `examples/stop_signal_demo.py`,
`examples/stop_signal_demo.ipynb`, their focused tests, and the notebook link or
description in `README.md`. It does not add native NIfTI ingestion to the
Boldtailor package, change the public modeling API, add feature chunking, or
retain an optional ROI mode.

The implementation assumes a machine with at least 32 GiB of memory. The real
two-session common mask currently contains about 289,425 voxels; the immutable
float64 signal arrays alone require about 1.5 GiB. The notebook reports its
voxel count and a signal-memory estimate before fitting.

## Spatial data flow

1. Discover the same exact raw events, preprocessed BOLD, brain-mask, and
   confounds files for sessions `ses-06` and `ses-08`.
2. Load both three-dimensional fMRIPrep masks and require matching shapes and
   affines.
3. Compute their voxelwise logical intersection and reject an empty result.
4. Construct a single `nilearn.maskers.NiftiMasker` with the intersection mask.
   The masker explicitly uses `standardize=False`, `detrend=False`, no spatial
   smoothing, no low- or high-pass filter, and no report generation. These
   settings prevent the extraction layer from silently preprocessing the
   fMRIPrep signals.
5. Fit the masker once and use it to transform both BOLD images into
   `time x common-brain-voxels` arrays. Require each BOLD image to match the
   common mask's spatial geometry and require both transformed arrays to have
   the common-mask feature count.
6. Pass the arrays, run-specific events, confounds, frame times, sources, and
   provenance metadata to `boldtailor.data.from_arrays` without chunking.
7. Fit the model once across both runs and use the fitted masker's
   `inverse_transform` method to reconstruct whole-brain result maps.

The helper replaces ROI-specific functions and names with common-mask and
whole-brain equivalents. There is no sphere center, radius, ROI bounding box,
or ROI-only reconstruction path.

## Model and contrasts

Each session continues to receive its own design matrix derived from the trial
types observed in that run. The model retains AR(1) noise and the existing HRF,
drift, confound, oversampling, and onset settings.

The named contrasts are:

- `successful_inhibition`: `stop_success - stop_failure`
- `stop_vs_go`: `(stop_success + stop_failure) - go_success`
- `go_success_vs_baseline`: `go_success`

The final expression tests the modeled `go_success` response against the
implicit baseline represented by the design intercept.

## Notebook presentation

The notebook displays compact, privacy-safe information rather than voxelwise
tables or absolute paths:

- discovered session labels and source basenames;
- mask shape, common voxel count, scan counts, transformed array shapes, and
  estimated signal-array memory;
- one run-specific design-matrix plot per session;
- contrast summaries for effect, variance, t statistic, z statistic, and
  directional one-sided p values;
- per-run and aggregate R-squared summaries;
- one unthresholded whole-brain z-statistic map for each named contrast;
- one unthresholded aggregate whole-brain R-squared map;
- selected diagnostics, lifecycle log fields, fingerprints, relative
  provenance paths, and publication summaries.

The z maps are deliberately unthresholded. The notebook states that the plots
are descriptive and do not imply a multiple-comparison-corrected inference.

## Derivative artifacts

Temporary publication remains the default. Persistent publication remains an
explicit option targeting exactly `<dataset>/derivatives/boldtailor`, with the
existing traversal, overlap, and symlink protections.

The artifact set retains dataset description, structured logs, provenance
projection, compact TSV summaries, and shareable configuration. ROI-specific
labels and filenames are replaced with whole-brain terminology. It adds
deterministic gzip-compressed NIfTI payloads for:

- the derived common brain mask;
- effect and z-statistic maps for all three contrasts;
- R-squared maps for each run and the aggregate fit.

Contrast filenames use stable alphanumeric labels
`successfulInhibition`, `stopVsGo`, and `goSuccessVsBaseline`. NIfTI payloads
use the explicit media type `application/gzip`; TSV and JSON media types remain
explicit. Image payloads must round-trip through nibabel with the common mask's
shape and affine. Because the existing `Artifact` API intentionally contains
only a relative path and immutable bytes, the example also publishes a
deterministic image-manifest TSV with each NIfTI artifact's relative path,
`application/gzip` media type, byte size, and SHA-256 digest.

## Provenance and logging

Both source brain masks remain first-class input sources. Analysis provenance
also records:

- the `intersection` mask-combination rule;
- common mask shape, affine, and voxel count;
- all explicit `NiftiMasker` preprocessing settings;
- transformed signal shapes and dtype;
- the 32 GiB execution assumption and estimated signal-array memory;
- the three contrast definitions;
- every published image's relative path, media type, size, and digest in the
  image-manifest derivative that is included in the transactional artifact set.

Notebook displays and shareable configuration must not expose the absolute
dataset root. Normalization, fit, publication, warning, and failure events
continue to use the existing structured logging and execution identifiers.

## Validation and errors

The helper fails with specific errors for mismatched mask geometry, empty mask
intersection, BOLD/mask geometry mismatch, scan-count disagreement, unexpected
transformed dimensionality, or inconsistent feature counts. Existing event,
timing, confound, source-path, output-overlap, and publication validation is
preserved.

No automatic downsampling, masking fallback, voxel truncation, or feature
chunking is permitted. A resource failure is surfaced rather than silently
changing the analysis.

## Testing and verification

Development follows strict RED-GREEN-Refactor history: behavioral pytest
changes are committed and observed failing before implementation is written.
Tests use small real NIfTI fixtures and functions rather than test classes.

Focused tests cover:

- common-mask intersection and geometry validation;
- explicit no-preprocessing masker configuration;
- identical feature ordering across runs;
- whole-brain inverse transformation;
- all three contrast definitions and result accessors;
- deterministic, readable NIfTI artifacts with correct geometry and media
  types;
- whole-brain naming with no ROI artifacts;
- privacy-safe configuration and complete provenance;
- temporary and persistent publication safety;
- complete notebook execution from both repository-root and notebook-directory
  kernels.

Final verification executes the notebook against real sessions `ses-06` and
`ses-08`, confirms the three contrast maps and R-squared map, and confirms that
temporary publication does not create
`<dataset>/derivatives/boldtailor`. It also runs focused and full pytest suites,
the full suite under `-W error`, Black, lock validation, package build,
repository contracts, the empty-initializer check, diff validation, and a clean
feature-worktree status check.
