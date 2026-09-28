# Simple single-trial estimation and RT validation

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

Date: 2026-09-26

Status: Conversational scope approved; this document records the proposed implementation details for review.

## Purpose and scope

Generate one beta map per stimulus presentation for all 12 runs of
`sub-07/ses-nsd10`, using the existing NSD CIFTI example and a reusable
array-level boldtailor estimator. Compare single-trial OLS with an explicitly
chosen fixed ridge penalty. Use reaction time only as an external descriptive
check of preserved trial-to-trial variation.

This is a simple single-trial model, not a reproduction of the full GLMsingle
algorithm. The user explicitly rejected tuning that rewards agreement among
repeated stimuli and requested simplicity. This milestone therefore supersedes
the earlier GLMsingle roadmap for this requested feature. It does not implement
HRF-library selection, learned GLMdenoise components, fractional-ridge tuning,
GCV, empirical Bayes, or automatic selection of an estimator using RT.

## Model

Keep the canonical SPM HRF and oversampling of 50 from the existing example.
Construct one unit-amplitude task column per event using its recorded onset and
duration. Preserve event row order within the ordered input runs; never merge
presentations with the same image ID. Use explicit frame times, including the
fMRIPrep `StartTime` offset. Do not round events to acquisition volumes.

Use the existing selected nuisance regressors: 24 motion terms, six retained
combined-mask aCompCor components, fMRIPrep cosine terms, non-steady-state
indicators, and a run intercept. Fit independent coefficients in each run.
Keep signals and coefficients in their native input units. The initial method
uses ordinary least squares and Euclidean ridge; it does not estimate an AR
model or advertise inferential p-values.

RT is absent from the fitted design and regularization selection. Its old
parametric column is a linear combination of the new trial columns. Preserve
RT, image IDs, correctness, and other event metadata in a separate trial table.

For a run with task matrix X, nuisance matrix N, and signals Y, use
`Y = X B + N G + E`. OLS uses Nilearn's array-level OLS engine. For ridge:

1. Project both X and Y out of the same nuisance span using a rank-revealing
   orthonormal basis; do not penalize nuisance coefficients.
2. Scale each projected task column to unit Euclidean norm.
3. Solve `min ||Y_res - X_scaled W||² + alpha ||W||²` by SVD.
4. Convert W back to the original task-column units to obtain B.
5. Refit G to `Y - X B` for predictions and R².

`alpha=0` is the OLS reference. The initial NSD comparison will use `alpha=0.1`
as a predeclared demonstration setting, not as an optimized value. With
orthogonal normalized columns it shrinks coefficients by `1 / 1.1`; with
correlated columns shrinkage depends on the design. The command accepts one
optional positive `--ridge-alpha` and always produces the OLS reference.
Changing alpha is an explicit new analysis, never an RT-driven search.

Reject task designs with zero supported columns or rank deficiency after
nuisance projection, and require positive residual degrees of freedom. Allow
redundant nuisance columns by projecting their span. This first implementation
does not silently distribute an unidentifiable effect between overlapping,
identical trial columns. Preserve NaNs for undefined statistics and for trial
betas at features with zero signal variance in that run.

## Public boundary

Add `fit_single_trials(data, *, ridge_alpha=0.0, run_labels=None)` in
`boldtailor.single_trial`. It consumes the existing `AnalysisData`, reads only
event onset/duration for fitting, uses all caller-supplied confound columns,
and appends its own intercept. Reject a supplied confound named `constant`.
Labels default to `run-01`, `run-02`, etc., and must be unique.

Return a small immutable `SingleTrialResult` containing per-run trial-beta
arrays (trials × features), defensive copies of designs and the trial table,
per-run and pooled R² for the full and nuisance-only models, raw ΔR²,
conditioning/rank diagnostics, model settings, and canonical provenance.
Do not attach conventional contrast statistics to regularized estimates.

The trial table includes `trial_id`, zero-based global `trial_index`,
zero-based `run_index`, `run_label`, zero-based `event_index`, and all original
event columns. Generated IDs such as `run-01_trial-0001` are independent of RT
and stimulus identity. Reject collisions with reserved table column names.

Keep numerical code free of filesystem I/O. Keep CIFTI loading, reconstruction,
RT diagnostics, and plotting in the NSD example. Leave existing conventional
and prepared-fit behavior unchanged; all `__init__.py` files stay empty.

## R² and RT diagnostics

Compute pooled R² as `1 - sum(SSE_run) / sum(SST_run)`, with SST centered
separately in each run. Report the full-model prediction at the requested
alpha, the nuisance-only OLS reference, and their raw difference. Do not pass
ridge output through the existing nested-OLS delta API, which refits OLS.
These are in-sample measures, not predictive validation scores.

For RT diagnostics, exclude invalid/nonpositive RTs only from the diagnostic;
never drop those trials from estimation. At each feature, also exclude
nonfinite beta values and record the number of usable trials. Center RT and
betas within each run on the same retained trials, then pool their centered
cross-products to calculate a Pearson correlation. Require at least three
usable trials and nonzero centered sums of squares; otherwise return NaN.

Save pooled, odd-run, even-run, and per-run correlations for each estimator.
Odd/even membership follows the numeric BIDS run entity. Select up to five
cortical vertices by absolute odd-run OLS correlation, resolving ties by
grayordinate index; use those same vertices for even-run scatterplots for both
estimators. All/odd/even maps are descriptive; the selected even-run plots are
the independent check of vertex selection. Do not report uncorrected vertex
p-values, declare significance from the largest correlation, or optimize ridge
from this comparison. A weak RT association is a scientific observation, not
an implementation-test failure.

## Output and memory

Publish beneath `/Volumes/extdata1/NSD/BIDS/derivatives/boldtailor`, using
`desc-singletrialOLS` and `desc-singletrialRidge` filename labels. Preserve the
existing conventional results. Use a multi-map `dscalar.nii` per run and
estimator, with trial IDs on the ScalarAxis and the unchanged BrainModelAxis.
The trial table maps every CIFTI scalar back to its source event. Trials are
not evenly sampled times, so do not use `dtseries.nii` for betas.

Include model/design sidecars, trial metadata, R² maps, RT diagnostic maps,
even-run scatterplots, and provenance. Record alpha, column normalization,
HRF, timing, native beta units, selected confounds, excluded diagnostic counts,
and the fixed vertex-selection rule. Output namespace collisions must fail
before expensive fitting. Preserve the existing derivative dataset description.

Use the existing feature-block approach. Assemble float32 output arrays and
artifacts incrementally by run/model, avoiding a full session float64 beta
copy. The existing publisher requires the complete artifact payload set in
memory: this is an explicit remaining memory cost, not streaming publication.
Never split or fit trial groups independently within a run. Never write
derivatives around `publish_artifact_set`.

## Acceptance

- Numerical parity with independent OLS and augmented least-squares ridge
  calculations, including nuisance projection and restoration of beta units.
- A change to RT or image IDs does not change any fitted coefficient.
- Events, trial IDs, and CIFTI scalar order match exactly across all blocks.
- RT tests catch correlations caused only by between-run mean differences.
- Changing even-run observations cannot change the selected plot vertices.
- Simulations include real trial variation and AR-correlated noise; they test
  implementation and quantify sensitivity without assuming ridge must win.
- All project tests pass; all NSD output axes, counts, and a sampled numerical
  reference calculation pass independently.

## Context

The broader architecture is documented in
`2026-08-07-general-first-level-fmri-package-design.md` and
`2026-08-11-fitlins-boldtailor-interoperability-design.md` in this directory.
The narrow new result type and optional ridge operation are additions; this
milestone does not activate their adaptive-model roadmap.

RT is a plausible check, not a guaranteed NSD effect:
[Yarkoni et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2622763/).
Independent vertex selection follows the concern described by
[Kriegeskorte et al. (2009)](https://www.nature.com/articles/nn.2303).
