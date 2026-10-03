# Comparing Boldtailor and GLMsingle

Both packages estimate a separate beta for each stimulus presentation and can
use a different HRF at each brain location. The main differences are how they
choose HRFs, handle nuisance variation, and set regularization.

The table describes Boldtailor's current single-trial workflow alongside the
published GLMsingle method. GLMsingle combines HRF-library selection,
data-derived nuisance regressors, and voxelwise fractional ridge; its paper
reports gains in beta reliability and downstream analyses on NSD, BOLD5000,
and StudyForrest. Those results are evidence for GLMsingle, not a comparison
against Boldtailor. [Prince et al. (2022)](https://elifesciences.org/articles/77599)

## Methods at a glance

| Choice | Boldtailor | Published GLMsingle workflow |
| --- | --- | --- |
| HRF library | NSD notebooks use canonical SPM plus 512 Sobol-sampled double-gamma candidates; parameter grids and custom libraries also supported | Default library of 20 empirically derived HRFs; custom libraries supported |
| HRF selection | Predict each omitted run using task-model amplitudes learned from other runs; pool prediction errors to choose one HRF per location | Fit single-trial models with each HRF and choose the highest in-sample R² per voxel |
| Confounds | Caller-supplied regressors; the NSD example uses 24 motion columns, six aCompCor components, cosines, and non-steady-state indicators | Polynomial drift terms and GLMdenoise PCs derived from a noise pool; number of PCs selected by cross-validation |
| Regularization | OLS, fixed ridge, or fractional ridge selected separately per grayordinate by trial-encoding prediction; shared-alpha CV also available | Fractional ridge selected separately per voxel by cross-validation |
| Cross-validation target | HRFs: nuisance-adjusted task-model time-series prediction. Fractional ridge: fixed OLS trial betas. Shared-alpha ridge: candidate-regularized trial betas | Reproducibility of beta estimates for repeated conditions, to choose denoising and ridge settings |
| Trial estimates | One coefficient per presentation, including repeats | One coefficient per presentation, including repeats |

GLMsingle's HRF selection itself does **not** use the repeated-condition
cross-validation criterion. This distinction is explicit in the
[GLMsingle FAQ](https://glmsingle.readthedocs.io/en/latest/wiki.html#why-isn-t-the-hrf-selection-cross-validated).
Boldtailor's selection and final fitting procedures are described in the
[user guide](user-guide.md#selecting-an-hrf-for-each-location).

## What repeated conditions mean

GLMsingle uses repeated conditions to choose denoising and ridge settings; it
does not force their final trial betas to be equal. Its tuning objective favors
the component of a response that reproduces across presentations. The authors
discuss the potential to remove trial-varying effects in the
[FAQ on repeated conditions](https://glmsingle.readthedocs.io/en/latest/wiki.html#in-glmsingle-the-glmdenoise-and-ridge-regression-rr-components-of-the-method-require-experimental-conditions-to-repeat-across-runs-how-should-i-think-about-whether-this-is-appropriate-for-my-experiment).

Boldtailor does not use image identity or repeated-condition beta agreement to
select an HRF. It assumes that each location's task-model response transfers
across runs. Final fits then estimate individual trial amplitudes, optionally
with fixed or CV-selected ridge. A changing mean response across runs, or a weak mean response,
can make this HRF-selection objective less informative. Avoiding repeat-based
tuning does not guarantee preservation of every trial-level effect: nuisance
regression and ridge still affect the estimates.

For ridge selection, Boldtailor fits a trial-level encoding model on training
runs and predicts omitted-run betas. The NSD model uses trial type and RT, with
run-specific training intercepts and within-run centered scoring by default.
It selects each grayordinate's fraction using its highest pooled held-out R²
against fixed OLS beta targets. The shared-alpha workflow uses the 90th percentile
across grayordinates and scores candidate-regularized targets. Neither objective
establishes which penalty
recovers the most accurate unobserved trial amplitudes. Image repeats are not
needed, but the encoding relationship is assumed to transfer across runs.

The repeated-condition requirement belongs to GLMsingle's automatic tuning
steps. HRF selection and fixed-setting fits can be used without that tuning;
it would be misleading to say GLMsingle cannot be used without repeats.

## Outputs and interpretation

Boldtailor's NSD example writes CIFTI beta series, trial tables, full and
confound-only R², optimized-minus-canonical R², HRF parameter/peak-time maps,
and separate odd/even selections. Fixed-penalty workflows use RT only for a
descriptive check. In the notebook's CV mode, RT and trial type tune ridge;
separate odd-to-even and even-to-odd evaluations assess encoding prediction
after tuning on the training half. HRF selection remains based on the
task-model response (the mean response under the default `TaskModel()`). Its general API also fits condition
contrasts with OLS or AR(1), can apply selected voxelwise HRFs to those
conventional GLMs, and accepts externally prepared designs.

GLMsingle exposes beta estimates and diagnostics for its component stages,
including HRF choice, noise PCs, and ridge fractions.
[GLMsingle output reference](https://glmsingle.readthedocs.io/en/latest/python.html#returns)
Both packages can analyze surface data supplied as arrays; Boldtailor's NSD
example additionally handles CIFTI loading and spatially matched exports.

Raw beta magnitudes need care when comparing packages. Both packages scale
kernels to unit peak, so a beta is the peak response to a unit event; remaining
scale differences come from signal units and nuisance handling.
[GLMsingle normalization FAQ](https://glmsingle.readthedocs.io/en/latest/wiki.html#if-the-hrf-changes-from-voxel-to-voxel-doesn-t-that-pose-some-interpretation-difficulties-or-confounding-issues).

Boldtailor's `ridge_alpha` is a numerical penalty;
`ridge_fraction` uses a norm ratio in its nuisance-projected, raw trial
coefficient basis. It does not apply GLMsingle's optional post-fit scaling and offset.
Match signal scaling, timing, nuisance regressors,
kernel normalization, and score definitions before comparing results.

Boldtailor's winning selection-CV score is used for model selection. Its
separate odd-to-even prediction maps provide an independent evaluation of
mean-response prediction. Neither those scores nor RT correlations establish
that its single-trial betas outperform GLMsingle. No matched comparison of
the two packages has been run in this repository.

## The local GLMsingle checkout

The development checkout at `/Users/poldrack/Dropbox/code/GLMsingle` has a
modular Python API that goes beyond the interface described in the 2022 paper
and online documentation. This description is pinned to revision
`1de98a92e80754ff549c87b5c5b815b1aab8148c` for this comparison.

Its `HRFLibrarySelector` accepts an arbitrary sampled library and selects by
in-sample R². `GLMDenoiser` and `FractionalRidgeEstimator` expose the other
stages independently. `AnalysisDataset.from_events` supports fractional
onsets and event-specific durations, and typed HDF5 results can be saved and
loaded. Thus, neither custom HRFs nor precise event timing is unique to
Boldtailor. The expanded parameter grid was inspired by that checkout's
`notebooks/component_generation.ipynb`.

For implementation details in that checkout, see its `README.md`,
`src/glmsingle/hrf/estimators.py`, `src/glmsingle/denoise/glmdenoise.py`,
`src/glmsingle/ridge/estimator.py`, and `src/glmsingle/persistence.py`.
Boldtailor's corresponding entry points are listed in the
[API reference](api.md#single-trial-fits).

See the [fractional ablation](validation/fractional-ridge-ablation-2026-09-28.md)
for the evidence and tradeoffs behind the current raw-basis, fixed-target defaults.
