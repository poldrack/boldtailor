# Comparing Boldtailor and GLMsingle

Both Boldtailor and GLMsingle estimate per-trial beta amplitudes and select a
location-specific HRF from a library of candidates. They differ primarily in how
they score candidate HRFs, construct nuisance regressors, and tune
regularization.

The table below compares Boldtailor's workflow with the published GLMsingle
method ([Prince et al., 2022](https://elifesciences.org/articles/77599)). Empirical
benchmarks in the GLMsingle paper evaluate GLMsingle on NSD, BOLD5000, and
StudyForrest; no matched head-to-head benchmark between the two packages is
included in this repository.

## Methods at a glance

| Aspect | Boldtailor | Published GLMsingle workflow |
| --- | --- | --- |
| HRF library | Default `default_hrf_library()`: canonical SPM, 512 timing-space Sobol double-gamma candidates, and GLMsingle's 20 empirical HRFs; parameter grids and custom libraries supported | Default library of 20 empirically derived HRFs; custom libraries supported |
| Entry point | Python array API plus `boldtailor run` CLI for BIDS + fMRIPrep fsLR-91k CIFTI sessions | MATLAB or Python `GLMestimatesingletrial` (plus modular classes in recent checkouts) |
| HRF selection | Leave-one-run-out task-model time-series prediction (`method="cv"`) or restricted maximum likelihood over random trial effects (`method="reml"`) | Highest in-sample single-trial $R^2$ per voxel (not cross-validated; see [GLMsingle FAQ](https://glmsingle.readthedocs.io/en/latest/wiki.html#why-isn-t-the-hrf-selection-cross-validated)) |
| Confounds | Caller-supplied confounds (e.g., fMRIPrep motion, aCompCor, cosines, spikes) plus optional `select_denoising()` (GLMdenoise-style noise-pool PCs with a `pcstop` rule and AR(1)-prewhitened F-test significance gate) | Polynomial drift terms plus GLMdenoise PCs from an ON-OFF $R^2$ noise pool, tuned by cross-validation |
| Regularization | OLS, fixed normalized ridge (`ridge_alpha`), per-feature fractional ridge (`ridge_fraction`) tuned against fixed OLS targets, or shared-alpha CV | Per-voxel fractional ridge tuned by repeated-condition beta consistency |
| Tuning objective | HRF CV: held-out task time-series prediction. Ridge CV: held-out trial-encoding prediction (e.g., trial type and RT) | Cross-validated reproducibility of trial betas across repeated conditions |
| Trial estimates | One coefficient per event presentation, including repeats | One coefficient per event presentation, including repeats |

## Data-derived noise regressors

**Attribution.** The denoising procedure follows GLMsingle's GLMdenoise
stage (Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M.J.,
Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. *eLife*, 11, e77599.
[doi:10.7554/eLife.77599](https://doi.org/10.7554/eLife.77599)), with the
deviations listed below. It is an independent reimplementation;
GLMsingle (Copyright (c) 2021, Kendrick Kay) is distributed under the BSD
3-Clause License, reproduced in `src/boldtailor/_resources/GLMsingle-LICENSE.txt`.
GLMsingle's authors have not reviewed or endorsed Boldtailor.

`select_denoising()` is an opt-in array-level stage modeled on GLMsingle's
GLMdenoise procedure ([GLMsingle source](https://github.com/cvnlab/GLMsingle/blob/main/glmsingle/glmsingle.py) ·
[GLMdenoise](https://pmc.ncbi.nlm.nih.gov/articles/PMC3865440/)). Like GLMsingle,
it selects HRFs once on all runs and freezes them, defines the noise pool from
in-sample canonical-HRF ON-OFF $R^2$ using a two-component Gaussian-mixture tail
threshold (`findtailthreshold`), scores features above that threshold (or the top
100 with a defined HRF if none pass), extracts per-run temporal PCs from the
pool, summarizes each candidate PC count by the median score across scored
features, and applies the `pcstop=1.05` stopping rule across counts 0–10.

Key deviations from GLMsingle:

- **Baseline nuisance**: Uses each run's supplied confounds plus an intercept
  rather than polynomial drifts.
- **Count-scoring objective**: Scores candidate PC counts by leave-one-run-out
  task-model time-series prediction against a baseline-projected target fixed
  across counts, rather than repeated-stimulus beta consistency, so repeated
  conditions are not required.
- **Anatomy-agnostic pool**: Omits GLMsingle's mean-intensity `bright` threshold;
  callers mask non-brain features before constructing `AnalysisData`.
- **Mixture threshold details**: Evaluates `robustrange` on all finite ON-OFF
  $R^2$ values (in $[0, 1]$ rather than percent), uses a fixed `random_state=0`,
  emits a `RuntimeWarning` when the mixture fit does not converge, and raises
  `ValueError` if the tail posterior never crosses 0.5.
- **Significance gate**: By default (`significance_gate=True`,
  `gate_noise_model="ar1"`), a positive `pcstop` count $k^* > 0$ is retained only
  if adding the first $k^*$ PCs per run yields $p < 0.05$ on in-sample
  AR(1)-prewhitened nested F-tests in significantly more scoring features than
  expected by chance (one-sided binomial test at $\alpha = 0.05$); otherwise 0
  PCs are selected. Because `pcstop` is a relative rule with no minimum
  improvement floor, it can select positive PC counts from tiny random gains on
  pure noise; the gate prevents this on synthetic white-noise and independent
  AR(1) checks. Pass `significance_gate=False` for GLMsingle's `pcstop`-only
  rule. See the [user guide](user-guide.md#task-guided-denoising) for details and
  caveats.

## Repeated conditions vs. task-model transfer

GLMsingle's automated GLMdenoise and fractional-ridge tuning require repeated
presentations of the same experimental condition across runs, favoring settings
that maximize consistency across repeats (see the
[GLMsingle FAQ on repeated conditions](https://glmsingle.readthedocs.io/en/latest/wiki.html#in-glmsingle-the-glmdenoise-and-ridge-regression-rr-components-of-the-method-require-experimental-conditions-to-repeat-across-runs-how-should-i-think-about-whether-this-is-appropriate-for-my-experiment)).
Fixed-setting GLMsingle fits and in-sample HRF selection can still be used
without repeats.

Boldtailor does not require repeated stimuli:
- **HRF selection** assumes either that task-model amplitudes transfer across
  runs (`method="cv"`) or that trial-to-trial amplitude variations follow a
  zero-mean random-effects covariance (`method="reml"`).
- **Ridge selection** tunes shrinkage by predicting held-out trial betas from
  observed trial covariates (such as condition and response time) using
  within-run centered OLS encoding. Fractional ridge scores against fixed OLS
  validation betas at each location; shared-alpha ridge scores against
  candidate-regularized validation betas at a spatial percentile.

## Beta scaling and outputs

Both packages scale each event's predicted HRF response to unit peak (and store
library kernels at unit peak), so single-trial betas represent peak BOLD
response amplitude in native signal units regardless of TR, oversampling, or
selected HRF ([GLMsingle normalization FAQ](https://glmsingle.readthedocs.io/en/latest/wiki.html#if-the-hrf-changes-from-voxel-to-voxel-doesn-t-that-pose-some-interpretation-difficulties-or-confounding-issues)).
In Boldtailor, Nilearn derivative and FIR bases keep Nilearn's scaling, and
user-supplied callable kernels are used as given.

Boldtailor's `ridge_fraction` constrains the L2 norm ratio of nuisance-projected
trial coefficients in the raw trial basis without post-fit affine scaling or
offset; `ridge_alpha` penalizes unit-norm nuisance-projected trial columns. See
the archived [fractional ridge ablation](archive/validation/fractional-ridge-ablation-2026-09-28.md)
for the tradeoffs behind the raw-basis, fixed-target defaults.

## Notes on the modular GLMsingle repository

Recent revisions of the GLMsingle repository (checked at commit
`1de98a92e80754ff549c87b5c5b815b1aab8148c`) provide modular Python classes
(`HRFLibrarySelector`, `GLMDenoiser`, `FractionalRidgeEstimator`, and
`AnalysisDataset.from_events`) supporting custom HRF libraries, sub-TR event
onsets, event-specific durations, and HDF5 persistence. Boldtailor's
`expanded_hrf_library()` parameter grid was adapted from that repository's
`notebooks/component_generation.ipynb`.
