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
| HRF library | Default `default_hrf_library()`: canonical SPM, 512 timing-space Sobol double-gamma candidates, and GLMsingle's 20 empirical HRFs; parameter grids and custom libraries also supported | Default library of 20 empirically derived HRFs; custom libraries supported |
| Command line | `boldtailor run --bids-dir ... --subject ... --session ... --task ...` runs the whole NSD-style workflow and writes a BIDS derivative with an HTML report | A MATLAB or Python function call, `GLMestimatesingletrial`, with options passed as a struct or dict |
| HRF selection | Predict each omitted run using task-model amplitudes learned from other runs; pool prediction errors to choose one HRF per location | Fit single-trial models with each HRF and choose the highest in-sample R² per voxel |
| Confounds | Caller-supplied regressors; the NSD example uses 24 motion columns, six aCompCor components, cosines, and non-steady-state indicators. Optional `select_denoising()` follows GLMdenoise: run-wise PCs of an ON-OFF R² noise pool (Gaussian-mixture tail threshold), with the count chosen by held-out task prediction and the `pcstop` rule, then kept only if it passes an F-test significance gate (a Boldtailor addition) | Polynomial drift terms and GLMdenoise PCs derived from a noise pool; number of PCs selected by cross-validation |
| Regularization | OLS, fixed ridge, or fractional ridge selected separately per grayordinate by trial-encoding prediction; shared-alpha CV also available | Fractional ridge selected separately per voxel by cross-validation |
| Cross-validation target | HRFs: nuisance-adjusted task-model time-series prediction. Fractional ridge: fixed OLS trial betas. Shared-alpha ridge: candidate-regularized trial betas | Reproducibility of beta estimates for repeated conditions, to choose denoising and ridge settings |
| Trial estimates | One coefficient per presentation, including repeats | One coefficient per presentation, including repeats |

GLMsingle's HRF selection itself does **not** use the repeated-condition
cross-validation criterion. This distinction is explicit in the
[GLMsingle FAQ](https://glmsingle.readthedocs.io/en/latest/wiki.html#why-isn-t-the-hrf-selection-cross-validated).
Boldtailor's selection and final fitting procedures are described in the
[user guide](user-guide.md#selecting-an-hrf-for-each-location).

## Data-derived noise regressors

**Attribution.** The denoising procedure follows GLMsingle's GLMdenoise
stage (Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M.J.,
Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. *eLife*, 11, e77599.
[doi:10.7554/eLife.77599](https://doi.org/10.7554/eLife.77599)), with the
deviations listed below. It is an independent reimplementation;
GLMsingle (Copyright (c) 2021, Kendrick Kay) is distributed under the BSD
3-Clause License, reproduced in `LICENSES/GLMsingle-BSD-3-Clause.txt`.
GLMsingle's authors have not reviewed or endorsed Boldtailor.

Boldtailor has an optional stage, `select_denoising()`, that follows
GLMsingle's GLMdenoise procedure as closely as Boldtailor's inputs allow. Like
GLMsingle, it selects HRFs once on all runs and freezes them, defines the
noise pool by ON-OFF R² (one canonical-HRF task regressor, a shared
coefficient, in-sample on all runs) below the `findtailthreshold`
Gaussian-mixture threshold, scores the features above that threshold (the
best 100 when none passes), computes run-wise temporal PCs of that single
pool, summarizes each count by the median over scored features, and stops
with the `pcstop` rule of `select_noise_regressors` (default 1.05, counts
0-10). The core is anatomy-agnostic: every input feature is a candidate and
masking is left to the input step.

Deviations, each with its reason:

- **Nuisance.** Each run's baseline confounds plus an intercept replace
  GLMsingle's polynomial drift terms, because Boldtailor analyses carry their
  own confounds.
- **Count scoring.** GLMsingle scores counts by cross-validated single-trial
  beta consistency across repeated conditions and skips GLMdenoise when
  conditions do not repeat. Boldtailor does not assume repeats, so it scores
  counts by leave-one-run-out task-model time-series prediction (pooled
  SSE/SST per feature across folds), close to the original GLMdenoise. Its
  task-guided criterion uses the caller's task model, including modulators.
- **Fixed target.** The held-out target never contains the candidate PCs, so
  more PCs cannot win by shrinking it.
- **Stopping rule.** The walk matches GLMsingle's `select_noise_regressors`
  except for a 64-eps roundoff slack that keeps numerically equal values
  equal.
- **Significance gate.** This gate is not part of GLMsingle. By default
  (`significance_gate=True`) the pcstop count `k*` is kept only if adding
  each run's first `k*` PCs passes per-feature F-tests: in-sample OLS fits on
  all runs, nested models with and without the PCs, frozen HRFs. More
  scoring features must have `p < gate_alpha` (0.05) than chance allows,
  by a one-sided binomial test at `gate_binomial_alpha` (0.05); otherwise
  zero PCs are chosen. The reason: pcstop is relative and has no absolute
  floor, so with time-series scoring it chose 6 PCs on independent noise in
  the predeclared validation. The F-tests are anti-conservative under
  autocorrelated noise (OLS, no prewhitening), and the binomial test treats
  features as independent, which is optimistic for correlated features; the
  gate is deliberately lenient. `significance_gate=False` gives GLMsingle's
  pcstop-only choice.

Boldtailor's stage is not run by default or in the NSD example, and it is
tuned sequentially: HRFs and the pool are frozen while counts are compared.
Pool refinement and voxelwise counts are not implemented. No matched
comparison with GLMsingle's denoising has been run. Synthetic checks in this
repository establish implementation behavior only: with shared noise one PC
was chosen, the gate kept it, and coefficient and outer-run recovery improved;
with independent noise only, pcstop chose 6 PCs on chance gains of about 2e-4
in median R², and the gate rejected them (1 of 10 features with `p < 0.05`),
so zero PCs were chosen. Without the gate this check failed. See the
[user guide](user-guide.md#task-guided-denoising).
[GLMsingle source](https://github.com/cvnlab/GLMsingle/blob/main/glmsingle/glmsingle.py) ·
[GLMdenoise](https://pmc.ncbi.nlm.nih.gov/articles/PMC3865440/)

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

Raw beta magnitudes need care when comparing packages. Every event's predicted response is scaled to a peak of one (kernels are
also stored at unit peak). A beta is therefore the peak BOLD response to that
presentation in signal units, independent of TR, oversampling, and event
duration, and comparable across grayordinates with different selected HRFs.
This matches GLMsingle's convention. Nilearn derivative and FIR bases are passed to Nilearn unchanged (sum-to-one for the canonical bases); user-supplied kernels are used exactly as given. Remaining
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

A local GLMsingle checkout has a
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
