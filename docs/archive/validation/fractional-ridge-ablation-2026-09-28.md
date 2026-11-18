# Leakage-safe fractional-ridge ablation

> **Dated validation record:** Results and test counts apply to the recorded
> revision/settings, not necessarily this checkout. See the
> [documentation index](../README.md) for current API and methods guidance.

Date: 2026-09-28. Experimental implementation: `4a5820f`.
Production fitting defaults are unchanged.

**Subsequent adoption (2026-09-28):** Following review of these results, the user
authorized adopting raw coefficient norms and fixed OLS targets for production
fractional ridge. Per-run fractions and no calibration were retained. The
experiment and recommendation below describe the evidence before that decision;
its archived `production_defaults_changed=false` records the experiment's scope.
See the [adoption record](fractional-defaults-2026-09-28.md) for migration details.

## Findings

Raw columns are the most promising change in this grid, particularly for
variable-duration events. Fixed OLS targets improve centered recovery and latent
prediction but can worsen absolute amplitude recovery. Affine calibration
partially restores amplitudes at the cost of increased within-run error. Pooled
training penalties alone provide little evidence of improvement. These results
do not support adopting all four changes as a package.

Each row below changes only the named component from the production-equivalent
baseline (candidate targets, normalized columns, per-run fractions, no
calibration). Values are paired mean differences ± SEM across ten seeds.
Negative RMSE changes and positive R² changes indicate improvement.

| Change | Δ beta RMSE | Δ within-run beta RMSE | Δ latent prediction R² |
|---|---:|---:|---:|
| Raw columns | −0.161 ± 0.036 | −0.144 ± 0.030 | +0.025 ± 0.009 |
| Fixed OLS targets | +0.250 ± 0.058 | −0.133 ± 0.046 | +0.060 ± 0.016 |
| Pooled training alpha | −0.028 ± 0.019 | +0.002 ± 0.017 | −0.002 ± 0.004 |
| Training affine calibration | −0.629 ± 0.059 | +0.531 ± 0.039 | −0.026 ± 0.020 |

The baseline's mean beta RMSE is 2.924, within-run beta RMSE 1.398, and latent
prediction R² 0.195. Absolute averages for selected combinations are:

| Targets / basis / scope / calibration | Beta RMSE | Within-run RMSE | Latent prediction R² |
|---|---:|---:|---:|
| Candidate / normalized / per-run / none | 2.924 | 1.398 | 0.195 |
| Candidate / raw / per-run / none | 2.762 | 1.254 | 0.220 |
| Fixed OLS / raw / per-run / none | 2.898 | 1.184 | 0.260 |
| Fixed OLS / raw / pooled / none | 2.851 | 1.196 | 0.258 |
| Fixed OLS / raw / pooled / affine | 2.083 | 1.620 | 0.211 |
| Unregularized OLS reference | 2.199 | 1.996 | 0.162 |

All 17 variants and paired uncertainties are in the archived summary tables.
The OLS reference illustrates why absolute RMSE alone is insufficient: it is
better than uncalibrated ridge on that measure but worse on centered recovery
and latent prediction.

### Conditions that change the interpretation

- **Column scaling:** the raw-column change reduces centered RMSE by
  0.287 ± 0.062 for variable durations, versus 0.001 ± 0.005 for uniform
  durations. At ISI 4 s the reduction is 0.293 ± 0.050; at ISI 12 s the
  change is +0.005 ± 0.036. Thus the overall benefit does not imply a universal
  improvement. The coefficient basis defines a different regularization prior.
  Centered recovery improves on average for all three feature types, but the
  weak-linear feature has essentially unchanged latent prediction R²
  (−0.001 ± 0.029), compared with +0.044 ± 0.021 for strong-linear and
  +0.031 ± 0.010 for nonlinear features.
- **Fixed targets:** with no run-offset increment, absolute RMSE changes by
  −0.099 ± 0.044; with increment one it changes by +0.600 ± 0.083.
  Centered recovery improves in both strata. Encoding run intercepts and
  centered scoring do not prevent shrinkage of trial-beta means in the first
  stage. Fixed targets select OLS in 18.9% of cases versus 43.9% for the
  baseline, and select fraction .1 in 26.3% versus 25.6%. These are conditional
  results for the tested candidate grid, not evidence that .1 is optimal.
- **Calibration:** in the high AR/heteroscedastic condition it increases
  centered RMSE by 1.024 ± 0.075, versus 0.037 ± 0.010 under low white noise.
  The baseline affine scale averages 2.285; centered amplitude projection
  increases from 0.548 to 0.765 while within-run correlation remains 0.413,
  as expected for positive affine rescaling. Restoration of amplitude can
  also amplify noise. Training-only estimation prevents leakage but does not
  guarantee recovery benefits.
- **Pooled penalties:** baseline-like centered changes are small in each
  one-factor stratum. Freezing a training-derived alpha is a useful controlled
  experiment, but this grid supplies no clear accuracy reason to replace the
  per-run convention.

Recommendation: retain production defaults while testing raw columns and fixed
OLS targets as separate options on real-data reliability and additional
simulations. Keep affine calibration explicitly optional and report raw and
calibrated results together. Do not infer a default from the best pooled average
or tune further changes on these same outer runs and call them independent
validation.

## Question and design

Does changing the validation target, coefficient basis, fraction scope, or
post-fit calibration improve recovery of known trial amplitudes?

The experiment crosses candidate-dependent versus fixed OLS validation targets,
normalized versus raw trial-design columns, and per-run versus pooled-training
fractions. Each of these eight variants is evaluated with and without
training-only affine calibration. An OLS reference makes 17 output variants.
This tests adaptations motivated by GLMsingle; it does not reproduce its
repeated-stimulus validation procedure or full denoising/HRF-selection pipeline.

Each synthetic dataset has six runs, 24 trials per run, and three features:
strong linear effects, weak linear effects, and linear effects plus an unmodeled
nonlinear component. All features also contain random trial-level variation.
The 32 scenarios cross ISI (4/12 seconds), uniform/variable event durations,
low white/high AR(1) heteroscedastic noise, matched/mismatched HRF, and run-offset
increments of zero/one. The noise factor changes both amplitude and covariance;
it cannot identify their separate effects. Ten seeds give 320 datasets.
Every method sees identical inputs within a dataset.

## Leakage controls

- Runs 0–3 select one fraction per feature using leave-one-run-out validation.
  Runs 4–5 are reserved for final evaluation. Fractions are 1, .8, .5, .2, .1;
  ties within 1e-12 favor less shrinkage.
- Pooled fractions use the coefficient norm over the block-diagonal training
  design. The corresponding feature-specific alpha is computed from training
  runs only and frozen for validation/test fits. Achieved held-out norm ratios
  are recorded and need not equal the requested training fraction.
- Per-run fractions reproduce the existing fitting convention: a held-out
  response determines its own beta-estimation penalty, but does not affect
  training penalties, encoding coefficients, calibration, or fraction selection
  in the outer split.
- Encoding uses run-specific training intercepts and within-run centered scoring.
  Centering held-out residuals defines the score; it does not update predictions
  or fit training parameters using held-out outcomes.
- Affine calibration maps training ridge betas to training OLS betas after
  selection. Its scale and offset are frozen for the two outer runs, including
  encoding predictions. Degenerate or nonpositive slopes use the identity map.
  Calibration is an ablation, not another choice selected on outer scores.
- Latent truth is used only to evaluate recovery. Perturbation tests verify that
  changing outer outcomes or latent truth cannot change selected fractions,
  inner scores, training penalties, or calibration.

## Measures and uncertainty

Beta RMSE includes run means; within-run beta RMSE removes each run's mean.
Amplitude ratios project estimated betas onto truth (one indicates correct
amplitude), with both raw and centered versions exported. Within-run correlation
measures shape independently of amplitude. Prediction R² is evaluated against
common OLS targets and separately against latent truth, always centered within
runs. Candidate-dependent inner scores are not used to compare recovery across
methods. `slope_rmse` compares estimated encoding slopes with slopes fitted to
the realized *training latent betas*, not population-generating coefficients;
this distinction particularly matters for the nonlinear feature.

Scenarios and features are averaged within each seed before computing the mean
and standard error across ten seeds. Paired differences use matching seeds.
These describe this simulation grid, not uncertainty across human participants.
Held-out fraction exports refer to estimates before affine calibration.

## Reproduction and artifacts

From the repository root:

```sh
uv run python -m examples.validation.fractional_ridge_ablation_simulation \
  --seeds 10 --output-dir /private/tmp/boldtailor-fractional-ablation-2026-09-28
uv run pytest tests examples/NSD -q -W error
```

The companion directory contains all 16,320 selected evaluation rows, 38,400
inner candidate rows, settings/package versions, seed means, paired seed
differences, summary tables, and scenario/feature-stratified seed means.

Verification: 860 tests passed with warnings treated as errors; Black and
whitespace checks passed. New tests were committed after observed failures
before each implementation step. Independent review found no blockers.
An independent numeric audit recomputed all 17 × 10 summary mean/SEM pairs
directly from the selected rows, averaging the 32 scenarios and three features
within each seed first; every value agreed within numerical tolerance.

This is a synthetic stress test with a fixed nuisance design and HRF fitting
basis. It does not establish performance on real fMRI, HRF-selection uncertainty,
or a population of subjects. Production changes require further evidence.
