# Within-run encoding evaluation, 2026-09-28

Production encoding now defaults to shared trial-level slopes with separate
training-run intercepts and centered residual scoring in held-out runs.
`encoding_mode="absolute"` retains the historical shared-intercept objective.
Returned predictions keep a training-derived reference level; held-out scoring
offsets are exported separately and never alter those predictions.

## Reproduction

```sh
uv run python examples/validation/ridge_objective_simulation.py --seeds 10 --output-dir /private/tmp/boldtailor-within-run-encoding-2026-09-28
```

The original 480 scenarios (10 seeds × 48 settings), six runs, canonical HRF,
alpha/fraction grids, and untouched outer runs are unchanged. See the
[September 27 design](ridge-objectives-2026-09-27.md) for generator details.
The fourth objective, `within_run`, combines within-run scoring with run-specific
training intercepts. `current`, `centered`, and `fixed_ols` explicitly fit using
`encoding_mode="absolute"`; `current` is the historical September 27 baseline.
All common numeric columns and selected values for those three objectives
reproduce the archived September 27 results exactly (maximum absolute difference
0 after reading CSVs). The old archive is unchanged.

The new archive contains 3,840 selected-model rows and 23,040 candidate-score
rows, plus settings, per-seed means, scenario summaries, and a deterministic
confounding stress experiment. [Download the tables and settings](within-run-encoding-2026-09-28/).

## Results

Each entry averages the 48 scenarios within a seed, then reports mean ± SEM
across ten seeds. Whole-beta amplitude is projection onto latent betas;
within-run amplitude and RMSE first remove each run's mean from both estimated
and latent betas. Unit amplitude indicates preservation. Neither RMSE alone nor
correlation alone captures all scientifically relevant distortion.

| Regularizer | Objective | Strongest boundary | Whole-beta RMSE | Whole-beta amplitude | Within-run RMSE | Within-run amplitude | Within-run correlation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | centered | 0.208 ± 0.049 | 1.974 ± 0.161 | 0.645 ± 0.043 | 0.519 ± 0.042 | 0.676 ± 0.040 | 0.657 ± 0.009 |
| alpha | current | 0.658 ± 0.026 | 4.195 ± 0.044 | 0.267 ± 0.018 | 0.493 ± 0.007 | 0.288 ± 0.022 | 0.564 ± 0.010 |
| alpha | fixed_ols | 0.000 ± 0.000 | 0.837 ± 0.018 | 1.003 ± 0.009 | 0.818 ± 0.017 | 0.984 ± 0.036 | 0.665 ± 0.012 |
| alpha | within_run | 0.171 ± 0.047 | 1.560 ± 0.141 | 0.713 ± 0.041 | 0.540 ± 0.042 | 0.738 ± 0.037 | 0.659 ± 0.009 |
| fraction | centered | 0.227 ± 0.048 | 1.807 ± 0.165 | 0.678 ± 0.042 | 0.542 ± 0.042 | 0.728 ± 0.038 | 0.656 ± 0.009 |
| fraction | current | 0.679 ± 0.024 | 3.853 ± 0.039 | 0.322 ± 0.017 | 0.477 ± 0.008 | 0.408 ± 0.022 | 0.562 ± 0.011 |
| fraction | fixed_ols | 0.000 ± 0.000 | 0.838 ± 0.018 | 1.004 ± 0.009 | 0.819 ± 0.017 | 0.985 ± 0.037 | 0.665 ± 0.012 |
| fraction | within_run | 0.183 ± 0.047 | 1.473 ± 0.121 | 0.732 ± 0.037 | 0.549 ± 0.039 | 0.772 ± 0.030 | 0.659 ± 0.010 |

Under run-offset increment one, strongest-boundary rates fall from 99.2%/100%
(alpha/fraction, historical objective) to 14.2%/14.6% with the new objective.
Centered scoring alone gives 21.2%/21.7%. New-objective whole-beta amplitude
ratios under offsets are 0.764/0.785, compared with 0.638/0.688 for centered
scoring alone.

The combined model improves whole-beta RMSE and amplitude preservation relative
to centered scoring alone, but does **not** uniformly improve recovery: its
within-run RMSE is slightly higher (0.540/0.549 versus 0.519/0.542 overall).
The historical objective's more severe shrinkage can also produce lower
within-run RMSE while substantially attenuating true variation. The scientific
justification for the new default is its explicit within-run estimand and
removal of between-run intercept confounding, not dominance on every metric.

## Predictor-mean/run-offset confounding stress test

A separate noiseless encoding experiment has six runs of five trials,
`X_r = arange(5) + 10*r`, and `beta_r = 3*X_r + offset*r`. Runs 0–3 train;
runs 4–5 evaluate. Offset is 0 or 20; there is no HRF or ridge fit in this
experiment, isolating the encoding model. The settings and results are in
`confounding-stress.csv`; `confounding_stress()` reproduces them.

With zero offset, all three tested objectives recover slope 3 and outer R² 1.
With offset 20, both shared-intercept fits estimate slope 4.968504; their outer
R² values are 0.518466 (absolute) and 0.569444 (centered scoring). The run-specific
intercept fit recovers slope 3 and within-run outer R² 1. This demonstrates why
centering validation residuals alone does not remove training-slope confounding.

## Verification and limitations

The full `uv run pytest tests examples/NSD -q -W error` suite passed 837 tests.
Four additional independent-review coverage tests passed afterward; they cover
fold-specific rank loss and outcome-only prediction isolation for both regularizers.
Black and whitespace checks passed. Independent review found no blocking defects.

Tests compare slopes and run intercepts to independent run-dummy OLS, and both
regularizer paths to independent candidate-beta and centered-loss oracles.
They cover pure run-offset invariance, retained slope/amplitude error, negative
and undefined scores, unequal run sizes, masks, rank loss, invalid modes,
held-out isolation, feature-block assembly, and exported SSE reconstruction.

Training requires full rank after within-run centering and positive residual
degrees of freedom. Predictors constant within each run cannot identify a
within-run effect. Test means affect the scoring statistic only; these scores
must not be interpreted as absolute held-out prediction of unknown run means.

Candidate-dependent target scale remains unresolved. Outer scores from different
objectives have different target/scoring definitions and are not a common
recovery criterion. The simulation still uses one feature, a fixed canonical
HRF, simple noise, and a restricted predictor generator; it does not establish
performance under realistic HRF mismatch, high-dimensional feature aggregation,
nonlinear signals, or correlated/heteroscedastic noise. Fractional outer fitting
continues to determine each test run's alpha from its observed BOLD while keeping
the selected fraction fixed. No large real-data analysis was rerun.
