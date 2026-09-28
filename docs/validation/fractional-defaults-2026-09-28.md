# Fractional-ridge defaults adopted on 2026-09-28

The user authorized updating production defaults following the
[320-dataset ablation](fractional-ridge-ablation-2026-09-28.md).
The chosen objective prioritizes within-run trial variation and encoding
prediction. This is a practical default choice from current evidence, not a
claim of universal superiority or real-data validation.

| Component | Adopted behavior |
|---|---|
| Fraction norm | Raw trial coefficients after nuisance projection |
| Inner validation targets | Fixed OLS betas, even if fraction 1 is absent from the grid |
| Outer NSD scoring targets | OLS betas using only training-selected HRFs |
| Fraction scope | One selected fraction per feature, converted to alpha separately per run |
| Encoding | Run-specific training intercepts and within-run centered scoring |
| Affine calibration | None |

Raw columns improved centered recovery most with variable-duration and
closely spaced events. Fixed targets improved centered recovery and latent
prediction but worsened absolute-amplitude error when run offsets differed.
The combined raw/fixed/per-run method had mean centered RMSE 1.184 versus
1.398 for the old baseline, and latent prediction R² 0.260 versus 0.195.
These are descriptive simulation results. Pooling showed little benefit, while
affine calibration amplified within-run error under high noise.

## Boundaries and migration

Training responses determine encoding coefficients; validation responses
provide only targets and scoring offsets. Inner HRF selection uses training
runs. Outer test runs remain excluded from fraction and HRF selection.
No held-out affine calibration or pooled-alpha estimation was introduced.
Nuisance coefficients remain unpenalized; exported betas retain native units.

The shared-alpha ridge API retains normalized columns and candidate-regularized
targets. Its numerical behavior is unchanged. `encoding_mode="absolute"`
continues to control encoding intercept/scoring behavior, but does not restore
the old fractional norm basis or target definition.

Recompute fractional tuning and fits. Old fraction maps and implied alphas
are not equivalent under the new raw basis. Provenance now identifies
`raw_trial_coefficients_after_nuisance_projection`,
`normalization="none_after_nuisance_projection"`, and
`validation_target="fixed_ols_betas"` for fractional CV.
The NSD outer result's `betas` retain selected shrinkage, while `targets` and
the exported `_targets.dscalar.nii` files hold actual OLS scoring targets.

## Verification

Independent augmented-regression/root-finding oracles test the raw coefficient
norm, implied penalties, nuisance fitting, canonical and selected-HRF APIs.
Nested scoring oracles test fixed OLS targets with canonical and optimized HRFs
in both encoding modes. Further tests cover grids without OLS, target invariance,
outer scoring/export agreement, and unchanged training fits under held-out
perturbations. Tests were observed failing and committed before implementation.
Independent code review found no correctness or leakage blockers.

Final verification: `uv run pytest tests examples/NSD -q -W error` passed all
861 tests in 164.35 seconds. Black and whitespace checks passed. The first full
run identified two older outer-oracle expectations that still used regularized
targets; their references were updated to the new OLS-target requirement and
the complete suite was rerun successfully.
