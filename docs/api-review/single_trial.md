# `boldtailor.single_trial`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/single_trial.py)

Single-trial shared or selected HRF models without repeated-stimulus tuning.

> Organization update (2026-11-16): The current public functions are parallel wrappers. Shared-HRF implementation helpers now live in [`_shared_hrf_fit.py`](../../src/boldtailor/_shared_hrf_fit.py); selected-HRF fitting lives in `_selected_hrf_fit.fit_selected_hrf_trials`.

## Definitions

- [`fit_single_trials`](#definition-1)
- [`_compile_runs`](#definition-2)
- [`_fit_runs`](#definition-3)
- [`_model_metadata`](#definition-4)
- [`_assemble_result`](#definition-5)
- [`fit_selected_hrfs`](#definition-6)

<a id="definition-1"></a>

## `fit_single_trials`

[Source, line 35](../../src/boldtailor/single_trial.py#L35)

```python
fit_single_trials(data: AnalysisData, *, ridge_alpha: float=0.0, run_labels: Sequence[str] | None=None, hrf_model: str | HrfCandidate='spm', ridge_fraction: float | np.ndarray | None=None, hrf: object=UNSET) -> SingleTrialResult
```

Estimate one native-unit beta per event row, independently in each run.

Nuisance regressors and an intercept are unpenalized. Positive alpha penalizes
trial coefficients after nuisance projection and column normalization. RT and
condition labels are retained as metadata and never enter the design.

Alternatively, ridge_fraction specifies the coefficient-norm ratio to OLS
in the raw trial-coefficient basis after nuisance projection, as a scalar or
one value per feature. Values must
be in (0, 1]; NaNs in a feature map exclude features. A positive ridge_alpha
cannot be combined with ridge_fraction. Fractional results include the
selected fractions and implied alpha for each run and feature.

``hrf_model`` is 'spm' or one identified ``HrfCandidate`` used everywhere;
the old keyword ``hrf`` is a deprecated alias.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:185](../../src/boldtailor/workflow/analysis.py#L185)
- [src/boldtailor/workflow/beta_series.py:232](../../src/boldtailor/workflow/beta_series.py#L232)
- [tests/test_fractional_ridge.py:169](../../tests/test_fractional_ridge.py#L169)
- [tests/test_fractional_ridge.py:170](../../tests/test_fractional_ridge.py#L170)

<a id="definition-2"></a>

## `_compile_runs`

[Source, line 81](../../src/boldtailor/single_trial.py#L81)

```python
_compile_runs(data, labels, hrf_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/single_trial.py:66](../../src/boldtailor/single_trial.py#L66)

<a id="definition-3"></a>

## `_fit_runs`

[Source, line 90](../../src/boldtailor/single_trial.py#L90)

```python
_fit_runs(compiled, signals, alpha, fractions)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/single_trial.py:67](../../src/boldtailor/single_trial.py#L67)

<a id="definition-4"></a>

## `_model_metadata`

[Source, line 101](../../src/boldtailor/single_trial.py#L101)

```python
_model_metadata(compiled, times, labels, alpha, hrf='spm')
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/single_trial.py:68](../../src/boldtailor/single_trial.py#L68)

<a id="definition-5"></a>

## `_assemble_result`

[Source, line 124](../../src/boldtailor/single_trial.py#L124)

```python
_assemble_result(compiled, trials, fits, alpha, provenance, fractions=None)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/single_trial.py:78](../../src/boldtailor/single_trial.py#L78)

<a id="definition-6"></a>

## `fit_selected_hrfs`

[Source, line 150](../../src/boldtailor/single_trial.py#L150)

```python
fit_selected_hrfs(data: AnalysisData, *, hrf_selection: HrfSelectionResult=UNSET, ridge_alpha: float=0.0, run_labels: Sequence[str] | None=None, feature_signature: str | None=None, ridge_fraction: float | np.ndarray | None=None, selection: HrfSelectionResult=UNSET) -> SingleTrialResult
```

Fit unrestricted trial betas using each feature's previously selected HRF.

The selection may come from different runs. A supplied spatial signature
must match; for anonymous arrays, the caller must preserve feature order.
OLS, fixed-alpha ridge, and fractional ridge use the identical selection
and peak-normalized HRFs. ridge_fraction follows fit_single_trials semantics;
its implied alpha is computed separately for each run and feature.
The old keyword ``selection`` is a deprecated alias of ``hrf_selection``.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:187](../../src/boldtailor/workflow/analysis.py#L187)
- [src/boldtailor/workflow/beta_series.py:233](../../src/boldtailor/workflow/beta_series.py#L233)
- [tests/test_lifecycle.py:346](../../tests/test_lifecycle.py#L346)
- [tests/test_lifecycle.py:151](../../tests/test_lifecycle.py#L151)
