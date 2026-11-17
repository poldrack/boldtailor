# `boldtailor.trial_encoding`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/trial_encoding.py)

Predict trial betas with shared slopes and explicit run-baseline semantics.

## Definitions

- [`_run_indices`](#definition-1)
- [`_predictor_arrays`](#definition-2)
- [`validate_encoding_mode`](#definition-3)
- [`encoding_metadata`](#definition-4)
- [`_center`](#definition-5)
- [`_within_design`](#definition-6)
- [`_training_design`](#definition-7)
- [`_fit_coefficients`](#definition-8)
- [`_fit_within_coefficients`](#definition-9)
- [`_predict_run`](#definition-10)
- [`evaluate_trial_encoding`](#definition-11)
- [`_indicator_frame`](#definition-12)
- [`_predictor_frame`](#definition-13)
- [`_censor_response_time_in_place`](#definition-14)
- [`_run_predictors`](#definition-15)
- [`trial_predictors`](#definition-16)

<a id="definition-1"></a>

## `_run_indices`

[Source, line 15](../../src/boldtailor/trial_encoding.py#L15)

```python
_run_indices(values, n_runs, name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:195](../../src/boldtailor/trial_encoding.py#L195)
- [src/boldtailor/trial_encoding.py:196](../../src/boldtailor/trial_encoding.py#L196)

<a id="definition-2"></a>

## `_predictor_arrays`

[Source, line 26](../../src/boldtailor/trial_encoding.py#L26)

```python
_predictor_arrays(predictors, n_trials)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:41](../../src/boldtailor/_ridge_cv.py#L41)
- [src/boldtailor/trial_encoding.py:199](../../src/boldtailor/trial_encoding.py#L199)

<a id="definition-3"></a>

## `validate_encoding_mode`

[Source, line 55](../../src/boldtailor/trial_encoding.py#L55)

```python
validate_encoding_mode(encoding_mode)
```

Reject ambiguous objectives before doing any model fitting.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:33](../../src/boldtailor/_ridge_cv.py#L33)
- [src/boldtailor/trial_encoding.py:117](../../src/boldtailor/trial_encoding.py#L117)
- [src/boldtailor/trial_encoding.py:187](../../src/boldtailor/trial_encoding.py#L187)
- [src/boldtailor/trial_encoding.py:67](../../src/boldtailor/trial_encoding.py#L67)

<a id="definition-4"></a>

## `encoding_metadata`

[Source, line 65](../../src/boldtailor/trial_encoding.py#L65)

```python
encoding_metadata(encoding_mode)
```

Describe the objective consistently in CV and exported artifacts.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:199](../../src/boldtailor/_ridge_cv.py#L199)
- [src/boldtailor/workflow/beta_series.py:599](../../src/boldtailor/workflow/beta_series.py#L599)
- [src/boldtailor/workflow/beta_series.py:562](../../src/boldtailor/workflow/beta_series.py#L562)
- [src/boldtailor/workflow/beta_series.py:652](../../src/boldtailor/workflow/beta_series.py#L652)

<a id="definition-5"></a>

## `_center`

[Source, line 95](../../src/boldtailor/trial_encoding.py#L95)

```python
_center(values)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:110](../../src/boldtailor/trial_encoding.py#L110)
- [src/boldtailor/trial_encoding.py:146](../../src/boldtailor/trial_encoding.py#L146)

<a id="definition-6"></a>

## `_within_design`

[Source, line 101](../../src/boldtailor/trial_encoding.py#L101)

```python
_within_design(arrays, masks, train)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:119](../../src/boldtailor/trial_encoding.py#L119)

<a id="definition-7"></a>

## `_training_design`

[Source, line 116](../../src/boldtailor/trial_encoding.py#L116)

```python
_training_design(arrays, masks, train, *, encoding_mode='absolute')
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:46](../../src/boldtailor/_ridge_cv.py#L46)
- [src/boldtailor/trial_encoding.py:201](../../src/boldtailor/trial_encoding.py#L201)

<a id="definition-8"></a>

## `_fit_coefficients`

[Source, line 132](../../src/boldtailor/trial_encoding.py#L132)

```python
_fit_coefficients(design, betas, masks, train)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:218](../../src/boldtailor/trial_encoding.py#L218)

<a id="definition-9"></a>

## `_fit_within_coefficients`

[Source, line 140](../../src/boldtailor/trial_encoding.py#L140)

```python
_fit_within_coefficients(design, betas, masks, train, run_means)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:214](../../src/boldtailor/trial_encoding.py#L214)

<a id="definition-10"></a>

## `_predict_run`

[Source, line 156](../../src/boldtailor/trial_encoding.py#L156)

```python
_predict_run(beta, values, mask, means, coefficients, *, encoding_mode)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:223](../../src/boldtailor/trial_encoding.py#L223)

<a id="definition-11"></a>

## `evaluate_trial_encoding`

[Source, line 175](../../src/boldtailor/trial_encoding.py#L175)

```python
evaluate_trial_encoding(beta_runs, predictors, *, train_runs, test_runs, encoding_mode='within_run')
```

Fit training slopes, then score within-run deviations or absolute levels.

Predictor rows correspond positionally to beta/event rows. Nonfinite
predictors exclude a trial from encoding only; its output row remains NaN.
The default fits separate training-run intercepts and centers test residuals
for scoring only. Returned predictions use a training-derived reference;
scoring offsets never alter predictions. Absolute mode preserves the shared
intercept and uncentered loss. Scores pool SSE/SST, retaining undefined R².

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:151](../../src/boldtailor/_ridge_cv.py#L151)
- [src/boldtailor/workflow/beta_series.py:287](../../src/boldtailor/workflow/beta_series.py#L287)
- [tests/workflow/test_beta_series.py:288](../../tests/workflow/test_beta_series.py#L288)

<a id="definition-12"></a>

## `_indicator_frame`

[Source, line 252](../../src/boldtailor/trial_encoding.py#L252)

```python
_indicator_frame(values, modulator, label)
```

One 0/1 column per non-reference level; missing values become NaN.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:271](../../src/boldtailor/trial_encoding.py#L271)

<a id="definition-13"></a>

## `_predictor_frame`

[Source, line 266](../../src/boldtailor/trial_encoding.py#L266)

```python
_predictor_frame(events, modulator, label)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:282](../../src/boldtailor/trial_encoding.py#L282)

<a id="definition-14"></a>

## `_censor_response_time_in_place`

[Source, line 275](../../src/boldtailor/trial_encoding.py#L275)

```python
_censor_response_time_in_place(table)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:284](../../src/boldtailor/trial_encoding.py#L284)

<a id="definition-15"></a>

## `_run_predictors`

[Source, line 281](../../src/boldtailor/trial_encoding.py#L281)

```python
_run_predictors(events, task_model, label)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:300](../../src/boldtailor/trial_encoding.py#L300)

<a id="definition-16"></a>

## `trial_predictors`

[Source, line 288](../../src/boldtailor/trial_encoding.py#L288)

```python
trial_predictors(events_tables, task_model, labels=None)
```

Predictor columns per run, named as the task-model regressors after task.

Unavailable behavior (NaN) excludes only the affected encoding rows.
``labels`` name the runs in errors (default ``run 1``, ``run 2``, ...).

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:492](../../src/boldtailor/workflow/beta_series.py#L492)
- [tests/workflow/test_beta_series.py:394](../../tests/workflow/test_beta_series.py#L394)
- [tests/workflow/test_beta_series.py:290](../../tests/workflow/test_beta_series.py#L290)
- [tests/workflow/test_beta_series.py:261](../../tests/workflow/test_beta_series.py#L261)
