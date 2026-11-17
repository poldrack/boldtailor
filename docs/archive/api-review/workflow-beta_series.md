# `boldtailor.workflow.beta_series`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/beta_series.py)

Fraction or shared-alpha tuning with independent odd/even evaluation.

## Definitions

- [`_settings`](#definition-1)
- [`_signature`](#definition-2)
- [`_score_block`](#definition-3)
- [`_global_scores`](#definition-4)
- [`_tune`](#definition-5)
- [`_fit_outer_trials`](#definition-6)
- [`_outer_block`](#definition-7)
- [`_evaluate`](#definition-8)
- [`_final_selection`](#definition-9)
- [`_final_fit`](#definition-10)
- [`fit_cv_beta_series`](#definition-11)
- [`tuning_provenance`](#definition-12)
- [`_identity_activities`](#definition-13)
- [`_alpha_activity`](#definition-14)
- [`link_final_provenance`](#definition-15)
- [`BetaModel`](#definition-16)
- [`BetaModel.fractional`](#definition-17)
- [`_fixed_models`](#definition-18)
- [`_cv_model`](#definition-19)
- [`fit_beta_models`](#definition-20)

<a id="definition-1"></a>

## `_settings`

[Source, line 49](../../src/boldtailor/workflow/beta_series.py#L49)

```python
_settings(runs, library, alphas, percentile, block_size, maximum, n_jobs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:487](../../src/boldtailor/workflow/beta_series.py#L487)

<a id="definition-2"></a>

## `_signature`

[Source, line 89](../../src/boldtailor/workflow/beta_series.py#L89)

```python
_signature(runs, indices)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:257](../../src/boldtailor/workflow/beta_series.py#L257)
- [src/boldtailor/workflow/beta_series.py:111](../../src/boldtailor/workflow/beta_series.py#L111)
- [src/boldtailor/workflow/beta_series.py:410](../../src/boldtailor/workflow/beta_series.py#L410)
- [src/boldtailor/workflow/beta_series.py:174](../../src/boldtailor/workflow/beta_series.py#L174)

<a id="definition-3"></a>

## `_score_block`

[Source, line 93](../../src/boldtailor/workflow/beta_series.py#L93)

```python
_score_block(indices, runs, root, predictors, library, alphas, fractional, encoding_mode, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `_global_scores`

[Source, line 116](../../src/boldtailor/workflow/beta_series.py#L116)

```python
_global_scores(runs, root, predictors, blocks, library, alphas, n_jobs, fractional, encoding_mode, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:204](../../src/boldtailor/workflow/beta_series.py#L204)

<a id="definition-5"></a>

## `_tune`

[Source, line 191](../../src/boldtailor/workflow/beta_series.py#L191)

```python
_tune(runs, root, predictors, blocks, library, alphas, percentile, n_jobs, fractional, encoding_mode, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:504](../../src/boldtailor/workflow/beta_series.py#L504)

<a id="definition-6"></a>

## `_fit_outer_trials`

[Source, line 230](../../src/boldtailor/workflow/beta_series.py#L230)

```python
_fit_outer_trials(data, selection, labels, signature, options)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:272](../../src/boldtailor/workflow/beta_series.py#L272)
- [src/boldtailor/workflow/beta_series.py:284](../../src/boldtailor/workflow/beta_series.py#L284)

<a id="definition-7"></a>

## `_outer_block`

[Source, line 242](../../src/boldtailor/workflow/beta_series.py#L242)

```python
_outer_block(indices, runs, root, predictors, library, alpha, train, test, fractional, encoding_mode, task_model, selection_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `_evaluate`

[Source, line 317](../../src/boldtailor/workflow/beta_series.py#L317)

```python
_evaluate(runs, root, predictors, blocks, library, alpha, train, test, n_jobs, fractional, encoding_mode, task_model, selection_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:525](../../src/boldtailor/workflow/beta_series.py#L525)

<a id="definition-9"></a>

## `_final_selection`

[Source, line 404](../../src/boldtailor/workflow/beta_series.py#L404)

```python
_final_selection(indices, runs, root, library, task_model, selection_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `_final_fit`

[Source, line 416](../../src/boldtailor/workflow/beta_series.py#L416)

```python
_final_fit(runs, root, blocks, library, alpha, n_jobs, fractional, task_model, selection_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:549](../../src/boldtailor/workflow/beta_series.py#L549)

<a id="definition-11"></a>

## `fit_cv_beta_series`

[Source, line 452](../../src/boldtailor/workflow/beta_series.py#L452)

```python
fit_cv_beta_series(runs, root, *, task_model, library, alphas=None, fractions=None, percentile=90.0, block_size=4096, max_grayordinates=None, n_jobs=1, encoding_mode='within_run', selection_model=None)
```

Tune per-feature fractions or one shared alpha, then evaluate and refit.

A library requests optimized HRFs; None uses canonical SPM. Sessions with
optimized HRFs need three or more runs in each half. Returned outer betas,
targets and predictions contain test runs only, in test_run_labels order.
Betas retain the selected regularization. Fractional targets are OLS fits
under training-selected HRFs; shared-alpha targets equal the fitted betas.
The task model drives fits and encoding predictors; selection_model (default
the task model) scores HRFs, for example without RT.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:725](../../src/boldtailor/workflow/beta_series.py#L725)

<a id="definition-12"></a>

## `tuning_provenance`

[Source, line 583](../../src/boldtailor/workflow/beta_series.py#L583)

```python
tuning_provenance(scores, selection)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:225](../../src/boldtailor/workflow/beta_series.py#L225)

<a id="definition-13"></a>

## `_identity_activities`

[Source, line 621](../../src/boldtailor/workflow/beta_series.py#L621)

```python
_identity_activities(record)
```

Embedded scoring activities without environment-only software keys.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:610](../../src/boldtailor/workflow/beta_series.py#L610)

<a id="definition-14"></a>

## `_alpha_activity`

[Source, line 629](../../src/boldtailor/workflow/beta_series.py#L629)

```python
_alpha_activity(selection)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:597](../../src/boldtailor/workflow/beta_series.py#L597)

<a id="definition-15"></a>

## `link_final_provenance`

[Source, line 642](../../src/boldtailor/workflow/beta_series.py#L642)

```python
link_final_provenance(result, decision)
```

Keep block fitting records, adding a reference to the saved decision.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:560](../../src/boldtailor/workflow/beta_series.py#L560)

<a id="definition-16"></a>

## `BetaModel`

[Source, line 673](../../src/boldtailor/workflow/beta_series.py#L673)

```python
@dataclass(frozen=True, kw_only=True)
class BetaModel
```

One trial-wise beta model and, for tuned ridge, how it was chosen.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
name: str
hrf: str
estimator: str
fit: Mapping[str, object]
tuning: Mapping[str, object] | None = None
evaluation: Mapping[str, object] | None = None
cv_provenance: Mapping[str, object] | None = None
predictors: tuple | None = None
```

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:741](../../src/boldtailor/workflow/beta_series.py#L741)
- [src/boldtailor/workflow/beta_series.py:700](../../src/boldtailor/workflow/beta_series.py#L700)
- [tests/workflow/test_plots.py:30](../../tests/workflow/test_plots.py#L30)

<a id="definition-17"></a>

## `BetaModel.fractional`

[Source, line 686](../../src/boldtailor/workflow/beta_series.py#L686)

```python
@property
BetaModel.fractional(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-18"></a>

## `_fixed_models`

[Source, line 693](../../src/boldtailor/workflow/beta_series.py#L693)

```python
_fixed_models(runs, root, blocks, settings, task_model, hrf, selected)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:761](../../src/boldtailor/workflow/beta_series.py#L761)

<a id="definition-19"></a>

## `_cv_model`

[Source, line 718](../../src/boldtailor/workflow/beta_series.py#L718)

```python
_cv_model(runs, root, settings, library, task_model, hrf)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:764](../../src/boldtailor/workflow/beta_series.py#L764)

<a id="definition-20"></a>

## `fit_beta_models`

[Source, line 753](../../src/boldtailor/workflow/beta_series.py#L753)

```python
fit_beta_models(runs, root, blocks, settings, library, selections, task_model)
```

OLS for both HRFs, plus the ridge variant the settings request.
