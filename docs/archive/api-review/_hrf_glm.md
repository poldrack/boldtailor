# `boldtailor._hrf_glm`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_hrf_glm.py)

Fit conventional models in HRF groups and restore input feature order.

## Definitions

- [`_FitContext`](#definition-1)
- [`_check_task_model`](#definition-2)
- [`_prepare`](#definition-3)
- [`_feature_groups`](#definition-4)
- [`_group_fits`](#definition-5)
- [`_assemble`](#definition-6)
- [`fit_selected_glm`](#definition-7)
- [`_validate_parent`](#definition-8)
- [`_ols_comparison`](#definition-9)
- [`selected_task_delta_r2`](#definition-10)
- [`_comparison_activity`](#definition-11)
- [`_comparison_identity`](#definition-12)

<a id="definition-1"></a>

## `_FitContext`

[Source, line 42](../../src/boldtailor/_hrf_glm.py#L42)

```python
@dataclass(frozen=True)
class _FitContext
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
groups: dict
nuisance: tuple
activity: dict
analysis_id: str | None
```

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:94](../../src/boldtailor/_hrf_glm.py#L94)

<a id="definition-2"></a>

## `_check_task_model`

[Source, line 49](../../src/boldtailor/_hrf_glm.py#L49)

```python
_check_task_model(model, selection)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:73](../../src/boldtailor/_hrf_glm.py#L73)

<a id="definition-3"></a>

## `_prepare`

[Source, line 71](../../src/boldtailor/_hrf_glm.py#L71)

```python
_prepare(data, model, selection, signature, model_settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:157](../../src/boldtailor/_hrf_glm.py#L157)
- [src/boldtailor/_hrf_glm.py:197](../../src/boldtailor/_hrf_glm.py#L197)

<a id="definition-4"></a>

## `_feature_groups`

[Source, line 102](../../src/boldtailor/_hrf_glm.py#L102)

```python
_feature_groups(selection)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:132](../../src/boldtailor/_hrf_glm.py#L132)
- [src/boldtailor/_hrf_glm.py:182](../../src/boldtailor/_hrf_glm.py#L182)
- [src/boldtailor/_hrf_glm.py:126](../../src/boldtailor/_hrf_glm.py#L126)
- [src/boldtailor/_hrf_glm.py:115](../../src/boldtailor/_hrf_glm.py#L115)

<a id="definition-5"></a>

## `_group_fits`

[Source, line 107](../../src/boldtailor/_hrf_glm.py#L107)

```python
_group_fits(data, model, selection, context)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:159](../../src/boldtailor/_hrf_glm.py#L159)

<a id="definition-6"></a>

## `_assemble`

[Source, line 119](../../src/boldtailor/_hrf_glm.py#L119)

```python
_assemble(data, model, selection, context, fits, provenance)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:163](../../src/boldtailor/_hrf_glm.py#L163)

<a id="definition-7"></a>

## `fit_selected_glm`

[Source, line 149](../../src/boldtailor/_hrf_glm.py#L149)

```python
fit_selected_glm(data: AnalysisData, model: ModelSpec, selection: HrfSelectionResult, signature: str | None) -> HrfAnalysisResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:47](../../src/boldtailor/fit.py#L47)

<a id="definition-8"></a>

## `_validate_parent`

[Source, line 166](../../src/boldtailor/_hrf_glm.py#L166)

```python
_validate_parent(data, result, context)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:200](../../src/boldtailor/_hrf_glm.py#L200)

<a id="definition-9"></a>

## `_ols_comparison`

[Source, line 179](../../src/boldtailor/_hrf_glm.py#L179)

```python
_ols_comparison(data, selection, context)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:201](../../src/boldtailor/_hrf_glm.py#L201)

<a id="definition-10"></a>

## `selected_task_delta_r2`

[Source, line 191](../../src/boldtailor/_hrf_glm.py#L191)

```python
selected_task_delta_r2(data: AnalysisData, model: ModelSpec, result: HrfAnalysisResult) -> TaskDeltaR2Result
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:77](../../src/boldtailor/fit.py#L77)

<a id="definition-11"></a>

## `_comparison_activity`

[Source, line 219](../../src/boldtailor/_hrf_glm.py#L219)

```python
_comparison_activity(model, comparison, context)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:209](../../src/boldtailor/_hrf_glm.py#L209)

<a id="definition-12"></a>

## `_comparison_identity`

[Source, line 229](../../src/boldtailor/_hrf_glm.py#L229)

```python
_comparison_identity(model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:213](../../src/boldtailor/_hrf_glm.py#L213)
