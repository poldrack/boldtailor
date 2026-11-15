# `boldtailor.fit`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/fit.py)

Dispatch conventional fits and nested-OLS task variance comparisons.

## Definitions

- [`fit`](#definition-1)
- [`task_delta_r2`](#definition-2)
- [`_fit_analysis`](#definition-3)
- [`_validate_parent_analysis`](#definition-4)
- [`_comparison_id`](#definition-5)
- [`_analysis_id`](#definition-6)
- [`_fit_activity`](#definition-7)
- [`_task_delta_r2_activity`](#definition-8)
- [`_run_diagnostic`](#definition-9)
- [`_design_provenance`](#definition-10)

<a id="definition-1"></a>

## `fit`

[Source, line 32](../../src/boldtailor/fit.py#L32)

```python
fit(data: AnalysisData, model: ModelSpec, *, hrf_selection: HrfSelectionResult | None=None, feature_signature: str | None=None) -> AnalysisResult | HrfAnalysisResult
```

Fit conventional contrasts, optionally using an HRF per feature.

A supplied selection replaces model.hrf_model. Its spatial signature must
match feature_signature; all other model settings retain their meaning.
Selected-HRF results expose group_design(run, hrf_id), recompiled on
request, instead of design_matrices.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:101](../../src/boldtailor/workflow/analysis.py#L101)
- [tests/test_denoising.py:745](../../tests/test_denoising.py#L745)
- [tests/test_denoising.py:746](../../tests/test_denoising.py#L746)
- [tests/test_denoising.py:753](../../tests/test_denoising.py#L753)

<a id="definition-2"></a>

## `task_delta_r2`

[Source, line 71](../../src/boldtailor/fit.py#L71)

```python
task_delta_r2(data: AnalysisData, model: ModelSpec, full_result: AnalysisResult | HrfAnalysisResult) -> TaskDeltaR2Result
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:102](../../src/boldtailor/workflow/analysis.py#L102)
- [tests/test_denoising.py:837](../../tests/test_denoising.py#L837)
- [tests/test_fit.py:519](../../tests/test_fit.py#L519)
- [tests/test_fit.py:679](../../tests/test_fit.py#L679)

<a id="definition-3"></a>

## `_fit_analysis`

[Source, line 110](../../src/boldtailor/fit.py#L110)

```python
_fit_analysis(data: AnalysisData, model: ModelSpec) -> tuple[tuple[CompiledDesign, ...], ConventionalFit]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:55](../../src/boldtailor/fit.py#L55)

<a id="definition-4"></a>

## `_validate_parent_analysis`

[Source, line 124](../../src/boldtailor/fit.py#L124)

```python
_validate_parent_analysis(expected_parent_id: str | None, full_result: AnalysisResult) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:83](../../src/boldtailor/fit.py#L83)

<a id="definition-5"></a>

## `_comparison_id`

[Source, line 138](../../src/boldtailor/fit.py#L138)

```python
_comparison_id(parent_id: str | None, model: ModelSpec) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:82](../../src/boldtailor/fit.py#L82)

<a id="definition-6"></a>

## `_analysis_id`

[Source, line 147](../../src/boldtailor/fit.py#L147)

```python
_analysis_id(data_id: str | None, model: Mapping[str, object] | None) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:144](../../src/boldtailor/fit.py#L144)
- [src/boldtailor/fit.py:52](../../src/boldtailor/fit.py#L52)
- [src/boldtailor/fit.py:81](../../src/boldtailor/fit.py#L81)

<a id="definition-7"></a>

## `_fit_activity`

[Source, line 156](../../src/boldtailor/fit.py#L156)

```python
_fit_activity(data: AnalysisData, model: Mapping[str, object], compiled: tuple[CompiledDesign, ...]) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:57](../../src/boldtailor/fit.py#L57)

<a id="definition-8"></a>

## `_task_delta_r2_activity`

[Source, line 167](../../src/boldtailor/fit.py#L167)

```python
_task_delta_r2_activity(data: AnalysisData, model: ModelSpec, compiled: tuple[CompiledDesign, ...], comparison: TaskDeltaR2Result, parent_id: str | None) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:99](../../src/boldtailor/fit.py#L99)

<a id="definition-9"></a>

## `_run_diagnostic`

[Source, line 195](../../src/boldtailor/fit.py#L195)

```python
_run_diagnostic(data: AnalysisData, compiled: CompiledDesign, run: int) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:162](../../src/boldtailor/fit.py#L162)
- [src/boldtailor/fit.py:175](../../src/boldtailor/fit.py#L175)

<a id="definition-10"></a>

## `_design_provenance`

[Source, line 216](../../src/boldtailor/fit.py#L216)

```python
_design_provenance(compiled: CompiledDesign) -> dict[str, int | float]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:64](../../src/boldtailor/fit.py#L64)
