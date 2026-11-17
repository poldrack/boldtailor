# `boldtailor.prepared_fit`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/prepared_fit.py)

Fit externally compiled designs and compare task versus nuisance variance.

## Definitions

- [`_PreparedFitSpec`](#definition-1)
- [`fit_prepared`](#definition-2)
- [`task_delta_r2_prepared`](#definition-3)
- [`_parent_fit_spec`](#definition-4)
- [`_stored_fit_spec`](#definition-5)
- [`_prepare_comparison_identity`](#definition-6)
- [`_prepared_comparison_id`](#definition-7)
- [`_validate_prepared_parent`](#definition-8)
- [`_nuisance_designs`](#definition-9)
- [`_prepared_delta_activity`](#definition-10)
- [`_prepared_delta_run_diagnostic`](#definition-11)
- [`_prepare_fit_spec`](#definition-12)
- [`_prepared_design_provenance`](#definition-13)
- [`_model_identity`](#definition-14)
- [`_fit_activity`](#definition-15)
- [`_run_diagnostic`](#definition-16)
- [`_role_counts`](#definition-17)

<a id="definition-1"></a>

## `_PreparedFitSpec`

[Source, line 43](../../src/boldtailor/prepared_fit.py#L43)

```python
@dataclass(frozen=True)
class _PreparedFitSpec
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
contrasts: Mapping[str, ContrastValue]
noise_model: str
model_metadata: Mapping[str, object]
```

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:280](../../src/boldtailor/prepared_fit.py#L280)

<a id="definition-2"></a>

## `fit_prepared`

[Source, line 49](../../src/boldtailor/prepared_fit.py#L49)

```python
fit_prepared(prepared: PreparedDesignAnalysis, *, contrasts: Mapping[str, ContrastValue], noise_model: str='ar1', model_metadata: Mapping[str, object] | None=None) -> AnalysisResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [tests/check_installed_package.py:27](../../tests/check_installed_package.py#L27)
- [tests/test_lifecycle.py:105](../../tests/test_lifecycle.py#L105)
- [tests/test_lifecycle.py:180](../../tests/test_lifecycle.py#L180)
- [tests/test_lifecycle.py:181](../../tests/test_lifecycle.py#L181)

<a id="definition-3"></a>

## `task_delta_r2_prepared`

[Source, line 79](../../src/boldtailor/prepared_fit.py#L79)

```python
task_delta_r2_prepared(prepared: PreparedDesignAnalysis, full_result: AnalysisResult, *, contrasts: Mapping[str, ContrastValue]=UNSET, noise_model: str=UNSET, model_metadata: Mapping[str, object] | None=UNSET) -> TaskDeltaR2Result
```

Nested-OLS task delta r-squared for a :func:`fit_prepared` result.

The contrasts, noise model, and model metadata are read from
``full_result``'s provenance. Passing them again is deprecated; supplied
values must match the stored model.

Usage in package/tests:

- [tests/test_lifecycle.py:184](../../tests/test_lifecycle.py#L184)
- [tests/test_lifecycle.py:185](../../tests/test_lifecycle.py#L185)
- [tests/test_prepared_fit.py:676](../../tests/test_prepared_fit.py#L676)
- [tests/test_prepared_fit.py:702](../../tests/test_prepared_fit.py#L702)

<a id="definition-4"></a>

## `_parent_fit_spec`

[Source, line 124](../../src/boldtailor/prepared_fit.py#L124)

```python
_parent_fit_spec(full_result: AnalysisResult, contrasts: object, noise_model: object, model_metadata: object) -> _PreparedFitSpec
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:94](../../src/boldtailor/prepared_fit.py#L94)

<a id="definition-5"></a>

## `_stored_fit_spec`

[Source, line 151](../../src/boldtailor/prepared_fit.py#L151)

```python
_stored_fit_spec(full_result: AnalysisResult) -> _PreparedFitSpec
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:130](../../src/boldtailor/prepared_fit.py#L130)

<a id="definition-6"></a>

## `_prepare_comparison_identity`

[Source, line 163](../../src/boldtailor/prepared_fit.py#L163)

```python
_prepare_comparison_identity(prepared: PreparedDesignAnalysis, fit_spec: _PreparedFitSpec, data_id: str | None) -> tuple[str | None, str | None]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:95](../../src/boldtailor/prepared_fit.py#L95)

<a id="definition-7"></a>

## `_prepared_comparison_id`

[Source, line 173](../../src/boldtailor/prepared_fit.py#L173)

```python
_prepared_comparison_id(data_id: str | None, parent_id: str | None, fit_spec: _PreparedFitSpec) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:169](../../src/boldtailor/prepared_fit.py#L169)

<a id="definition-8"></a>

## `_validate_prepared_parent`

[Source, line 186](../../src/boldtailor/prepared_fit.py#L186)

```python
_validate_prepared_parent(prepared: PreparedDesignAnalysis, full_result: AnalysisResult, parent_id: str | None) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:99](../../src/boldtailor/prepared_fit.py#L99)

<a id="definition-9"></a>

## `_nuisance_designs`

[Source, line 202](../../src/boldtailor/prepared_fit.py#L202)

```python
_nuisance_designs(prepared: PreparedDesignAnalysis) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:100](../../src/boldtailor/prepared_fit.py#L100)

<a id="definition-10"></a>

## `_prepared_delta_activity`

[Source, line 230](../../src/boldtailor/prepared_fit.py#L230)

```python
_prepared_delta_activity(prepared: PreparedDesignAnalysis, fit_spec: _PreparedFitSpec, nuisance_designs: tuple[pd.DataFrame, ...], comparison: TaskDeltaR2Result, parent_id: str) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:113](../../src/boldtailor/prepared_fit.py#L113)

<a id="definition-11"></a>

## `_prepared_delta_run_diagnostic`

[Source, line 260](../../src/boldtailor/prepared_fit.py#L260)

```python
_prepared_delta_run_diagnostic(prepared: PreparedDesignAnalysis, nuisance_design: pd.DataFrame, run: int) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:248](../../src/boldtailor/prepared_fit.py#L248)

<a id="definition-12"></a>

## `_prepare_fit_spec`

[Source, line 270](../../src/boldtailor/prepared_fit.py#L270)

```python
_prepare_fit_spec(contrasts: Mapping[str, ContrastValue], noise_model: str, model_metadata: Mapping[str, object] | None) -> _PreparedFitSpec
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:138](../../src/boldtailor/prepared_fit.py#L138)
- [src/boldtailor/prepared_fit.py:156](../../src/boldtailor/prepared_fit.py#L156)
- [src/boldtailor/prepared_fit.py:57](../../src/boldtailor/prepared_fit.py#L57)

<a id="definition-13"></a>

## `_prepared_design_provenance`

[Source, line 287](../../src/boldtailor/prepared_fit.py#L287)

```python
_prepared_design_provenance(prepared: PreparedDesignAnalysis) -> tuple[dict[str, str], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:72](../../src/boldtailor/prepared_fit.py#L72)

<a id="definition-14"></a>

## `_model_identity`

[Source, line 296](../../src/boldtailor/prepared_fit.py#L296)

```python
_model_identity(prepared: PreparedDesignAnalysis, fit_spec: _PreparedFitSpec) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:58](../../src/boldtailor/prepared_fit.py#L58)
- [src/boldtailor/prepared_fit.py:168](../../src/boldtailor/prepared_fit.py#L168)

<a id="definition-15"></a>

## `_fit_activity`

[Source, line 309](../../src/boldtailor/prepared_fit.py#L309)

```python
_fit_activity(prepared: PreparedDesignAnalysis, model: Mapping[str, object]) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:67](../../src/boldtailor/prepared_fit.py#L67)

<a id="definition-16"></a>

## `_run_diagnostic`

[Source, line 321](../../src/boldtailor/prepared_fit.py#L321)

```python
_run_diagnostic(prepared: PreparedDesignAnalysis, run: int) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:265](../../src/boldtailor/prepared_fit.py#L265)
- [src/boldtailor/prepared_fit.py:317](../../src/boldtailor/prepared_fit.py#L317)

<a id="definition-17"></a>

## `_role_counts`

[Source, line 340](../../src/boldtailor/prepared_fit.py#L340)

```python
_role_counts(roles: Mapping[str, str]) -> dict[str, int]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:332](../../src/boldtailor/prepared_fit.py#L332)
