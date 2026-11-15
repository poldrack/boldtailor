# `boldtailor.design`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/design.py)

Compile event-based task and nuisance design matrices.

## Definitions

- [`CompiledDesign`](#definition-1)
- [`compile_designs`](#definition-2)
- [`compile_nuisance_designs`](#definition-3)
- [`_compile_run`](#definition-4)
- [`_make_task_model_design`](#definition-5)
- [`_compile_nuisance_run`](#definition-6)
- [`_make_design_matrix`](#definition-7)
- [`kernel_task_columns`](#definition-8)
- [`_make_nuisance_matrix`](#definition-9)
- [`_validate_design_matrix`](#definition-10)
- [`_select_modeled_events`](#definition-11)
- [`_select_confounds`](#definition-12)

<a id="definition-1"></a>

## `CompiledDesign`

[Source, line 28](../../src/boldtailor/design.py#L28)

```python
@dataclass(frozen=True)
class CompiledDesign
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
matrix: pd.DataFrame
excluded_event_count: int
min_onset_cutoff: float
```

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:81](../../src/boldtailor/_hrf_glm_design.py#L81)
- [src/boldtailor/_hrf_glm_design.py:91](../../src/boldtailor/_hrf_glm_design.py#L91)
- [src/boldtailor/design.py:71](../../src/boldtailor/design.py#L71)
- [src/boldtailor/design.py:107](../../src/boldtailor/design.py#L107)

<a id="definition-2"></a>

## `compile_designs`

[Source, line 34](../../src/boldtailor/design.py#L34)

```python
compile_designs(data: AnalysisData, model: ModelSpec) -> tuple[CompiledDesign, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:30](../../src/boldtailor/_hrf_glm_design.py#L30)
- [src/boldtailor/fit.py:114](../../src/boldtailor/fit.py#L114)
- [src/boldtailor/fit.py:85](../../src/boldtailor/fit.py#L85)
- [tests/test_design.py:231](../../tests/test_design.py#L231)

<a id="definition-3"></a>

## `compile_nuisance_designs`

[Source, line 43](../../src/boldtailor/design.py#L43)

```python
compile_nuisance_designs(data: AnalysisData, model: ModelSpec) -> tuple[CompiledDesign, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:24](../../src/boldtailor/_hrf_glm_design.py#L24)
- [src/boldtailor/fit.py:86](../../src/boldtailor/fit.py#L86)
- [tests/test_design.py:75](../../tests/test_design.py#L75)
- [tests/test_design.py:104](../../tests/test_design.py#L104)

<a id="definition-4"></a>

## `_compile_run`

[Source, line 53](../../src/boldtailor/design.py#L53)

```python
_compile_run(frame_times: np.ndarray, events: pd.DataFrame, confounds: pd.DataFrame, model: ModelSpec, run: int) -> CompiledDesign
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/design.py:38](../../src/boldtailor/design.py#L38)

<a id="definition-5"></a>

## `_make_task_model_design`

[Source, line 78](../../src/boldtailor/design.py#L78)

```python
_make_task_model_design(frame_times: np.ndarray, events: pd.DataFrame, confounds: pd.DataFrame | None, model: ModelSpec, run: int) -> pd.DataFrame
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/design.py:67](../../src/boldtailor/design.py#L67)

<a id="definition-6"></a>

## `_compile_nuisance_run`

[Source, line 98](../../src/boldtailor/design.py#L98)

```python
_compile_nuisance_run(frame_times: np.ndarray, confounds: pd.DataFrame, model: ModelSpec, run: int) -> CompiledDesign
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/design.py:48](../../src/boldtailor/design.py#L48)

<a id="definition-7"></a>

## `_make_design_matrix`

[Source, line 114](../../src/boldtailor/design.py#L114)

```python
_make_design_matrix(frame_times: np.ndarray, events: pd.DataFrame, confounds: pd.DataFrame | None, model: ModelSpec, run: int) -> pd.DataFrame
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/design.py:65](../../src/boldtailor/design.py#L65)

<a id="definition-8"></a>

## `kernel_task_columns`

[Source, line 142](../../src/boldtailor/design.py#L142)

```python
kernel_task_columns(frame_times, events, kernel, model, run)
```

Nilearn condition columns for one peak-one kernel, in Nilearn's order.

Each event's amplitude is scaled so its response peaks at one. Task
columns are built apart from nuisance columns: Nilearn appends the
callable's name, which must neither rename contrasts nor hit a confound.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:88](../../src/boldtailor/_hrf_glm_design.py#L88)
- [src/boldtailor/design.py:123](../../src/boldtailor/design.py#L123)

<a id="definition-9"></a>

## `_make_nuisance_matrix`

[Source, line 168](../../src/boldtailor/design.py#L168)

```python
_make_nuisance_matrix(frame_times: np.ndarray, confounds: pd.DataFrame | None, model: ModelSpec, run: int) -> pd.DataFrame
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/design.py:94](../../src/boldtailor/design.py#L94)
- [src/boldtailor/design.py:105](../../src/boldtailor/design.py#L105)
- [src/boldtailor/design.py:124](../../src/boldtailor/design.py#L124)

<a id="definition-10"></a>

## `_validate_design_matrix`

[Source, line 192](../../src/boldtailor/design.py#L192)

```python
_validate_design_matrix(design: pd.DataFrame, run: int) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:80](../../src/boldtailor/_hrf_glm_design.py#L80)
- [src/boldtailor/_hrf_glm_design.py:90](../../src/boldtailor/_hrf_glm_design.py#L90)
- [src/boldtailor/design.py:70](../../src/boldtailor/design.py#L70)
- [src/boldtailor/design.py:106](../../src/boldtailor/design.py#L106)

<a id="definition-11"></a>

## `_select_modeled_events`

[Source, line 200](../../src/boldtailor/design.py#L200)

```python
_select_modeled_events(events: pd.DataFrame, frame_times: np.ndarray, min_onset: float, run: int) -> tuple[pd.DataFrame, int, float]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:67](../../src/boldtailor/_hrf_glm_design.py#L67)
- [src/boldtailor/_hrf_glm_design.py:85](../../src/boldtailor/_hrf_glm_design.py#L85)
- [src/boldtailor/design.py:61](../../src/boldtailor/design.py#L61)

<a id="definition-12"></a>

## `_select_confounds`

[Source, line 221](../../src/boldtailor/design.py#L221)

```python
_select_confounds(confounds: pd.DataFrame, names: tuple[str, ...], run: int) -> pd.DataFrame | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/design.py:60](../../src/boldtailor/design.py#L60)
- [src/boldtailor/design.py:104](../../src/boldtailor/design.py#L104)
