# `boldtailor.results`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/results.py)

Own conventional contrast maps, R² diagnostics and result builders.

## Definitions

- [`_NilearnContrast`](#definition-1)
- [`_NilearnContrast.effect_size`](#definition-2)
- [`_NilearnContrast.effect_variance`](#definition-3)
- [`_NilearnContrast.stat`](#definition-4)
- [`_NilearnContrast.z_score`](#definition-5)
- [`_NilearnContrast.p_value`](#definition-6)
- [`_ContrastResult`](#definition-7)
- [`_ContrastResult.__post_init__`](#definition-8)
- [`owned_contrasts`](#definition-9)
- [`_AnalysisAccessors`](#definition-10)
- [`_AnalysisAccessors.contrast_names`](#definition-11)
- [`_AnalysisAccessors.run_r2`](#definition-12)
- [`_AnalysisAccessors.r2`](#definition-13)
- [`_AnalysisAccessors.provenance`](#definition-14)
- [`_AnalysisAccessors.effect`](#definition-15)
- [`_AnalysisAccessors.variance`](#definition-16)
- [`_AnalysisAccessors.stat`](#definition-17)
- [`_AnalysisAccessors.z_score`](#definition-18)
- [`_AnalysisAccessors.one_sided_p_value`](#definition-19)
- [`_AnalysisAccessors._contrast`](#definition-20)
- [`AnalysisResult`](#definition-21)
- [`AnalysisResult.__post_init__`](#definition-22)
- [`AnalysisResult.design_matrices`](#definition-23)
- [`AnalysisResult.design_provenance`](#definition-24)
- [`TaskDeltaR2Result`](#definition-25)
- [`TaskDeltaR2Result.__post_init__`](#definition-26)
- [`TaskDeltaR2Result.full_r2`](#definition-27)
- [`TaskDeltaR2Result.nuisance_r2`](#definition-28)
- [`TaskDeltaR2Result.raw_delta_r2`](#definition-29)
- [`TaskDeltaR2Result.delta_r2`](#definition-30)
- [`TaskDeltaR2Result.negative_voxel_count`](#definition-31)
- [`TaskDeltaR2Result.raw_min`](#definition-32)
- [`TaskDeltaR2Result.nuisance_design_matrices`](#definition-33)
- [`TaskDeltaR2Result.provenance`](#definition-34)
- [`make_result`](#definition-35)
- [`make_task_delta_r2_result`](#definition-36)
- [`_validate_r2_pair`](#definition-37)
- [`_validate_r2_array`](#definition-38)
- [`_delta_r2_arrays`](#definition-39)
- [`_copy_nuisance_designs`](#definition-40)
- [`_is_finite_numeric_design`](#definition-41)
- [`contrast_result`](#definition-42)
- [`mask_contrast`](#definition-43)
- [`mask_contrast.masked`](#definition-44)

<a id="definition-1"></a>

## `_NilearnContrast`

[Source, line 24](../../src/boldtailor/results.py#L24)

```python
class _NilearnContrast(Protocol)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-2"></a>

## `_NilearnContrast.effect_size`

[Source, line 25](../../src/boldtailor/results.py#L25)

```python
_NilearnContrast.effect_size(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `_NilearnContrast.effect_variance`

[Source, line 27](../../src/boldtailor/results.py#L27)

```python
_NilearnContrast.effect_variance(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `_NilearnContrast.stat`

[Source, line 29](../../src/boldtailor/results.py#L29)

```python
_NilearnContrast.stat(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `_NilearnContrast.z_score`

[Source, line 31](../../src/boldtailor/results.py#L31)

```python
_NilearnContrast.z_score(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `_NilearnContrast.p_value`

[Source, line 33](../../src/boldtailor/results.py#L33)

```python
_NilearnContrast.p_value(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `_ContrastResult`

[Source, line 37](../../src/boldtailor/results.py#L37)

```python
@dataclass(frozen=True, kw_only=True)
class _ContrastResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
effect: np.ndarray
variance: np.ndarray
stat: np.ndarray
z_score: np.ndarray
one_sided_p_value: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:129](../../src/boldtailor/_hrf_glm.py#L129)
- [src/boldtailor/results.py:299](../../src/boldtailor/results.py#L299)
- [src/boldtailor/results.py:316](../../src/boldtailor/results.py#L316)

<a id="definition-8"></a>

## `_ContrastResult.__post_init__`

[Source, line 44](../../src/boldtailor/results.py#L44)

```python
_ContrastResult.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `owned_contrasts`

[Source, line 48](../../src/boldtailor/results.py#L48)

```python
owned_contrasts(contrasts: Mapping[str, _ContrastResult]) -> Mapping[str, _ContrastResult]
```

Read-only name-to-contrast mapping; each contrast owns its arrays.

Usage in package/tests:

- [src/boldtailor/hrf_glm_results.py:33](../../src/boldtailor/hrf_glm_results.py#L33)
- [src/boldtailor/results.py:113](../../src/boldtailor/results.py#L113)

<a id="definition-10"></a>

## `_AnalysisAccessors`

[Source, line 57](../../src/boldtailor/results.py#L57)

```python
class _AnalysisAccessors
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
_contrasts: Mapping[str, _ContrastResult]
_run_r2: tuple[np.ndarray, ...]
_r2: np.ndarray
_provenance: ProvenanceRecord
```

<a id="definition-11"></a>

## `_AnalysisAccessors.contrast_names`

[Source, line 64](../../src/boldtailor/results.py#L64)

```python
@property
_AnalysisAccessors.contrast_names(self) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-12"></a>

## `_AnalysisAccessors.run_r2`

[Source, line 68](../../src/boldtailor/results.py#L68)

```python
@property
_AnalysisAccessors.run_r2(self) -> tuple[np.ndarray, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-13"></a>

## `_AnalysisAccessors.r2`

[Source, line 72](../../src/boldtailor/results.py#L72)

```python
@property
_AnalysisAccessors.r2(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-14"></a>

## `_AnalysisAccessors.provenance`

[Source, line 76](../../src/boldtailor/results.py#L76)

```python
@property
_AnalysisAccessors.provenance(self) -> ProvenanceRecord
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-15"></a>

## `_AnalysisAccessors.effect`

[Source, line 79](../../src/boldtailor/results.py#L79)

```python
_AnalysisAccessors.effect(self, name: str) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-16"></a>

## `_AnalysisAccessors.variance`

[Source, line 82](../../src/boldtailor/results.py#L82)

```python
_AnalysisAccessors.variance(self, name: str) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-17"></a>

## `_AnalysisAccessors.stat`

[Source, line 85](../../src/boldtailor/results.py#L85)

```python
_AnalysisAccessors.stat(self, name: str) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-18"></a>

## `_AnalysisAccessors.z_score`

[Source, line 88](../../src/boldtailor/results.py#L88)

```python
_AnalysisAccessors.z_score(self, name: str) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-19"></a>

## `_AnalysisAccessors.one_sided_p_value`

[Source, line 91](../../src/boldtailor/results.py#L91)

```python
_AnalysisAccessors.one_sided_p_value(self, name: str) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-20"></a>

## `_AnalysisAccessors._contrast`

[Source, line 94](../../src/boldtailor/results.py#L94)

```python
_AnalysisAccessors._contrast(self, name: str) -> _ContrastResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-21"></a>

## `AnalysisResult`

[Source, line 102](../../src/boldtailor/results.py#L102)

```python
@dataclass(frozen=True, kw_only=True)
class AnalysisResult(_AnalysisAccessors)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
_contrasts: Mapping[str, _ContrastResult]
_design_matrices: tuple[pd.DataFrame, ...]
_design_provenance: tuple[DesignProvenance, ...]
_run_r2: tuple[np.ndarray, ...]
_r2: np.ndarray
_provenance: ProvenanceRecord
```

Usage in package/tests:

- [src/boldtailor/results.py:199](../../src/boldtailor/results.py#L199)

<a id="definition-22"></a>

## `AnalysisResult.__post_init__`

[Source, line 110](../../src/boldtailor/results.py#L110)

```python
AnalysisResult.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-23"></a>

## `AnalysisResult.design_matrices`

[Source, line 123](../../src/boldtailor/results.py#L123)

```python
@property
AnalysisResult.design_matrices(self) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-24"></a>

## `AnalysisResult.design_provenance`

[Source, line 127](../../src/boldtailor/results.py#L127)

```python
@property
AnalysisResult.design_provenance(self) -> tuple[DesignProvenance, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-25"></a>

## `TaskDeltaR2Result`

[Source, line 132](../../src/boldtailor/results.py#L132)

```python
@dataclass(frozen=True, kw_only=True)
class TaskDeltaR2Result
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
_full_r2: np.ndarray
_nuisance_r2: np.ndarray
_raw_delta_r2: np.ndarray
_delta_r2: np.ndarray
_negative_voxel_count: int
_raw_min: float
_nuisance_design_matrices: tuple[pd.DataFrame, ...]
_provenance: ProvenanceRecord
```

Usage in package/tests:

- [src/boldtailor/results.py:220](../../src/boldtailor/results.py#L220)

<a id="definition-26"></a>

## `TaskDeltaR2Result.__post_init__`

[Source, line 142](../../src/boldtailor/results.py#L142)

```python
TaskDeltaR2Result.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-27"></a>

## `TaskDeltaR2Result.full_r2`

[Source, line 156](../../src/boldtailor/results.py#L156)

```python
@property
TaskDeltaR2Result.full_r2(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-28"></a>

## `TaskDeltaR2Result.nuisance_r2`

[Source, line 160](../../src/boldtailor/results.py#L160)

```python
@property
TaskDeltaR2Result.nuisance_r2(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-29"></a>

## `TaskDeltaR2Result.raw_delta_r2`

[Source, line 164](../../src/boldtailor/results.py#L164)

```python
@property
TaskDeltaR2Result.raw_delta_r2(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-30"></a>

## `TaskDeltaR2Result.delta_r2`

[Source, line 168](../../src/boldtailor/results.py#L168)

```python
@property
TaskDeltaR2Result.delta_r2(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-31"></a>

## `TaskDeltaR2Result.negative_voxel_count`

[Source, line 172](../../src/boldtailor/results.py#L172)

```python
@property
TaskDeltaR2Result.negative_voxel_count(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-32"></a>

## `TaskDeltaR2Result.raw_min`

[Source, line 176](../../src/boldtailor/results.py#L176)

```python
@property
TaskDeltaR2Result.raw_min(self) -> float
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-33"></a>

## `TaskDeltaR2Result.nuisance_design_matrices`

[Source, line 180](../../src/boldtailor/results.py#L180)

```python
@property
TaskDeltaR2Result.nuisance_design_matrices(self) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-34"></a>

## `TaskDeltaR2Result.provenance`

[Source, line 184](../../src/boldtailor/results.py#L184)

```python
@property
TaskDeltaR2Result.provenance(self) -> ProvenanceRecord
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-35"></a>

## `make_result`

[Source, line 191](../../src/boldtailor/results.py#L191)

```python
make_result(contrasts: Mapping[str, _ContrastResult], designs: tuple[pd.DataFrame, ...], design_provenance: tuple[DesignProvenance, ...], run_r2: tuple[np.ndarray, ...], r2: np.ndarray, provenance: ProvenanceRecord) -> AnalysisResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:61](../../src/boldtailor/fit.py#L61)
- [src/boldtailor/prepared_fit.py:69](../../src/boldtailor/prepared_fit.py#L69)

<a id="definition-36"></a>

## `make_task_delta_r2_result`

[Source, line 209](../../src/boldtailor/results.py#L209)

```python
make_task_delta_r2_result(*, full_r2: np.ndarray, nuisance_r2: np.ndarray, nuisance_designs: tuple[pd.DataFrame, ...], provenance: ProvenanceRecord, allow_undefined: bool=False) -> TaskDeltaR2Result
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:202](../../src/boldtailor/_hrf_glm.py#L202)
- [src/boldtailor/fit.py:93](../../src/boldtailor/fit.py#L93)
- [src/boldtailor/prepared_fit.py:107](../../src/boldtailor/prepared_fit.py#L107)
- [tests/test_fit.py:580](../../tests/test_fit.py#L580)

<a id="definition-37"></a>

## `_validate_r2_pair`

[Source, line 232](../../src/boldtailor/results.py#L232)

```python
_validate_r2_pair(full_r2: object, nuisance_r2: object, allow_undefined: bool) -> tuple[np.ndarray, np.ndarray]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/results.py:217](../../src/boldtailor/results.py#L217)

<a id="definition-38"></a>

## `_validate_r2_array`

[Source, line 244](../../src/boldtailor/results.py#L244)

```python
_validate_r2_array(values: object, name: str, allow_undefined: bool=False) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/results.py:237](../../src/boldtailor/results.py#L237)
- [src/boldtailor/results.py:238](../../src/boldtailor/results.py#L238)

<a id="definition-39"></a>

## `_delta_r2_arrays`

[Source, line 257](../../src/boldtailor/results.py#L257)

```python
_delta_r2_arrays(full: np.ndarray, nuisance: np.ndarray, allow_undefined: bool=False) -> tuple[np.ndarray, np.ndarray]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/results.py:218](../../src/boldtailor/results.py#L218)

<a id="definition-40"></a>

## `_copy_nuisance_designs`

[Source, line 273](../../src/boldtailor/results.py#L273)

```python
_copy_nuisance_designs(designs: tuple[pd.DataFrame, ...]) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/results.py:227](../../src/boldtailor/results.py#L227)

<a id="definition-41"></a>

## `_is_finite_numeric_design`

[Source, line 284](../../src/boldtailor/results.py#L284)

```python
_is_finite_numeric_design(frame: pd.DataFrame) -> bool
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/results.py:279](../../src/boldtailor/results.py#L279)

<a id="definition-42"></a>

## `contrast_result`

[Source, line 298](../../src/boldtailor/results.py#L298)

```python
contrast_result(contrast: _NilearnContrast) -> _ContrastResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:145](../../src/boldtailor/_conventional.py#L145)

<a id="definition-43"></a>

## `mask_contrast`

[Source, line 308](../../src/boldtailor/results.py#L308)

```python
mask_contrast(result: _ContrastResult, undefined: np.ndarray) -> _ContrastResult
```

Return a copy with every statistic set to NaN where ``undefined`` is True.

Usage in package/tests:

- [src/boldtailor/_conventional.py:61](../../src/boldtailor/_conventional.py#L61)

<a id="definition-44"></a>

## `mask_contrast.masked`

[Source, line 311](../../src/boldtailor/results.py#L311)

```python
mask_contrast.masked(values: np.ndarray) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.
