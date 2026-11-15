# `boldtailor.hrf_glm_results`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/hrf_glm_results.py)

Conventional contrast results with spatially varying HRF designs.

## Definitions

- [`HrfAnalysisResult`](#definition-1)
- [`HrfAnalysisResult.__post_init__`](#definition-2)
- [`HrfAnalysisResult.hrf_selection`](#definition-3)
- [`HrfAnalysisResult.hrf_indices`](#definition-4)
- [`HrfAnalysisResult.selection_provenance`](#definition-5)
- [`HrfAnalysisResult.group_design`](#definition-6)
- [`HrfAnalysisResult.group_design_provenance`](#definition-7)

<a id="definition-1"></a>

## `HrfAnalysisResult`

[Source, line 17](../../src/boldtailor/hrf_glm_results.py#L17)

```python
@dataclass(frozen=True, kw_only=True)
class HrfAnalysisResult(_AnalysisAccessors)
```

Contrast maps in input feature order; designs rebuilt per (run, HRF ID).

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
_contrasts: Mapping[str, _ContrastResult]
_run_r2: tuple[np.ndarray, ...]
_r2: np.ndarray
_provenance: ProvenanceRecord
_selection: HrfSelectionResult
_group_design_provenance: Mapping[tuple[int, int], dict]
_group_builder: Callable[[int, int], pd.DataFrame] = field(compare=False, repr=False)
```

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:136](../../src/boldtailor/_hrf_glm.py#L136)

<a id="definition-2"></a>

## `HrfAnalysisResult.__post_init__`

[Source, line 30](../../src/boldtailor/hrf_glm_results.py#L30)

```python
HrfAnalysisResult.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `HrfAnalysisResult.hrf_selection`

[Source, line 40](../../src/boldtailor/hrf_glm_results.py#L40)

```python
@property
HrfAnalysisResult.hrf_selection(self) -> HrfSelectionResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `HrfAnalysisResult.hrf_indices`

[Source, line 44](../../src/boldtailor/hrf_glm_results.py#L44)

```python
@property
HrfAnalysisResult.hrf_indices(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `HrfAnalysisResult.selection_provenance`

[Source, line 48](../../src/boldtailor/hrf_glm_results.py#L48)

```python
@property
HrfAnalysisResult.selection_provenance(self) -> ProvenanceRecord
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `HrfAnalysisResult.group_design`

[Source, line 51](../../src/boldtailor/hrf_glm_results.py#L51)

```python
HrfAnalysisResult.group_design(self, run: int, hrf_id: int) -> pd.DataFrame
```

Recompiled design for a fitted (run, HRF ID); KeyError otherwise.

<a id="definition-7"></a>

## `HrfAnalysisResult.group_design_provenance`

[Source, line 56](../../src/boldtailor/hrf_glm_results.py#L56)

```python
@property
HrfAnalysisResult.group_design_provenance(self) -> dict
```

No source docstring. Read the linked implementation and call sites for the precise contract.
