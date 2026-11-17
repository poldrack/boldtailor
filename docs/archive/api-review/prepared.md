# `boldtailor.prepared`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/prepared.py)

Own externally compiled designs and explicit column roles.

## Definitions

- [`PreparedDesignAnalysis`](#definition-1)
- [`PreparedDesignAnalysis.from_arrays`](#definition-2)
- [`PreparedDesignAnalysis.n_runs`](#definition-3)
- [`PreparedDesignAnalysis.n_features`](#definition-4)
- [`PreparedDesignAnalysis.signals`](#definition-5)
- [`PreparedDesignAnalysis.design_matrices`](#definition-6)
- [`PreparedDesignAnalysis.frame_times`](#definition-7)
- [`PreparedDesignAnalysis.timing_source`](#definition-8)
- [`PreparedDesignAnalysis.column_roles`](#definition-9)
- [`PreparedDesignAnalysis.run_metadata`](#definition-10)
- [`PreparedDesignAnalysis.run_design_fingerprints`](#definition-11)
- [`PreparedDesignAnalysis.design_fingerprint`](#definition-12)
- [`PreparedDesignAnalysis.provenance`](#definition-13)
- [`_prepare_analysis`](#definition-14)
- [`_normalization_activity`](#definition-15)
- [`_run_design_fingerprint`](#definition-16)
- [`_aggregate_design_fingerprint`](#definition-17)
- [`_update_hash_with_chunk`](#definition-18)
- [`_canonical_json`](#definition-19)
- [`_prepare_designs`](#definition-20)
- [`_as_design_runs`](#definition-21)
- [`_prepare_design`](#definition-22)
- [`_validate_columns`](#definition-23)
- [`_validate_design_values`](#definition-24)
- [`_prepare_roles`](#definition-25)
- [`_as_role_runs`](#definition-26)
- [`_prepare_run_roles`](#definition-27)
- [`_prepare_metadata`](#definition-28)
- [`_as_metadata_runs`](#definition-29)
- [`_freeze_metadata`](#definition-30)
- [`_prepare_provenance_metadata`](#definition-31)

<a id="definition-1"></a>

## `PreparedDesignAnalysis`

[Source, line 45](../../src/boldtailor/prepared.py#L45)

```python
@dataclass(frozen=True)
class PreparedDesignAnalysis
```

Owned inputs with defensive public table accessors.

Package fitting code reads `_design_matrices` and `_column_roles` directly
to avoid copying all runs per diagnostic. It must never mutate those values.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
_signals: tuple[np.ndarray, ...]
_design_matrices: tuple[pd.DataFrame, ...]
_frame_times: tuple[np.ndarray, ...]
_timing_source: str
_column_roles: tuple[Mapping[str, ColumnRole], ...]
_run_metadata: tuple[Mapping[str, object], ...]
_run_design_fingerprints: tuple[str, ...]
_design_fingerprint: str
_provenance: ProvenanceRecord
```

<a id="definition-2"></a>

## `PreparedDesignAnalysis.from_arrays`

[Source, line 63](../../src/boldtailor/prepared.py#L63)

```python
@classmethod
PreparedDesignAnalysis.from_arrays(cls, signals: np.ndarray | Sequence[np.ndarray], design_matrices: pd.DataFrame | Sequence[pd.DataFrame], *, tr: float | None=None, frame_times: np.ndarray | Sequence[np.ndarray] | None=None, column_roles: Mapping[str, str] | Sequence[Mapping[str, str]], sources: Sequence[RunSources] | None=None, run_metadata: Mapping[str, object] | Sequence[Mapping[str, object]] | None=None, provenance_metadata: Mapping[str, object] | None=None) -> PreparedDesignAnalysis
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `PreparedDesignAnalysis.n_runs`

[Source, line 90](../../src/boldtailor/prepared.py#L90)

```python
@property
PreparedDesignAnalysis.n_runs(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `PreparedDesignAnalysis.n_features`

[Source, line 94](../../src/boldtailor/prepared.py#L94)

```python
@property
PreparedDesignAnalysis.n_features(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `PreparedDesignAnalysis.signals`

[Source, line 98](../../src/boldtailor/prepared.py#L98)

```python
@property
PreparedDesignAnalysis.signals(self) -> tuple[np.ndarray, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `PreparedDesignAnalysis.design_matrices`

[Source, line 102](../../src/boldtailor/prepared.py#L102)

```python
@property
PreparedDesignAnalysis.design_matrices(self) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `PreparedDesignAnalysis.frame_times`

[Source, line 106](../../src/boldtailor/prepared.py#L106)

```python
@property
PreparedDesignAnalysis.frame_times(self) -> tuple[np.ndarray, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `PreparedDesignAnalysis.timing_source`

[Source, line 110](../../src/boldtailor/prepared.py#L110)

```python
@property
PreparedDesignAnalysis.timing_source(self) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `PreparedDesignAnalysis.column_roles`

[Source, line 114](../../src/boldtailor/prepared.py#L114)

```python
@property
PreparedDesignAnalysis.column_roles(self) -> tuple[Mapping[str, ColumnRole], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `PreparedDesignAnalysis.run_metadata`

[Source, line 118](../../src/boldtailor/prepared.py#L118)

```python
@property
PreparedDesignAnalysis.run_metadata(self) -> tuple[Mapping[str, object], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-11"></a>

## `PreparedDesignAnalysis.run_design_fingerprints`

[Source, line 122](../../src/boldtailor/prepared.py#L122)

```python
@property
PreparedDesignAnalysis.run_design_fingerprints(self) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-12"></a>

## `PreparedDesignAnalysis.design_fingerprint`

[Source, line 126](../../src/boldtailor/prepared.py#L126)

```python
@property
PreparedDesignAnalysis.design_fingerprint(self) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-13"></a>

## `PreparedDesignAnalysis.provenance`

[Source, line 130](../../src/boldtailor/prepared.py#L130)

```python
@property
PreparedDesignAnalysis.provenance(self) -> ProvenanceRecord
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-14"></a>

## `_prepare_analysis`

[Source, line 134](../../src/boldtailor/prepared.py#L134)

```python
_prepare_analysis(*, cls: type[PreparedDesignAnalysis], signals: np.ndarray | Sequence[np.ndarray], design_matrices: pd.DataFrame | Sequence[pd.DataFrame], tr: float | None, frame_times: np.ndarray | Sequence[np.ndarray] | None, column_roles: Mapping[str, str] | Sequence[Mapping[str, str]], sources: Sequence[RunSources] | None, run_metadata: Mapping[str, object] | Sequence[Mapping[str, object]] | None, provenance_metadata: Mapping[str, object] | None) -> PreparedDesignAnalysis
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:77](../../src/boldtailor/prepared.py#L77)

<a id="definition-15"></a>

## `_normalization_activity`

[Source, line 224](../../src/boldtailor/prepared.py#L224)

```python
_normalization_activity(*, timing_source: str, designs: Sequence[pd.DataFrame], roles: Sequence[Mapping[str, ColumnRole]], n_features: int, run_fingerprints: Sequence[str], design_fingerprint: str, run_metadata: Sequence[Mapping[str, object]], provenance_metadata: Mapping[str, object] | None) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:181](../../src/boldtailor/prepared.py#L181)

<a id="definition-16"></a>

## `_run_design_fingerprint`

[Source, line 259](../../src/boldtailor/prepared.py#L259)

```python
_run_design_fingerprint(design: pd.DataFrame, frame_times: np.ndarray, roles: Mapping[str, ColumnRole]) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:177](../../src/boldtailor/prepared.py#L177)

<a id="definition-17"></a>

## `_aggregate_design_fingerprint`

[Source, line 284](../../src/boldtailor/prepared.py#L284)

```python
_aggregate_design_fingerprint(run_fingerprints: Sequence[str]) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:180](../../src/boldtailor/prepared.py#L180)

<a id="definition-18"></a>

## `_update_hash_with_chunk`

[Source, line 292](../../src/boldtailor/prepared.py#L292)

```python
_update_hash_with_chunk(hasher: Any, payload: bytes) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:265](../../src/boldtailor/prepared.py#L265)
- [src/boldtailor/prepared.py:272](../../src/boldtailor/prepared.py#L272)
- [src/boldtailor/prepared.py:273](../../src/boldtailor/prepared.py#L273)
- [src/boldtailor/prepared.py:277](../../src/boldtailor/prepared.py#L277)

<a id="definition-19"></a>

## `_canonical_json`

[Source, line 297](../../src/boldtailor/prepared.py#L297)

```python
_canonical_json(payload: Mapping[str, object]) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:272](../../src/boldtailor/prepared.py#L272)
- [src/boldtailor/prepared.py:288](../../src/boldtailor/prepared.py#L288)

<a id="definition-20"></a>

## `_prepare_designs`

[Source, line 303](../../src/boldtailor/prepared.py#L303)

```python
_prepare_designs(values: pd.DataFrame | Sequence[pd.DataFrame], signals: Sequence[np.ndarray]) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:165](../../src/boldtailor/prepared.py#L165)

<a id="definition-21"></a>

## `_as_design_runs`

[Source, line 315](../../src/boldtailor/prepared.py#L315)

```python
_as_design_runs(values: pd.DataFrame | Sequence[pd.DataFrame]) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:307](../../src/boldtailor/prepared.py#L307)

<a id="definition-22"></a>

## `_prepare_design`

[Source, line 326](../../src/boldtailor/prepared.py#L326)

```python
_prepare_design(design: pd.DataFrame, n_rows: int, run: int) -> pd.DataFrame
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:310](../../src/boldtailor/prepared.py#L310)

<a id="definition-23"></a>

## `_validate_columns`

[Source, line 339](../../src/boldtailor/prepared.py#L339)

```python
_validate_columns(columns: pd.Index, run: int) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:331](../../src/boldtailor/prepared.py#L331)

<a id="definition-24"></a>

## `_validate_design_values`

[Source, line 346](../../src/boldtailor/prepared.py#L346)

```python
_validate_design_values(design: pd.DataFrame, run: int) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:334](../../src/boldtailor/prepared.py#L334)

<a id="definition-25"></a>

## `_prepare_roles`

[Source, line 356](../../src/boldtailor/prepared.py#L356)

```python
_prepare_roles(values: Mapping[str, str] | Sequence[Mapping[str, str]], designs: Sequence[pd.DataFrame]) -> tuple[Mapping[str, ColumnRole], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:166](../../src/boldtailor/prepared.py#L166)

<a id="definition-26"></a>

## `_as_role_runs`

[Source, line 368](../../src/boldtailor/prepared.py#L368)

```python
_as_role_runs(values: Mapping[str, str] | Sequence[Mapping[str, str]]) -> tuple[Mapping[str, str], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:360](../../src/boldtailor/prepared.py#L360)

<a id="definition-27"></a>

## `_prepare_run_roles`

[Source, line 381](../../src/boldtailor/prepared.py#L381)

```python
_prepare_run_roles(roles: Mapping[str, str], columns: pd.Index, run: int) -> Mapping[str, ColumnRole]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:363](../../src/boldtailor/prepared.py#L363)

<a id="definition-28"></a>

## `_prepare_metadata`

[Source, line 393](../../src/boldtailor/prepared.py#L393)

```python
_prepare_metadata(values: Mapping[str, object] | Sequence[Mapping[str, object]] | None, run_count: int) -> tuple[Mapping[str, object], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:167](../../src/boldtailor/prepared.py#L167)

<a id="definition-29"></a>

## `_as_metadata_runs`

[Source, line 404](../../src/boldtailor/prepared.py#L404)

```python
_as_metadata_runs(values: Mapping[str, object] | Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:399](../../src/boldtailor/prepared.py#L399)

<a id="definition-30"></a>

## `_freeze_metadata`

[Source, line 415](../../src/boldtailor/prepared.py#L415)

```python
_freeze_metadata(values: Mapping[str, object]) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:401](../../src/boldtailor/prepared.py#L401)

<a id="definition-31"></a>

## `_prepare_provenance_metadata`

[Source, line 421](../../src/boldtailor/prepared.py#L421)

```python
_prepare_provenance_metadata(values: Mapping[str, object] | None) -> Mapping[str, object] | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:168](../../src/boldtailor/prepared.py#L168)
