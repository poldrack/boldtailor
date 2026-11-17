# `boldtailor.data`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/data.py)

Own and validate run signals, events, confounds, timing and provenance.

## Definitions

- [`AnalysisData`](#definition-1)
- [`AnalysisData.n_runs`](#definition-2)
- [`AnalysisData.n_features`](#definition-3)
- [`AnalysisData.signals`](#definition-4)
- [`AnalysisData.events`](#definition-5)
- [`AnalysisData.confounds`](#definition-6)
- [`AnalysisData.frame_times`](#definition-7)
- [`AnalysisData.timing_source`](#definition-8)
- [`AnalysisData.provenance`](#definition-9)
- [`from_arrays`](#definition-10)
- [`_as_signal_runs`](#definition-11)
- [`_as_table_runs`](#definition-12)
- [`_validate_run_count`](#definition-13)
- [`run_labels_for`](#definition-14)
- [`_prepare_signal`](#definition-15)
- [`_validate_feature_counts`](#definition-16)
- [`_prepare_events`](#definition-17)
- [`_prepare_confounds`](#definition-18)
- [`_prepare_sources`](#definition-19)
- [`_anonymous_sources`](#definition-20)
- [`_normalization_activity`](#definition-21)
- [`_owned_table`](#definition-22)
- [`_prepare_timing`](#definition-23)
- [`_prepare_tr`](#definition-24)
- [`_as_frame_time_runs`](#definition-25)
- [`_prepare_frame_times`](#definition-26)

<a id="definition-1"></a>

## `AnalysisData`

[Source, line 34](../../src/boldtailor/data.py#L34)

```python
@dataclass(frozen=True)
class AnalysisData
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
_signals: tuple[np.ndarray, ...]
_events: tuple[pd.DataFrame, ...]
_confounds: tuple[pd.DataFrame, ...]
_frame_times: tuple[np.ndarray, ...]
_timing_source: str
_provenance: ProvenanceRecord
```

Usage in package/tests:

- [src/boldtailor/data.py:130](../../src/boldtailor/data.py#L130)
- [src/boldtailor/denoising.py:580](../../src/boldtailor/denoising.py#L580)

<a id="definition-2"></a>

## `AnalysisData.n_runs`

[Source, line 43](../../src/boldtailor/data.py#L43)

```python
@property
AnalysisData.n_runs(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `AnalysisData.n_features`

[Source, line 47](../../src/boldtailor/data.py#L47)

```python
@property
AnalysisData.n_features(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `AnalysisData.signals`

[Source, line 51](../../src/boldtailor/data.py#L51)

```python
@property
AnalysisData.signals(self) -> tuple[np.ndarray, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `AnalysisData.events`

[Source, line 55](../../src/boldtailor/data.py#L55)

```python
@property
AnalysisData.events(self) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `AnalysisData.confounds`

[Source, line 59](../../src/boldtailor/data.py#L59)

```python
@property
AnalysisData.confounds(self) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `AnalysisData.frame_times`

[Source, line 63](../../src/boldtailor/data.py#L63)

```python
@property
AnalysisData.frame_times(self) -> tuple[np.ndarray, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `AnalysisData.timing_source`

[Source, line 67](../../src/boldtailor/data.py#L67)

```python
@property
AnalysisData.timing_source(self) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `AnalysisData.provenance`

[Source, line 71](../../src/boldtailor/data.py#L71)

```python
@property
AnalysisData.provenance(self) -> ProvenanceRecord
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `from_arrays`

[Source, line 75](../../src/boldtailor/data.py#L75)

```python
from_arrays(signals: np.ndarray | Sequence[np.ndarray], events: pd.DataFrame | Sequence[pd.DataFrame], *, tr: float | None=None, frame_times: np.ndarray | Sequence[np.ndarray] | None=None, confounds: pd.DataFrame | Sequence[pd.DataFrame] | None=None, sources: Sequence[RunSources] | None=None, provenance_metadata: Mapping[str, object] | None=None) -> AnalysisData
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:342](../../src/boldtailor/_hrf_cv.py#L342)
- [src/boldtailor/workflow/inputs.py:203](../../src/boldtailor/workflow/inputs.py#L203)
- [tests/conftest.py:148](../../tests/conftest.py#L148)
- [tests/conftest.py:282](../../tests/conftest.py#L282)

<a id="definition-11"></a>

## `_as_signal_runs`

[Source, line 151](../../src/boldtailor/data.py#L151)

```python
_as_signal_runs(values: np.ndarray | Sequence[np.ndarray]) -> tuple[np.ndarray, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:93](../../src/boldtailor/data.py#L93)
- [src/boldtailor/prepared.py:154](../../src/boldtailor/prepared.py#L154)

<a id="definition-12"></a>

## `_as_table_runs`

[Source, line 162](../../src/boldtailor/data.py#L162)

```python
_as_table_runs(values: pd.DataFrame | Sequence[pd.DataFrame]) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:244](../../src/boldtailor/data.py#L244)
- [src/boldtailor/data.py:94](../../src/boldtailor/data.py#L94)

<a id="definition-13"></a>

## `_validate_run_count`

[Source, line 170](../../src/boldtailor/data.py#L170)

```python
_validate_run_count(signals: Sequence[object], values: Sequence[object], name: str) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:245](../../src/boldtailor/data.py#L245)
- [src/boldtailor/data.py:322](../../src/boldtailor/data.py#L322)
- [src/boldtailor/data.py:95](../../src/boldtailor/data.py#L95)
- [src/boldtailor/prepared.py:308](../../src/boldtailor/prepared.py#L308)

<a id="definition-14"></a>

## `run_labels_for`

[Source, line 177](../../src/boldtailor/data.py#L177)

```python
run_labels_for(data: AnalysisData, run_labels: Sequence[str] | None) -> tuple[str, ...]
```

Return validated run labels, defaulting to ``run-01``, ``run-02``, ...

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:215](../../src/boldtailor/_denoising_cv.py#L215)
- [src/boldtailor/_denoising_pool.py:73](../../src/boldtailor/_denoising_pool.py#L73)
- [src/boldtailor/_ridge_cv.py:253](../../src/boldtailor/_ridge_cv.py#L253)
- [src/boldtailor/_selected_hrf_fit.py:155](../../src/boldtailor/_selected_hrf_fit.py#L155)

<a id="definition-15"></a>

## `_prepare_signal`

[Source, line 199](../../src/boldtailor/data.py#L199)

```python
_prepare_signal(values: np.ndarray, run: int) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:103](../../src/boldtailor/data.py#L103)
- [src/boldtailor/prepared.py:162](../../src/boldtailor/prepared.py#L162)

<a id="definition-16"></a>

## `_validate_feature_counts`

[Source, line 214](../../src/boldtailor/data.py#L214)

```python
_validate_feature_counts(signals: tuple[np.ndarray, ...]) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:105](../../src/boldtailor/data.py#L105)
- [src/boldtailor/prepared.py:164](../../src/boldtailor/prepared.py#L164)

<a id="definition-17"></a>

## `_prepare_events`

[Source, line 219](../../src/boldtailor/data.py#L219)

```python
_prepare_events(frame: pd.DataFrame, run: int) -> pd.DataFrame
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:107](../../src/boldtailor/data.py#L107)

<a id="definition-18"></a>

## `_prepare_confounds`

[Source, line 238](../../src/boldtailor/data.py#L238)

```python
_prepare_confounds(confounds: pd.DataFrame | Sequence[pd.DataFrame] | None, signals: tuple[np.ndarray, ...]) -> tuple[pd.DataFrame, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:109](../../src/boldtailor/data.py#L109)

<a id="definition-19"></a>

## `_prepare_sources`

[Source, line 256](../../src/boldtailor/data.py#L256)

```python
_prepare_sources(*, run_count: int, sources: Sequence[RunSources] | None, include_confounds: bool) -> tuple[RunSources, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:96](../../src/boldtailor/data.py#L96)
- [src/boldtailor/prepared.py:155](../../src/boldtailor/prepared.py#L155)

<a id="definition-20"></a>

## `_anonymous_sources`

[Source, line 270](../../src/boldtailor/data.py#L270)

```python
_anonymous_sources(include_confounds: bool) -> RunSources
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:263](../../src/boldtailor/data.py#L263)

<a id="definition-21"></a>

## `_normalization_activity`

[Source, line 278](../../src/boldtailor/data.py#L278)

```python
_normalization_activity(*, timing_source: str, n_runs: int, n_features: int, provenance_metadata: Mapping[str, object] | None) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:115](../../src/boldtailor/data.py#L115)

<a id="definition-22"></a>

## `_owned_table`

[Source, line 298](../../src/boldtailor/data.py#L298)

```python
_owned_table(frame: pd.DataFrame) -> pd.DataFrame
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:480](../../src/boldtailor/_denoising_cv.py#L480)
- [src/boldtailor/_denoising_cv.py:481](../../src/boldtailor/_denoising_cv.py#L481)
- [src/boldtailor/data.py:225](../../src/boldtailor/data.py#L225)
- [src/boldtailor/data.py:56](../../src/boldtailor/data.py#L56)

<a id="definition-23"></a>

## `_prepare_timing`

[Source, line 308](../../src/boldtailor/data.py#L308)

```python
_prepare_timing(tr: float | None, frame_times: np.ndarray | Sequence[np.ndarray] | None, signals: tuple[np.ndarray, ...]) -> tuple[tuple[np.ndarray, ...], str]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:110](../../src/boldtailor/data.py#L110)
- [src/boldtailor/prepared.py:171](../../src/boldtailor/prepared.py#L171)

<a id="definition-24"></a>

## `_prepare_tr`

[Source, line 330](../../src/boldtailor/data.py#L330)

```python
_prepare_tr(tr: object) -> float
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:316](../../src/boldtailor/data.py#L316)

<a id="definition-25"></a>

## `_as_frame_time_runs`

[Source, line 339](../../src/boldtailor/data.py#L339)

```python
_as_frame_time_runs(values: np.ndarray | Sequence[np.ndarray] | None) -> tuple[np.ndarray, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:321](../../src/boldtailor/data.py#L321)

<a id="definition-26"></a>

## `_prepare_frame_times`

[Source, line 350](../../src/boldtailor/data.py#L350)

```python
_prepare_frame_times(values: np.ndarray, n_timepoints: int, run: int) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:324](../../src/boldtailor/data.py#L324)
