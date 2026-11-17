# `boldtailor.hrf_results`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/hrf_results.py)

Owned results for HRF selection and independent evaluation.

## Definitions

- [`HrfSelectionResult`](#definition-1)
- [`HrfSelectionResult.__post_init__`](#definition-2)
- [`HrfSelectionResult.eligibility`](#definition-3)
- [`HrfSelectionResult.parameter_bound_flags`](#definition-4)
- [`HrfSelectionResult.at_parameter_bound`](#definition-5)
- [`HrfSelectionResult.parameter_bound_table`](#definition-6)
- [`HrfEvaluationResult`](#definition-7)
- [`HrfEvaluationResult.__post_init__`](#definition-8)

<a id="definition-1"></a>

## `HrfSelectionResult`

[Source, line 15](../../src/boldtailor/hrf_results.py#L15)

```python
@dataclass(frozen=True, kw_only=True)
class HrfSelectionResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
hrf_indices: np.ndarray
cv_r2: np.ndarray
canonical_cv_r2: np.ndarray
delta_cv_r2: np.ndarray
library: HrfLibrary
_eligibility: pd.DataFrame
run_labels: tuple[str, ...]
feature_signature: str | None
provenance: ProvenanceRecord
task_model: TaskModel = TaskModel()
```

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:113](../../src/boldtailor/hrf_selection.py#L113)

<a id="definition-2"></a>

## `HrfSelectionResult.__post_init__`

[Source, line 27](../../src/boldtailor/hrf_results.py#L27)

```python
HrfSelectionResult.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `HrfSelectionResult.eligibility`

[Source, line 36](../../src/boldtailor/hrf_results.py#L36)

```python
@property
HrfSelectionResult.eligibility(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `HrfSelectionResult.parameter_bound_flags`

[Source, line 40](../../src/boldtailor/hrf_results.py#L40)

```python
@property
HrfSelectionResult.parameter_bound_flags(self) -> np.ndarray
```

``(n_features, 6, 2)`` edge flags of each selected custom kernel.

Parameters follow ``PARAMETER_NAMES[:6]``; the last axis is
``[low, high]``. True within 2 % of the library box width of an edge;
canonical (0) and ineligible (-1) features are all False.

<a id="definition-5"></a>

## `HrfSelectionResult.at_parameter_bound`

[Source, line 52](../../src/boldtailor/hrf_results.py#L52)

```python
@property
HrfSelectionResult.at_parameter_bound(self) -> np.ndarray
```

Features flagged at an edge of any informative library parameter.

<a id="definition-6"></a>

## `HrfSelectionResult.parameter_bound_table`

[Source, line 60](../../src/boldtailor/hrf_results.py#L60)

```python
HrfSelectionResult.parameter_bound_table(self) -> pd.DataFrame
```

Fraction of custom picks (ID > 0) flagged at each parameter edge.

<a id="definition-7"></a>

## `HrfEvaluationResult`

[Source, line 74](../../src/boldtailor/hrf_results.py#L74)

```python
@dataclass(frozen=True, kw_only=True)
class HrfEvaluationResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
training_selection: HrfSelectionResult
training_amplitudes: np.ndarray
test_r2: np.ndarray
canonical_test_r2: np.ndarray
delta_test_r2: np.ndarray
train_runs: tuple[int, ...]
test_runs: tuple[int, ...]
provenance: ProvenanceRecord
amplitude_names: tuple[str, ...] = ('task',)
```

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:283](../../src/boldtailor/hrf_selection.py#L283)

<a id="definition-8"></a>

## `HrfEvaluationResult.__post_init__`

[Source, line 85](../../src/boldtailor/hrf_results.py#L85)

```python
HrfEvaluationResult.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.
