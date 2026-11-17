# `boldtailor.ridge_results`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/ridge_results.py)

Owned numerical results for trial encoding and ridge selection.

## Definitions

- [`TrialEncodingResult`](#definition-1)
- [`TrialEncodingResult.__post_init__`](#definition-2)
- [`RidgeSelection`](#definition-3)
- [`RidgeSelection.__post_init__`](#definition-4)
- [`CandidateScores`](#definition-5)
- [`CandidateScores.__post_init__`](#definition-6)
- [`FractionSelection`](#definition-7)
- [`FractionSelection.__post_init__`](#definition-8)
- [`paired_scores`](#definition-9)
- [`scoring_mask`](#definition-10)

<a id="definition-1"></a>

## `TrialEncodingResult`

[Source, line 13](../../src/boldtailor/ridge_results.py#L13)

```python
@dataclass(frozen=True, kw_only=True)
class TrialEncodingResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
coefficients: np.ndarray
predictor_means: np.ndarray
predictor_names: tuple[str, ...]
train_runs: tuple[int, ...]
test_runs: tuple[int, ...]
trial_masks: tuple[np.ndarray, ...]
predictions: tuple[np.ndarray, ...]
run_sse: np.ndarray
run_sst: np.ndarray
r2: np.ndarray
encoding_mode: str
train_run_predictor_means: np.ndarray
train_run_intercepts: np.ndarray
scoring_offsets: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/trial_encoding.py:234](../../src/boldtailor/trial_encoding.py#L234)

<a id="definition-2"></a>

## `TrialEncodingResult.__post_init__`

[Source, line 29](../../src/boldtailor/ridge_results.py#L29)

```python
TrialEncodingResult.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `RidgeSelection`

[Source, line 49](../../src/boldtailor/ridge_results.py#L49)

```python
@dataclass(frozen=True, kw_only=True)
class RidgeSelection
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
ridge_alpha: float
alphas: tuple[float, ...]
objective_scores: np.ndarray
percentile: float
scoring_mask: np.ndarray
at_boundary: bool = False
```

Usage in package/tests:

- [src/boldtailor/ridge_selection.py:70](../../src/boldtailor/ridge_selection.py#L70)

<a id="definition-4"></a>

## `RidgeSelection.__post_init__`

[Source, line 57](../../src/boldtailor/ridge_results.py#L57)

```python
RidgeSelection.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `CandidateScores`

[Source, line 65](../../src/boldtailor/ridge_results.py#L65)

```python
@dataclass(frozen=True, kw_only=True)
class CandidateScores
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
regularization: Literal['normalized_ridge', 'fractional_ridge']
grid: tuple[float, ...]
cv_r2: np.ndarray
fold_sse: np.ndarray
fold_sst: np.ndarray
fold_hrf_indices: np.ndarray
trial_masks: tuple[np.ndarray, ...]
run_labels: tuple[str, ...]
provenance: ProvenanceRecord
```

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:285](../../src/boldtailor/_ridge_cv.py#L285)
- [src/boldtailor/workflow/beta_series.py:178](../../src/boldtailor/workflow/beta_series.py#L178)

<a id="definition-6"></a>

## `CandidateScores.__post_init__`

[Source, line 76](../../src/boldtailor/ridge_results.py#L76)

```python
CandidateScores.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `FractionSelection`

[Source, line 86](../../src/boldtailor/ridge_results.py#L86)

```python
@dataclass(frozen=True, kw_only=True)
class FractionSelection
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
fractions: tuple[float, ...]
ridge_fraction: np.ndarray
selected_r2: np.ndarray
fraction_indices: np.ndarray
scoring_mask: np.ndarray
at_boundary: np.ndarray | None = None
```

Usage in package/tests:

- [src/boldtailor/fractional_ridge.py:67](../../src/boldtailor/fractional_ridge.py#L67)

<a id="definition-8"></a>

## `FractionSelection.__post_init__`

[Source, line 94](../../src/boldtailor/ridge_results.py#L94)

```python
FractionSelection.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `paired_scores`

[Source, line 103](../../src/boldtailor/ridge_results.py#L103)

```python
paired_scores(scores, grid, *, kind, grid_name)
```

Score matrix and grid from CandidateScores, or from an array plus grid.

Usage in package/tests:

- [src/boldtailor/fractional_ridge.py:41](../../src/boldtailor/fractional_ridge.py#L41)
- [src/boldtailor/ridge_selection.py:52](../../src/boldtailor/ridge_selection.py#L52)

<a id="definition-10"></a>

## `scoring_mask`

[Source, line 120](../../src/boldtailor/ridge_results.py#L120)

```python
scoring_mask(scores, feature_mask)
```

Features finite at every candidate, restricted by an optional mask.

Usage in package/tests:

- [src/boldtailor/fractional_ridge.py:48](../../src/boldtailor/fractional_ridge.py#L48)
- [src/boldtailor/ridge_selection.py:60](../../src/boldtailor/ridge_selection.py#L60)
