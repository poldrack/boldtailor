# `boldtailor.single_trial_results`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/single_trial_results.py)

Owned, read-only single-trial numerical results.

## Definitions

- [`SharedTrialDesign`](#definition-1)
- [`SharedTrialDesign.__post_init__`](#definition-2)
- [`SharedTrialDesign.matrices`](#definition-3)
- [`SelectedTrialDesign`](#definition-4)
- [`SelectedTrialDesign.__post_init__`](#definition-5)
- [`SelectedTrialDesign.matrix`](#definition-6)
- [`SingleTrialResult`](#definition-7)
- [`SingleTrialResult.__post_init__`](#definition-8)
- [`SingleTrialResult.trial_table`](#definition-9)
- [`SingleTrialResult.diagnostics`](#definition-10)

<a id="definition-1"></a>

## `SharedTrialDesign`

[Source, line 16](../../src/boldtailor/single_trial_results.py#L16)

```python
@dataclass(frozen=True, kw_only=True)
class SharedTrialDesign
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
_matrices: tuple[pd.DataFrame, ...]
```

Usage in package/tests:

- [src/boldtailor/single_trial.py:132](../../src/boldtailor/single_trial.py#L132)

<a id="definition-2"></a>

## `SharedTrialDesign.__post_init__`

[Source, line 19](../../src/boldtailor/single_trial_results.py#L19)

```python
SharedTrialDesign.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `SharedTrialDesign.matrices`

[Source, line 23](../../src/boldtailor/single_trial_results.py#L23)

```python
@property
SharedTrialDesign.matrices(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `SelectedTrialDesign`

[Source, line 28](../../src/boldtailor/single_trial_results.py#L28)

```python
@dataclass(frozen=True, kw_only=True)
class SelectedTrialDesign
```

Selected HRF IDs; each fitted (run, HRF) design is rebuilt on request.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
hrf_indices: np.ndarray
design_fingerprint: str
selection_provenance: ProvenanceRecord
_rebuild: Callable[[int, int], np.ndarray] = field(compare=False, repr=False)
```

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:172](../../src/boldtailor/_selected_hrf_fit.py#L172)

<a id="definition-5"></a>

## `SelectedTrialDesign.__post_init__`

[Source, line 36](../../src/boldtailor/single_trial_results.py#L36)

```python
SelectedTrialDesign.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `SelectedTrialDesign.matrix`

[Source, line 39](../../src/boldtailor/single_trial_results.py#L39)

```python
SelectedTrialDesign.matrix(self, run: int, hrf_id: int) -> np.ndarray
```

Trial columns then nuisance columns; KeyError for unfitted pairs.

<a id="definition-7"></a>

## `SingleTrialResult`

[Source, line 45](../../src/boldtailor/single_trial_results.py#L45)

```python
@dataclass(frozen=True, kw_only=True)
class SingleTrialResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
run_betas: tuple[np.ndarray, ...]
_trial_table: pd.DataFrame
design: SharedTrialDesign | SelectedTrialDesign
run_full_r2: tuple[np.ndarray, ...]
run_nuisance_r2: tuple[np.ndarray, ...]
full_r2: np.ndarray
nuisance_r2: np.ndarray
delta_r2: np.ndarray
_diagnostics: tuple[dict, ...]
ridge_alpha: float | None
provenance: ProvenanceRecord
ridge_fraction: np.ndarray | None = None
run_ridge_alphas: tuple[np.ndarray, ...] | None = None
```

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:185](../../src/boldtailor/_selected_hrf_fit.py#L185)
- [src/boldtailor/single_trial.py:129](../../src/boldtailor/single_trial.py#L129)

<a id="definition-8"></a>

## `SingleTrialResult.__post_init__`

[Source, line 60](../../src/boldtailor/single_trial_results.py#L60)

```python
SingleTrialResult.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `SingleTrialResult.trial_table`

[Source, line 73](../../src/boldtailor/single_trial_results.py#L73)

```python
@property
SingleTrialResult.trial_table(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `SingleTrialResult.diagnostics`

[Source, line 77](../../src/boldtailor/single_trial_results.py#L77)

```python
@property
SingleTrialResult.diagnostics(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.
