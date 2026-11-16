# `boldtailor._selected_hrf_fit`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_selected_hrf_fit.py)

Apply an identified HRF assignment, fitting only its feature groups.

## Definitions

- [`_tables`](#definition-1)
- [`_update_design_hash`](#definition-2)
- [`_fit_group`](#definition-3)
- [`GroupRunFit`](#definition-4)
- [`_fit_run`](#definition-5)
- [`_fit_activity`](#definition-6)
- [`_rebuilder`](#definition-7)
- [`_rebuilder.rebuild`](#definition-8)
- [`fit_groups`](#definition-9)
- [`_result`](#definition-10)

<a id="definition-1"></a>

## `_tables`

[Source, line 31](../../src/boldtailor/_selected_hrf_fit.py#L31)

```python
_tables(data, labels)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:156](../../src/boldtailor/_selected_hrf_fit.py#L156)

<a id="definition-2"></a>

## `_update_design_hash`

[Source, line 40](../../src/boldtailor/_selected_hrf_fit.py#L40)

```python
_update_design_hash(digest, run_index, cid, x, nuisance)
```

Feed one (run, HRF) design into the digest without retaining it.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:83](../../src/boldtailor/_selected_hrf_fit.py#L83)

<a id="definition-3"></a>

## `_fit_group`

[Source, line 48](../../src/boldtailor/_selected_hrf_fit.py#L48)

```python
_fit_group(run, x, y, alpha, fractions)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:76](../../src/boldtailor/_selected_hrf_fit.py#L76)

<a id="definition-4"></a>

## `GroupRunFit`

[Source, line 55](../../src/boldtailor/_selected_hrf_fit.py#L55)

```python
@dataclass(frozen=True, kw_only=True)
class GroupRunFit
```

One run's fits over every HRF group, assembled in feature order.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
betas: np.ndarray
full_sse: np.ndarray
nuisance_sse: np.ndarray
total_ss: np.ndarray
diagnostics: list[dict]
alphas: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:92](../../src/boldtailor/_selected_hrf_fit.py#L92)

<a id="definition-5"></a>

## `_fit_run`

[Source, line 66](../../src/boldtailor/_selected_hrf_fit.py#L66)

```python
_fit_run(run, y, ids, alpha, run_index, digest, fractions=None) -> GroupRunFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:160](../../src/boldtailor/_selected_hrf_fit.py#L160)

<a id="definition-6"></a>

## `_fit_activity`

[Source, line 102](../../src/boldtailor/_selected_hrf_fit.py#L102)

```python
_fit_activity(data, selection, labels, alpha, assignment, digest, fractions=None)
```

Finish the design digest (fed run-major, HRF-ascending) and describe the fit.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:163](../../src/boldtailor/_selected_hrf_fit.py#L163)

<a id="definition-7"></a>

## `_rebuilder`

[Source, line 129](../../src/boldtailor/_selected_hrf_fit.py#L129)

```python
_rebuilder(runs, ids)
```

Rebuild fitted designs from the prepared runs' cached trial matrices.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:176](../../src/boldtailor/_selected_hrf_fit.py#L176)

<a id="definition-8"></a>

## `_rebuilder.rebuild`

[Source, line 133](../../src/boldtailor/_selected_hrf_fit.py#L133)

```python
_rebuilder.rebuild(run, hrf_id)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `fit_groups`

[Source, line 143](../../src/boldtailor/_selected_hrf_fit.py#L143)

```python
fit_groups(data: AnalysisData, *, selection: HrfSelectionResult, ridge_alpha: float, run_labels: Sequence[str] | None, feature_signature: str | None, ridge_fraction: float | np.ndarray | None=None) -> SingleTrialResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `_result`

[Source, line 181](../../src/boldtailor/_selected_hrf_fit.py#L181)

```python
_result(fits, trials, design, alpha, fractions, provenance)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:178](../../src/boldtailor/_selected_hrf_fit.py#L178)
