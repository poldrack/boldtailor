# `boldtailor._ridge_cv`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_ridge_cv.py)

Training-only HRF selection and streamed candidate beta-series prediction.

## Definitions

- [`_validate`](#definition-1)
- [`RunBetaPath`](#definition-2)
- [`RunBetaPath.betas_at`](#definition-3)
- [`run_tables`](#definition-4)
- [`prepare_run_beta_path`](#definition-5)
- [`_beta_path`](#definition-6)
- [`_fold_selection`](#definition-7)
- [`_score_fold`](#definition-8)
- [`_fingerprint`](#definition-9)
- [`_provenance`](#definition-10)
- [`score_candidates`](#definition-11)

<a id="definition-1"></a>

## `_validate`

[Source, line 32](../../src/boldtailor/_ridge_cv.py#L32)

```python
_validate(data, predictors, library, signature, *, encoding_mode)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:250](../../src/boldtailor/_ridge_cv.py#L250)

<a id="definition-2"></a>

## `RunBetaPath`

[Source, line 56](../../src/boldtailor/_ridge_cv.py#L56)

```python
@dataclass(frozen=True)
class RunBetaPath
```

Prepared HRF groups, placed back into their original feature columns.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
shape: tuple[int, int]
groups: tuple
```

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:93](../../src/boldtailor/_ridge_cv.py#L93)

<a id="definition-3"></a>

## `RunBetaPath.betas_at`

[Source, line 62](../../src/boldtailor/_ridge_cv.py#L62)

```python
RunBetaPath.betas_at(self, value)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `run_tables`

[Source, line 69](../../src/boldtailor/_ridge_cv.py#L69)

```python
run_tables(data)
```

Copy every run's events and confounds once for a per-run loop.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:78](../../src/boldtailor/_ridge_cv.py#L78)
- [src/boldtailor/_ridge_cv.py:124](../../src/boldtailor/_ridge_cv.py#L124)

<a id="definition-5"></a>

## `prepare_run_beta_path`

[Source, line 76](../../src/boldtailor/_ridge_cv.py#L76)

```python
prepare_run_beta_path(data, run_index, prepared, ids, label, *, fractional=False)
```

Prepare each selected HRF group once for this run and fold.

<a id="definition-6"></a>

## `_beta_path`

[Source, line 82](../../src/boldtailor/_ridge_cv.py#L82)

```python
_beta_path(tables, r, prepared, ids, label, fractional)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:79](../../src/boldtailor/_ridge_cv.py#L79)
- [src/boldtailor/_ridge_cv.py:126](../../src/boldtailor/_ridge_cv.py#L126)

<a id="definition-7"></a>

## `_fold_selection`

[Source, line 96](../../src/boldtailor/_ridge_cv.py#L96)

```python
_fold_selection(data, library, labels, signature, train)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:122](../../src/boldtailor/_ridge_cv.py#L122)

<a id="definition-8"></a>

## `_score_fold`

[Source, line 108](../../src/boldtailor/_ridge_cv.py#L108)

```python
_score_fold(data, predictors, prepared, library, labels, signature, alphas, test, *, fractional=False, encoding_mode='within_run')
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:256](../../src/boldtailor/_ridge_cv.py#L256)

<a id="definition-9"></a>

## `_fingerprint`

[Source, line 171](../../src/boldtailor/_ridge_cv.py#L171)

```python
_fingerprint(arrays, names)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:206](../../src/boldtailor/_ridge_cv.py#L206)
- [src/boldtailor/_ridge_cv.py:207](../../src/boldtailor/_ridge_cv.py#L207)

<a id="definition-10"></a>

## `_provenance`

[Source, line 180](../../src/boldtailor/_ridge_cv.py#L180)

```python
_provenance(data, arrays, columns, labels, alphas, library, signature, folds, ids, *, fractional=False, encoding_mode='within_run')
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:272](../../src/boldtailor/_ridge_cv.py#L272)

<a id="definition-11"></a>

## `score_candidates`

[Source, line 239](../../src/boldtailor/_ridge_cv.py#L239)

```python
score_candidates(data, predictors, alphas, library, run_labels, feature_signature, *, fractional=False, encoding_mode='within_run')
```

No source docstring. Read the linked implementation and call sites for the precise contract.
