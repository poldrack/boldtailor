# `boldtailor.reliability`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/reliability.py)

Descriptive agreement between selected HRF shapes, per grayordinate.

## Module contract

Descriptive agreement between selected HRF shapes, per grayordinate.

Correlations use the full stored library time grid, without shifting curves,
so they ignore amplitude scale and offset. None of these scores is an ICC.

## Definitions

- [`library_indices`](#definition-1)
- [`finite_mean`](#definition-2)
- [`_curve_correlation_matrix`](#definition-3)
- [`curve_correlations`](#definition-4)
- [`_session_indices`](#definition-5)
- [`_parameter_sd`](#definition-6)
- [`_pair_scores`](#definition-7)
- [`_peak_time_mean`](#definition-8)
- [`compare_hrfs`](#definition-9)

<a id="definition-1"></a>

## `library_indices`

[Source, line 23](../../src/boldtailor/reliability.py#L23)

```python
library_indices(values, size)
```

Integer library indices, with NaN or -1 (no selection) mapped to -1.

Usage in package/tests:

- [src/boldtailor/reliability.py:68](../../src/boldtailor/reliability.py#L68)
- [src/boldtailor/reliability.py:69](../../src/boldtailor/reliability.py#L69)
- [src/boldtailor/reliability.py:90](../../src/boldtailor/reliability.py#L90)
- [src/boldtailor/workflow/plots.py:93](../../src/boldtailor/workflow/plots.py#L93)

<a id="definition-2"></a>

## `finite_mean`

[Source, line 40](../../src/boldtailor/reliability.py#L40)

```python
finite_mean(values, axis=0)
```

Mean over finite entries; NaN where none are finite.

Usage in package/tests:

- [src/boldtailor/reliability.py:121](../../src/boldtailor/reliability.py#L121)
- [src/boldtailor/reliability.py:98](../../src/boldtailor/reliability.py#L98)
- [src/boldtailor/reliability.py:153](../../src/boldtailor/reliability.py#L153)
- [src/boldtailor/reliability.py:154](../../src/boldtailor/reliability.py#L154)

<a id="definition-3"></a>

## `_curve_correlation_matrix`

[Source, line 51](../../src/boldtailor/reliability.py#L51)

```python
_curve_correlation_matrix(curves)
```

Pearson r between every pair of curves; NaN for constant curves.

Usage in package/tests:

- [src/boldtailor/reliability.py:72](../../src/boldtailor/reliability.py#L72)
- [src/boldtailor/reliability.py:134](../../src/boldtailor/reliability.py#L134)

<a id="definition-4"></a>

## `curve_correlations`

[Source, line 61](../../src/boldtailor/reliability.py#L61)

```python
curve_correlations(library, ids_a, ids_b)
```

Pearson r on the common stored time grid, including padded tails.

Rows are a/b, a/canonical, and b/canonical (``CORRELATION_NAMES`` for an
odd/even split). Each pair retains NaN if either selection is missing or
either curve is constant. Curves are not shifted in time.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:191](../../src/boldtailor/workflow/outputs.py#L191)
- [src/boldtailor/workflow/run.py:266](../../src/boldtailor/workflow/run.py#L266)

<a id="definition-5"></a>

## `_session_indices`

[Source, line 81](../../src/boldtailor/reliability.py#L81)

```python
_session_indices(library, hrf_indices, sessions)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reliability.py:133](../../src/boldtailor/reliability.py#L133)

<a id="definition-6"></a>

## `_parameter_sd`

[Source, line 93](../../src/boldtailor/reliability.py#L93)

```python
_parameter_sd(library, ids)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reliability.py:139](../../src/boldtailor/reliability.py#L139)

<a id="definition-7"></a>

## `_pair_scores`

[Source, line 108](../../src/boldtailor/reliability.py#L108)

```python
_pair_scores(lookup, ids, canonical, pairs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reliability.py:137](../../src/boldtailor/reliability.py#L137)

<a id="definition-8"></a>

## `_peak_time_mean`

[Source, line 118](../../src/boldtailor/reliability.py#L118)

```python
_peak_time_mean(library, ids)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reliability.py:150](../../src/boldtailor/reliability.py#L150)

<a id="definition-9"></a>

## `compare_hrfs`

[Source, line 124](../../src/boldtailor/reliability.py#L124)

```python
compare_hrfs(library, hrf_indices, sessions)
```

Pearson r over the full stored time grid, with no shifting of curves.

Every unordered session pair receives equal weight. Its canonical baseline
is the mean of the two session-to-SPM correlations at that grayordinate.
Summary correlations (``SUMMARY_NAMES``) require at least one valid pair.
All scores are descriptive; shared sessions make pairs dependent.
