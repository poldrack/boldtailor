# `boldtailor._numerics`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_numerics.py)

Shared numerical policy: rank tolerance, column bases and projection error.

## Module contract

Shared numerical policy: rank tolerance, column bases and projection error.

Callers keep their own statistical requirements (full column rank, redundant
columns allowed, ...); this module only fixes how rank and degeneracy are
judged numerically, so that related kernels agree.

## Definitions

- [`rank_tolerance`](#definition-1)
- [`_unit_columns`](#definition-2)
- [`conditioned_basis`](#definition-3)
- [`orthonormal_basis`](#definition-4)
- [`residualise`](#definition-5)
- [`_amplified_energy`](#definition-6)
- [`project_out`](#definition-7)

<a id="definition-1"></a>

## `rank_tolerance`

[Source, line 36](../../src/boldtailor/_numerics.py#L36)

```python
rank_tolerance(shape, spectral_norm) -> float
```

Singular values at or below this are treated as zero.

Usage in package/tests:

- [src/boldtailor/_numerics.py:65](../../src/boldtailor/_numerics.py#L65)
- [src/boldtailor/_reml_terms.py:114](../../src/boldtailor/_reml_terms.py#L114)
- [tests/test_numerics.py:56](../../tests/test_numerics.py#L56)
- [tests/test_numerics_properties.py:96](../../tests/test_numerics_properties.py#L96)

<a id="definition-2"></a>

## `_unit_columns`

[Source, line 41](../../src/boldtailor/_numerics.py#L41)

```python
_unit_columns(matrix) -> np.ndarray
```

Nonzero columns scaled to unit norm, which leaves their span unchanged.

Each column is divided by its largest magnitude first, so that columns in
extreme units neither overflow nor underflow when their norm is taken.

Usage in package/tests:

- [src/boldtailor/_numerics.py:61](../../src/boldtailor/_numerics.py#L61)

<a id="definition-3"></a>

## `conditioned_basis`

[Source, line 52](../../src/boldtailor/_numerics.py#L52)

```python
conditioned_basis(matrix) -> tuple[np.ndarray, np.ndarray]
```

Orthonormal basis of the column span, and ``s_max / s_i`` of each vector.

Columns are normalised first, so the basis, the rank decision and the
amplification depend on the span and the collinearity of the columns, not
on their units. Only exactly zero columns are dropped for their size.
The basis is row-major, like the copies that projections store, so that
products with either round identically.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:43](../../src/boldtailor/_ar1_moments.py#L43)
- [src/boldtailor/_numerics.py:71](../../src/boldtailor/_numerics.py#L71)
- [src/boldtailor/_reml_terms.py:42](../../src/boldtailor/_reml_terms.py#L42)
- [src/boldtailor/ar1.py:87](../../src/boldtailor/ar1.py#L87)

<a id="definition-4"></a>

## `orthonormal_basis`

[Source, line 69](../../src/boldtailor/_numerics.py#L69)

```python
orthonormal_basis(matrix) -> np.ndarray
```

Orthonormal basis of the column span of ``matrix``.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:85](../../src/boldtailor/_ar1_moments.py#L85)
- [tests/test_numerics.py:25](../../tests/test_numerics.py#L25)
- [tests/test_numerics.py:61](../../tests/test_numerics.py#L61)
- [tests/test_numerics.py:74](../../tests/test_numerics.py#L74)

<a id="definition-5"></a>

## `residualise`

[Source, line 74](../../src/boldtailor/_numerics.py#L74)

```python
residualise(signals, basis) -> np.ndarray
```

Signals minus their projection on an orthonormal ``basis``.

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:60](../../src/boldtailor/_reml_terms.py#L60)
- [src/boldtailor/_reml_terms.py:105](../../src/boldtailor/_reml_terms.py#L105)
- [tests/test_numerics.py:62](../../tests/test_numerics.py#L62)
- [tests/test_numerics.py:104](../../tests/test_numerics.py#L104)

<a id="definition-6"></a>

## `_amplified_energy`

[Source, line 79](../../src/boldtailor/_numerics.py#L79)

```python
_amplified_energy(energy, coordinates, amplification)
```

Squared norm of the signals with basis coordinates amplified.

Usage in package/tests:

- [src/boldtailor/_numerics.py:97](../../src/boldtailor/_numerics.py#L97)

<a id="definition-7"></a>

## `project_out`

[Source, line 85](../../src/boldtailor/_numerics.py#L85)

```python
project_out(signals, basis, amplification)
```

Residual off ``basis`` with its rounding and uncertainty energy floors.

Returns ``(residual, rounding_floor, uncertainty_floor)``; the floors are
squared norms, per feature, with ``rounding_floor <= uncertainty_floor``.
See ``PROJECTION_ERROR_FACTOR``.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:35](../../src/boldtailor/_ar1_moments.py#L35)
- [src/boldtailor/_reml_terms.py:46](../../src/boldtailor/_reml_terms.py#L46)
- [tests/test_numerics.py:114](../../tests/test_numerics.py#L114)
- [tests/test_numerics.py:132](../../tests/test_numerics.py#L132)
