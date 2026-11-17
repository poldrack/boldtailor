# `boldtailor._reml_criterion`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_reml_criterion.py)

Profiled restricted likelihood for scalar REML.

## Module contract

Profiled restricted likelihood for scalar REML.

The criterion depends on the groups only through their pooled statistics:
all singular values, the stacked squared coordinates, the summed residual sum
of squares and the summed dof. Pooling once makes every evaluation a single
vectorised pass, whatever the number of groups.

## Definitions

- [`PooledTerms`](#definition-1)
- [`pool_terms`](#definition-2)
- [`_scaled`](#definition-3)
- [`pooled_quadratic`](#definition-4)
- [`pooled_criterion`](#definition-5)
- [`pooled_criterion_at_zero`](#definition-6)
- [`pooled_slope`](#definition-7)
- [`pooled_value_and_slope`](#definition-8)
- [`profiled_variance`](#definition-9)
- [`criterion`](#definition-10)
- [`criterion_slope`](#definition-11)

<a id="definition-1"></a>

## `PooledTerms`

[Source, line 18](../../src/boldtailor/_reml_criterion.py#L18)

```python
@dataclass(frozen=True)
class PooledTerms
```

Sufficient statistics of all groups together.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
singular2: np.ndarray
coordinates2: np.ndarray
residual_ss: np.ndarray
dof: int
```

Usage in package/tests:

- [src/boldtailor/_reml_criterion.py:29](../../src/boldtailor/_reml_criterion.py#L29)
- [src/boldtailor/_reml_search.py:126](../../src/boldtailor/_reml_search.py#L126)
- [tests/test_reml_scientific_properties.py:48](../../tests/test_reml_scientific_properties.py#L48)
- [tests/test_reml_scientific_properties.py:115](../../tests/test_reml_scientific_properties.py#L115)

<a id="definition-2"></a>

## `pool_terms`

[Source, line 27](../../src/boldtailor/_reml_criterion.py#L27)

```python
pool_terms(terms: Sequence[RemlTerms]) -> PooledTerms
```

Concatenate singular values and coordinates; sum residuals and dof.

Usage in package/tests:

- [src/boldtailor/_reml_criterion.py:89](../../src/boldtailor/_reml_criterion.py#L89)
- [src/boldtailor/_reml_criterion.py:95](../../src/boldtailor/_reml_criterion.py#L95)
- [src/boldtailor/_reml_criterion.py:100](../../src/boldtailor/_reml_criterion.py#L100)
- [src/boldtailor/_reml_search.py:275](../../src/boldtailor/_reml_search.py#L275)

<a id="definition-3"></a>

## `_scaled`

[Source, line 37](../../src/boldtailor/_reml_criterion.py#L37)

```python
_scaled(ratios, singular2)
```

ratio * s2 with shape (..., k, features).

Usage in package/tests:

- [src/boldtailor/_reml_criterion.py:50](../../src/boldtailor/_reml_criterion.py#L50)
- [src/boldtailor/_reml_criterion.py:65](../../src/boldtailor/_reml_criterion.py#L65)
- [src/boldtailor/_reml_criterion.py:76](../../src/boldtailor/_reml_criterion.py#L76)
- [src/boldtailor/_reml_criterion.py:44](../../src/boldtailor/_reml_criterion.py#L44)

<a id="definition-4"></a>

## `pooled_quadratic`

[Source, line 42](../../src/boldtailor/_reml_criterion.py#L42)

```python
pooled_quadratic(ratios, pooled: PooledTerms) -> np.ndarray
```

Q(lambda), the whitened residual quadratic form, shaped like ``ratios``.

Usage in package/tests:

- [src/boldtailor/_reml_criterion.py:90](../../src/boldtailor/_reml_criterion.py#L90)
- [src/boldtailor/reml.py:113](../../src/boldtailor/reml.py#L113)
- [tests/test_reml_scientific_properties.py:77](../../tests/test_reml_scientific_properties.py#L77)
- [tests/test_reml_scientific_properties.py:78](../../tests/test_reml_scientific_properties.py#L78)

<a id="definition-5"></a>

## `pooled_criterion`

[Source, line 48](../../src/boldtailor/_reml_criterion.py#L48)

```python
pooled_criterion(ratios, pooled: PooledTerms) -> np.ndarray
```

N log(Q/N) + sum log(1 + lambda s2) from pooled statistics.

Usage in package/tests:

- [src/boldtailor/_reml_criterion.py:95](../../src/boldtailor/_reml_criterion.py#L95)
- [src/boldtailor/_reml_search.py:147](../../src/boldtailor/_reml_search.py#L147)
- [tests/test_reml_scientific_properties.py:72](../../tests/test_reml_scientific_properties.py#L72)

<a id="definition-6"></a>

## `pooled_criterion_at_zero`

[Source, line 57](../../src/boldtailor/_reml_criterion.py#L57)

```python
pooled_criterion_at_zero(pooled: PooledTerms) -> np.ndarray
```

The criterion at lambda = 0, N log(energy / N), without (k, features) work.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:234](../../src/boldtailor/_reml_search.py#L234)
- [src/boldtailor/_reml_search.py:210](../../src/boldtailor/_reml_search.py#L210)

<a id="definition-7"></a>

## `pooled_slope`

[Source, line 63](../../src/boldtailor/_reml_criterion.py#L63)

```python
pooled_slope(ratios, pooled: PooledTerms) -> np.ndarray
```

Derivative of the criterion in log(lambda), from pooled statistics.

Usage in package/tests:

- [src/boldtailor/_reml_criterion.py:100](../../src/boldtailor/_reml_criterion.py#L100)
- [src/boldtailor/_reml_search.py:138](../../src/boldtailor/_reml_search.py#L138)
- [tests/test_reml_scientific_properties.py:85](../../tests/test_reml_scientific_properties.py#L85)

<a id="definition-8"></a>

## `pooled_value_and_slope`

[Source, line 74](../../src/boldtailor/_reml_criterion.py#L74)

```python
pooled_value_and_slope(ratios, pooled: PooledTerms)
```

Criterion and its slope in log(lambda), sharing one pass of temporaries.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:113](../../src/boldtailor/_reml_search.py#L113)

<a id="definition-9"></a>

## `profiled_variance`

[Source, line 87](../../src/boldtailor/_reml_criterion.py#L87)

```python
profiled_variance(ratios, terms: Sequence[RemlTerms]) -> np.ndarray
```

Profiled noise variance Q(lambda) / N, shaped like ``ratios``.

Usage in package/tests:

- [tests/test_reml_criterion.py:63](../../tests/test_reml_criterion.py#L63)
- [tests/test_reml_statistics.py:82](../../tests/test_reml_statistics.py#L82)

<a id="definition-10"></a>

## `criterion`

[Source, line 93](../../src/boldtailor/_reml_criterion.py#L93)

```python
criterion(ratios, terms: Sequence[RemlTerms]) -> np.ndarray
```

Profiled REML criterion N log(Q/N) + sum log(1 + lambda s2).

Usage in package/tests:

- [tests/test_reml_criterion.py:43](../../tests/test_reml_criterion.py#L43)
- [tests/test_reml_criterion.py:69](../../tests/test_reml_criterion.py#L69)
- [tests/test_reml_criterion.py:92](../../tests/test_reml_criterion.py#L92)
- [tests/test_reml_criterion.py:93](../../tests/test_reml_criterion.py#L93)

<a id="definition-11"></a>

## `criterion_slope`

[Source, line 98](../../src/boldtailor/_reml_criterion.py#L98)

```python
criterion_slope(ratios, terms: Sequence[RemlTerms]) -> np.ndarray
```

Derivative of ``criterion`` with respect to log(lambda).

Usage in package/tests:

- [tests/test_reml_criterion.py:119](../../tests/test_reml_criterion.py#L119)
- [tests/test_reml_criterion.py:109](../../tests/test_reml_criterion.py#L109)
- [tests/test_reml_search.py:98](../../tests/test_reml_search.py#L98)
- [tests/test_reml_search.py:181](../../tests/test_reml_search.py#L181)
