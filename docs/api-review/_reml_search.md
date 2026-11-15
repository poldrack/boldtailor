# `boldtailor._reml_search`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_reml_search.py)

Vectorised search for the REML variance ratio.

## Module contract

Vectorised search for the REML variance ratio.

A log grid evaluates the criterion and its analytic slope in log(lambda).
Bisection refines the slope root inside every grid cell where the slope turns
from negative to positive, so each local minimum the grid resolves competes,
not only the one nearest the best grid point. The lowest root replaces the
best grid point unless it is worse beyond floating-point rounding, and the
result is compared explicitly with lambda = 0: a positive ratio must improve
on zero by more than the criterion's rounding noise.

A positive ratio is a verified stationary point only when a root was
accepted. Otherwise the best grid point stands and the optimum is flagged
``unrefined``; the upper edge of the grid (``at_upper``) is one such case.

When the random covariance is a multiple of the identity on the whole
restricted space, the covariance is (sigma2 + tau2 c) I and the criterion is
constant in lambda. Such fits are flagged unidentifiable instead of searched.

Memory: features are searched in batches so that every temporary of the
search, of shape (grid rows, k, features) or (k, crossings), and each
retained (grid, features) table stays within ``_WORKING_BYTES``; a few such
arrays are alive at once. A single feature is the smallest batch, so the
floor is 8 k bytes per temporary. The pooled statistics themselves, of shape
(k, features), are inputs and are not covered.

The search is not a certified global minimisation: a minimum and a maximum
that both fall strictly inside one grid cell leave no sign change and are not
refined. The grid spacing never exceeds log(1e16) / (grid_size - 1), about
0.31 in log(lambda) by default, whatever the spread of the singular values.

## Definitions

- [`RatioOptimum`](#definition-1)
- [`RatioOptimum.__post_init__`](#definition-2)
- [`_search_grid`](#definition-3)
- [`_log_grid`](#definition-4)
- [`_chunk_rows`](#definition-5)
- [`_batch_features`](#definition-6)
- [`_grid_chunks`](#definition-7)
- [`_on_grid`](#definition-8)
- [`_take`](#definition-9)
- [`_bisect_slope`](#definition-10)
- [`_refined`](#definition-11)
- [`_stacked`](#definition-12)
- [`_crossing_roots`](#definition-13)
- [`_lowest_root`](#definition-14)
- [`_rounding_slack`](#definition-15)
- [`_best_grid_point`](#definition-16)
- [`_best_positive`](#definition-17)
- [`_boundary_optimum`](#definition-18)
- [`_confounded`](#definition-19)
- [`_search_batch`](#definition-20)
- [`_joined`](#definition-21)
- [`_searched_optimum`](#definition-22)
- [`minimize_ratio`](#definition-23)

<a id="definition-1"></a>

## `RatioOptimum`

[Source, line 59](../../src/boldtailor/_reml_search.py#L59)

```python
@dataclass(frozen=True)
class RatioOptimum
```

Per-feature variance-ratio optimum and where it sits.

``unrefined`` marks a positive ratio that is a grid point, not a slope
root; ``at_upper`` marks those at the upper edge of the grid.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
ratio: np.ndarray
criterion: np.ndarray
at_zero: np.ndarray
at_upper: np.ndarray
unrefined: np.ndarray
unidentifiable: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/_reml_search.py:208](../../src/boldtailor/_reml_search.py#L208)
- [src/boldtailor/_reml_search.py:236](../../src/boldtailor/_reml_search.py#L236)
- [src/boldtailor/_reml_search.py:248](../../src/boldtailor/_reml_search.py#L248)

<a id="definition-2"></a>

## `RatioOptimum.__post_init__`

[Source, line 73](../../src/boldtailor/_reml_search.py#L73)

```python
RatioOptimum.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `_search_grid`

[Source, line 78](../../src/boldtailor/_reml_search.py#L78)

```python
_search_grid(singular2, size)
```

Log-ratio grid spanning 1e-8 / max(s2) to 1e8 / min(s2).

``size`` points cover a single singular value; a wider spectrum adds
points so that the spacing never exceeds that base step.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:92](../../src/boldtailor/_reml_search.py#L92)
- [src/boldtailor/_reml_search.py:255](../../src/boldtailor/_reml_search.py#L255)

<a id="definition-4"></a>

## `_log_grid`

[Source, line 90](../../src/boldtailor/_reml_search.py#L90)

```python
_log_grid(terms: Sequence[RemlTerms], size: int) -> np.ndarray
```

The search grid of the pooled singular values.

Usage in package/tests:

- [tests/test_reml_search.py:175](../../tests/test_reml_search.py#L175)
- [tests/test_reml_search.py:123](../../tests/test_reml_search.py#L123)

<a id="definition-5"></a>

## `_chunk_rows`

[Source, line 95](../../src/boldtailor/_reml_search.py#L95)

```python
_chunk_rows(pooled: PooledTerms) -> int
```

Grid rows per chunk so one (rows, k, features) temporary stays bounded.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:109](../../src/boldtailor/_reml_search.py#L109)

<a id="definition-6"></a>

## `_batch_features`

[Source, line 101](../../src/boldtailor/_reml_search.py#L101)

```python
_batch_features(pooled: PooledTerms, grid_points: int) -> int
```

Features per batch: one (k, features) temporary and one (grid, features)
table within the budget, and never fewer than one feature.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:158](../../src/boldtailor/_reml_search.py#L158)
- [src/boldtailor/_reml_search.py:257](../../src/boldtailor/_reml_search.py#L257)

<a id="definition-7"></a>

## `_grid_chunks`

[Source, line 107](../../src/boldtailor/_reml_search.py#L107)

```python
_grid_chunks(pooled: PooledTerms, log_grid)
```

Criterion and slope of each chunk of grid rows, for every feature.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:120](../../src/boldtailor/_reml_search.py#L120)

<a id="definition-8"></a>

## `_on_grid`

[Source, line 118](../../src/boldtailor/_reml_search.py#L118)

```python
_on_grid(pooled: PooledTerms, log_grid)
```

Criterion and slope on the grid, each of shape (grid, features).

Usage in package/tests:

- [src/boldtailor/_reml_search.py:191](../../src/boldtailor/_reml_search.py#L191)

<a id="definition-9"></a>

## `_take`

[Source, line 124](../../src/boldtailor/_reml_search.py#L124)

```python
_take(pooled: PooledTerms, features) -> PooledTerms
```

The pooled statistics of the given feature columns.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:145](../../src/boldtailor/_reml_search.py#L145)
- [src/boldtailor/_reml_search.py:260](../../src/boldtailor/_reml_search.py#L260)

<a id="definition-10"></a>

## `_bisect_slope`

[Source, line 134](../../src/boldtailor/_reml_search.py#L134)

```python
_bisect_slope(pooled, low, high, iterations)
```

Vectorised bisection for the zero of the slope in log ratio.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:146](../../src/boldtailor/_reml_search.py#L146)

<a id="definition-11"></a>

## `_refined`

[Source, line 143](../../src/boldtailor/_reml_search.py#L143)

```python
_refined(pooled, log_grid, cell, feature, iterations)
```

Slope root and its criterion for each (cell, feature) crossing.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:160](../../src/boldtailor/_reml_search.py#L160)

<a id="definition-12"></a>

## `_stacked`

[Source, line 150](../../src/boldtailor/_reml_search.py#L150)

```python
_stacked(parts, position)
```

Concatenate one output of the refined chunks; empty when there are none.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:165](../../src/boldtailor/_reml_search.py#L165)

<a id="definition-13"></a>

## `_crossing_roots`

[Source, line 155](../../src/boldtailor/_reml_search.py#L155)

```python
_crossing_roots(pooled, log_grid, slopes, iterations)
```

Feature, slope root and criterion of every cell where the slope turns up.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:193](../../src/boldtailor/_reml_search.py#L193)

<a id="definition-14"></a>

## `_lowest_root`

[Source, line 168](../../src/boldtailor/_reml_search.py#L168)

```python
_lowest_root(n_features, feature, root, value)
```

Per feature, its lowest-criterion root; value inf where it has none.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:194](../../src/boldtailor/_reml_search.py#L194)

<a id="definition-15"></a>

## `_rounding_slack`

[Source, line 178](../../src/boldtailor/_reml_search.py#L178)

```python
_rounding_slack(value, dof)
```

Criterion rounding noise. On a flat criterion a grid point can sit a few
ulps below the better-determined slope root, which must still win; at an
exact rounding tie with lambda = 0 the result may therefore flip to zero.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:195](../../src/boldtailor/_reml_search.py#L195)
- [src/boldtailor/_reml_search.py:235](../../src/boldtailor/_reml_search.py#L235)

<a id="definition-16"></a>

## `_best_grid_point`

[Source, line 185](../../src/boldtailor/_reml_search.py#L185)

```python
_best_grid_point(values)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:192](../../src/boldtailor/_reml_search.py#L192)

<a id="definition-17"></a>

## `_best_positive`

[Source, line 190](../../src/boldtailor/_reml_search.py#L190)

```python
_best_positive(pooled, log_grid, iterations)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:233](../../src/boldtailor/_reml_search.py#L233)

<a id="definition-18"></a>

## `_boundary_optimum`

[Source, line 204](../../src/boldtailor/_reml_search.py#L204)

```python
_boundary_optimum(pooled, ratio, *, at_zero) -> RatioOptimum
```

An optimum that needs no search; its criterion is the value at zero.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:277](../../src/boldtailor/_reml_search.py#L277)
- [src/boldtailor/_reml_search.py:279](../../src/boldtailor/_reml_search.py#L279)

<a id="definition-19"></a>

## `_confounded`

[Source, line 218](../../src/boldtailor/_reml_search.py#L218)

```python
_confounded(pooled: PooledTerms) -> bool
```

True when the random covariance is c I on the whole restricted space.

That needs a retained direction for every restricted degree of freedom
and one common singular value, up to the rounding of the decompositions
that produced them (relative error about dof * eps per singular value).

Usage in package/tests:

- [src/boldtailor/_reml_search.py:278](../../src/boldtailor/_reml_search.py#L278)
- [tests/test_reml_scientific_properties.py:116](../../tests/test_reml_scientific_properties.py#L116)

<a id="definition-20"></a>

## `_search_batch`

[Source, line 231](../../src/boldtailor/_reml_search.py#L231)

```python
_search_batch(pooled, log_grid, iterations) -> RatioOptimum
```

Search one batch of features and compare the result with zero.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:260](../../src/boldtailor/_reml_search.py#L260)

<a id="definition-21"></a>

## `_joined`

[Source, line 246](../../src/boldtailor/_reml_search.py#L246)

```python
_joined(parts: Sequence[RatioOptimum]) -> RatioOptimum
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:259](../../src/boldtailor/_reml_search.py#L259)

<a id="definition-22"></a>

## `_searched_optimum`

[Source, line 253](../../src/boldtailor/_reml_search.py#L253)

```python
_searched_optimum(pooled, grid_size, iterations) -> RatioOptimum
```

Search the features batch by batch within the working budget.

Usage in package/tests:

- [src/boldtailor/_reml_search.py:280](../../src/boldtailor/_reml_search.py#L280)

<a id="definition-23"></a>

## `minimize_ratio`

[Source, line 264](../../src/boldtailor/_reml_search.py#L264)

```python
minimize_ratio(terms: Sequence[RemlTerms] | PooledTerms, *, grid_size: int=121, iterations: int=48) -> RatioOptimum
```

Minimise the profiled criterion over lambda >= 0 for every feature.

``iterations`` is the number of slope bisections inside each bracket.
Unidentifiable fits have a NaN ratio and the criterion's constant value.

Usage in package/tests:

- [src/boldtailor/reml.py:110](../../src/boldtailor/reml.py#L110)
- [tests/test_reml.py:42](../../tests/test_reml.py#L42)
- [tests/test_reml_scientific_properties.py:93](../../tests/test_reml_scientific_properties.py#L93)
- [tests/test_reml_search.py:48](../../tests/test_reml_search.py#L48)
