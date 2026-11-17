# `boldtailor._ar1_moments`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_ar1_moments.py)

Lag-one residual autocorrelation and its projection-induced bias correction.

## Module contract

Lag-one residual autocorrelation and its projection-induced bias correction.

With ``M = I - QQ'`` the residual-forming matrix of a design with orthonormal
basis ``Q``, the estimator is ``e'Le / e'De`` for residuals ``e = My``, where
``L`` has 0.5 on the first off-diagonals and ``D`` is the identity without its
last diagonal entry. Its expectation under AR(1) noise is approximated by
``tr(MLM R(rho)) / tr(MDM R(rho))``, inverted on a fixed rho grid.

## Definitions

- [`lag_moments`](#definition-1)
- [`basis_lag_ratio`](#definition-2)
- [`residual_lag_ratio`](#definition-3)
- [`_lag_sums`](#definition-4)
- [`_sandwich`](#definition-5)
- [`_moment_matrices`](#definition-6)
- [`moment_polynomials`](#definition-7)
- [`_evaluate`](#definition-8)
- [`basis_expected_lag_ratio`](#definition-9)
- [`expected_lag_ratio`](#definition-10)
- [`_resolution`](#definition-11)
- [`_branches`](#definition-12)
- [`_pooling_branch`](#definition-13)
- [`_branch_inverse`](#definition-14)
- [`invert_lag_ratio`](#definition-15)
- [`correct_lag_ratio`](#definition-16)
- [`finite_median`](#definition-17)
- [`pool_rho`](#definition-18)

<a id="definition-1"></a>

## `lag_moments`

[Source, line 22](../../src/boldtailor/_ar1_moments.py#L22)

```python
lag_moments(residuals)
```

Per-feature numerator ``e'Le`` and denominator ``e'De``.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:36](../../src/boldtailor/_ar1_moments.py#L36)
- [tests/test_ar1.py:477](../../tests/test_ar1.py#L477)

<a id="definition-2"></a>

## `basis_lag_ratio`

[Source, line 28](../../src/boldtailor/_ar1_moments.py#L28)

```python
basis_lag_ratio(signals, basis, amplification) -> np.ndarray
```

Lag-one ratio of residuals after projecting out ``basis``; NaN if degenerate.

A feature is degenerate when its denominator does not exceed the
uncertainty floor of the projection (``boldtailor._numerics.project_out``).

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:43](../../src/boldtailor/_ar1_moments.py#L43)
- [src/boldtailor/ar1.py:126](../../src/boldtailor/ar1.py#L126)

<a id="definition-3"></a>

## `residual_lag_ratio`

[Source, line 41](../../src/boldtailor/_ar1_moments.py#L41)

```python
residual_lag_ratio(signals, design) -> np.ndarray
```

Lag-one ratio of OLS residuals per feature; NaN where degenerate.

<a id="definition-4"></a>

## `_lag_sums`

[Source, line 46](../../src/boldtailor/_ar1_moments.py#L46)

```python
_lag_sums(matrix)
```

Coefficients c_k with tr(matrix R(rho)) = sum_k c_k rho**k.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:69](../../src/boldtailor/_ar1_moments.py#L69)

<a id="definition-5"></a>

## `_sandwich`

[Source, line 53](../../src/boldtailor/_ar1_moments.py#L53)

```python
_sandwich(basis, matrix)
```

M A M for M = I - QQ', using only (scans x rank) products.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:69](../../src/boldtailor/_ar1_moments.py#L69)

<a id="definition-6"></a>

## `_moment_matrices`

[Source, line 59](../../src/boldtailor/_ar1_moments.py#L59)

```python
_moment_matrices(n)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:68](../../src/boldtailor/_ar1_moments.py#L68)

<a id="definition-7"></a>

## `moment_polynomials`

[Source, line 66](../../src/boldtailor/_ar1_moments.py#L66)

```python
moment_polynomials(basis)
```

Polynomial coefficients in rho of tr(MLM R) and tr(MDM R).

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:78](../../src/boldtailor/_ar1_moments.py#L78)
- [tests/test_ar1.py:478](../../tests/test_ar1.py#L478)

<a id="definition-8"></a>

## `_evaluate`

[Source, line 72](../../src/boldtailor/_ar1_moments.py#L72)

```python
_evaluate(coefficients, rhos)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:80](../../src/boldtailor/_ar1_moments.py#L80)

<a id="definition-9"></a>

## `basis_expected_lag_ratio`

[Source, line 76](../../src/boldtailor/_ar1_moments.py#L76)

```python
basis_expected_lag_ratio(basis, rhos) -> np.ndarray
```

tr(M L M R(rho)) / tr(M D M R(rho)) for each rho.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:85](../../src/boldtailor/_ar1_moments.py#L85)
- [src/boldtailor/ar1.py:88](../../src/boldtailor/ar1.py#L88)
- [tests/test_reml_scientific_properties.py:141](../../tests/test_reml_scientific_properties.py#L141)
- [tests/test_reml_scientific_properties.py:133](../../tests/test_reml_scientific_properties.py#L133)

<a id="definition-10"></a>

## `expected_lag_ratio`

[Source, line 83](../../src/boldtailor/_ar1_moments.py#L83)

```python
expected_lag_ratio(design, rhos) -> np.ndarray
```

tr(M L M R(rho)) / tr(M D M R(rho)) for each rho.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:157](../../src/boldtailor/_ar1_moments.py#L157)
- [tests/test_ar1.py:358](../../tests/test_ar1.py#L358)
- [tests/test_ar1.py:380](../../tests/test_ar1.py#L380)
- [tests/test_ar1.py:73](../../tests/test_ar1.py#L73)

<a id="definition-11"></a>

## `_resolution`

[Source, line 88](../../src/boldtailor/_ar1_moments.py#L88)

```python
_resolution(expected) -> float
```

Differences of the dimensionless lag ratio at or below this are rounding.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:147](../../src/boldtailor/_ar1_moments.py#L147)

<a id="definition-12"></a>

## `_branches`

[Source, line 93](../../src/boldtailor/_ar1_moments.py#L93)

```python
_branches(expected, slack)
```

Maximal runs of grid cells with one direction, as (first, last, direction).

``first`` and ``last`` index the run's end points. Direction is +1 or -1
for a resolved rise or fall and 0 where the differences are rounding.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:113](../../src/boldtailor/_ar1_moments.py#L113)

<a id="definition-13"></a>

## `_pooling_branch`

[Source, line 107](../../src/boldtailor/_ar1_moments.py#L107)

```python
_pooling_branch(expected, grid, slack, rho_range)
```

The resolved monotone branch spanning ``rho_range``, or ``None``.

The range is cut to the grid, which must be increasing.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:147](../../src/boldtailor/_ar1_moments.py#L147)

<a id="definition-14"></a>

## `_branch_inverse`

[Source, line 119](../../src/boldtailor/_ar1_moments.py#L119)

```python
_branch_inverse(values, levels, rhos, direction)
```

Rho interpolated on one branch, read in its rising direction; values
beyond the branch take the rho of its nearer end.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:151](../../src/boldtailor/_ar1_moments.py#L151)

<a id="definition-15"></a>

## `invert_lag_ratio`

[Source, line 126](../../src/boldtailor/_ar1_moments.py#L126)

```python
invert_lag_ratio(values, expected, grid, rho_range=RHO_RANGE) -> np.ndarray
```

Map observed ratios to rho through ``expected`` on an increasing ``grid``.

The curve must rise or fall, by more than rounding at every step, across
the whole of ``rho_range``, in which estimates are pooled. That branch, extended as
far as it stays monotone, is inverted by interpolation for every value,
and values beyond it are clipped to its ends, exactly as a fully monotone
curve clips at the ends of the grid; individual estimates are not
confined to ``rho_range``. A bend outside the range therefore only
shortens the usable branch. No finite value is dropped and
the order of the values is kept, so the pooled median is not biased by
which values happen to sit at levels the bend repeats.

A curve that is nonfinite, flat, or turns inside ``rho_range``
identifies nothing there and gives NaN for every value. NaN observations
stay NaN.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:158](../../src/boldtailor/_ar1_moments.py#L158)
- [src/boldtailor/ar1.py:127](../../src/boldtailor/ar1.py#L127)
- [tests/test_ar1.py:202](../../tests/test_ar1.py#L202)
- [tests/test_ar1.py:229](../../tests/test_ar1.py#L229)

<a id="definition-16"></a>

## `correct_lag_ratio`

[Source, line 155](../../src/boldtailor/_ar1_moments.py#L155)

```python
correct_lag_ratio(values, design, rho_range=RHO_RANGE) -> np.ndarray
```

Invert the expected lag ratio on the fixed rho grid; NaN stays NaN.

Usage in package/tests:

- [tests/test_ar1.py:78](../../tests/test_ar1.py#L78)

<a id="definition-17"></a>

## `finite_median`

[Source, line 161](../../src/boldtailor/_ar1_moments.py#L161)

```python
finite_median(values) -> float
```

Median of the finite values; NaN if there are none.

Usage in package/tests:

- [src/boldtailor/_ar1_moments.py:170](../../src/boldtailor/_ar1_moments.py#L170)
- [src/boldtailor/ar1.py:133](../../src/boldtailor/ar1.py#L133)

<a id="definition-18"></a>

## `pool_rho`

[Source, line 168](../../src/boldtailor/_ar1_moments.py#L168)

```python
pool_rho(values, rho_range=RHO_RANGE) -> float
```

Median of finite values clipped to ``rho_range``; 0.0 if none are finite.

Usage in package/tests:

- [src/boldtailor/ar1.py:132](../../src/boldtailor/ar1.py#L132)
- [tests/test_ar1.py:95](../../tests/test_ar1.py#L95)
- [tests/test_ar1.py:96](../../tests/test_ar1.py#L96)
- [tests/test_ar1.py:97](../../tests/test_ar1.py#L97)
