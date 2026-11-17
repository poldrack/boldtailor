# `boldtailor.ar1`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/ar1.py)

AR(1) whitening and bias-corrected noise autocorrelation estimates.

## Definitions

- [`_check_rho`](#definition-1)
- [`whiten`](#definition-2)
- [`RhoEstimate`](#definition-3)
- [`RhoEstimate.__post_init__`](#definition-4)
- [`RhoReference`](#definition-5)
- [`RhoReference.__post_init__`](#definition-6)
- [`_check_2d`](#definition-7)
- [`_check_matrix`](#definition-8)
- [`prepare_reference`](#definition-9)
- [`_group_reference`](#definition-10)
- [`_validated`](#definition-11)
- [`_checked_range`](#definition-12)
- [`_group_estimates`](#definition-13)
- [`_pooled`](#definition-14)
- [`estimate_rho`](#definition-15)

<a id="definition-1"></a>

## `_check_rho`

[Source, line 21](../../src/boldtailor/ar1.py#L21)

```python
_check_rho(rho)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/ar1.py:28](../../src/boldtailor/ar1.py#L28)

<a id="definition-2"></a>

## `whiten`

[Source, line 26](../../src/boldtailor/ar1.py#L26)

```python
whiten(values: np.ndarray, rho: float) -> np.ndarray
```

Whiten rows by AR(1) noise so that W R(rho) W' is the identity.

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:41](../../src/boldtailor/_reml_terms.py#L41)
- [src/boldtailor/_reml_terms.py:104](../../src/boldtailor/_reml_terms.py#L104)
- [src/boldtailor/_reml_terms.py:42](../../src/boldtailor/_reml_terms.py#L42)
- [src/boldtailor/_reml_terms.py:60](../../src/boldtailor/_reml_terms.py#L60)

<a id="definition-3"></a>

## `RhoEstimate`

[Source, line 36](../../src/boldtailor/ar1.py#L36)

```python
@dataclass(frozen=True)
class RhoEstimate
```

Per-group, per-feature AR(1) estimates and their pooled value.

``pooled`` is the median of the finite ``values`` clipped to
``rho_range``. ``pooled_at_bound`` says that it sits on a limit of that
range, where it is a bound on rho and not a measurement of it.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
values: np.ndarray
raw: np.ndarray
pooled: float
rho_range: tuple[float, float]
pooled_at_bound: bool
```

Usage in package/tests:

- [src/boldtailor/ar1.py:163](../../src/boldtailor/ar1.py#L163)

<a id="definition-4"></a>

## `RhoEstimate.__post_init__`

[Source, line 50](../../src/boldtailor/ar1.py#L50)

```python
RhoEstimate.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `RhoReference`

[Source, line 55](../../src/boldtailor/ar1.py#L55)

```python
@dataclass(frozen=True)
class RhoReference
```

Signal-independent work of one reference design, reusable across signals.

``basis`` is an orthonormal basis ``(scans, rank)`` of the design,
``amplification`` the ratio of the design's largest singular value to
that of each basis vector (it scales the degeneracy floor), and
``expected`` the expected residual lag-one ratio on the fixed rho grid.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
basis: np.ndarray
amplification: np.ndarray
expected: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/ar1.py:89](../../src/boldtailor/ar1.py#L89)

<a id="definition-6"></a>

## `RhoReference.__post_init__`

[Source, line 68](../../src/boldtailor/ar1.py#L68)

```python
RhoReference.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `_check_2d`

[Source, line 72](../../src/boldtailor/ar1.py#L72)

```python
_check_2d(values, name, where)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/ar1.py:78](../../src/boldtailor/ar1.py#L78)
- [src/boldtailor/ar1.py:94](../../src/boldtailor/ar1.py#L94)

<a id="definition-8"></a>

## `_check_matrix`

[Source, line 77](../../src/boldtailor/ar1.py#L77)

```python
_check_matrix(values, name, where='')
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/ar1.py:86](../../src/boldtailor/ar1.py#L86)
- [src/boldtailor/ar1.py:96](../../src/boldtailor/ar1.py#L96)

<a id="definition-9"></a>

## `prepare_reference`

[Source, line 83](../../src/boldtailor/ar1.py#L83)

```python
prepare_reference(design: np.ndarray) -> RhoReference
```

Precompute what ``estimate_rho`` needs from one reference design.

Usage in package/tests:

- [src/boldtailor/ar1.py:97](../../src/boldtailor/ar1.py#L97)
- [tests/test_ar1.py:143](../../tests/test_ar1.py#L143)
- [tests/test_ar1.py:153](../../tests/test_ar1.py#L153)
- [tests/test_ar1.py:534](../../tests/test_ar1.py#L534)

<a id="definition-10"></a>

## `_group_reference`

[Source, line 92](../../src/boldtailor/ar1.py#L92)

```python
_group_reference(signals, reference, index)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/ar1.py:109](../../src/boldtailor/ar1.py#L109)

<a id="definition-11"></a>

## `_validated`

[Source, line 103](../../src/boldtailor/ar1.py#L103)

```python
_validated(signals, reference)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/ar1.py:156](../../src/boldtailor/ar1.py#L156)

<a id="definition-12"></a>

## `_checked_range`

[Source, line 115](../../src/boldtailor/ar1.py#L115)

```python
_checked_range(rho_range) -> tuple[float, float]
```

``(low, high)`` as floats, inside the correction grid and increasing.

Usage in package/tests:

- [src/boldtailor/ar1.py:155](../../src/boldtailor/ar1.py#L155)

<a id="definition-13"></a>

## `_group_estimates`

[Source, line 125](../../src/boldtailor/ar1.py#L125)

```python
_group_estimates(signals, reference, rho_range)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/ar1.py:158](../../src/boldtailor/ar1.py#L158)

<a id="definition-14"></a>

## `_pooled`

[Source, line 130](../../src/boldtailor/ar1.py#L130)

```python
_pooled(values, rho_range)
```

The pooled value and whether it sits on a limit of ``rho_range``.

Usage in package/tests:

- [src/boldtailor/ar1.py:162](../../src/boldtailor/ar1.py#L162)

<a id="definition-15"></a>

## `estimate_rho`

[Source, line 137](../../src/boldtailor/ar1.py#L137)

```python
estimate_rho(signals: Sequence[np.ndarray], reference: Sequence[np.ndarray | RhoReference], *, rho_range: tuple[float, float]=RHO_RANGE) -> RhoEstimate
```

Estimate AR(1) rho per group after residualising on its reference.

Each reference is a design array ``(scans, columns)`` or a
``RhoReference`` from ``prepare_reference``; both give identical results.

``rho_range`` is the admissible range of the pooled value. The pooled
median is clipped to it, and the bias-correction curve only has to be
monotone across it. The default lower limit of -0.2 excludes strong
negative autocorrelation, which is not plausible for fMRI noise; widen
the range for other data. The default upper limit, 0.95, is the end of
the correction grid. Per-feature ``values`` are not confined to the range.

Usage in package/tests:

- [tests/test_ar1.py:87](../../tests/test_ar1.py#L87)
- [tests/test_ar1.py:126](../../tests/test_ar1.py#L126)
- [tests/test_ar1.py:138](../../tests/test_ar1.py#L138)
- [tests/test_ar1.py:295](../../tests/test_ar1.py#L295)
