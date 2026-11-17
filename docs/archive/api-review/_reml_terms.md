# `boldtailor._reml_terms`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_reml_terms.py)

Per-group sufficient statistics for scalar REML.

## Definitions

- [`FixedProjection`](#definition-1)
- [`FixedProjection.__post_init__`](#definition-2)
- [`project_fixed`](#definition-3)
- [`reproject_signals`](#definition-4)
- [`RemlTerms`](#definition-5)
- [`RemlTerms.__post_init__`](#definition-6)
- [`_smaller_gram`](#definition-7)
- [`_spectral_norm`](#definition-8)
- [`_projected_random`](#definition-9)
- [`_random_directions`](#definition-10)
- [`_orthogonal_energy`](#definition-11)
- [`reml_terms`](#definition-12)

<a id="definition-1"></a>

## `FixedProjection`

[Source, line 18](../../src/boldtailor/_reml_terms.py#L18)

```python
@dataclass(frozen=True)
class FixedProjection
```

Whitened signals with the whitened fixed design projected out.

``energy`` at or below ``rounding_floor`` is projection rounding; at or
below ``uncertainty_floor`` it cannot be told apart from the error of the
computed fixed span.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
basis: np.ndarray
signals: np.ndarray
energy: np.ndarray
rounding_floor: np.ndarray
uncertainty_floor: np.ndarray
dof: int
rho: float
```

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:47](../../src/boldtailor/_reml_terms.py#L47)
- [src/boldtailor/_reml_terms.py:62](../../src/boldtailor/_reml_terms.py#L62)

<a id="definition-2"></a>

## `FixedProjection.__post_init__`

[Source, line 34](../../src/boldtailor/_reml_terms.py#L34)

```python
FixedProjection.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `project_fixed`

[Source, line 39](../../src/boldtailor/_reml_terms.py#L39)

```python
project_fixed(signals, fixed_design, rho: float=0.0) -> FixedProjection
```

Whiten by AR(1) and remove the span of the fixed design.

Usage in package/tests:

- [src/boldtailor/reml.py:69](../../src/boldtailor/reml.py#L69)
- [tests/test_reml.py:39](../../tests/test_reml.py#L39)
- [tests/test_reml_criterion.py:21](../../tests/test_reml_criterion.py#L21)
- [tests/test_reml_criterion.py:80](../../tests/test_reml_criterion.py#L80)

<a id="definition-4"></a>

## `reproject_signals`

[Source, line 58](../../src/boldtailor/_reml_terms.py#L58)

```python
reproject_signals(projection: FixedProjection, signals) -> FixedProjection
```

Whiten and project ``signals`` onto an existing fixed-design basis.

Usage in package/tests:

- [src/boldtailor/reml.py:85](../../src/boldtailor/reml.py#L85)

<a id="definition-5"></a>

## `RemlTerms`

[Source, line 74](../../src/boldtailor/_reml_terms.py#L74)

```python
@dataclass(frozen=True)
class RemlTerms
```

Sufficient statistics of one group for the REML criterion.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
singular2: np.ndarray
coordinates2: np.ndarray
residual_ss: np.ndarray
dof: int
energy: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:138](../../src/boldtailor/_reml_terms.py#L138)

<a id="definition-6"></a>

## `RemlTerms.__post_init__`

[Source, line 83](../../src/boldtailor/_reml_terms.py#L83)

```python
RemlTerms.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `_smaller_gram`

[Source, line 87](../../src/boldtailor/_reml_terms.py#L87)

```python
_smaller_gram(matrix)
```

Z'Z or ZZ', whichever is smaller; both share the nonzero eigenvalues.

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:99](../../src/boldtailor/_reml_terms.py#L99)

<a id="definition-8"></a>

## `_spectral_norm`

[Source, line 93](../../src/boldtailor/_reml_terms.py#L93)

```python
_spectral_norm(matrix)
```

Largest singular value from the smaller Gram matrix, without a second SVD.

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:105](../../src/boldtailor/_reml_terms.py#L105)

<a id="definition-9"></a>

## `_projected_random`

[Source, line 103](../../src/boldtailor/_reml_terms.py#L103)

```python
_projected_random(projection, random_design)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:112](../../src/boldtailor/_reml_terms.py#L112)

<a id="definition-10"></a>

## `_random_directions`

[Source, line 108](../../src/boldtailor/_reml_terms.py#L108)

```python
_random_directions(projection, random_design)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:135](../../src/boldtailor/_reml_terms.py#L135)

<a id="definition-11"></a>

## `_orthogonal_energy`

[Source, line 118](../../src/boldtailor/_reml_terms.py#L118)

```python
_orthogonal_energy(projection, directions, coordinates)
```

Energy outside the retained random span, without subtracting energies.

At full restricted rank the complement has dimension zero. Otherwise
forming the residual preserves small orthogonal signals that disappear
in ``||y||² - ||U'y||²``. Its rounding error is at the amplitude scale.

Usage in package/tests:

- [src/boldtailor/_reml_terms.py:141](../../src/boldtailor/_reml_terms.py#L141)

<a id="definition-12"></a>

## `reml_terms`

[Source, line 131](../../src/boldtailor/_reml_terms.py#L131)

```python
reml_terms(projection: FixedProjection, random_design: np.ndarray | None) -> RemlTerms
```

Reduce a projected group to singular values and squared coordinates.

Usage in package/tests:

- [src/boldtailor/reml.py:107](../../src/boldtailor/reml.py#L107)
- [tests/test_reml.py:39](../../tests/test_reml.py#L39)
- [tests/test_reml_criterion.py:21](../../tests/test_reml_criterion.py#L21)
- [tests/test_reml_criterion.py:80](../../tests/test_reml_criterion.py#L80)
