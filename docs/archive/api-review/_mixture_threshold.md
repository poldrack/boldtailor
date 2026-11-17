# `boldtailor._mixture_threshold`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_mixture_threshold.py)

Two-component Gaussian-mixture tail threshold (GLMsingle's findtailthreshold).

## Module contract

Two-component Gaussian-mixture tail threshold (GLMsingle's findtailthreshold).

Follows GLMsingle's ``findtailthreshold``: a two-component
``sklearn.mixture.GaussianMixture`` (``tol=1e-10``, ``reg_covar=0``, 3
initialisations) is fit to the raw finite values of any 1-D statistic,
subsampled to at most ``MAX_VALUES`` values. Posteriors are evaluated on
500 evenly spaced points over GLMsingle's ``robustrange`` of the values,
widened to include both fitted means. The tail component is the one whose
posterior exceeds 0.5 at the right end of that grid, and the threshold is
the rightmost grid point where its posterior is at most 0.5.

Deviations, recorded in ``to_dict()``: ``random_state=0`` and a seeded
subsample make the result reproducible; convergence warnings are captured
and recorded (``converged``, ``n_iter``) instead of printed; failures raise
``ValueError`` rather than asserting.

This module reimplements procedures from GLMsingle
(https://github.com/cvnlab/GLMsingle), Copyright (c) 2021, Kendrick Kay,
distributed under the BSD 3-Clause License; see
src/boldtailor/_resources/GLMsingle-LICENSE.txt for the copyright
notice, conditions, and disclaimer.
It is an independent reimplementation, not a copy of GLMsingle
code. Follows GLMsingle's findtailthreshold and robustrange. Reference: Prince,
J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M.J., Kay, K.N. (2022).
Improving the accuracy of single-trial fMRI response estimates using GLMsingle.
eLife, 11, e77599. https://doi.org/10.7554/eLife.77599

## Definitions

- [`MixtureThreshold`](#definition-1)
- [`MixtureThreshold.to_dict`](#definition-2)
- [`robust_range`](#definition-3)
- [`_finite_values`](#definition-4)
- [`_subsample`](#definition-5)
- [`_fit`](#definition-6)
- [`_grid`](#definition-7)
- [`_threshold`](#definition-8)
- [`_ordered`](#definition-9)
- [`mixture_threshold`](#definition-10)

<a id="definition-1"></a>

## `MixtureThreshold`

[Source, line 48](../../src/boldtailor/_mixture_threshold.py#L48)

```python
@dataclass(frozen=True, kw_only=True)
class MixtureThreshold
```

Threshold and fitted components, ordered by ascending mean.

``tail`` indexes (in that order) the component dominating the right end.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
threshold: float
means: tuple[float, float]
sds: tuple[float, float]
weights: tuple[float, float]
tail: int
n_values: int
n_fitted: int
converged: bool
n_iter: int
```

Usage in package/tests:

- [src/boldtailor/_mixture_threshold.py:170](../../src/boldtailor/_mixture_threshold.py#L170)

<a id="definition-2"></a>

## `MixtureThreshold.to_dict`

[Source, line 64](../../src/boldtailor/_mixture_threshold.py#L64)

```python
MixtureThreshold.to_dict(self) -> dict
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `robust_range`

[Source, line 82](../../src/boldtailor/_mixture_threshold.py#L82)

```python
robust_range(values) -> tuple[float, float]
```

GLMsingle ``robustrange``: data range with outliers trimmed recursively.

Bounds are median +/- 5x the median-to-10th/90th percentile distance;
values beyond them are dropped and the range recomputed. Otherwise each
end is the absolute extreme if within 1.1x the median-to-0.1/99.9th
percentile distance, else that percentile.

Usage in package/tests:

- [src/boldtailor/_mixture_threshold.py:139](../../src/boldtailor/_mixture_threshold.py#L139)
- [tests/test_mixture_threshold.py:195](../../tests/test_mixture_threshold.py#L195)
- [tests/test_mixture_threshold.py:198](../../tests/test_mixture_threshold.py#L198)
- [tests/test_mixture_threshold.py:203](../../tests/test_mixture_threshold.py#L203)

<a id="definition-4"></a>

## `_finite_values`

[Source, line 104](../../src/boldtailor/_mixture_threshold.py#L104)

```python
_finite_values(values) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_mixture_threshold.py:164](../../src/boldtailor/_mixture_threshold.py#L164)

<a id="definition-5"></a>

## `_subsample`

[Source, line 122](../../src/boldtailor/_mixture_threshold.py#L122)

```python
_subsample(finite)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_mixture_threshold.py:165](../../src/boldtailor/_mixture_threshold.py#L165)

<a id="definition-6"></a>

## `_fit`

[Source, line 129](../../src/boldtailor/_mixture_threshold.py#L129)

```python
_fit(fitted)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_mixture_threshold.py:166](../../src/boldtailor/_mixture_threshold.py#L166)

<a id="definition-7"></a>

## `_grid`

[Source, line 138](../../src/boldtailor/_mixture_threshold.py#L138)

```python
_grid(finite, means)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_mixture_threshold.py:168](../../src/boldtailor/_mixture_threshold.py#L168)

<a id="definition-8"></a>

## `_threshold`

[Source, line 143](../../src/boldtailor/_mixture_threshold.py#L143)

```python
_threshold(model, grid)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_mixture_threshold.py:168](../../src/boldtailor/_mixture_threshold.py#L168)

<a id="definition-9"></a>

## `_ordered`

[Source, line 154](../../src/boldtailor/_mixture_threshold.py#L154)

```python
_ordered(values, order)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_mixture_threshold.py:172](../../src/boldtailor/_mixture_threshold.py#L172)
- [src/boldtailor/_mixture_threshold.py:174](../../src/boldtailor/_mixture_threshold.py#L174)
- [src/boldtailor/_mixture_threshold.py:173](../../src/boldtailor/_mixture_threshold.py#L173)

<a id="definition-10"></a>

## `mixture_threshold`

[Source, line 158](../../src/boldtailor/_mixture_threshold.py#L158)

```python
mixture_threshold(values) -> MixtureThreshold
```

GLMsingle-style tail threshold of the finite entries of ``values``.

Raises ``ValueError`` for fewer than two distinct finite values, a
failed fit, or a tail posterior that never reaches 0.5; no fallback.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:151](../../src/boldtailor/_denoising_pool.py#L151)
- [tests/test_denoising.py:377](../../tests/test_denoising.py#L377)
- [tests/test_denoising_pool.py:265](../../tests/test_denoising_pool.py#L265)
- [tests/test_mixture_threshold.py:107](../../tests/test_mixture_threshold.py#L107)
