# `boldtailor._denoising_pool`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_denoising_pool.py)

GLMsingle ON-OFF R² noise pool, scoring features, and run-wise pool PCA.

## Module contract

GLMsingle ON-OFF R² noise pool, scoring features, and run-wise pool PCA.

The pool statistic is GLMsingle's ON-OFF R²: one task regressor (every
trial, amplitude 1) convolved with the library's canonical HRF (candidate
0), a coefficient shared by all runs, and each run's baseline confounds plus
intercept as run-specific nuisance, fit in-sample on all runs (GLMsingle
uses polynomial drift terms; Boldtailor analyses carry confounds instead).
It ignores the caller's modulators and HRF choices. Every input feature is a
candidate: the core is anatomy-agnostic and takes no mask. The pool holds
finite values below the threshold (GLMsingle ``badR2``), the scoring
features finite values above it with a defined HRF (``pcR2cutoff``, same
threshold), or the 100 best when none passes. The threshold is fixed or,
with ``"auto"``, GLMsingle's two-component Gaussian-mixture tail threshold
of the finite values (see :mod:`boldtailor._mixture_threshold`). Pool time
series are projected off the run's baseline confounds plus intercept,
numerically zero columns are discarded, and the rest are scaled to unit L2
norm before a reduced SVD.

This module reimplements procedures from GLMsingle
(https://github.com/cvnlab/GLMsingle), Copyright (c) 2021, Kendrick Kay,
distributed under the BSD 3-Clause License; see
src/boldtailor/_resources/GLMsingle-LICENSE.txt for the copyright
notice, conditions, and disclaimer.
It is an independent reimplementation, not a copy of
GLMsingle code. Follows the GLMdenoise noise-pool/PC procedure used by
GLMsingle (ON-OFF R² noise pool, run-wise temporal PCs of the pool).
Reference: Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr,
M.J., Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. eLife, 11, e77599.
https://doi.org/10.7554/eLife.77599

## Definitions

- [`_onoff_terms`](#definition-1)
- [`onoff_r2`](#definition-2)
- [`PoolMasks`](#definition-3)
- [`PoolMasks.__post_init__`](#definition-4)
- [`PoolMasks.pool_size`](#definition-5)
- [`PoolMasks.scoring_size`](#definition-6)
- [`validate_threshold`](#definition-7)
- [`validate_feature_mask`](#definition-8)
- [`_validate_statistic`](#definition-9)
- [`_mixture`](#definition-10)
- [`_best_features`](#definition-11)
- [`pool_masks`](#definition-12)
- [`RunComponents`](#definition-13)
- [`RunComponents.__post_init__`](#definition-14)
- [`RunComponents.rank`](#definition-15)
- [`RunComponents.splits_tied_block`](#definition-16)
- [`RunComponents.unavailable_reason`](#definition-17)
- [`RunComponents.prefix`](#definition-18)
- [`_validate_count`](#definition-19)
- [`_projected_pool`](#definition-20)
- [`_rank_tolerance`](#definition-21)
- [`_signed`](#definition-22)
- [`_svd`](#definition-23)
- [`run_components`](#definition-24)
- [`analysis_components`](#definition-25)

<a id="definition-1"></a>

## `_onoff_terms`

[Source, line 56](../../src/boldtailor/_denoising_pool.py#L56)

```python
_onoff_terms(run, y, label)
```

Projected ON column, projected BOLD, and its energy for one run.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:75](../../src/boldtailor/_denoising_pool.py#L75)

<a id="definition-2"></a>

## `onoff_r2`

[Source, line 67](../../src/boldtailor/_denoising_pool.py#L67)

```python
onoff_r2(data, library, *, run_labels=None) -> np.ndarray
```

GLMsingle ON-OFF R² per feature over all runs; NaN for zero energy.

``1 - sum_r ||M_r y_r - M_r x_r b||^2 / sum_r ||M_r y_r||^2`` with ``M_r``
projecting off run r's confounds plus intercept and ``b`` shared.

Usage in package/tests:

- [src/boldtailor/denoising.py:155](../../src/boldtailor/denoising.py#L155)
- [tests/test_denoising.py:73](../../tests/test_denoising.py#L73)
- [tests/test_denoising_cv.py:114](../../tests/test_denoising_cv.py#L114)
- [tests/test_denoising_pool.py:42](../../tests/test_denoising_pool.py#L42)

<a id="definition-3"></a>

## `PoolMasks`

[Source, line 92](../../src/boldtailor/_denoising_pool.py#L92)

```python
@dataclass(frozen=True, kw_only=True)
class PoolMasks
```

Pool and scoring feature masks fixed for every candidate PC count.

``mixture`` is the fitted mixture when the threshold was automatic.
``fallback`` marks GLMsingle's best-100 scoring set, which may overlap
the pool; otherwise the masks are disjoint.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
pool: np.ndarray
scoring: np.ndarray
threshold: float
mixture: MixtureThreshold | None = None
fallback: bool = False
```

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:190](../../src/boldtailor/_denoising_pool.py#L190)
- [tests/test_denoising_pool.py:243](../../tests/test_denoising_pool.py#L243)
- [tests/test_denoising_pool.py:242](../../tests/test_denoising_pool.py#L242)

<a id="definition-4"></a>

## `PoolMasks.__post_init__`

[Source, line 106](../../src/boldtailor/_denoising_pool.py#L106)

```python
PoolMasks.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `PoolMasks.pool_size`

[Source, line 114](../../src/boldtailor/_denoising_pool.py#L114)

```python
@property
PoolMasks.pool_size(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `PoolMasks.scoring_size`

[Source, line 118](../../src/boldtailor/_denoising_pool.py#L118)

```python
@property
PoolMasks.scoring_size(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `validate_threshold`

[Source, line 122](../../src/boldtailor/_denoising_pool.py#L122)

```python
validate_threshold(value, name='pool_r2_threshold') -> float | str
```

``"auto"`` (mixture rule) or a finite real number, as a float.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:181](../../src/boldtailor/_denoising_pool.py#L181)
- [src/boldtailor/denoising.py:124](../../src/boldtailor/denoising.py#L124)

<a id="definition-8"></a>

## `validate_feature_mask`

[Source, line 131](../../src/boldtailor/_denoising_pool.py#L131)

```python
validate_feature_mask(mask, n_features, name='pool') -> np.ndarray
```

Read-only copy of a Boolean vector with one entry per feature.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:217](../../src/boldtailor/_denoising_cv.py#L217)
- [src/boldtailor/_denoising_pool.py:298](../../src/boldtailor/_denoising_pool.py#L298)

<a id="definition-9"></a>

## `_validate_statistic`

[Source, line 139](../../src/boldtailor/_denoising_pool.py#L139)

```python
_validate_statistic(statistic, n_features)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:183](../../src/boldtailor/_denoising_pool.py#L183)

<a id="definition-10"></a>

## `_mixture`

[Source, line 149](../../src/boldtailor/_denoising_pool.py#L149)

```python
_mixture(values, context)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:185](../../src/boldtailor/_denoising_pool.py#L185)

<a id="definition-11"></a>

## `_best_features`

[Source, line 159](../../src/boldtailor/_denoising_pool.py#L159)

```python
_best_features(scores, candidates)
```

The ``FALLBACK_SIZE`` candidates with the highest scores.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:192](../../src/boldtailor/_denoising_pool.py#L192)

<a id="definition-12"></a>

## `pool_masks`

[Source, line 167](../../src/boldtailor/_denoising_pool.py#L167)

```python
pool_masks(statistic: np.ndarray, threshold: float | str, *, hrf_indices: np.ndarray, context: str='the noise pool') -> PoolMasks
```

Pool: finite statistic < threshold; scoring: > threshold, HRF defined.

With ``"auto"`` the threshold is the mixture tail threshold of all finite
values; a failed fit raises naming ``context``. When no feature passes,
the 100 highest-scoring candidates are scored (``fallback``). An empty
pool or scoring set is returned as such; callers decide if it is fatal.

Usage in package/tests:

- [src/boldtailor/denoising.py:157](../../src/boldtailor/denoising.py#L157)
- [tests/test_denoising.py:74](../../tests/test_denoising.py#L74)
- [tests/test_denoising.py:407](../../tests/test_denoising.py#L407)
- [tests/test_denoising_cv.py:115](../../tests/test_denoising_cv.py#L115)

<a id="definition-13"></a>

## `RunComponents`

[Source, line 203](../../src/boldtailor/_denoising_pool.py#L203)

```python
@dataclass(frozen=True, kw_only=True)
class RunComponents
```

Temporal PCs (time x rank) of one run's normalized, projected pool.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
components: np.ndarray
singular_values: np.ndarray
rank_tolerance: float
pool_size: int
retained_columns: int
```

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:303](../../src/boldtailor/_denoising_pool.py#L303)
- [tests/test_denoising_gate.py:478](../../tests/test_denoising_gate.py#L478)
- [tests/test_denoising_gate.py:610](../../tests/test_denoising_gate.py#L610)

<a id="definition-14"></a>

## `RunComponents.__post_init__`

[Source, line 212](../../src/boldtailor/_denoising_pool.py#L212)

```python
RunComponents.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-15"></a>

## `RunComponents.rank`

[Source, line 216](../../src/boldtailor/_denoising_pool.py#L216)

```python
@property
RunComponents.rank(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-16"></a>

## `RunComponents.splits_tied_block`

[Source, line 219](../../src/boldtailor/_denoising_pool.py#L219)

```python
RunComponents.splits_tied_block(self, count: int) -> bool
```

True when s[count-1] and s[count] agree within the rank tolerance.

<a id="definition-17"></a>

## `RunComponents.unavailable_reason`

[Source, line 226](../../src/boldtailor/_denoising_pool.py#L226)

```python
RunComponents.unavailable_reason(self, count: int) -> str
```

Empty when ``count`` leading PCs are uniquely defined on this run.

<a id="definition-18"></a>

## `RunComponents.prefix`

[Source, line 241](../../src/boldtailor/_denoising_pool.py#L241)

```python
RunComponents.prefix(self, count: int) -> np.ndarray
```

Read-only leading ``count`` PCs; raises when they are not unique.

<a id="definition-19"></a>

## `_validate_count`

[Source, line 249](../../src/boldtailor/_denoising_pool.py#L249)

```python
_validate_count(count) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:228](../../src/boldtailor/_denoising_pool.py#L228)

<a id="definition-20"></a>

## `_projected_pool`

[Source, line 255](../../src/boldtailor/_denoising_pool.py#L255)

```python
_projected_pool(signal, confounds, pool)
```

Unit-norm projected pool columns and each column's norm amplification.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:299](../../src/boldtailor/_denoising_pool.py#L299)

<a id="definition-21"></a>

## `_rank_tolerance`

[Source, line 266](../../src/boldtailor/_denoising_pool.py#L266)

```python
_rank_tolerance(normalized, amplification)
```

``max(shape) * eps * scale`` as elsewhere, with scale >= s[0].

Projection roundoff is ``eps * |y_j|`` per column; normalization by the
residual norm amplifies it by ``|y_j| / |r_j|``. The 2-norm of those
factors bounds the normalized perturbation and is at least sqrt(columns),
hence at least the largest singular value.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:301](../../src/boldtailor/_denoising_pool.py#L301)

<a id="definition-22"></a>

## `_signed`

[Source, line 277](../../src/boldtailor/_denoising_pool.py#L277)

```python
_signed(u)
```

Flip each column so its largest absolute entry is positive.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:304](../../src/boldtailor/_denoising_pool.py#L304)

<a id="definition-23"></a>

## `_svd`

[Source, line 283](../../src/boldtailor/_denoising_pool.py#L283)

```python
_svd(matrix)
```

Reduced SVD; falls back to LAPACK gesvd when gesdd does not converge.

gesdd (NumPy's driver) can fail on well-conditioned input with some
LAPACK builds; gesvd is slower but more robust and gives the same
decomposition.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:300](../../src/boldtailor/_denoising_pool.py#L300)

<a id="definition-24"></a>

## `run_components`

[Source, line 296](../../src/boldtailor/_denoising_pool.py#L296)

```python
run_components(signal, confounds, pool) -> RunComponents
```

Normalized pool PCA for one run; ``confounds`` excludes the intercept.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:315](../../src/boldtailor/_denoising_pool.py#L315)
- [tests/test_denoising_cv.py:625](../../tests/test_denoising_cv.py#L625)
- [tests/test_denoising_gate.py:414](../../tests/test_denoising_gate.py#L414)
- [tests/test_denoising_pool.py:370](../../tests/test_denoising_pool.py#L370)

<a id="definition-25"></a>

## `analysis_components`

[Source, line 312](../../src/boldtailor/_denoising_pool.py#L312)

```python
analysis_components(data, pool) -> tuple[RunComponents, ...]
```

Run-specific normalized pool PCs for every run of an AnalysisData.

Usage in package/tests:

- [src/boldtailor/denoising.py:167](../../src/boldtailor/denoising.py#L167)
- [tests/test_denoising.py:222](../../tests/test_denoising.py#L222)
- [tests/test_denoising_cv.py:528](../../tests/test_denoising_cv.py#L528)
- [tests/test_denoising_cv.py:116](../../tests/test_denoising_cv.py#L116)
