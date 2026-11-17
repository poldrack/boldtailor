# `boldtailor._denoising_gate`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_denoising_gate.py)

Per-feature nested F-test significance gate on the pcstop PC count.

## Module contract

Per-feature nested F-test significance gate on the pcstop PC count.

This gate is a Boldtailor addition, not part of GLMsingle. GLMsingle's
pcstop rule is relative and has no absolute floor, so chance gains on
independent noise can choose a positive count. After pcstop chooses ``k*``,
every feature pcstop scored is fit in-sample on all runs with the frozen HRF
under two nested models: reduced (task regressors shared across runs plus
each run's baseline confounds, intercept, and missing-value indicators) and
full (reduced plus each run's first ``k*`` PCs as run-specific columns). Per
feature, ``F = ((SSE_r - SSE_f) / df1) / (SSE_f / df2)`` with ``df1 =
rank(full) - rank(reduced)`` and ``df2 = N - rank(full)``, using the
existing rank tolerances. With ``m`` of ``n`` tested features at
``p < alpha``, ``k*`` is kept only if the one-sided binomial test of ``m``
against ``alpha`` gives ``p < binomial_alpha``; otherwise the count is 0.
Features whose stacked design is rank deficient, has ``df2 <= 0``, or gains
no PC columns (``df1 <= 0``), and scoring features with a numerically zero
target in any run (which pcstop does not score), are excluded from ``n``.

By default (``noise_model="ar1"``) the F-tests are AR(1) prewhitened,
following Nilearn's first-level ``noise_model="ar1"`` convention as far as
the pooled multi-run design allows. Per run and feature, the lag-1
coefficient is estimated from the full model's OLS residuals in that run by
Nilearn's Yule-Walker estimator and truncated to Nilearn's 1/100 bins (as
``nilearn.glm.first_level.run_glm`` does). That run's data and all of its
design columns (task, baseline, indicators, PCs) are whitened by
``nilearn.glm.ARModel`` (``x[t] - rho * x[t - 1]``, first scan unscaled),
and both nested models are refit on the stacked whitened runs with the same
degrees of freedom (whitening is invertible, so ranks are unchanged).
``nilearn.glm.first_level.first_level._yule_walker`` is private: there is no
public Nilearn path to the coefficients of residuals from a pooled
multi-run fit (``run_glm`` fits one design), so it is imported directly;
Nilearn is pinned below 0.15. ``noise_model="ols"`` skips the whitening.

The F-test asks whether the PCs explain variance in the scoring features
(variance explained), not whether removing them improves task prediction.
Without prewhitening, PCs of independent autocorrelated noise pass: in an
illustrative probe (20k features, 12 runs, no shared noise; not a
calibration) the OLS gate kept independent AR(1) noise with coefficient
0.5. Prewhitening addresses lag-1 autocorrelation only: AR(1) can
under-whiten higher-order autocorrelation. The gate is in-sample, the
binomial test treats features as independent, which is optimistic for
spatially correlated features, and with few tested features its decision is
coarse (with n = 6, one feature can flip it). The gate reuses the
frozen-HRF scoring terms of :mod:`boldtailor._denoising_cv` (whose GLMsingle
attribution applies to that module).

## Definitions

- [`_alpha`](#definition-1)
- [`validate_gate`](#definition-2)
- [`validate_noise_model`](#definition-3)
- [`_Fit`](#definition-4)
- [`_rank_reason`](#definition-5)
- [`_stacked_fit`](#definition-6)
- [`ar1_coefficients`](#definition-7)
- [`_off`](#definition-8)
- [`_whitened_run`](#definition-9)
- [`_whitened_pieces`](#definition-10)
- [`_pooled_sse`](#definition-11)
- [`_GroupTest`](#definition-12)
- [`_nested_sse`](#definition-13)
- [`_group_test`](#definition-14)
- [`_arrays`](#definition-15)
- [`_record_test_results_in_place`](#definition-16)
- [`_feature_tests`](#definition-17)
- [`_gate`](#definition-18)
- [`apply_gate`](#definition-19)

<a id="definition-1"></a>

## `_alpha`

[Source, line 71](../../src/boldtailor/_denoising_gate.py#L71)

```python
_alpha(value, name) -> float
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:83](../../src/boldtailor/_denoising_gate.py#L83)
- [src/boldtailor/_denoising_gate.py:84](../../src/boldtailor/_denoising_gate.py#L84)

<a id="definition-2"></a>

## `validate_gate`

[Source, line 77](../../src/boldtailor/_denoising_gate.py#L77)

```python
validate_gate(enabled, alpha, binomial_alpha) -> tuple[bool, float, float]
```

``significance_gate`` (a bool) and its two levels in (0, 1).

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:329](../../src/boldtailor/_denoising_gate.py#L329)
- [src/boldtailor/denoising.py:126](../../src/boldtailor/denoising.py#L126)
- [tests/test_denoising_gate.py:205](../../tests/test_denoising_gate.py#L205)
- [tests/test_denoising_gate.py:206](../../tests/test_denoising_gate.py#L206)

<a id="definition-3"></a>

## `validate_noise_model`

[Source, line 88](../../src/boldtailor/_denoising_gate.py#L88)

```python
validate_noise_model(value) -> str
```

``gate_noise_model``: ``"ar1"`` (prewhitened) or ``"ols"``.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:330](../../src/boldtailor/_denoising_gate.py#L330)
- [src/boldtailor/denoising.py:126](../../src/boldtailor/denoising.py#L126)
- [tests/test_denoising_gate.py:230](../../tests/test_denoising_gate.py#L230)
- [tests/test_denoising_gate.py:236](../../tests/test_denoising_gate.py#L236)

<a id="definition-4"></a>

## `_Fit`

[Source, line 99](../../src/boldtailor/_denoising_gate.py#L99)

```python
@dataclass(frozen=True)
class _Fit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
sse: np.ndarray
rank: int
residuals: tuple[np.ndarray, ...]
```

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:136](../../src/boldtailor/_denoising_gate.py#L136)

<a id="definition-5"></a>

## `_rank_reason`

[Source, line 105](../../src/boldtailor/_denoising_gate.py#L105)

```python
_rank_reason(x, raw, dof) -> str
```

Why the projected task design is not estimable, or empty when it is.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:127](../../src/boldtailor/_denoising_gate.py#L127)

<a id="definition-6"></a>

## `_stacked_fit`

[Source, line 114](../../src/boldtailor/_denoising_gate.py#L114)

```python
_stacked_fit(group, bases) -> tuple[_Fit | None, str]
```

Pooled task fit after projecting each run off its nuisance and ``bases``.

Returns ``(None, reason)`` when the stacked task design is rank
deficient after projection or leaves no residual degrees of freedom.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:231](../../src/boldtailor/_denoising_gate.py#L231)
- [src/boldtailor/_denoising_gate.py:232](../../src/boldtailor/_denoising_gate.py#L232)

<a id="definition-7"></a>

## `ar1_coefficients`

[Source, line 142](../../src/boldtailor/_denoising_gate.py#L142)

```python
ar1_coefficients(residuals) -> np.ndarray
```

Binned lag-1 coefficients per column of (scans x features) residuals.

Nilearn's Yule-Walker estimate, truncated to ``1 / AR1_BINS`` as
``run_glm`` does. Columns without residual energy get 0 (no whitening).

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:219](../../src/boldtailor/_denoising_gate.py#L219)
- [tests/test_denoising_gate.py:307](../../tests/test_denoising_gate.py#L307)

<a id="definition-8"></a>

## `_off`

[Source, line 155](../../src/boldtailor/_denoising_gate.py#L155)

```python
_off(nuisance, x, y)
```

``x`` and ``y`` projected off the span of ``nuisance``.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:173](../../src/boldtailor/_denoising_gate.py#L173)

<a id="definition-9"></a>

## `_whitened_run`

[Source, line 161](../../src/boldtailor/_denoising_gate.py#L161)

```python
_whitened_run(terms, basis, rho, cols)
```

One run's reduced and full pieces for features ``cols``, AR(1) whitened.

Nilearn's ``ARModel`` whitens the run's design ``[task, baseline,
indicators, PCs]`` and its BOLD; each model's whitened task design and
BOLD are then projected off that model's whitened nuisance columns.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:182](../../src/boldtailor/_denoising_gate.py#L182)

<a id="definition-10"></a>

## `_whitened_pieces`

[Source, line 176](../../src/boldtailor/_denoising_gate.py#L176)

```python
_whitened_pieces(group, bases, rho)
```

Reduced and full ``(features, x, y)`` pieces per run and coefficient bin.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:221](../../src/boldtailor/_denoising_gate.py#L221)

<a id="definition-11"></a>

## `_pooled_sse`

[Source, line 188](../../src/boldtailor/_denoising_gate.py#L188)

```python
_pooled_sse(pieces, n, k)
```

Per-feature SSE of the pooled task fit over whitened run pieces.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:222](../../src/boldtailor/_denoising_gate.py#L222)

<a id="definition-12"></a>

## `_GroupTest`

[Source, line 205](../../src/boldtailor/_denoising_gate.py#L205)

```python
@dataclass(frozen=True)
class _GroupTest
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
features: np.ndarray
f: np.ndarray | None = None
df1: int = 0
df2: int = 0
reason: str = ''
rho: np.ndarray | None = None
```

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:245](../../src/boldtailor/_denoising_gate.py#L245)
- [src/boldtailor/_denoising_gate.py:235](../../src/boldtailor/_denoising_gate.py#L235)
- [src/boldtailor/_denoising_gate.py:238](../../src/boldtailor/_denoising_gate.py#L238)

<a id="definition-13"></a>

## `_nested_sse`

[Source, line 214](../../src/boldtailor/_denoising_gate.py#L214)

```python
_nested_sse(group, bases, fits, noise_model)
```

(SSE reduced, SSE full, rho per run x feature or None).

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:240](../../src/boldtailor/_denoising_gate.py#L240)

<a id="definition-14"></a>

## `_group_test`

[Source, line 226](../../src/boldtailor/_denoising_gate.py#L226)

```python
_group_test(setup, group, count, noise_model) -> _GroupTest
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:283](../../src/boldtailor/_denoising_gate.py#L283)

<a id="definition-15"></a>

## `_arrays`

[Source, line 251](../../src/boldtailor/_denoising_gate.py#L251)

```python
_arrays(n_runs, n_features)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:279](../../src/boldtailor/_denoising_gate.py#L279)
- [src/boldtailor/_denoising_gate.py:334](../../src/boldtailor/_denoising_gate.py#L334)

<a id="definition-16"></a>

## `_record_test_results_in_place`

[Source, line 261](../../src/boldtailor/_denoising_gate.py#L261)

```python
_record_test_results_in_place(values, exclusions, hrf_id, test, zero)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:284](../../src/boldtailor/_denoising_gate.py#L284)

<a id="definition-17"></a>

## `_feature_tests`

[Source, line 278](../../src/boldtailor/_denoising_gate.py#L278)

```python
_feature_tests(setup, count, noise_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:336](../../src/boldtailor/_denoising_gate.py#L336)

<a id="definition-18"></a>

## `_gate`

[Source, line 288](../../src/boldtailor/_denoising_gate.py#L288)

```python
_gate(settings, count, decision, values, exclusions=(), m=0, binomial_p=np.nan)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:343](../../src/boldtailor/_denoising_gate.py#L343)
- [src/boldtailor/_denoising_gate.py:335](../../src/boldtailor/_denoising_gate.py#L335)
- [src/boldtailor/_denoising_gate.py:340](../../src/boldtailor/_denoising_gate.py#L340)

<a id="definition-19"></a>

## `apply_gate`

[Source, line 312](../../src/boldtailor/_denoising_gate.py#L312)

```python
apply_gate(setup: ScoringSetup, count: int, *, enabled, alpha, binomial_alpha, noise_model) -> SignificanceGate
```

Keep the pcstop ``count`` only if the PCs pass the F-test gate.

``count`` must be available in every run (as a pcstop choice is).
``noise_model`` is ``"ar1"`` (prewhitened F-tests) or ``"ols"``.
With the gate disabled the count is returned untested; a zero count
skips the gate. With no testable feature the count is rejected.

Usage in package/tests:

- [src/boldtailor/denoising.py:185](../../src/boldtailor/denoising.py#L185)
- [tests/test_denoising_gate.py:89](../../tests/test_denoising_gate.py#L89)
