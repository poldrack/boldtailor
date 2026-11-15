# `boldtailor.diagnostics`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/diagnostics.py)

Descriptive diagnostics of fitted trial betas: one-sample t and RT correlation.

## Module contract

Descriptive diagnostics of fitted trial betas: one-sample t and RT correlation.

Inputs are lists with one ``(trials, features)`` beta array per run. Nothing
here fits a model, and no p-value is corrected for multiple comparisons.

## Definitions

- [`_beta_arrays`](#definition-1)
- [`_finite_mean`](#definition-2)
- [`_sample_variance`](#definition-3)
- [`one_sample_t`](#definition-4)
- [`_validate_runs`](#definition-5)
- [`_centered`](#definition-6)
- [`pearson_correlation`](#definition-7)
- [`_run_moments`](#definition-8)
- [`correlate_rt`](#definition-9)
- [`even_run_points`](#definition-10)

<a id="definition-1"></a>

## `_beta_arrays`

[Source, line 15](../../src/boldtailor/diagnostics.py#L15)

```python
_beta_arrays(run_betas)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/diagnostics.py:66](../../src/boldtailor/diagnostics.py#L66)

<a id="definition-2"></a>

## `_finite_mean`

[Source, line 24](../../src/boldtailor/diagnostics.py#L24)

```python
_finite_mean(arrays)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/diagnostics.py:67](../../src/boldtailor/diagnostics.py#L67)

<a id="definition-3"></a>

## `_sample_variance`

[Source, line 35](../../src/boldtailor/diagnostics.py#L35)

```python
_sample_variance(arrays, mean, count)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/diagnostics.py:68](../../src/boldtailor/diagnostics.py#L68)

<a id="definition-4"></a>

## `one_sample_t`

[Source, line 55](../../src/boldtailor/diagnostics.py#L55)

```python
one_sample_t(run_betas)
```

Pool finite trial betas across runs, assuming independent observations.

Each input has shape (trials, grayordinates). Trials have equal weight;
no run-wise centering is applied. Return mean, signed t, two-sided p,
observation count and degrees of freedom, keyed by ``ONE_SAMPLE_T_NAMES``.
P-values are uncorrected and ignore trial covariance and HRF/ridge
selection. Fewer than two trials or zero sample variance gives NaN t/p; no
finite trials gives NaN in all maps. These are descriptive maps relative to
the fitted model's zero.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:302](../../src/boldtailor/workflow/run.py#L302)

<a id="definition-5"></a>

## `_validate_runs`

[Source, line 85](../../src/boldtailor/diagnostics.py#L85)

```python
_validate_runs(beta_runs, rt_runs, run_numbers)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/diagnostics.py:147](../../src/boldtailor/diagnostics.py#L147)

<a id="definition-6"></a>

## `_centered`

[Source, line 109](../../src/boldtailor/diagnostics.py#L109)

```python
_centered(beta, rt)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/diagnostics.py:132](../../src/boldtailor/diagnostics.py#L132)
- [src/boldtailor/diagnostics.py:170](../../src/boldtailor/diagnostics.py#L170)

<a id="definition-7"></a>

## `pearson_correlation`

[Source, line 121](../../src/boldtailor/diagnostics.py#L121)

```python
pearson_correlation(cross, beta_ss, rt_ss, counts)
```

Pearson r from centered cross/sum-of-squares; NaN below 3 trials or no variance.

Usage in package/tests:

- [src/boldtailor/diagnostics.py:150](../../src/boldtailor/diagnostics.py#L150)
- [src/boldtailor/diagnostics.py:160](../../src/boldtailor/diagnostics.py#L160)

<a id="definition-8"></a>

## `_run_moments`

[Source, line 129](../../src/boldtailor/diagnostics.py#L129)

```python
_run_moments(beta_runs, rt_runs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/diagnostics.py:148](../../src/boldtailor/diagnostics.py#L148)

<a id="definition-9"></a>

## `correlate_rt`

[Source, line 140](../../src/boldtailor/diagnostics.py#L140)

```python
correlate_rt(beta_runs, rt_runs, *, run_numbers)
```

Pearson r after matched-mask within-run centering; no p-values.

Valid trials have a finite beta and a finite positive RT. Return per-run,
all-run, odd-run, and even-run correlations (by BIDS run number) and the
matching trial counts. Fewer than three trials or zero variance gives NaN.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:260](../../src/boldtailor/workflow/analysis.py#L260)

<a id="definition-10"></a>

## `even_run_points`

[Source, line 165](../../src/boldtailor/diagnostics.py#L165)

```python
even_run_points(beta_runs, rt_runs, run_numbers, vertex)
```

Return matched-mask, within-run centered (RT, beta) points from even runs.

Usage in package/tests:

- [src/boldtailor/workflow/plots.py:195](../../src/boldtailor/workflow/plots.py#L195)
