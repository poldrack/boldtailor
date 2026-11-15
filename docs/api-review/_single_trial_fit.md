# `boldtailor._single_trial_fit`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_single_trial_fit.py)

Run-wise OLS and normalized ridge with an unpenalized nuisance span.

## Definitions

- [`TrialRunFit`](#definition-1)
- [`ProjectedTrialDesign`](#definition-2)
- [`PreparedTrialBetas`](#definition-3)
- [`PreparedTrialBetas.betas_at`](#definition-4)
- [`prepare_trial_betas`](#definition-5)
- [`validate_alpha`](#definition-6)
- [`nuisance_span`](#definition-7)
- [`residual_column_scale`](#definition-8)
- [`design_diagnostics`](#definition-9)
- [`project_trial_design`](#definition-10)
- [`fit_trial_run`](#definition-11)
- [`r_squared`](#definition-12)

<a id="definition-1"></a>

## `TrialRunFit`

[Source, line 11](../../src/boldtailor/_single_trial_fit.py#L11)

```python
@dataclass(frozen=True)
class TrialRunFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
betas: np.ndarray
nuisance_betas: np.ndarray
full_sse: np.ndarray
nuisance_sse: np.ndarray
total_ss: np.ndarray
diagnostics: dict
```

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:161](../../src/boldtailor/_fractional_ridge.py#L161)
- [src/boldtailor/_single_trial_fit.py:139](../../src/boldtailor/_single_trial_fit.py#L139)

<a id="definition-2"></a>

## `ProjectedTrialDesign`

[Source, line 21](../../src/boldtailor/_single_trial_fit.py#L21)

```python
@dataclass(frozen=True)
class ProjectedTrialDesign
```

SVD of unit-norm trial columns after removing the nuisance span.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
nuisance_basis: np.ndarray
column_scale: np.ndarray
left_vectors: np.ndarray
singular_values: np.ndarray
right_vectors: np.ndarray
diagnostics: dict
```

Usage in package/tests:

- [src/boldtailor/_single_trial_fit.py:114](../../src/boldtailor/_single_trial_fit.py#L114)

<a id="definition-3"></a>

## `PreparedTrialBetas`

[Source, line 33](../../src/boldtailor/_single_trial_fit.py#L33)

```python
@dataclass(frozen=True)
class PreparedTrialBetas
```

Signal coordinates shared by all normalized ridge candidates.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
design: ProjectedTrialDesign
coordinates: np.ndarray
varying: np.ndarray
shape: tuple[int, int]
```

Usage in package/tests:

- [src/boldtailor/_single_trial_fit.py:61](../../src/boldtailor/_single_trial_fit.py#L61)

<a id="definition-4"></a>

## `PreparedTrialBetas.betas_at`

[Source, line 41](../../src/boldtailor/_single_trial_fit.py#L41)

```python
PreparedTrialBetas.betas_at(self, alpha)
```

Return trial-by-feature coefficients without retaining the output.

<a id="definition-5"></a>

## `prepare_trial_betas`

[Source, line 53](../../src/boldtailor/_single_trial_fit.py#L53)

```python
prepare_trial_betas(x, nuisance, signals)
```

Project signals and factor the normalized trial design once.

Usage in package/tests:

- [src/boldtailor/_single_trial_fit.py:121](../../src/boldtailor/_single_trial_fit.py#L121)
- [tests/oracles.py:283](../../tests/oracles.py#L283)

<a id="definition-6"></a>

## `validate_alpha`

[Source, line 64](../../src/boldtailor/_single_trial_fit.py#L64)

```python
validate_alpha(alpha)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:44](../../src/boldtailor/_fractional_ridge.py#L44)
- [src/boldtailor/_single_trial_fit.py:119](../../src/boldtailor/_single_trial_fit.py#L119)
- [src/boldtailor/_single_trial_fit.py:43](../../src/boldtailor/_single_trial_fit.py#L43)
- [src/boldtailor/ridge_selection.py:22](../../src/boldtailor/ridge_selection.py#L22)

<a id="definition-7"></a>

## `nuisance_span`

[Source, line 72](../../src/boldtailor/_single_trial_fit.py#L72)

```python
nuisance_span(nuisance)
```

Orthonormal basis of the nuisance column span and its rank.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:257](../../src/boldtailor/_denoising_pool.py#L257)
- [src/boldtailor/_fractional_ridge.py:101](../../src/boldtailor/_fractional_ridge.py#L101)
- [src/boldtailor/_single_trial_fit.py:110](../../src/boldtailor/_single_trial_fit.py#L110)

<a id="definition-8"></a>

## `residual_column_scale`

[Source, line 80](../../src/boldtailor/_single_trial_fit.py#L80)

```python
residual_column_scale(x, q)
```

Column norms of the nuisance-residualized design; reject unsupported ones.

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:102](../../src/boldtailor/_fractional_ridge.py#L102)
- [src/boldtailor/_single_trial_fit.py:111](../../src/boldtailor/_single_trial_fit.py#L111)

<a id="definition-9"></a>

## `design_diagnostics`

[Source, line 89](../../src/boldtailor/_single_trial_fit.py#L89)

```python
design_diagnostics(x, normalized_s, nuisance_rank)
```

Rank and degrees-of-freedom checks from normalized singular values.

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:106](../../src/boldtailor/_fractional_ridge.py#L106)
- [src/boldtailor/_single_trial_fit.py:113](../../src/boldtailor/_single_trial_fit.py#L113)

<a id="definition-10"></a>

## `project_trial_design`

[Source, line 109](../../src/boldtailor/_single_trial_fit.py#L109)

```python
project_trial_design(x, nuisance)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:130](../../src/boldtailor/_hrf_cv.py#L130)
- [src/boldtailor/_single_trial_fit.py:56](../../src/boldtailor/_single_trial_fit.py#L56)

<a id="definition-11"></a>

## `fit_trial_run`

[Source, line 117](../../src/boldtailor/_single_trial_fit.py#L117)

```python
fit_trial_run(x, nuisance, signals, *, alpha)
```

Fit all trials together; alpha acts on unit-norm projected columns.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:50](../../src/boldtailor/_selected_hrf_fit.py#L50)
- [src/boldtailor/single_trial.py:93](../../src/boldtailor/single_trial.py#L93)

<a id="definition-12"></a>

## `r_squared`

[Source, line 144](../../src/boldtailor/_single_trial_fit.py#L144)

```python
r_squared(sse, total_ss)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:288](../../src/boldtailor/_ridge_cv.py#L288)
- [src/boldtailor/_selected_hrf_fit.py:183](../../src/boldtailor/_selected_hrf_fit.py#L183)
- [src/boldtailor/_selected_hrf_fit.py:184](../../src/boldtailor/_selected_hrf_fit.py#L184)
- [src/boldtailor/_selected_hrf_fit.py:189](../../src/boldtailor/_selected_hrf_fit.py#L189)
