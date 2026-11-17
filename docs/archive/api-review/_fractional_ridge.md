# `boldtailor._fractional_ridge`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_fractional_ridge.py)

Fractional ridge in the nuisance-projected, raw trial-coefficient basis.

## Definitions

- [`fraction_grid`](#definition-1)
- [`fraction_map`](#definition-2)
- [`regularization`](#definition-3)
- [`fraction_metadata`](#definition-4)
- [`PreparedFractionBetas`](#definition-5)
- [`PreparedFractionBetas.solve`](#definition-6)
- [`PreparedFractionBetas.betas_at`](#definition-7)
- [`prepare_fraction_betas`](#definition-8)
- [`_normalized_singular_values`](#definition-9)
- [`fraction_alphas`](#definition-10)
- [`fit_fraction_run`](#definition-11)

<a id="definition-1"></a>

## `fraction_grid`

[Source, line 20](../../src/boldtailor/_fractional_ridge.py#L20)

```python
fraction_grid(values)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fractional_ridge.py:44](../../src/boldtailor/fractional_ridge.py#L44)
- [src/boldtailor/fractional_ridge.py:97](../../src/boldtailor/fractional_ridge.py#L97)
- [tests/oracles.py:290](../../tests/oracles.py#L290)

<a id="definition-2"></a>

## `fraction_map`

[Source, line 31](../../src/boldtailor/_fractional_ridge.py#L31)

```python
fraction_map(values, n_features)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:49](../../src/boldtailor/_fractional_ridge.py#L49)
- [src/boldtailor/_fractional_ridge.py:81](../../src/boldtailor/_fractional_ridge.py#L81)

<a id="definition-3"></a>

## `regularization`

[Source, line 43](../../src/boldtailor/_fractional_ridge.py#L43)

```python
regularization(ridge_alpha, ridge_fraction, n_features)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:153](../../src/boldtailor/_selected_hrf_fit.py#L153)
- [src/boldtailor/single_trial.py:64](../../src/boldtailor/single_trial.py#L64)

<a id="definition-4"></a>

## `fraction_metadata`

[Source, line 52](../../src/boldtailor/_fractional_ridge.py#L52)

```python
fraction_metadata(fractions)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:125](../../src/boldtailor/_selected_hrf_fit.py#L125)
- [src/boldtailor/single_trial.py:70](../../src/boldtailor/single_trial.py#L70)

<a id="definition-5"></a>

## `PreparedFractionBetas`

[Source, line 66](../../src/boldtailor/_fractional_ridge.py#L66)

```python
@dataclass(frozen=True)
class PreparedFractionBetas
```

Raw trial-coefficient basis and signal coordinates for fractional ridge.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
trial_design: np.ndarray
nuisance: np.ndarray
signals: np.ndarray
projected_signals: np.ndarray
singular_values: np.ndarray
right_vectors: np.ndarray
ols_coordinates: np.ndarray
valid: np.ndarray
diagnostics: dict
```

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:114](../../src/boldtailor/_fractional_ridge.py#L114)

<a id="definition-6"></a>

## `PreparedFractionBetas.solve`

[Source, line 79](../../src/boldtailor/_fractional_ridge.py#L79)

```python
PreparedFractionBetas.solve(self, fractions)
```

Return trial betas and implied alpha for each feature's fraction.

<a id="definition-7"></a>

## `PreparedFractionBetas.betas_at`

[Source, line 93](../../src/boldtailor/_fractional_ridge.py#L93)

```python
PreparedFractionBetas.betas_at(self, fraction)
```

Evaluate a fraction without retaining candidate outputs.

<a id="definition-8"></a>

## `prepare_fraction_betas`

[Source, line 98](../../src/boldtailor/_fractional_ridge.py#L98)

```python
prepare_fraction_betas(x, nuisance, signals)
```

Validate in normalized coordinates, then prepare the raw coefficient SVD.

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:149](../../src/boldtailor/_fractional_ridge.py#L149)
- [tests/oracles.py:291](../../tests/oracles.py#L291)

<a id="definition-9"></a>

## `_normalized_singular_values`

[Source, line 119](../../src/boldtailor/_fractional_ridge.py#L119)

```python
_normalized_singular_values(s, vt, scale)
```

Singular values of the unit-norm columns, from the raw SVD factors.

X = U S Vt, so X / scale shares its singular values with the small
k-by-k matrix S Vt / scale; no second factorization of the T-by-k design.

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:107](../../src/boldtailor/_fractional_ridge.py#L107)

<a id="definition-10"></a>

## `fraction_alphas`

[Source, line 128](../../src/boldtailor/_fractional_ridge.py#L128)

```python
fraction_alphas(s, ols, fractions)
```

Bracket using singular-value extrema, then bisect in log-alpha space.

Usage in package/tests:

- [src/boldtailor/_fractional_ridge.py:87](../../src/boldtailor/_fractional_ridge.py#L87)

<a id="definition-11"></a>

## `fit_fraction_run`

[Source, line 148](../../src/boldtailor/_fractional_ridge.py#L148)

```python
fit_fraction_run(x, nuisance, signals, *, fractions)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:51](../../src/boldtailor/_selected_hrf_fit.py#L51)
- [src/boldtailor/single_trial.py:95](../../src/boldtailor/single_trial.py#L95)
