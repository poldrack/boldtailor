# `boldtailor._fit_diagnostics`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_fit_diagnostics.py)

Common diagnostics for conventional and prepared GLM fits.

## Definitions

- [`validate_nested_ols_delta`](#definition-1)
- [`rank_warnings`](#definition-2)
- [`validate_result_dimensions`](#definition-3)
- [`nested_ols_delta`](#definition-4)
- [`delta_r2_activity`](#definition-5)
- [`delta_r2_identity`](#definition-6)

<a id="definition-1"></a>

## `validate_nested_ols_delta`

[Source, line 10](../../src/boldtailor/_fit_diagnostics.py#L10)

```python
validate_nested_ols_delta(raw_delta_r2)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fit_diagnostics.py:55](../../src/boldtailor/_fit_diagnostics.py#L55)
- [src/boldtailor/results.py:269](../../src/boldtailor/results.py#L269)

<a id="definition-2"></a>

## `rank_warnings`

[Source, line 15](../../src/boldtailor/_fit_diagnostics.py#L15)

```python
rank_warnings(rank, columns, run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:202](../../src/boldtailor/fit.py#L202)
- [src/boldtailor/prepared_fit.py:336](../../src/boldtailor/prepared_fit.py#L336)

<a id="definition-3"></a>

## `validate_result_dimensions`

[Source, line 23](../../src/boldtailor/_fit_diagnostics.py#L23)

```python
validate_result_dimensions(data, result, *, input_label)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/fit.py:84](../../src/boldtailor/fit.py#L84)
- [src/boldtailor/prepared_fit.py:199](../../src/boldtailor/prepared_fit.py#L199)

<a id="definition-4"></a>

## `nested_ols_delta`

[Source, line 44](../../src/boldtailor/_fit_diagnostics.py#L44)

```python
nested_ols_delta(signals, full_designs, nuisance_designs, *, allow_undefined)
```

Return (full R2, nuisance R2) from nested OLS fits; guard monotonicity.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:185](../../src/boldtailor/_hrf_glm.py#L185)
- [src/boldtailor/fit.py:87](../../src/boldtailor/fit.py#L87)
- [src/boldtailor/prepared_fit.py:101](../../src/boldtailor/prepared_fit.py#L101)

<a id="definition-5"></a>

## `delta_r2_activity`

[Source, line 59](../../src/boldtailor/_fit_diagnostics.py#L59)

```python
delta_r2_activity(*, name, parent_id, inferential_noise_model, nuisance_model, undefined_features)
```

Provenance activity shared by every task delta-R2 entry point.

Usage in package/tests:

- [src/boldtailor/_fit_diagnostics.py:79](../../src/boldtailor/_fit_diagnostics.py#L79)
- [src/boldtailor/_hrf_glm.py:220](../../src/boldtailor/_hrf_glm.py#L220)
- [src/boldtailor/fit.py:178](../../src/boldtailor/fit.py#L178)
- [src/boldtailor/prepared_fit.py:238](../../src/boldtailor/prepared_fit.py#L238)

<a id="definition-6"></a>

## `delta_r2_identity`

[Source, line 77](../../src/boldtailor/_fit_diagnostics.py#L77)

```python
delta_r2_identity(*, name, inferential_noise_model, nuisance_model)
```

Hashed identity of a delta-R2 comparison (activity minus run-specific keys).

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:230](../../src/boldtailor/_hrf_glm.py#L230)
- [src/boldtailor/fit.py:139](../../src/boldtailor/fit.py#L139)
- [src/boldtailor/prepared_fit.py:178](../../src/boldtailor/prepared_fit.py#L178)
