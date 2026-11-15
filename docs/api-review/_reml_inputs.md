# `boldtailor._reml_inputs`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_reml_inputs.py)

Input validation for the public scalar REML entry points.

## Definitions

- [`_check_finite`](#definition-1)
- [`_check_matrix`](#definition-2)
- [`_check_group`](#definition-3)
- [`_check_random`](#definition-4)
- [`_as_random`](#definition-5)
- [`validate_groups`](#definition-6)
- [`checked_random`](#definition-7)
- [`feature_rho`](#definition-8)

<a id="definition-1"></a>

## `_check_finite`

[Source, line 6](../../src/boldtailor/_reml_inputs.py#L6)

```python
_check_finite(values, name, index)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_inputs.py:14](../../src/boldtailor/_reml_inputs.py#L14)

<a id="definition-2"></a>

## `_check_matrix`

[Source, line 11](../../src/boldtailor/_reml_inputs.py#L11)

```python
_check_matrix(values, name, index)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_inputs.py:18](../../src/boldtailor/_reml_inputs.py#L18)
- [src/boldtailor/_reml_inputs.py:19](../../src/boldtailor/_reml_inputs.py#L19)
- [src/boldtailor/_reml_inputs.py:27](../../src/boldtailor/_reml_inputs.py#L27)

<a id="definition-3"></a>

## `_check_group`

[Source, line 17](../../src/boldtailor/_reml_inputs.py#L17)

```python
_check_group(group, index)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_inputs.py:43](../../src/boldtailor/_reml_inputs.py#L43)

<a id="definition-4"></a>

## `_check_random`

[Source, line 24](../../src/boldtailor/_reml_inputs.py#L24)

```python
_check_random(group, random, label, index)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_inputs.py:56](../../src/boldtailor/_reml_inputs.py#L56)

<a id="definition-5"></a>

## `_as_random`

[Source, line 34](../../src/boldtailor/_reml_inputs.py#L34)

```python
_as_random(random)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_reml_inputs.py:52](../../src/boldtailor/_reml_inputs.py#L52)

<a id="definition-6"></a>

## `validate_groups`

[Source, line 38](../../src/boldtailor/_reml_inputs.py#L38)

```python
validate_groups(groups)
```

Raise ValueError unless the groups are finite, 2-D and share features.

Usage in package/tests:

- [src/boldtailor/reml.py:190](../../src/boldtailor/reml.py#L190)
- [src/boldtailor/reml.py:282](../../src/boldtailor/reml.py#L282)

<a id="definition-7"></a>

## `checked_random`

[Source, line 50](../../src/boldtailor/_reml_inputs.py#L50)

```python
checked_random(groups, random_design, label)
```

One validated float random design (or None) per group from any iterable.

Usage in package/tests:

- [src/boldtailor/reml.py:191](../../src/boldtailor/reml.py#L191)
- [src/boldtailor/reml.py:267](../../src/boldtailor/reml.py#L267)

<a id="definition-8"></a>

## `feature_rho`

[Source, line 60](../../src/boldtailor/_reml_inputs.py#L60)

```python
feature_rho(rho, n_features)
```

Broadcast rho to one stationary value per feature.

Usage in package/tests:

- [src/boldtailor/reml.py:170](../../src/boldtailor/reml.py#L170)
