# `boldtailor._scalars`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_scalars.py)

Scalar type predicates that never treat booleans as numbers.

## Definitions

- [`is_boolean`](#definition-1)
- [`is_integer`](#definition-2)
- [`is_real`](#definition-3)

<a id="definition-1"></a>

## `is_boolean`

[Source, line 8](../../src/boldtailor/_scalars.py#L8)

```python
is_boolean(value: object) -> bool
```

True for Python and NumPy booleans.

Usage in package/tests:

- [src/boldtailor/_scalars.py:15](../../src/boldtailor/_scalars.py#L15)
- [src/boldtailor/_scalars.py:20](../../src/boldtailor/_scalars.py#L20)
- [src/boldtailor/model.py:312](../../src/boldtailor/model.py#L312)

<a id="definition-2"></a>

## `is_integer`

[Source, line 13](../../src/boldtailor/_scalars.py#L13)

```python
is_integer(value: object) -> bool
```

True for integral numbers (including NumPy integers) that are not booleans.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:64](../../src/boldtailor/_denoising_cv.py#L64)
- [src/boldtailor/_denoising_pool.py:250](../../src/boldtailor/_denoising_pool.py#L250)
- [src/boldtailor/hrf_library.py:84](../../src/boldtailor/hrf_library.py#L84)
- [src/boldtailor/hrf_library.py:116](../../src/boldtailor/hrf_library.py#L116)

<a id="definition-3"></a>

## `is_real`

[Source, line 18](../../src/boldtailor/_scalars.py#L18)

```python
is_real(value: object) -> bool
```

True for real numbers (including NumPy scalars) that are not booleans.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:71](../../src/boldtailor/_denoising_cv.py#L71)
- [src/boldtailor/_denoising_gate.py:72](../../src/boldtailor/_denoising_gate.py#L72)
- [src/boldtailor/_denoising_pool.py:126](../../src/boldtailor/_denoising_pool.py#L126)
- [src/boldtailor/_fractional_ridge.py:33](../../src/boldtailor/_fractional_ridge.py#L33)
