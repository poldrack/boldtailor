# `boldtailor._arrays`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_arrays.py)

Owned NumPy arrays with protection against accidental writes.

## Definitions

- [`readonly_array`](#definition-1)
- [`rebind`](#definition-2)
- [`own_fields`](#definition-3)
- [`own_array_tuples`](#definition-4)
- [`own_tuples`](#definition-5)

<a id="definition-1"></a>

## `readonly_array`

[Source, line 6](../../src/boldtailor/_arrays.py#L6)

```python
readonly_array(values, *, dtype=np.float64)
```

Copy values into a plain array; callers should copy again before editing.

Usage in package/tests:

- [src/boldtailor/_arrays.py:24](../../src/boldtailor/_arrays.py#L24)
- [src/boldtailor/_arrays.py:34](../../src/boldtailor/_arrays.py#L34)
- [src/boldtailor/_denoising_pool.py:85](../../src/boldtailor/_denoising_pool.py#L85)
- [src/boldtailor/_denoising_pool.py:136](../../src/boldtailor/_denoising_pool.py#L136)

<a id="definition-2"></a>

## `rebind`

[Source, line 13](../../src/boldtailor/_arrays.py#L13)

```python
rebind(instance, **values)
```

Set fields of a frozen dataclass from its own ``__post_init__``.

Usage in package/tests:

- [src/boldtailor/_arrays.py:21](../../src/boldtailor/_arrays.py#L21)
- [src/boldtailor/_arrays.py:31](../../src/boldtailor/_arrays.py#L31)
- [src/boldtailor/_arrays.py:42](../../src/boldtailor/_arrays.py#L42)
- [src/boldtailor/_denoising_cv.py:478](../../src/boldtailor/_denoising_cv.py#L478)

<a id="definition-3"></a>

## `own_fields`

[Source, line 19](../../src/boldtailor/_arrays.py#L19)

```python
own_fields(instance, names, *, dtype=np.float64)
```

Replace each named field with a read-only owned array.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:142](../../src/boldtailor/_denoising_cv.py#L142)
- [src/boldtailor/_denoising_cv.py:143](../../src/boldtailor/_denoising_cv.py#L143)
- [src/boldtailor/_denoising_cv.py:144](../../src/boldtailor/_denoising_cv.py#L144)
- [src/boldtailor/_denoising_cv.py:334](../../src/boldtailor/_denoising_cv.py#L334)

<a id="definition-4"></a>

## `own_array_tuples`

[Source, line 29](../../src/boldtailor/_arrays.py#L29)

```python
own_array_tuples(instance, names, *, dtype=np.float64)
```

Replace each named field with a tuple of read-only owned arrays.

Usage in package/tests:

- [src/boldtailor/denoising_results.py:47](../../src/boldtailor/denoising_results.py#L47)
- [src/boldtailor/denoising_results.py:172](../../src/boldtailor/denoising_results.py#L172)
- [src/boldtailor/hrf_glm_results.py:37](../../src/boldtailor/hrf_glm_results.py#L37)
- [src/boldtailor/results.py:119](../../src/boldtailor/results.py#L119)

<a id="definition-5"></a>

## `own_tuples`

[Source, line 40](../../src/boldtailor/_arrays.py#L40)

```python
own_tuples(instance, names)
```

Replace each named sequence field with an immutable tuple.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:145](../../src/boldtailor/_denoising_cv.py#L145)
- [src/boldtailor/_denoising_cv.py:333](../../src/boldtailor/_denoising_cv.py#L333)
- [src/boldtailor/_denoising_cv.py:477](../../src/boldtailor/_denoising_cv.py#L477)
- [src/boldtailor/denoising_results.py:46](../../src/boldtailor/denoising_results.py#L46)
