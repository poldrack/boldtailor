# `boldtailor._deprecation`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_deprecation.py)

One-release aliases for renamed keywords and functions.

## Definitions

- [`_Unset`](#definition-1)
- [`_Unset.__repr__`](#definition-2)
- [`deprecated`](#definition-3)
- [`renamed_keyword`](#definition-4)

<a id="definition-1"></a>

## `_Unset`

[Source, line 6](../../src/boldtailor/_deprecation.py#L6)

```python
class _Unset
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_deprecation.py:11](../../src/boldtailor/_deprecation.py#L11)

<a id="definition-2"></a>

## `_Unset.__repr__`

[Source, line 7](../../src/boldtailor/_deprecation.py#L7)

```python
_Unset.__repr__(self) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `deprecated`

[Source, line 14](../../src/boldtailor/_deprecation.py#L14)

```python
deprecated(message: str, *, stacklevel: int=3) -> None
```

Emit a DeprecationWarning attributed to the caller's caller.

Usage in package/tests:

- [src/boldtailor/_deprecation.py:30](../../src/boldtailor/_deprecation.py#L30)
- [src/boldtailor/hrf_selection.py:158](../../src/boldtailor/hrf_selection.py#L158)
- [src/boldtailor/prepared_fit.py:133](../../src/boldtailor/prepared_fit.py#L133)

<a id="definition-4"></a>

## `renamed_keyword`

[Source, line 19](../../src/boldtailor/_deprecation.py#L19)

```python
renamed_keyword(owner, old, new, old_value, new_value, *, default=UNSET)
```

Value for ``new``, accepting ``old`` for one release with a warning.

Supplying both raises TypeError; ``default`` is the new keyword's default.

Usage in package/tests:

- [src/boldtailor/single_trial.py:60](../../src/boldtailor/single_trial.py#L60)
- [src/boldtailor/single_trial.py:171](../../src/boldtailor/single_trial.py#L171)
