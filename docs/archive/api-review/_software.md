# `boldtailor._software`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_software.py)

Software versions for provenance, resolved only when requested.

## Definitions

- [`package_version`](#definition-1)
- [`software_environment`](#definition-2)

<a id="definition-1"></a>

## `package_version`

[Source, line 6](../../src/boldtailor/_software.py#L6)

```python
package_version(name: str) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_software.py:23](../../src/boldtailor/_software.py#L23)
- [src/boldtailor/bids_provenance.py:159](../../src/boldtailor/bids_provenance.py#L159)
- [src/boldtailor/bids_provenance.py:310](../../src/boldtailor/bids_provenance.py#L310)

<a id="definition-2"></a>

## `software_environment`

[Source, line 15](../../src/boldtailor/_software.py#L15)

```python
software_environment() -> dict[str, str]
```

Interpreter, platform, and package versions, resolved at call time.

Usage in package/tests:

- [src/boldtailor/_fit_lifecycle.py:46](../../src/boldtailor/_fit_lifecycle.py#L46)
- [src/boldtailor/data.py:291](../../src/boldtailor/data.py#L291)
- [src/boldtailor/prepared.py:252](../../src/boldtailor/prepared.py#L252)
- [tests/test_data.py:339](../../tests/test_data.py#L339)
