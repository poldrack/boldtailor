# `boldtailor._denoising_identity`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_denoising_identity.py)

Content fingerprints that tie a denoising result to its analysis data.

## Module contract

Content fingerprints that tie a denoising result to its analysis data.

Anonymous arrays carry no source metadata, so identity is checked on content:
each run's signal, time grid, baseline confounds, and events. A changed
feature order, run order, or confound table changes a fingerprint.

## Definitions

- [`RunIdentity`](#definition-1)
- [`RunIdentity.to_dict`](#definition-2)
- [`array_digest`](#definition-3)
- [`table_digest`](#definition-4)
- [`run_identities`](#definition-5)
- [`_check_runs`](#definition-6)
- [`_check_run`](#definition-7)
- [`check_identity`](#definition-8)

<a id="definition-1"></a>

## `RunIdentity`

[Source, line 17](../../src/boldtailor/_denoising_identity.py#L17)

```python
@dataclass(frozen=True)
class RunIdentity
```

Fingerprints of one run of the analysis a result was selected on.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
n_scans: int
signal: str
frame_times: str
confounds: str
confound_columns: tuple[str, ...]
events: str
```

Usage in package/tests:

- [src/boldtailor/_denoising_identity.py:60](../../src/boldtailor/_denoising_identity.py#L60)

<a id="definition-2"></a>

## `RunIdentity.to_dict`

[Source, line 27](../../src/boldtailor/_denoising_identity.py#L27)

```python
RunIdentity.to_dict(self) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `array_digest`

[Source, line 38](../../src/boldtailor/_denoising_identity.py#L38)

```python
array_digest(values) -> str
```

sha256 of an array's shape and little-endian float64 values.

Usage in package/tests:

- [src/boldtailor/_denoising_identity.py:62](../../src/boldtailor/_denoising_identity.py#L62)
- [src/boldtailor/_denoising_identity.py:63](../../src/boldtailor/_denoising_identity.py#L63)

<a id="definition-4"></a>

## `table_digest`

[Source, line 45](../../src/boldtailor/_denoising_identity.py#L45)

```python
table_digest(frame: pd.DataFrame) -> str
```

sha256 of a table's row count, column names, dtypes, and values.

Usage in package/tests:

- [src/boldtailor/_denoising_identity.py:64](../../src/boldtailor/_denoising_identity.py#L64)
- [src/boldtailor/_denoising_identity.py:66](../../src/boldtailor/_denoising_identity.py#L66)

<a id="definition-5"></a>

## `run_identities`

[Source, line 58](../../src/boldtailor/_denoising_identity.py#L58)

```python
run_identities(data) -> tuple[RunIdentity, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_identity.py:117](../../src/boldtailor/_denoising_identity.py#L117)
- [src/boldtailor/denoising.py:420](../../src/boldtailor/denoising.py#L420)
- [src/boldtailor/denoising.py:240](../../src/boldtailor/denoising.py#L240)
- [src/boldtailor/denoising.py:372](../../src/boldtailor/denoising.py#L372)

<a id="definition-6"></a>

## `_check_runs`

[Source, line 74](../../src/boldtailor/_denoising_identity.py#L74)

```python
_check_runs(expected, actual, n_features, data)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_identity.py:118](../../src/boldtailor/_denoising_identity.py#L118)

<a id="definition-7"></a>

## `_check_run`

[Source, line 101](../../src/boldtailor/_denoising_identity.py#L101)

```python
_check_run(expected, actual, label)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_identity.py:120](../../src/boldtailor/_denoising_identity.py#L120)

<a id="definition-8"></a>

## `check_identity`

[Source, line 115](../../src/boldtailor/_denoising_identity.py#L115)

```python
check_identity(expected, n_features, data, labels) -> None
```

Raise unless ``data`` is the analysis the result was selected on.

Usage in package/tests:

- [src/boldtailor/denoising.py:493](../../src/boldtailor/denoising.py#L493)
