# `boldtailor.publication`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/publication.py)

Stage and publish artifact sets with locking and rollback.

## Definitions

- [`is_control_directory`](#definition-1)
- [`PublicationError`](#definition-2)
- [`PublicationError.__init__`](#definition-3)
- [`Artifact`](#definition-4)
- [`Artifact.__post_init__`](#definition-5)
- [`_Transaction`](#definition-6)
- [`_Transaction.stage`](#definition-7)
- [`_Transaction.backups`](#definition-8)
- [`publish_artifact_set`](#definition-9)
- [`_unless_kept`](#definition-10)
- [`_publish_locked`](#definition-11)
- [`_prepare_artifacts`](#definition-12)
- [`_validate_artifact_path`](#definition-13)
- [`_own_bytes`](#definition-14)
- [`_reject_duplicate_paths`](#definition-15)
- [`_validate_lock_timeout`](#definition-16)
- [`_resolve_destination`](#definition-17)
- [`_reject_source_overlap`](#definition-18)
- [`_check_target`](#definition-19)
- [`_make_transaction_directory`](#definition-20)
- [`_stage_artifacts`](#definition-21)
- [`_backup_existing`](#definition-22)
- [`_promote_staged`](#definition-23)
- [`_rollback`](#definition-24)
- [`_handle_failure`](#definition-25)
- [`_recovery_directory`](#definition-26)
- [`_record_failure`](#definition-27)
- [`_remove_transaction`](#definition-28)

<a id="definition-1"></a>

## `is_control_directory`

[Source, line 20](../../src/boldtailor/publication.py#L20)

```python
is_control_directory(path: str | os.PathLike[str]) -> bool
```

True for the ``<destination>.boldtailor`` sibling holding lock and staging.

Usage in package/tests:

- [tests/test_publication.py:750](../../tests/test_publication.py#L750)
- [tests/test_publication.py:751](../../tests/test_publication.py#L751)

<a id="definition-2"></a>

## `PublicationError`

[Source, line 25](../../src/boldtailor/publication.py#L25)

```python
class PublicationError(RuntimeError)
```

Publication failed; inspect the cause and any retained rollback backups.

Usage in package/tests:

- [src/boldtailor/publication.py:296](../../src/boldtailor/publication.py#L296)
- [src/boldtailor/publication.py:104](../../src/boldtailor/publication.py#L104)

<a id="definition-3"></a>

## `PublicationError.__init__`

[Source, line 28](../../src/boldtailor/publication.py#L28)

```python
PublicationError.__init__(self, message: str, *, recovery_directory: Path | None=None, rollback_errors: Sequence[Exception]=()) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `Artifact`

[Source, line 41](../../src/boldtailor/publication.py#L41)

```python
@dataclass(frozen=True, slots=True)
class Artifact
```

One immutable in-memory artifact addressed relative to a destination.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
path: str
payload: bytes
```

Usage in package/tests:

- [src/boldtailor/cifti.py:26](../../src/boldtailor/cifti.py#L26)
- [src/boldtailor/workflow/artifacts.py:16](../../src/boldtailor/workflow/artifacts.py#L16)
- [src/boldtailor/workflow/artifacts.py:23](../../src/boldtailor/workflow/artifacts.py#L23)
- [src/boldtailor/workflow/artifacts.py:30](../../src/boldtailor/workflow/artifacts.py#L30)

<a id="definition-5"></a>

## `Artifact.__post_init__`

[Source, line 47](../../src/boldtailor/publication.py#L47)

```python
Artifact.__post_init__(self) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `_Transaction`

[Source, line 53](../../src/boldtailor/publication.py#L53)

```python
@dataclass(slots=True)
class _Transaction
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
control: Path
identifier: str = field(default_factory=lambda: str(uuid4()))
created: list[Path] = field(default_factory=list)
backed_up: list[str] = field(default_factory=list)
promoted: list[str] = field(default_factory=list)
```

Usage in package/tests:

- [src/boldtailor/publication.py:126](../../src/boldtailor/publication.py#L126)

<a id="definition-7"></a>

## `_Transaction.stage`

[Source, line 61](../../src/boldtailor/publication.py#L61)

```python
@property
_Transaction.stage(self) -> Path
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `_Transaction.backups`

[Source, line 65](../../src/boldtailor/publication.py#L65)

```python
@property
_Transaction.backups(self) -> Path
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `publish_artifact_set`

[Source, line 69](../../src/boldtailor/publication.py#L69)

```python
publish_artifact_set(destination: str | os.PathLike[str], artifacts: Iterable[Artifact], *, source_paths: Iterable[str | os.PathLike[str]]=(), overwrite: bool=False, lock_timeout: float=30.0, keep_existing: Iterable[str]=()) -> tuple[Path, ...]
```

Publish a file set under one writer lock, rolling back on failure.

Paths in ``keep_existing`` (shared files such as a dataset description)
are written only if absent once the lock is held; an existing file there
is left untouched and omitted from the returned paths.

Lock, staging, backups, and the failure ledger live in the sibling
``<destination>.boldtailor/`` directory, not in the dataset. Concurrent
readers may observe a partially replaced set.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1128](../../src/boldtailor/workflow/outputs.py#L1128)
- [tests/test_publication.py:161](../../tests/test_publication.py#L161)
- [tests/test_publication.py:172](../../tests/test_publication.py#L172)
- [tests/test_publication.py:194](../../tests/test_publication.py#L194)

<a id="definition-10"></a>

## `_unless_kept`

[Source, line 107](../../src/boldtailor/publication.py#L107)

```python
_unless_kept(destination: Path, artifacts: tuple[Artifact, ...], keep: frozenset[str]) -> tuple[Artifact, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:99](../../src/boldtailor/publication.py#L99)

<a id="definition-11"></a>

## `_publish_locked`

[Source, line 117](../../src/boldtailor/publication.py#L117)

```python
_publish_locked(destination: Path, control: Path, artifacts: tuple[Artifact, ...], *, overwrite: bool) -> tuple[Path, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:100](../../src/boldtailor/publication.py#L100)

<a id="definition-12"></a>

## `_prepare_artifacts`

[Source, line 138](../../src/boldtailor/publication.py#L138)

```python
_prepare_artifacts(artifacts: Iterable[Artifact]) -> tuple[Artifact, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:88](../../src/boldtailor/publication.py#L88)

<a id="definition-13"></a>

## `_validate_artifact_path`

[Source, line 148](../../src/boldtailor/publication.py#L148)

```python
_validate_artifact_path(path: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:48](../../src/boldtailor/publication.py#L48)

<a id="definition-14"></a>

## `_own_bytes`

[Source, line 158](../../src/boldtailor/publication.py#L158)

```python
_own_bytes(payload: object) -> bytes
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:49](../../src/boldtailor/publication.py#L49)

<a id="definition-15"></a>

## `_reject_duplicate_paths`

[Source, line 164](../../src/boldtailor/publication.py#L164)

```python
_reject_duplicate_paths(artifacts: Sequence[Artifact]) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:144](../../src/boldtailor/publication.py#L144)

<a id="definition-16"></a>

## `_validate_lock_timeout`

[Source, line 176](../../src/boldtailor/publication.py#L176)

```python
_validate_lock_timeout(timeout: float) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:90](../../src/boldtailor/publication.py#L90)

<a id="definition-17"></a>

## `_resolve_destination`

[Source, line 186](../../src/boldtailor/publication.py#L186)

```python
_resolve_destination(destination: str | os.PathLike[str]) -> Path
```

Resolve the destination (and links such as macOS /tmp) before any checks.

Usage in package/tests:

- [src/boldtailor/publication.py:91](../../src/boldtailor/publication.py#L91)

<a id="definition-18"></a>

## `_reject_source_overlap`

[Source, line 194](../../src/boldtailor/publication.py#L194)

```python
_reject_source_overlap(targets: Sequence[Path], sources: Sequence[Path]) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:95](../../src/boldtailor/publication.py#L95)

<a id="definition-19"></a>

## `_check_target`

[Source, line 202](../../src/boldtailor/publication.py#L202)

```python
_check_target(destination: Path, path: str, *, overwrite: bool) -> None
```

Refuse symlinks inside the destination and unrequested overwrites.

Usage in package/tests:

- [src/boldtailor/publication.py:125](../../src/boldtailor/publication.py#L125)

<a id="definition-20"></a>

## `_make_transaction_directory`

[Source, line 216](../../src/boldtailor/publication.py#L216)

```python
_make_transaction_directory(transaction: _Transaction, path: Path) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:222](../../src/boldtailor/publication.py#L222)
- [src/boldtailor/publication.py:242](../../src/boldtailor/publication.py#L242)

<a id="definition-21"></a>

## `_stage_artifacts`

[Source, line 221](../../src/boldtailor/publication.py#L221)

```python
_stage_artifacts(transaction: _Transaction, artifacts: Sequence[Artifact]) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:128](../../src/boldtailor/publication.py#L128)

<a id="definition-22"></a>

## `_backup_existing`

[Source, line 232](../../src/boldtailor/publication.py#L232)

```python
_backup_existing(destination: Path, transaction: _Transaction, artifacts: Sequence[Artifact]) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:130](../../src/boldtailor/publication.py#L130)

<a id="definition-23"></a>

## `_promote_staged`

[Source, line 249](../../src/boldtailor/publication.py#L249)

```python
_promote_staged(destination: Path, transaction: _Transaction, artifacts: Sequence[Artifact]) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:131](../../src/boldtailor/publication.py#L131)

<a id="definition-24"></a>

## `_rollback`

[Source, line 261](../../src/boldtailor/publication.py#L261)

```python
_rollback(destination: Path, transaction: _Transaction) -> tuple[Exception, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:282](../../src/boldtailor/publication.py#L282)

<a id="definition-25"></a>

## `_handle_failure`

[Source, line 277](../../src/boldtailor/publication.py#L277)

```python
_handle_failure(destination: Path, transaction: _Transaction, error: Exception) -> PublicationError
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:133](../../src/boldtailor/publication.py#L133)

<a id="definition-26"></a>

## `_recovery_directory`

[Source, line 301](../../src/boldtailor/publication.py#L301)

```python
_recovery_directory(transaction: _Transaction) -> Path | None
```

The retained directory holding originals, else the staged outputs.

Usage in package/tests:

- [src/boldtailor/publication.py:283](../../src/boldtailor/publication.py#L283)

<a id="definition-27"></a>

## `_record_failure`

[Source, line 309](../../src/boldtailor/publication.py#L309)

```python
_record_failure(control: Path, identifier: str, error: Exception, *, rollback_errors: Sequence[Exception]=(), recovery_directory: str | None=None) -> None
```

Append one fixed-field line to the best-effort failure ledger.

Usage in package/tests:

- [src/boldtailor/publication.py:284](../../src/boldtailor/publication.py#L284)
- [src/boldtailor/publication.py:102](../../src/boldtailor/publication.py#L102)

<a id="definition-28"></a>

## `_remove_transaction`

[Source, line 337](../../src/boldtailor/publication.py#L337)

```python
_remove_transaction(transaction: _Transaction) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/publication.py:134](../../src/boldtailor/publication.py#L134)
- [src/boldtailor/publication.py:293](../../src/boldtailor/publication.py#L293)
