# `boldtailor.logging`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/logging.py)

Bind fit context and collect structured lifecycle events.

## Definitions

- [`bind_context`](#definition-1)
- [`emit_event`](#definition-2)
- [`_emit_record`](#definition-3)
- [`append_event_history`](#definition-4)
- [`_context_updates`](#definition-5)
- [`_make_event`](#definition-6)
- [`_history_entry`](#definition-7)
- [`_timestamp`](#definition-8)
- [`_error_code`](#definition-9)

<a id="definition-1"></a>

## `bind_context`

[Source, line 24](../../src/boldtailor/logging.py#L24)

```python
@contextmanager
bind_context(*, execution_id: str | None=None, data_id: str | None=None, analysis_id: str | None=None, run_index: int | None=None, inherit: bool=True) -> Iterator[None]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fit_lifecycle.py:58](../../src/boldtailor/_fit_lifecycle.py#L58)
- [src/boldtailor/data.py:87](../../src/boldtailor/data.py#L87)
- [src/boldtailor/prepared.py:148](../../src/boldtailor/prepared.py#L148)
- [tests/test_logging.py:20](../../tests/test_logging.py#L20)

<a id="definition-2"></a>

## `emit_event`

[Source, line 46](../../src/boldtailor/logging.py#L46)

```python
emit_event(event: str, *, stage: str, level: int=logging.INFO, error: Exception | str | None=None, execution_id: str | None=None, data_id: str | None=None, analysis_id: str | None=None, run_index: int | None=None) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fit_lifecycle.py:64](../../src/boldtailor/_fit_lifecycle.py#L64)
- [src/boldtailor/_fit_lifecycle.py:71](../../src/boldtailor/_fit_lifecycle.py#L71)
- [src/boldtailor/data.py:90](../../src/boldtailor/data.py#L90)
- [src/boldtailor/data.py:139](../../src/boldtailor/data.py#L139)

<a id="definition-3"></a>

## `_emit_record`

[Source, line 71](../../src/boldtailor/logging.py#L71)

```python
_emit_record(payload: Mapping[str, object], *, level: int=logging.INFO) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fit_lifecycle.py:80](../../src/boldtailor/_fit_lifecycle.py#L80)
- [src/boldtailor/data.py:147](../../src/boldtailor/data.py#L147)
- [src/boldtailor/logging.py:67](../../src/boldtailor/logging.py#L67)
- [src/boldtailor/prepared.py:220](../../src/boldtailor/prepared.py#L220)

<a id="definition-4"></a>

## `append_event_history`

[Source, line 80](../../src/boldtailor/logging.py#L80)

```python
append_event_history(history: Sequence[Mapping[str, object]], event: Mapping[str, object]) -> tuple[Mapping[str, object], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fit_lifecycle.py:63](../../src/boldtailor/_fit_lifecycle.py#L63)
- [src/boldtailor/_fit_lifecycle.py:47](../../src/boldtailor/_fit_lifecycle.py#L47)
- [src/boldtailor/data.py:88](../../src/boldtailor/data.py#L88)
- [src/boldtailor/data.py:128](../../src/boldtailor/data.py#L128)

<a id="definition-5"></a>

## `_context_updates`

[Source, line 90](../../src/boldtailor/logging.py#L90)

```python
_context_updates(*, execution_id: str | None, data_id: str | None, analysis_id: str | None, run_index: int | None) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/logging.py:32](../../src/boldtailor/logging.py#L32)
- [src/boldtailor/logging.py:129](../../src/boldtailor/logging.py#L129)

<a id="definition-6"></a>

## `_make_event`

[Source, line 109](../../src/boldtailor/logging.py#L109)

```python
_make_event(event: str, *, stage: str, level: int=logging.INFO, error: Exception | str | None=None, execution_id: str | None=None, data_id: str | None=None, analysis_id: str | None=None, run_index: int | None=None) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fit_lifecycle.py:40](../../src/boldtailor/_fit_lifecycle.py#L40)
- [src/boldtailor/data.py:121](../../src/boldtailor/data.py#L121)
- [src/boldtailor/logging.py:57](../../src/boldtailor/logging.py#L57)
- [src/boldtailor/prepared.py:191](../../src/boldtailor/prepared.py#L191)

<a id="definition-7"></a>

## `_history_entry`

[Source, line 141](../../src/boldtailor/logging.py#L141)

```python
_history_entry(event: Mapping[str, object]) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/logging.py:84](../../src/boldtailor/logging.py#L84)

<a id="definition-8"></a>

## `_timestamp`

[Source, line 157](../../src/boldtailor/logging.py#L157)

```python
_timestamp() -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/logging.py:121](../../src/boldtailor/logging.py#L121)

<a id="definition-9"></a>

## `_error_code`

[Source, line 168](../../src/boldtailor/logging.py#L168)

```python
_error_code(error: Exception | str) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/logging.py:137](../../src/boldtailor/logging.py#L137)
- [src/boldtailor/publication.py:320](../../src/boldtailor/publication.py#L320)
