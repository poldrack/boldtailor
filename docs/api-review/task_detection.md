# `boldtailor.task_detection`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/task_detection.py)

Detect a task model from event tables: response time and categorical trial type.

## Definitions

- [`_run_labels`](#definition-1)
- [`_table_levels`](#definition-2)
- [`observed_levels`](#definition-3)
- [`_everywhere`](#definition-4)
- [`_automatic`](#definition-5)
- [`_resolved`](#definition-6)
- [`_check_columns`](#definition-7)
- [`detect_task_model`](#definition-8)
- [`task_model_notes`](#definition-9)

<a id="definition-1"></a>

## `_run_labels`

[Source, line 8](../../src/boldtailor/task_detection.py#L8)

```python
_run_labels(tables, labels)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/task_detection.py:69](../../src/boldtailor/task_detection.py#L69)
- [src/boldtailor/task_detection.py:22](../../src/boldtailor/task_detection.py#L22)

<a id="definition-2"></a>

## `_table_levels`

[Source, line 12](../../src/boldtailor/task_detection.py#L12)

```python
_table_levels(table, column, label)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/task_detection.py:23](../../src/boldtailor/task_detection.py#L23)

<a id="definition-3"></a>

## `observed_levels`

[Source, line 19](../../src/boldtailor/task_detection.py#L19)

```python
observed_levels(tables, column, labels=None)
```

Canonical non-missing level names of ``column`` across every table.

Usage in package/tests:

- [src/boldtailor/task_detection.py:37](../../src/boldtailor/task_detection.py#L37)
- [src/boldtailor/task_detection.py:49](../../src/boldtailor/task_detection.py#L49)
- [src/boldtailor/task_detection.py:80](../../src/boldtailor/task_detection.py#L80)

<a id="definition-4"></a>

## `_everywhere`

[Source, line 28](../../src/boldtailor/task_detection.py#L28)

```python
_everywhere(column, tables)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/task_detection.py:34](../../src/boldtailor/task_detection.py#L34)
- [src/boldtailor/task_detection.py:36](../../src/boldtailor/task_detection.py#L36)
- [src/boldtailor/task_detection.py:79](../../src/boldtailor/task_detection.py#L79)

<a id="definition-5"></a>

## `_automatic`

[Source, line 32](../../src/boldtailor/task_detection.py#L32)

```python
_automatic(tables, labels)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/task_detection.py:71](../../src/boldtailor/task_detection.py#L71)

<a id="definition-6"></a>

## `_resolved`

[Source, line 45](../../src/boldtailor/task_detection.py#L45)

```python
_resolved(modulator, tables, labels)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/task_detection.py:73](../../src/boldtailor/task_detection.py#L73)

<a id="definition-7"></a>

## `_check_columns`

[Source, line 54](../../src/boldtailor/task_detection.py#L54)

```python
_check_columns(tables, labels, modulators)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/task_detection.py:72](../../src/boldtailor/task_detection.py#L72)

<a id="definition-8"></a>

## `detect_task_model`

[Source, line 61](../../src/boldtailor/task_detection.py#L61)

```python
detect_task_model(events_tables, modulators=None, labels=None)
```

The task model from event columns, or explicit modulators checked per run.

Detection takes ``response_time`` and a categorical ``trial_type`` with at
least two levels; unresolved categorical modulators take their levels from
every run. ``labels`` name the runs in errors.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:152](../../src/boldtailor/workflow/inputs.py#L152)
- [src/boldtailor/workflow/run.py:98](../../src/boldtailor/workflow/run.py#L98)
- [tests/test_task_detection.py:32](../../tests/test_task_detection.py#L32)
- [tests/test_task_detection.py:35](../../tests/test_task_detection.py#L35)

<a id="definition-9"></a>

## `task_model_notes`

[Source, line 76](../../src/boldtailor/task_detection.py#L76)

```python
task_model_notes(events_tables, modulators=None)
```

Why automatic detection left out a column that every run has.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:101](../../src/boldtailor/workflow/run.py#L101)
- [tests/test_task_detection.py:51](../../tests/test_task_detection.py#L51)
- [tests/test_task_detection.py:92](../../tests/test_task_detection.py#L92)
- [tests/test_task_detection.py:97](../../tests/test_task_detection.py#L97)
