# `boldtailor.workflow.artifacts`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/artifacts.py)

In-memory JSON, TSV, NPZ, PNG, and CIFTI artifacts for workflow outputs.

## Definitions

- [`json_artifact`](#definition-1)
- [`table_artifact`](#definition-2)
- [`npz_artifact`](#definition-3)
- [`figure_artifact`](#definition-4)
- [`dataset_description`](#definition-5)
- [`parameter_artifact`](#definition-6)
- [`scalar_map`](#definition-7)

<a id="definition-1"></a>

## `json_artifact`

[Source, line 14](../../src/boldtailor/workflow/artifacts.py#L14)

```python
json_artifact(path, value)
```

Indented, strict (no NaN) JSON with a trailing newline.

Usage in package/tests:

- [src/boldtailor/workflow/artifacts.py:42](../../src/boldtailor/workflow/artifacts.py#L42)
- [src/boldtailor/workflow/outputs.py:136](../../src/boldtailor/workflow/outputs.py#L136)
- [src/boldtailor/workflow/outputs.py:241](../../src/boldtailor/workflow/outputs.py#L241)
- [src/boldtailor/workflow/outputs.py:334](../../src/boldtailor/workflow/outputs.py#L334)

<a id="definition-2"></a>

## `table_artifact`

[Source, line 21](../../src/boldtailor/workflow/artifacts.py#L21)

```python
table_artifact(path, frame)
```

Tab-separated table without the index; missing values as ``n/a``.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:170](../../src/boldtailor/workflow/outputs.py#L170)
- [src/boldtailor/workflow/outputs.py:316](../../src/boldtailor/workflow/outputs.py#L316)
- [src/boldtailor/workflow/outputs.py:331](../../src/boldtailor/workflow/outputs.py#L331)
- [src/boldtailor/workflow/outputs.py:494](../../src/boldtailor/workflow/outputs.py#L494)

<a id="definition-3"></a>

## `npz_artifact`

[Source, line 26](../../src/boldtailor/workflow/artifacts.py#L26)

```python
npz_artifact(path, **arrays)
```

Compressed NPZ of named arrays (load with ``allow_pickle=False``).

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:129](../../src/boldtailor/workflow/outputs.py#L129)
- [src/boldtailor/workflow/outputs.py:618](../../src/boldtailor/workflow/outputs.py#L618)
- [src/boldtailor/workflow/outputs.py:171](../../src/boldtailor/workflow/outputs.py#L171)
- [src/boldtailor/workflow/outputs.py:493](../../src/boldtailor/workflow/outputs.py#L493)

<a id="definition-4"></a>

## `figure_artifact`

[Source, line 33](../../src/boldtailor/workflow/artifacts.py#L33)

```python
figure_artifact(path, figure)
```

PNG at 130 dpi; the figure is left intact (the report embeds it later).

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1061](../../src/boldtailor/workflow/outputs.py#L1061)

<a id="definition-5"></a>

## `dataset_description`

[Source, line 40](../../src/boldtailor/workflow/artifacts.py#L40)

```python
dataset_description(name)
```

BIDS ``dataset_description.json`` for a boldtailor derivative.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1120](../../src/boldtailor/workflow/outputs.py#L1120)

<a id="definition-6"></a>

## `parameter_artifact`

[Source, line 53](../../src/boldtailor/workflow/artifacts.py#L53)

```python
parameter_artifact(brain, library, ids, path)
```

Look up the exact selected library entry, preserving undefined features.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:181](../../src/boldtailor/workflow/outputs.py#L181)
- [src/boldtailor/workflow/outputs.py:678](../../src/boldtailor/workflow/outputs.py#L678)

<a id="definition-7"></a>

## `scalar_map`

[Source, line 64](../../src/boldtailor/workflow/artifacts.py#L64)

```python
scalar_map(stem, space_entity, brain, descriptor, statistic, values, names)
```

A ``<space>_desc-<descriptor>_stat-<statistic>`` dense scalar map.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:39](../../src/boldtailor/workflow/outputs.py#L39)
