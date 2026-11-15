# `boldtailor.workflow.summaries`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/summaries.py)

Small tables for the HTML report, built from results already in memory.

## Definitions

- [`median`](#definition-1)
- [`activation_summary`](#definition-2)
- [`rt_summary`](#definition-3)
- [`library_head`](#definition-4)
- [`selected_hrfs`](#definition-5)
- [`encoding_scores`](#definition-6)
- [`ridge_boundary`](#definition-7)
- [`input_tables`](#definition-8)

<a id="definition-1"></a>

## `median`

[Source, line 14](../../src/boldtailor/workflow/summaries.py#L14)

```python
median(values)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/summaries.py:22](../../src/boldtailor/workflow/summaries.py#L22)
- [src/boldtailor/workflow/summaries.py:30](../../src/boldtailor/workflow/summaries.py#L30)
- [src/boldtailor/workflow/summaries.py:69](../../src/boldtailor/workflow/summaries.py#L69)

<a id="definition-2"></a>

## `activation_summary`

[Source, line 19](../../src/boldtailor/workflow/summaries.py#L19)

```python
activation_summary(activation)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `rt_summary`

[Source, line 26](../../src/boldtailor/workflow/summaries.py#L26)

```python
rt_summary(activation, beta_models)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `library_head`

[Source, line 38](../../src/boldtailor/workflow/summaries.py#L38)

```python
library_head(library, rows=10)
```

The first rows of the HRF library parameter table.

Usage in package/tests:

- [src/boldtailor/workflow/summaries.py:91](../../src/boldtailor/workflow/summaries.py#L91)

<a id="definition-5"></a>

## `selected_hrfs`

[Source, line 43](../../src/boldtailor/workflow/summaries.py#L43)

```python
selected_hrfs(selections, library, n_features, top=10)
```

The most often selected HRFs (all runs) with their peak times.

<a id="definition-6"></a>

## `encoding_scores`

[Source, line 63](../../src/boldtailor/workflow/summaries.py#L63)

```python
encoding_scores(beta_models)
```

Median outer (held-out half) encoding R² per tuned model and split.

<a id="definition-7"></a>

## `ridge_boundary`

[Source, line 78](../../src/boldtailor/workflow/summaries.py#L78)

```python
ridge_boundary(beta_models)
```

Share of scored grayordinates whose ridge winner is a grid endpoint.

<a id="definition-8"></a>

## `input_tables`

[Source, line 86](../../src/boldtailor/workflow/summaries.py#L86)

```python
input_tables(runs, task_model, library)
```

Run summary, confound names and library head for the inputs section.
