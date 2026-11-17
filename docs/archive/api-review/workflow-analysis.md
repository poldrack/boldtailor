# `boldtailor.workflow.analysis`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/analysis.py)

Blockwise public-API calls: HRF selection, GLMs, and beta series.

## Definitions

- [`_signature`](#definition-1)
- [`_split_selections`](#definition-2)
- [`_select_block`](#definition-3)
- [`select_hrfs`](#definition-4)
- [`selection_maps`](#definition-5)
- [`_glm_block`](#definition-6)
- [`_jobs`](#definition-7)
- [`fit_glms`](#definition-8)
- [`_update_result_metadata`](#definition-9)
- [`_fitted_pairs`](#definition-10)
- [`_trial_designs`](#definition-11)
- [`_beta_block`](#definition-12)
- [`fit_beta_series`](#definition-13)
- [`_rt_correlations`](#definition-14)

<a id="definition-1"></a>

## `_signature`

[Source, line 23](../../src/boldtailor/workflow/analysis.py#L23)

```python
_signature(runs, indices)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:40](../../src/boldtailor/workflow/analysis.py#L40)
- [src/boldtailor/workflow/analysis.py:96](../../src/boldtailor/workflow/analysis.py#L96)
- [src/boldtailor/workflow/analysis.py:190](../../src/boldtailor/workflow/analysis.py#L190)

<a id="definition-2"></a>

## `_split_selections`

[Source, line 27](../../src/boldtailor/workflow/analysis.py#L27)

```python
_split_selections(data, runs, options)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:45](../../src/boldtailor/workflow/analysis.py#L45)

<a id="definition-3"></a>

## `_select_block`

[Source, line 35](../../src/boldtailor/workflow/analysis.py#L35)

```python
_select_block(indices, runs, root, library, task_model, splits)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `select_hrfs`

[Source, line 49](../../src/boldtailor/workflow/analysis.py#L49)

```python
select_hrfs(runs, root, blocks, library, *, task_model, n_jobs=1, splits=True)
```

Select on all runs and, with ``splits``, on each half; retain the objects.

The GLM keeps the full task model; pass a subset (for example
selection_task_model(task_model, False)) to score HRFs without RT. Without
``splits`` each bundle holds only ``all``, which needs just two runs.

<a id="definition-5"></a>

## `selection_maps`

[Source, line 68](../../src/boldtailor/workflow/analysis.py#L68)

```python
selection_maps(selections, n_features)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:233](../../src/boldtailor/workflow/outputs.py#L233)

<a id="definition-6"></a>

## `_glm_block`

[Source, line 89](../../src/boldtailor/workflow/analysis.py#L89)

```python
_glm_block(job, runs, root, model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `_jobs`

[Source, line 122](../../src/boldtailor/workflow/analysis.py#L122)

```python
_jobs(blocks, selections)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:139](../../src/boldtailor/workflow/analysis.py#L139)
- [src/boldtailor/workflow/analysis.py:239](../../src/boldtailor/workflow/analysis.py#L239)

<a id="definition-8"></a>

## `fit_glms`

[Source, line 129](../../src/boldtailor/workflow/analysis.py#L129)

```python
fit_glms(runs, root, blocks, model, *, selections=None, n_jobs=1)
```

Fit one contrast per task regressor using SPM or the selected HRF per feature.

<a id="definition-9"></a>

## `_update_result_metadata`

[Source, line 148](../../src/boldtailor/workflow/analysis.py#L148)

```python
_update_result_metadata(result, block, indices)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:143](../../src/boldtailor/workflow/analysis.py#L143)
- [src/boldtailor/workflow/analysis.py:251](../../src/boldtailor/workflow/analysis.py#L251)

<a id="definition-10"></a>

## `_fitted_pairs`

[Source, line 158](../../src/boldtailor/workflow/analysis.py#L158)

```python
_fitted_pairs(hrf_indices, n_runs)
```

Every (run, HRF) pair a selected fit used, run-major, HRF ascending.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:168](../../src/boldtailor/workflow/analysis.py#L168)
- [src/boldtailor/workflow/analysis.py:108](../../src/boldtailor/workflow/analysis.py#L108)

<a id="definition-11"></a>

## `_trial_designs`

[Source, line 164](../../src/boldtailor/workflow/analysis.py#L164)

```python
_trial_designs(result, runs, selection)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:197](../../src/boldtailor/workflow/analysis.py#L197)

<a id="definition-12"></a>

## `_beta_block`

[Source, line 178](../../src/boldtailor/workflow/analysis.py#L178)

```python
_beta_block(job, runs, root, alpha, task_model, fractions=None)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-13"></a>

## `fit_beta_series`

[Source, line 203](../../src/boldtailor/workflow/analysis.py#L203)

```python
fit_beta_series(runs, root, blocks, *, task_model, selections=None, ridge_alpha=0.0, ridge_fraction=None, n_jobs=1)
```

Fit raw trials at the supplied penalty; tuning is a separate operation.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:439](../../src/boldtailor/workflow/beta_series.py#L439)
- [src/boldtailor/workflow/beta_series.py:704](../../src/boldtailor/workflow/beta_series.py#L704)

<a id="definition-14"></a>

## `_rt_correlations`

[Source, line 257](../../src/boldtailor/workflow/analysis.py#L257)

```python
_rt_correlations(result, runs, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:252](../../src/boldtailor/workflow/analysis.py#L252)
