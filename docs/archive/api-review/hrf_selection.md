# `boldtailor.hrf_selection`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/hrf_selection.py)

Select HRFs by held-out task-model prediction across runs, without repeated images.

## Definitions

- [`_validate`](#definition-1)
- [`_activity`](#definition-2)
- [`_provenance`](#definition-3)
- [`_select`](#definition-4)
- [`select_hrfs`](#definition-5)
- [`select_hrf`](#definition-6)
- [`_select_hrfs`](#definition-7)
- [`_fold_indices`](#definition-8)
- [`_predict`](#definition-9)
- [`evaluate_hrf_split`](#definition-10)
- [`_train_selection`](#definition-11)
- [`_evaluate_split`](#definition-12)
- [`_held_out_scores`](#definition-13)

<a id="definition-1"></a>

## `_validate`

[Source, line 28](../../src/boldtailor/hrf_selection.py#L28)

```python
_validate(library, signature, batch, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:165](../../src/boldtailor/hrf_selection.py#L165)
- [src/boldtailor/hrf_selection.py:258](../../src/boldtailor/hrf_selection.py#L258)

<a id="definition-2"></a>

## `_activity`

[Source, line 39](../../src/boldtailor/hrf_selection.py#L39)

```python
_activity(runs, library, labels, signature, task_model, name, **extra)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:96](../../src/boldtailor/hrf_selection.py#L96)
- [src/boldtailor/hrf_selection.py:269](../../src/boldtailor/hrf_selection.py#L269)

<a id="definition-3"></a>

## `_provenance`

[Source, line 69](../../src/boldtailor/hrf_selection.py#L69)

```python
_provenance(operation, data, activity)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:112](../../src/boldtailor/hrf_selection.py#L112)
- [src/boldtailor/hrf_selection.py:281](../../src/boldtailor/hrf_selection.py#L281)

<a id="definition-4"></a>

## `_select`

[Source, line 74](../../src/boldtailor/hrf_selection.py#L74)

```python
_select(operation, data, runs, signals, library, labels, signature, batch, eligibility_runs, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:170](../../src/boldtailor/hrf_selection.py#L170)
- [src/boldtailor/hrf_selection.py:241](../../src/boldtailor/hrf_selection.py#L241)

<a id="definition-5"></a>

## `select_hrfs`

[Source, line 127](../../src/boldtailor/hrf_selection.py#L127)

```python
select_hrfs(data: AnalysisData, *, library: HrfLibrary | None=None, run_labels: Sequence[str] | None=None, feature_signature: str | None=None, candidate_batch_size: int=32, task_model: TaskModel=TaskModel()) -> HrfSelectionResult
```

Choose each feature's HRF by leave-one-run-out task-model prediction.

This is a selection statistic. Confounds and missing-value indicators are
profiled in each run; task-model amplitudes are learned only from other
runs. Anonymous arrays without feature_signature require the caller to
preserve feature order.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:99](../../src/boldtailor/_ridge_cv.py#L99)
- [src/boldtailor/denoising.py:148](../../src/boldtailor/denoising.py#L148)
- [src/boldtailor/hrf_selection.py:159](../../src/boldtailor/hrf_selection.py#L159)
- [src/boldtailor/workflow/beta_series.py:261](../../src/boldtailor/workflow/beta_series.py#L261)

<a id="definition-6"></a>

## `select_hrf`

[Source, line 156](../../src/boldtailor/hrf_selection.py#L156)

```python
select_hrf(data: AnalysisData, **options) -> HrfSelectionResult
```

Deprecated alias of :func:`select_hrfs`, kept for one release.

<a id="definition-7"></a>

## `_select_hrfs`

[Source, line 162](../../src/boldtailor/hrf_selection.py#L162)

```python
_select_hrfs(operation, data, library, run_labels, feature_signature, batch, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:145](../../src/boldtailor/hrf_selection.py#L145)

<a id="definition-8"></a>

## `_fold_indices`

[Source, line 184](../../src/boldtailor/hrf_selection.py#L184)

```python
_fold_indices(values, n_runs, minimum, name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:259](../../src/boldtailor/hrf_selection.py#L259)
- [src/boldtailor/hrf_selection.py:260](../../src/boldtailor/hrf_selection.py#L260)

<a id="definition-9"></a>

## `_predict`

[Source, line 195](../../src/boldtailor/hrf_selection.py#L195)

```python
_predict(a, b, c, energy, train, test)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:299](../../src/boldtailor/hrf_selection.py#L299)

<a id="definition-10"></a>

## `evaluate_hrf_split`

[Source, line 208](../../src/boldtailor/hrf_selection.py#L208)

```python
evaluate_hrf_split(data: AnalysisData, *, library: HrfLibrary | None=None, train_runs: Sequence[int], test_runs: Sequence[int], run_labels: Sequence[str] | None=None, feature_signature: str | None=None, task_model: TaskModel=TaskModel(), candidate_batch_size: int=32) -> HrfEvaluationResult
```

Select within training runs, then freeze HRF and amplitudes for test runs.

``candidate_batch_size`` bounds memory exactly as in :func:`select_hrfs`;
it never changes the result.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:30](../../src/boldtailor/workflow/analysis.py#L30)
- [src/boldtailor/workflow/analysis.py:31](../../src/boldtailor/workflow/analysis.py#L31)
- [tests/test_lifecycle.py:355](../../tests/test_lifecycle.py#L355)
- [tests/test_lifecycle.py:167](../../tests/test_lifecycle.py#L167)

<a id="definition-11"></a>

## `_train_selection`

[Source, line 238](../../src/boldtailor/hrf_selection.py#L238)

```python
_train_selection(data, runs, library, labels, signature, task_model, split, batch)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:265](../../src/boldtailor/hrf_selection.py#L265)

<a id="definition-12"></a>

## `_evaluate_split`

[Source, line 255](../../src/boldtailor/hrf_selection.py#L255)

```python
_evaluate_split(operation, data, library, split, run_labels, feature_signature, task_model, batch)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:226](../../src/boldtailor/hrf_selection.py#L226)

<a id="definition-13"></a>

## `_held_out_scores`

[Source, line 296](../../src/boldtailor/hrf_selection.py#L296)

```python
_held_out_scores(data, runs, selection, task_model, split, batch)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:268](../../src/boldtailor/hrf_selection.py#L268)
