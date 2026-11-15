# `boldtailor._single_trial_design`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_single_trial_design.py)

Compile event-wise task columns without behavioral or condition pooling.

## Definitions

- [`compile_trial_run`](#definition-1)
- [`run_trial_table`](#definition-2)
- [`trial_table`](#definition-3)
- [`check_trial_ids`](#definition-4)
- [`_validate_events`](#definition-5)
- [`_nuisance_matrix`](#definition-6)

<a id="definition-1"></a>

## `compile_trial_run`

[Source, line 18](../../src/boldtailor/_single_trial_design.py#L18)

```python
compile_trial_run(events, frame_times, confounds, run_label, *, hrf='spm')
```

Return task and nuisance matrices plus a row-aligned trial table.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:115](../../src/boldtailor/_hrf_cv.py#L115)
- [src/boldtailor/_ridge_cv.py:87](../../src/boldtailor/_ridge_cv.py#L87)
- [src/boldtailor/single_trial.py:83](../../src/boldtailor/single_trial.py#L83)
- [tests/conftest.py:89](../../tests/conftest.py#L89)

<a id="definition-2"></a>

## `run_trial_table`

[Source, line 30](../../src/boldtailor/_single_trial_design.py#L30)

```python
run_trial_table(events, run_label)
```

One run's events led by trial_id, run_label, and event_index.

Usage in package/tests:

- [src/boldtailor/_single_trial_design.py:24](../../src/boldtailor/_single_trial_design.py#L24)
- [src/boldtailor/_single_trial_design.py:43](../../src/boldtailor/_single_trial_design.py#L43)

<a id="definition-3"></a>

## `trial_table`

[Source, line 40](../../src/boldtailor/_single_trial_design.py#L40)

```python
trial_table(events_per_run, labels)
```

All runs' trials in run order, with global trial_index and run_index.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:34](../../src/boldtailor/_selected_hrf_fit.py#L34)
- [src/boldtailor/single_trial.py:77](../../src/boldtailor/single_trial.py#L77)

<a id="definition-4"></a>

## `check_trial_ids`

[Source, line 51](../../src/boldtailor/_single_trial_design.py#L51)

```python
check_trial_ids(trial_ids, nuisance_columns)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_selected_hrf_fit.py:36](../../src/boldtailor/_selected_hrf_fit.py#L36)
- [src/boldtailor/_single_trial_design.py:25](../../src/boldtailor/_single_trial_design.py#L25)

<a id="definition-5"></a>

## `_validate_events`

[Source, line 56](../../src/boldtailor/_single_trial_design.py#L56)

```python
_validate_events(events, times, run_label)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:171](../../src/boldtailor/_hrf_cv.py#L171)
- [src/boldtailor/_hrf_cv.py:180](../../src/boldtailor/_hrf_cv.py#L180)
- [src/boldtailor/_selected_hrf_fit.py:33](../../src/boldtailor/_selected_hrf_fit.py#L33)
- [src/boldtailor/_single_trial_design.py:22](../../src/boldtailor/_single_trial_design.py#L22)

<a id="definition-6"></a>

## `_nuisance_matrix`

[Source, line 76](../../src/boldtailor/_single_trial_design.py#L76)

```python
_nuisance_matrix(confounds, n_scans)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:257](../../src/boldtailor/_denoising_pool.py#L257)
- [src/boldtailor/_hrf_cv.py:174](../../src/boldtailor/_hrf_cv.py#L174)
- [src/boldtailor/_hrf_cv.py:182](../../src/boldtailor/_hrf_cv.py#L182)
- [src/boldtailor/_single_trial_design.py:23](../../src/boldtailor/_single_trial_design.py#L23)
