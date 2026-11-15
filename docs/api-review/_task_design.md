# `boldtailor._task_design`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_task_design.py)

Expand raw trials into Nilearn conditions and let Nilearn build the columns.

## Definitions

- [`_numeric`](#definition-1)
- [`_numeric_amplitudes`](#definition-2)
- [`categorical_names`](#definition-3)
- [`_categorical_amplitudes`](#definition-4)
- [`_modulator_amplitudes`](#definition-5)
- [`expand_events`](#definition-6)
- [`task_columns`](#definition-7)
- [`run_task_columns`](#definition-8)

<a id="definition-1"></a>

## `_numeric`

[Source, line 19](../../src/boldtailor/_task_design.py#L19)

```python
_numeric(events, column, run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_task_design.py:29](../../src/boldtailor/_task_design.py#L29)

<a id="definition-2"></a>

## `_numeric_amplitudes`

[Source, line 28](../../src/boldtailor/_task_design.py#L28)

```python
_numeric_amplitudes(events, modulator, run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_task_design.py:86](../../src/boldtailor/_task_design.py#L86)

<a id="definition-3"></a>

## `categorical_names`

[Source, line 45](../../src/boldtailor/_task_design.py#L45)

```python
categorical_names(values, modulator)
```

Canonical level per trial (None when missing), checked against the declared levels.

Usage in package/tests:

- [src/boldtailor/_task_design.py:69](../../src/boldtailor/_task_design.py#L69)
- [src/boldtailor/trial_encoding.py:255](../../src/boldtailor/trial_encoding.py#L255)
- [src/boldtailor/workflow/inputs.py:82](../../src/boldtailor/workflow/inputs.py#L82)
- [tests/test_task_design.py:295](../../tests/test_task_design.py#L295)

<a id="definition-4"></a>

## `_categorical_amplitudes`

[Source, line 63](../../src/boldtailor/_task_design.py#L63)

```python
_categorical_amplitudes(events, modulator, run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_task_design.py:85](../../src/boldtailor/_task_design.py#L85)

<a id="definition-5"></a>

## `_modulator_amplitudes`

[Source, line 83](../../src/boldtailor/_task_design.py#L83)

```python
_modulator_amplitudes(events, modulator, run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_task_design.py:98](../../src/boldtailor/_task_design.py#L98)

<a id="definition-6"></a>

## `expand_events`

[Source, line 89](../../src/boldtailor/_task_design.py#L89)

```python
expand_events(events, task_model, run=0)
```

One Nilearn condition per task-model regressor; amplitudes become modulation.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:173](../../src/boldtailor/_hrf_cv.py#L173)
- [src/boldtailor/_hrf_cv.py:185](../../src/boldtailor/_hrf_cv.py#L185)
- [src/boldtailor/_task_design.py:149](../../src/boldtailor/_task_design.py#L149)
- [tests/test_task_design.py:38](../../tests/test_task_design.py#L38)

<a id="definition-7"></a>

## `task_columns`

[Source, line 117](../../src/boldtailor/_task_design.py#L117)

```python
task_columns(expanded, frame_times, hrf, *, min_onset=MIN_ONSET, oversampling=OVERSAMPLING)
```

Nilearn task columns for one HRF: no drift, no constant, semantic names.

Plain 'spm' and 'glover' become their peak-one kernels before Nilearn, and
kernel amplitudes are scaled so each event's response peaks at one.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:72](../../src/boldtailor/_hrf_cv.py#L72)
- [src/boldtailor/_task_design.py:150](../../src/boldtailor/_task_design.py#L150)
- [tests/test_task_design.py:135](../../tests/test_task_design.py#L135)
- [tests/test_task_design.py:161](../../tests/test_task_design.py#L161)

<a id="definition-8"></a>

## `run_task_columns`

[Source, line 144](../../src/boldtailor/_task_design.py#L144)

```python
run_task_columns(events, task_model, frame_times, hrf, *, run, min_onset, oversampling)
```

Task columns for one run; data errors name the run exactly once.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:70](../../src/boldtailor/_hrf_glm_design.py#L70)
- [src/boldtailor/design.py:85](../../src/boldtailor/design.py#L85)
- [tests/test_denoising_cv.py:159](../../tests/test_denoising_cv.py#L159)
- [tests/test_denoising_cv.py:324](../../tests/test_denoising_cv.py#L324)
