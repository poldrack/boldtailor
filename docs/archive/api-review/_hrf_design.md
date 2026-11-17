# `boldtailor._hrf_design`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_hrf_design.py)

One identified HRF per convolution; bases are never orthogonalized.

## Module contract

One identified HRF per convolution; bases are never orthogonalized.

Kernels are stored at unit peak, and every event's amplitude is scaled so its
predicted response peaks at one. Nilearn's convolution does not multiply by
the sampling step, so without the event scale a beta would depend on TR,
oversampling, and duration. A beta is the peak BOLD response to that
presentation in signal units.

## Definitions

- [`_glover_kernel`](#definition-1)
- [`hrf_model`](#definition-2)
- [`hrf_kernel`](#definition-3)
- [`resolve_hrf`](#definition-4)
- [`identified_candidate`](#definition-5)
- [`hrf_metadata`](#definition-6)
- [`boxcar_response_peak`](#definition-7)
- [`_count_scales`](#definition-8)
- [`_realized_counts`](#definition-9)
- [`event_response_scales`](#definition-10)
- [`frame_tr`](#definition-11)
- [`scale_event_amplitudes`](#definition-12)
- [`convolve_events`](#definition-13)
- [`stimulus_regressor`](#definition-14)
- [`_boxcar_sampling`](#definition-15)
- [`trial_regressors`](#definition-16)
- [`trial_regressors.sampled`](#definition-17)

<a id="definition-1"></a>

## `_glover_kernel`

[Source, line 35](../../src/boldtailor/_hrf_design.py#L35)

```python
_glover_kernel(tr, oversampling=OVERSAMPLING)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-2"></a>

## `hrf_model`

[Source, line 43](../../src/boldtailor/_hrf_design.py#L43)

```python
hrf_model(candidate)
```

Peak-one kernel callable for 'spm', 'glover', or an HrfCandidate.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:75](../../src/boldtailor/_hrf_cv.py#L75)
- [src/boldtailor/_hrf_design.py:153](../../src/boldtailor/_hrf_design.py#L153)
- [src/boldtailor/_hrf_design.py:57](../../src/boldtailor/_hrf_design.py#L57)
- [src/boldtailor/_hrf_design.py:62](../../src/boldtailor/_hrf_design.py#L62)

<a id="definition-3"></a>

## `hrf_kernel`

[Source, line 55](../../src/boldtailor/_hrf_design.py#L55)

```python
hrf_kernel(model, tr, oversampling=OVERSAMPLING)
```

Read-only peak-one kernel for a basis name or candidate.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:213](../../src/boldtailor/_hrf_design.py#L213)

<a id="definition-4"></a>

## `resolve_hrf`

[Source, line 60](../../src/boldtailor/_hrf_design.py#L60)

```python
resolve_hrf(model)
```

Peak-one callable for plain 'spm'/'glover'; other bases pass to Nilearn.

Usage in package/tests:

- [src/boldtailor/_task_design.py:126](../../src/boldtailor/_task_design.py#L126)
- [src/boldtailor/design.py:121](../../src/boldtailor/design.py#L121)

<a id="definition-5"></a>

## `identified_candidate`

[Source, line 65](../../src/boldtailor/_hrf_design.py#L65)

```python
identified_candidate(candidate)
```

The identified candidate for 'spm' or an HrfCandidate; others are rejected.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:75](../../src/boldtailor/_hrf_design.py#L75)
- [src/boldtailor/_single_trial_design.py:21](../../src/boldtailor/_single_trial_design.py#L21)

<a id="definition-6"></a>

## `hrf_metadata`

[Source, line 74](../../src/boldtailor/_hrf_design.py#L74)

```python
hrf_metadata(candidate)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/single_trial.py:111](../../src/boldtailor/single_trial.py#L111)

<a id="definition-7"></a>

## `boxcar_response_peak`

[Source, line 87](../../src/boldtailor/_hrf_design.py#L87)

```python
boxcar_response_peak(kernel, count)
```

Exact peak of a ``count``-sample unit boxcar convolved with the kernel.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:103](../../src/boldtailor/_hrf_design.py#L103)

<a id="definition-8"></a>

## `_count_scales`

[Source, line 101](../../src/boldtailor/_hrf_design.py#L101)

```python
_count_scales(kernel, counts)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:132](../../src/boldtailor/_hrf_design.py#L132)
- [src/boldtailor/_hrf_design.py:214](../../src/boldtailor/_hrf_design.py#L214)

<a id="definition-9"></a>

## `_realized_counts`

[Source, line 107](../../src/boldtailor/_hrf_design.py#L107)

```python
_realized_counts(onsets, durations, frame_times, oversampling, min_onset)
```

Samples Nilearn's boxcar occupies for each event on its oversampled grid.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:131](../../src/boldtailor/_hrf_design.py#L131)

<a id="definition-10"></a>

## `event_response_scales`

[Source, line 121](../../src/boldtailor/_hrf_design.py#L121)

```python
event_response_scales(kernel_fn, onsets, durations, frame_times, oversampling=OVERSAMPLING, min_onset=MIN_ONSET)
```

Amplitudes giving each event's realized oversampled response a unit peak.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:142](../../src/boldtailor/_hrf_design.py#L142)
- [src/boldtailor/_hrf_design.py:154](../../src/boldtailor/_hrf_design.py#L154)

<a id="definition-11"></a>

## `frame_tr`

[Source, line 135](../../src/boldtailor/_hrf_design.py#L135)

```python
frame_tr(frame_times)
```

Nilearn's kernel TR: the minimal frame spacing.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:212](../../src/boldtailor/_hrf_design.py#L212)
- [src/boldtailor/_hrf_design.py:130](../../src/boldtailor/_hrf_design.py#L130)

<a id="definition-12"></a>

## `scale_event_amplitudes`

[Source, line 140](../../src/boldtailor/_hrf_design.py#L140)

```python
scale_event_amplitudes(events, kernel_fn, frame_times, oversampling, min_onset)
```

Events with modulation scaled so each response peaks at one.

Usage in package/tests:

- [src/boldtailor/_task_design.py:128](../../src/boldtailor/_task_design.py#L128)
- [src/boldtailor/design.py:150](../../src/boldtailor/design.py#L150)

<a id="definition-13"></a>

## `convolve_events`

[Source, line 149](../../src/boldtailor/_hrf_design.py#L149)

```python
convolve_events(onsets, durations, times, candidate, name='stimulus')
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:173](../../src/boldtailor/_hrf_design.py#L173)
- [tests/denoising_fixtures.py:85](../../tests/denoising_fixtures.py#L85)
- [tests/test_denoising_validation.py:108](../../tests/test_denoising_validation.py#L108)
- [tests/test_recovery.py:119](../../tests/test_recovery.py#L119)

<a id="definition-14"></a>

## `stimulus_regressor`

[Source, line 167](../../src/boldtailor/_hrf_design.py#L167)

```python
stimulus_regressor(events, frame_times, candidate)
```

Convolve unit-amplitude presentations on the actual run sampling grid.

Usage in package/tests:

- [tests/oracles.py:210](../../tests/oracles.py#L210)
- [tests/test_hrf_selection.py:210](../../tests/test_hrf_selection.py#L210)
- [tests/test_hrf_selection.py:197](../../tests/test_hrf_selection.py#L197)
- [tests/test_hrf_selection.py:86](../../tests/test_hrf_selection.py#L86)

<a id="definition-15"></a>

## `_boxcar_sampling`

[Source, line 177](../../src/boldtailor/_hrf_design.py#L177)

```python
@lru_cache(maxsize=64)
_boxcar_sampling(timing_bytes, times_bytes, oversampling=OVERSAMPLING, min_onset=MIN_ONSET)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_design.py:112](../../src/boldtailor/_hrf_design.py#L112)
- [src/boldtailor/_hrf_design.py:209](../../src/boldtailor/_hrf_design.py#L209)

<a id="definition-16"></a>

## `trial_regressors`

[Source, line 194](../../src/boldtailor/_hrf_design.py#L194)

```python
trial_regressors(events, frame_times, candidate)
```

Convolve unit-peak boxcars using kernel prefix sums on Nilearn's grid.

A sampled boxcar [a,b) convolved with h equals H[j-a]-H[j-b], where H
is the cumulative kernel. Evaluate only the two samples needed for each
acquisition's linear interpolation. Every candidate, including canonical
SPM, takes this path; it is pinned to Nilearn's compute_regressor.

Usage in package/tests:

- [src/boldtailor/_single_trial_design.py:26](../../src/boldtailor/_single_trial_design.py#L26)

<a id="definition-17"></a>

## `trial_regressors.sampled`

[Source, line 217](../../src/boldtailor/_hrf_design.py#L217)

```python
trial_regressors.sampled(indices)
```

No source docstring. Read the linked implementation and call sites for the precise contract.
