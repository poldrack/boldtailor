# `boldtailor._hrf_cv`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_hrf_cv.py)

Nuisance-projected sufficient statistics and data-independent design caches.

## Definitions

- [`_basis`](#definition-1)
- [`_profiled_basis`](#definition-2)
- [`_check_task_rank`](#definition-3)
- [`RunDesign`](#definition-4)
- [`RunDesign.__init__`](#definition-5)
- [`RunDesign.task_design`](#definition-6)
- [`RunDesign._build`](#definition-7)
- [`RunDesign.block`](#definition-8)
- [`RunDesign.eligible`](#definition-9)
- [`RunDesign.trial_matrix`](#definition-10)
- [`RunDesign.trial_eligible`](#definition-11)
- [`_cached_design`](#definition-12)
- [`_validate_onsets`](#definition-13)
- [`_run_inputs`](#definition-14)
- [`_labeled_inputs`](#definition-15)
- [`prepare_runs`](#definition-16)
- [`_batched_products`](#definition-17)
- [`signal_statistics`](#definition-18)
- [`pooled_amplitude`](#definition-19)
- [`prediction_loss`](#definition-20)
- [`loro_scores`](#definition-21)
- [`choose_eligible`](#definition-22)
- [`choose_eligible.check`](#definition-23)
- [`subset_runs`](#definition-24)

<a id="definition-1"></a>

## `_basis`

[Source, line 25](../../src/boldtailor/_hrf_cv.py#L25)

```python
_basis(columns)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:157](../../src/boldtailor/_denoising_gate.py#L157)
- [src/boldtailor/_hrf_cv.py:38](../../src/boldtailor/_hrf_cv.py#L38)
- [src/boldtailor/_hrf_cv.py:66](../../src/boldtailor/_hrf_cv.py#L66)
- [src/boldtailor/_hrf_cv.py:49](../../src/boldtailor/_hrf_cv.py#L49)

<a id="definition-2"></a>

## `_profiled_basis`

[Source, line 30](../../src/boldtailor/_hrf_cv.py#L30)

```python
_profiled_basis(p, q)
```

Orthonormal basis of profiled columns after confound projection.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:84](../../src/boldtailor/_hrf_cv.py#L84)

<a id="definition-3"></a>

## `_check_task_rank`

[Source, line 44](../../src/boldtailor/_hrf_cv.py#L44)

```python
_check_task_rank(xr, x, dof)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:257](../../src/boldtailor/_denoising_cv.py#L257)
- [src/boldtailor/_denoising_gate.py:108](../../src/boldtailor/_denoising_gate.py#L108)
- [src/boldtailor/_hrf_cv.py:87](../../src/boldtailor/_hrf_cv.py#L87)

<a id="definition-4"></a>

## `RunDesign`

[Source, line 55](../../src/boldtailor/_hrf_cv.py#L55)

```python
class RunDesign
```

Cached timing, modulator amplitudes, and confounds: never retains BOLD.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:161](../../src/boldtailor/_hrf_cv.py#L161)

<a id="definition-5"></a>

## `RunDesign.__init__`

[Source, line 58](../../src/boldtailor/_hrf_cv.py#L58)

```python
RunDesign.__init__(self, expanded, times, nuisance, library, task_model, fingerprint)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `RunDesign.task_design`

[Source, line 70](../../src/boldtailor/_hrf_cv.py#L70)

```python
RunDesign.task_design(self, candidate_id)
```

Nilearn task and profiled columns for one candidate on this run.

<a id="definition-7"></a>

## `RunDesign._build`

[Source, line 80](../../src/boldtailor/_hrf_cv.py#L80)

```python
RunDesign._build(self, candidate_id)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `RunDesign.block`

[Source, line 90](../../src/boldtailor/_hrf_cv.py#L90)

```python
RunDesign.block(self, candidate_id)
```

Projected and raw task columns, profiled basis, A; zeros if ineligible.

<a id="definition-9"></a>

## `RunDesign.eligible`

[Source, line 107](../../src/boldtailor/_hrf_cv.py#L107)

```python
RunDesign.eligible(self, candidate_id)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `RunDesign.trial_matrix`

[Source, line 112](../../src/boldtailor/_hrf_cv.py#L112)

```python
@lru_cache(maxsize=256)
RunDesign.trial_matrix(self, candidate_id)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-11"></a>

## `RunDesign.trial_eligible`

[Source, line 126](../../src/boldtailor/_hrf_cv.py#L126)

```python
RunDesign.trial_eligible(self, candidate_id)
```

Estimability of the single-trial design; not used by selection.

<a id="definition-12"></a>

## `_cached_design`

[Source, line 138](../../src/boldtailor/_hrf_cv.py#L138)

```python
@lru_cache(maxsize=32)
_cached_design(expanded_bytes, names_bytes, times_bytes, nuisance_bytes, n_columns, library, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:209](../../src/boldtailor/_hrf_cv.py#L209)

<a id="definition-13"></a>

## `_validate_onsets`

[Source, line 164](../../src/boldtailor/_hrf_cv.py#L164)

```python
_validate_onsets(events, times)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:172](../../src/boldtailor/_hrf_cv.py#L172)
- [src/boldtailor/_hrf_cv.py:181](../../src/boldtailor/_hrf_cv.py#L181)

<a id="definition-14"></a>

## `_run_inputs`

[Source, line 170](../../src/boldtailor/_hrf_cv.py#L170)

```python
_run_inputs(events, times, confounds, task_model, r)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:196](../../src/boldtailor/_hrf_cv.py#L196)

<a id="definition-15"></a>

## `_labeled_inputs`

[Source, line 177](../../src/boldtailor/_hrf_cv.py#L177)

```python
_labeled_inputs(events, times, confounds, task_model, label)
```

As ``_run_inputs``, but every message leads with ``run '<label>'``.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:198](../../src/boldtailor/_hrf_cv.py#L198)

<a id="definition-16"></a>

## `prepare_runs`

[Source, line 188](../../src/boldtailor/_hrf_cv.py#L188)

```python
prepare_runs(data, library, task_model=TaskModel(), labels=None)
```

Cached run designs; with ``labels``, design errors name runs by label.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:220](../../src/boldtailor/_denoising_cv.py#L220)
- [src/boldtailor/_denoising_pool.py:74](../../src/boldtailor/_denoising_pool.py#L74)
- [src/boldtailor/_ridge_cv.py:254](../../src/boldtailor/_ridge_cv.py#L254)
- [src/boldtailor/_selected_hrf_fit.py:157](../../src/boldtailor/_selected_hrf_fit.py#L157)

<a id="definition-17"></a>

## `_batched_products`

[Source, line 213](../../src/boldtailor/_hrf_cv.py#L213)

```python
_batched_products(run, yr, batch_size)
```

B for every candidate as (H, K, F), multiplying batch_size candidates at once.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:242](../../src/boldtailor/_hrf_cv.py#L242)

<a id="definition-18"></a>

## `signal_statistics`

[Source, line 224](../../src/boldtailor/_hrf_cv.py#L224)

```python
signal_statistics(runs, signals, batch_size)
```

Project Y once per run; form A, B, C per candidate and confound-only energy.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:86](../../src/boldtailor/hrf_selection.py#L86)
- [src/boldtailor/hrf_selection.py:298](../../src/boldtailor/hrf_selection.py#L298)

<a id="definition-19"></a>

## `pooled_amplitude`

[Source, line 248](../../src/boldtailor/_hrf_cv.py#L248)

```python
pooled_amplitude(a_sum, b_sum)
```

Solve pooled normal equations per candidate; singular ones get zeros and False.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:298](../../src/boldtailor/_denoising_cv.py#L298)
- [src/boldtailor/_denoising_gate.py:131](../../src/boldtailor/_denoising_gate.py#L131)
- [src/boldtailor/_denoising_gate.py:194](../../src/boldtailor/_denoising_gate.py#L194)
- [src/boldtailor/_denoising_pool.py:78](../../src/boldtailor/_denoising_pool.py#L78)

<a id="definition-20"></a>

## `prediction_loss`

[Source, line 258](../../src/boldtailor/_hrf_cv.py#L258)

```python
prediction_loss(a, b, c, amplitude)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_cv.py:278](../../src/boldtailor/_hrf_cv.py#L278)
- [src/boldtailor/hrf_selection.py:199](../../src/boldtailor/hrf_selection.py#L199)

<a id="definition-21"></a>

## `loro_scores`

[Source, line 269](../../src/boldtailor/_hrf_cv.py#L269)

```python
loro_scores(a, b, c, energy)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:88](../../src/boldtailor/hrf_selection.py#L88)

<a id="definition-22"></a>

## `choose_eligible`

[Source, line 287](../../src/boldtailor/_hrf_cv.py#L287)

```python
choose_eligible(scores, runs)
```

Validate winners lazily; structural exclusions apply to every feature.

Usage in package/tests:

- [src/boldtailor/hrf_selection.py:87](../../src/boldtailor/hrf_selection.py#L87)

<a id="definition-23"></a>

## `choose_eligible.check`

[Source, line 292](../../src/boldtailor/_hrf_cv.py#L292)

```python
choose_eligible.check(cid)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-24"></a>

## `subset_runs`

[Source, line 338](../../src/boldtailor/_hrf_cv.py#L338)

```python
subset_runs(data, indices)
```

Subset run data and source records together without relabeling features.

Usage in package/tests:

- [src/boldtailor/_ridge_cv.py:100](../../src/boldtailor/_ridge_cv.py#L100)
