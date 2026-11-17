# `boldtailor._denoising_cv`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_denoising_cv.py)

Leave-one-run-out scoring of temporal-PC counts and GLMsingle's pcstop rule.

## Module contract

Leave-one-run-out scoring of temporal-PC counts and GLMsingle's pcstop rule.

HRFs, the noise pool, its run-wise PCs, and the scoring features are
full-data inputs, frozen for every fold and count (as GLMsingle selects HRFs
and the pool once before GLMdenoise). For each held-out run and candidate
count, shared task coefficients are fit across the training runs after
projecting both the task design and BOLD off each training run's baseline
confounds, intercept, missing-value indicators (convolved with the frozen
HRF), and that run's leading PCs. The held-out run's BOLD never enters its
fold's fit, and its PCs never enter the fit or the target: the target is
projected off its baseline confounds, intercept, and indicators only, so the
target, denominator, and scored features are identical for every count.

Each scored feature's SSE and SST are pooled across folds,
``r2_f(k) = 1 - sum SSE / sum SST``, and the performance of a count is the
median of ``r2_f(k)`` over scored features (GLMsingle uses the median). The
count follows GLMsingle's ``select_noise_regressors`` with ``pcstop``.
Deviation, stated: GLMsingle scores counts by cross-validated single-trial
beta consistency across repeated conditions (and disables denoising without
repeats); repeats are not assumed here, so counts are scored by held-out
time-series prediction, close to the original GLMdenoise. Indicator
coefficients are profiled on held-out BOLD (conditional prediction), and the
scores are selection statistics, not independent performance estimates.

This module reimplements procedures from GLMsingle
(https://github.com/cvnlab/GLMsingle), Copyright (c) 2021, Kendrick Kay,
distributed under the BSD 3-Clause License; see
src/boldtailor/_resources/GLMsingle-LICENSE.txt for the copyright
notice, conditions, and disclaimer.
It is an independent reimplementation, not a copy of
GLMsingle code. Follows GLMsingle's select_noise_regressors stopping rule
(pcstop) and the GLMdenoise cross-validated choice of the PC count.
Reference: Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr,
M.J., Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. eLife, 11, e77599.
https://doi.org/10.7554/eLife.77599

## Definitions

- [`validate_counts`](#definition-1)
- [`validate_pcstop`](#definition-2)
- [`_check_inputs`](#definition-3)
- [`_Terms`](#definition-4)
- [`_HrfGroup`](#definition-5)
- [`ScoringSetup`](#definition-6)
- [`ScoringSetup.__post_init__`](#definition-7)
- [`ScoringSetup.n_runs`](#definition-8)
- [`_run_terms`](#definition-9)
- [`_hrf_groups`](#definition-10)
- [`_target_summary`](#definition-11)
- [`_checked_scored`](#definition-12)
- [`prepare_scoring`](#definition-13)
- [`_extra_basis`](#definition-14)
- [`_training_statistics`](#definition-15)
- [`_run_statistics`](#definition-16)
- [`_training_reason`](#definition-17)
- [`_fold_coefficients`](#definition-18)
- [`_heldout_sse`](#definition-19)
- [`CountScore`](#definition-20)
- [`CountScore.__post_init__`](#definition-21)
- [`CountScore.fold_median`](#definition-22)
- [`_pooled`](#definition-23)
- [`_aggregate_reason`](#definition-24)
- [`score_count`](#definition-25)
- [`fold_score_table`](#definition-26)
- [`score_table`](#definition-27)
- [`choose_count`](#definition-28)
- [`CountSelection`](#definition-29)
- [`CountSelection.__post_init__`](#definition-30)
- [`select_component_count`](#definition-31)

<a id="definition-1"></a>

## `validate_counts`

[Source, line 57](../../src/boldtailor/_denoising_cv.py#L57)

```python
validate_counts(counts) -> tuple[int, ...]
```

Sorted, deduplicated nonnegative integer counts that include zero.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:498](../../src/boldtailor/_denoising_cv.py#L498)
- [src/boldtailor/denoising.py:123](../../src/boldtailor/denoising.py#L123)
- [tests/test_denoising_cv.py:247](../../tests/test_denoising_cv.py#L247)
- [tests/test_denoising_cv.py:248](../../tests/test_denoising_cv.py#L248)

<a id="definition-2"></a>

## `validate_pcstop`

[Source, line 69](../../src/boldtailor/_denoising_cv.py#L69)

```python
validate_pcstop(value) -> float
```

GLMsingle's ``pcstop``: a finite real number of at least 1.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:498](../../src/boldtailor/_denoising_cv.py#L498)
- [src/boldtailor/denoising.py:125](../../src/boldtailor/denoising.py#L125)
- [tests/test_denoising_cv.py:260](../../tests/test_denoising_cv.py#L260)
- [tests/test_denoising_cv.py:263](../../tests/test_denoising_cv.py#L263)

<a id="definition-3"></a>

## `_check_inputs`

[Source, line 76](../../src/boldtailor/_denoising_cv.py#L76)

```python
_check_inputs(data, indices, components, scoring)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:219](../../src/boldtailor/_denoising_cv.py#L219)

<a id="definition-4"></a>

## `_Terms`

[Source, line 96](../../src/boldtailor/_denoising_cv.py#L96)

```python
@dataclass(frozen=True)
class _Terms
```

One run's task design and BOLD for one HRF, off baseline + indicators.

``baseline`` (confounds plus intercept) and ``profiled`` (indicators) are
orthonormal bases of the nuisance span that ``x`` and ``y`` are off.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
x: np.ndarray
raw: np.ndarray
y: np.ndarray
profiled: np.ndarray
baseline: np.ndarray
nuisance_rank: int
zero: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:165](../../src/boldtailor/_denoising_cv.py#L165)

<a id="definition-5"></a>

## `_HrfGroup`

[Source, line 113](../../src/boldtailor/_denoising_cv.py#L113)

```python
@dataclass(frozen=True)
class _HrfGroup
```

Scoring features sharing one frozen HRF, with every run's terms.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
hrf_id: int
features: np.ndarray
runs: tuple[_Terms, ...]
```

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:176](../../src/boldtailor/_denoising_cv.py#L176)

<a id="definition-6"></a>

## `ScoringSetup`

[Source, line 122](../../src/boldtailor/_denoising_cv.py#L122)

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class ScoringSetup
```

Frozen full-data inputs and cached projected terms for every run.

``zero_target`` (runs x features) marks numerically zero held-out
targets; ``scored`` is ``scoring`` minus features with a zero target in
any run, fixed for every fold and count. ``target_energy`` (runs x
features) is each held-out target's energy, NaN outside ``scored``.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
run_labels: tuple[str, ...]
hrf_indices: np.ndarray
components: tuple[RunComponents, ...]
scoring: np.ndarray
scored: np.ndarray
zero_target: np.ndarray
target_energy: np.ndarray
n_regressors: int
groups: tuple[_HrfGroup, ...] = field(repr=False)
```

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:225](../../src/boldtailor/_denoising_cv.py#L225)

<a id="definition-7"></a>

## `ScoringSetup.__post_init__`

[Source, line 141](../../src/boldtailor/_denoising_cv.py#L141)

```python
ScoringSetup.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `ScoringSetup.n_runs`

[Source, line 148](../../src/boldtailor/_denoising_cv.py#L148)

```python
@property
ScoringSetup.n_runs(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `_run_terms`

[Source, line 152](../../src/boldtailor/_denoising_cv.py#L152)

```python
_run_terms(run, hrf_id, y, label)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:173](../../src/boldtailor/_denoising_cv.py#L173)

<a id="definition-10"></a>

## `_hrf_groups`

[Source, line 168](../../src/boldtailor/_denoising_cv.py#L168)

```python
_hrf_groups(data, runs, labels, indices, scoring)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:221](../../src/boldtailor/_denoising_cv.py#L221)

<a id="definition-11"></a>

## `_target_summary`

[Source, line 180](../../src/boldtailor/_denoising_cv.py#L180)

```python
_target_summary(groups, n_runs, n_features)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:222](../../src/boldtailor/_denoising_cv.py#L222)

<a id="definition-12"></a>

## `_checked_scored`

[Source, line 190](../../src/boldtailor/_denoising_cv.py#L190)

```python
_checked_scored(scoring, zero, labels)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:223](../../src/boldtailor/_denoising_cv.py#L223)

<a id="definition-13"></a>

## `prepare_scoring`

[Source, line 201](../../src/boldtailor/_denoising_cv.py#L201)

```python
prepare_scoring(data, *, hrf_indices, components, scoring, task_model, library, run_labels=None) -> ScoringSetup
```

Validate frozen inputs and cache every run's projected terms.

Messages name runs by ``run_labels`` (default ``run-01``, ...).

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:501](../../src/boldtailor/_denoising_cv.py#L501)
- [tests/test_denoising_cv.py:121](../../tests/test_denoising_cv.py#L121)
- [tests/test_denoising_gate.py:355](../../tests/test_denoising_gate.py#L355)
- [tests/test_denoising_gate.py:422](../../tests/test_denoising_gate.py#L422)

<a id="definition-14"></a>

## `_extra_basis`

[Source, line 241](../../src/boldtailor/_denoising_cv.py#L241)

```python
_extra_basis(pcs, profiled)
```

Orthonormal PC directions outside the (orthogonal) indicator span.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:253](../../src/boldtailor/_denoising_cv.py#L253)
- [src/boldtailor/_denoising_gate.py:228](../../src/boldtailor/_denoising_gate.py#L228)

<a id="definition-15"></a>

## `_training_statistics`

[Source, line 251](../../src/boldtailor/_denoising_cv.py#L251)

```python
_training_statistics(terms, pcs)
```

A and B after also projecting off the PCs; raises if not estimable.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:271](../../src/boldtailor/_denoising_cv.py#L271)

<a id="definition-16"></a>

## `_run_statistics`

[Source, line 261](../../src/boldtailor/_denoising_cv.py#L261)

```python
_run_statistics(setup, run, count)
```

Per HRF group (A, B) for one run as a training run, or a reason.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:367](../../src/boldtailor/_denoising_cv.py#L367)

<a id="definition-17"></a>

## `_training_reason`

[Source, line 280](../../src/boldtailor/_denoising_cv.py#L280)

```python
_training_reason(setup, per_run, held_out)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:289](../../src/boldtailor/_denoising_cv.py#L289)

<a id="definition-18"></a>

## `_fold_coefficients`

[Source, line 287](../../src/boldtailor/_denoising_cv.py#L287)

```python
_fold_coefficients(setup, per_run, held_out)
```

Shared task coefficients from the training runs, or a reason.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:372](../../src/boldtailor/_denoising_cv.py#L372)

<a id="definition-19"></a>

## `_heldout_sse`

[Source, line 305](../../src/boldtailor/_denoising_cv.py#L305)

```python
_heldout_sse(setup, held_out, beta)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:376](../../src/boldtailor/_denoising_cv.py#L376)

<a id="definition-20"></a>

## `CountScore`

[Source, line 316](../../src/boldtailor/_denoising_cv.py#L316)

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class CountScore
```

Per-fold coefficients and errors, pooled feature R², and median perf.

Arrays are indexed by held-out run. ``perf`` and ``feature_r2`` are NaN
unless the count is available in every fold.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
count: int
reason: str
fold_reasons: tuple[str, ...]
coefficients: np.ndarray
fold_sse: np.ndarray
fold_sst: np.ndarray
feature_r2: np.ndarray
perf: float
```

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:381](../../src/boldtailor/_denoising_cv.py#L381)

<a id="definition-21"></a>

## `CountScore.__post_init__`

[Source, line 332](../../src/boldtailor/_denoising_cv.py#L332)

```python
CountScore.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-22"></a>

## `CountScore.fold_median`

[Source, line 336](../../src/boldtailor/_denoising_cv.py#L336)

```python
CountScore.fold_median(self, scored, held_out) -> float
```

Median held-out R² of one fold over ``scored`` (diagnostic).

<a id="definition-23"></a>

## `_pooled`

[Source, line 342](../../src/boldtailor/_denoising_cv.py#L342)

```python
_pooled(setup, sse, available)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:380](../../src/boldtailor/_denoising_cv.py#L380)

<a id="definition-24"></a>

## `_aggregate_reason`

[Source, line 352](../../src/boldtailor/_denoising_cv.py#L352)

```python
_aggregate_reason(setup, reasons)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:377](../../src/boldtailor/_denoising_cv.py#L377)

<a id="definition-25"></a>

## `score_count`

[Source, line 360](../../src/boldtailor/_denoising_cv.py#L360)

```python
score_count(setup: ScoringSetup, count: int) -> CountScore
```

Score one count in every fold; an invalid zero count raises.

A positive count is unavailable in a fold when a training run cannot
supply that many unique PCs or a projected task design loses rank or
residual degrees of freedom; it must be available in every fold.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:510](../../src/boldtailor/_denoising_cv.py#L510)
- [tests/test_denoising_cv.py:345](../../tests/test_denoising_cv.py#L345)
- [tests/test_denoising_cv.py:396](../../tests/test_denoising_cv.py#L396)
- [tests/test_denoising_cv.py:546](../../tests/test_denoising_cv.py#L546)

<a id="definition-26"></a>

## `fold_score_table`

[Source, line 396](../../src/boldtailor/_denoising_cv.py#L396)

```python
fold_score_table(setup, scores) -> pd.DataFrame
```

Per fold and count: eligibility, reason, fold median R², and sizes.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:517](../../src/boldtailor/_denoising_cv.py#L517)

<a id="definition-27"></a>

## `score_table`

[Source, line 418](../../src/boldtailor/_denoising_cv.py#L418)

```python
score_table(scores) -> pd.DataFrame
```

Per count: eligibility, median performance, curve, and reasons.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:511](../../src/boldtailor/_denoising_cv.py#L511)

<a id="definition-28"></a>

## `choose_count`

[Source, line 433](../../src/boldtailor/_denoising_cv.py#L433)

```python
choose_count(counts, perf, pcstop) -> int
```

GLMsingle ``select_noise_regressors`` over the available counts.

With ``curve = perf - perf[0]``, walk counts in increasing order keeping
the best curve value so far and its count; stop at the first count where
``best * pcstop >= max(curve)`` and return the count holding that best.
NaN marks an unavailable count (skipped). Zero is always a candidate,
and ``max(curve) <= 0`` chooses 0. This matches GLMsingle's walk over
``p = 0 .. numpcstotry``, except for a roundoff slack of 64 eps (as in
HRF choice) that keeps numerically equal values equal.

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:513](../../src/boldtailor/_denoising_cv.py#L513)
- [tests/test_denoising_cv.py:272](../../tests/test_denoising_cv.py#L272)
- [tests/test_denoising_cv.py:273](../../tests/test_denoising_cv.py#L273)
- [tests/test_denoising_cv.py:274](../../tests/test_denoising_cv.py#L274)

<a id="definition-29"></a>

## `CountSelection`

[Source, line 466](../../src/boldtailor/_denoising_cv.py#L466)

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class CountSelection
```

Chosen count, the scoring setup, per-count scores, and score tables.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
n_components: int
counts: tuple[int, ...]
setup: ScoringSetup
count_scores: tuple[CountScore, ...]
fold_scores: pd.DataFrame
scores: pd.DataFrame
```

Usage in package/tests:

- [src/boldtailor/_denoising_cv.py:512](../../src/boldtailor/_denoising_cv.py#L512)

<a id="definition-30"></a>

## `CountSelection.__post_init__`

[Source, line 476](../../src/boldtailor/_denoising_cv.py#L476)

```python
CountSelection.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-31"></a>

## `select_component_count`

[Source, line 485](../../src/boldtailor/_denoising_cv.py#L485)

```python
select_component_count(data, *, hrf_indices, components, scoring, task_model, library, counts, pcstop, run_labels=None) -> CountSelection
```

Leave-one-run-out PC-count choice by pcstop; requires three runs.

Usage in package/tests:

- [src/boldtailor/denoising.py:173](../../src/boldtailor/denoising.py#L173)
- [tests/test_denoising.py:219](../../tests/test_denoising.py#L219)
- [tests/test_denoising_cv.py:136](../../tests/test_denoising_cv.py#L136)
- [tests/test_denoising_gate.py:71](../../tests/test_denoising_gate.py#L71)
