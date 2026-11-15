# `boldtailor.denoising`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/denoising.py)

Opt-in task-guided denoising following GLMsingle's GLMdenoise stage.

## Module contract

Opt-in task-guided denoising following GLMsingle's GLMdenoise stage.

``select_denoising`` follows GLMsingle with stated deviations. It selects
per-feature HRFs once on all runs and freezes them; computes GLMsingle's
ON-OFF R² (canonical HRF, one task regressor, shared coefficient, each run's
confounds plus intercept as nuisance; GLMsingle uses polynomials); pools
features below GLMsingle's Gaussian-mixture tail threshold (by default) and
scores features above it (the 100 best when none passes); extracts run-wise
PCs of that single pool; and chooses one PC count with GLMsingle's pcstop
rule over the median of fold-pooled held-out R², then (a Boldtailor
addition, not part of GLMsingle) keeps that count only if the PCs pass a
per-feature F-test gate (AR(1) prewhitened by default) with a binomial test
across features. Counts are
scored by leave-one-run-out time-series prediction against a fixed target,
because repeated conditions are not assumed. Every input feature is a
candidate: the core is anatomy-agnostic. ``with_denoising`` appends the chosen PCs to the
baseline confounds of the same analysis for ordinary HRF selection and
fitting.

This module reimplements procedures from GLMsingle
(https://github.com/cvnlab/GLMsingle), Copyright (c) 2021, Kendrick Kay,
distributed under the BSD 3-Clause License; see
src/boldtailor/_resources/GLMsingle-LICENSE.txt for the copyright
notice, conditions, and disclaimer.
It is an independent reimplementation, not a copy of
GLMsingle code. Follows the GLMdenoise stage of GLMsingle (noise pool,
run-wise PCs, cross-validated PC count, select_noise_regressors).
Reference: Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr,
M.J., Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. eLife, 11, e77599.
https://doi.org/10.7554/eLife.77599

## Definitions

- [`_check_data`](#definition-1)
- [`_Settings`](#definition-2)
- [`_check_models`](#definition-3)
- [`_settings`](#definition-4)
- [`_check_runs`](#definition-5)
- [`_digest`](#definition-6)
- [`_pool`](#definition-7)
- [`_count_selection`](#definition-8)
- [`_public_tables`](#definition-9)
- [`_diagnostics`](#definition-10)
- [`_folds`](#definition-11)
- [`_baseline_confounds`](#definition-12)
- [`_excluded_counts`](#definition-13)
- [`_floats`](#definition-14)
- [`_rho_summary`](#definition-15)
- [`_noise_model_record`](#definition-16)
- [`_gate_record`](#definition-17)
- [`_pool_records`](#definition-18)
- [`_count_records`](#definition-19)
- [`_selection_activity`](#definition-20)
- [`_provenance`](#definition-21)
- [`_select`](#definition-22)
- [`_assemble`](#definition-23)
- [`select_denoising`](#definition-24)
- [`_already_applied`](#definition-25)
- [`_check_application`](#definition-26)
- [`_augmented_confounds`](#definition-27)
- [`_augmentation_activity`](#definition-28)
- [`_annotated`](#definition-29)
- [`_annotated_run`](#definition-30)
- [`_annotated_record`](#definition-31)
- [`with_denoising`](#definition-32)

<a id="definition-1"></a>

## `_check_data`

[Source, line 84](../../src/boldtailor/denoising.py#L84)

```python
_check_data(data)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:117](../../src/boldtailor/denoising.py#L117)
- [src/boldtailor/denoising.py:486](../../src/boldtailor/denoising.py#L486)

<a id="definition-2"></a>

## `_Settings`

[Source, line 90](../../src/boldtailor/denoising.py#L90)

```python
@dataclass(frozen=True)
class _Settings
```

Validated, owned selection settings.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
task_model: TaskModel
library: HrfLibrary
counts: tuple[int, ...]
threshold: float | str
pcstop: float
gate: tuple[bool, float, float, str]
feature_signature: str | None
```

Usage in package/tests:

- [src/boldtailor/denoising.py:120](../../src/boldtailor/denoising.py#L120)

<a id="definition-3"></a>

## `_check_models`

[Source, line 102](../../src/boldtailor/denoising.py#L102)

```python
_check_models(task_model, library, signature)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:118](../../src/boldtailor/denoising.py#L118)

<a id="definition-4"></a>

## `_settings`

[Source, line 111](../../src/boldtailor/denoising.py#L111)

```python
_settings(data, task_model, library, grid, signature)
```

Validate everything before any fitting.

``grid`` is (counts, r2 threshold, pcstop, (gate, alpha, binomial alpha,
noise model)).

Usage in package/tests:

- [src/boldtailor/denoising.py:466](../../src/boldtailor/denoising.py#L466)

<a id="definition-5"></a>

## `_check_runs`

[Source, line 131](../../src/boldtailor/denoising.py#L131)

```python
_check_runs(data, labels, settings)
```

At least three runs; every run design valid, errors named by label.

Usage in package/tests:

- [src/boldtailor/denoising.py:468](../../src/boldtailor/denoising.py#L468)

<a id="definition-6"></a>

## `_digest`

[Source, line 139](../../src/boldtailor/denoising.py#L139)

```python
_digest(values, dtype) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:292](../../src/boldtailor/denoising.py#L292)
- [src/boldtailor/denoising.py:303](../../src/boldtailor/denoising.py#L303)
- [src/boldtailor/denoising.py:304](../../src/boldtailor/denoising.py#L304)
- [src/boldtailor/denoising.py:320](../../src/boldtailor/denoising.py#L320)

<a id="definition-7"></a>

## `_pool`

[Source, line 146](../../src/boldtailor/denoising.py#L146)

```python
_pool(data, labels, settings)
```

Frozen full-data HRFs, ON-OFF R², masks, and run-wise pool PCs.

Usage in package/tests:

- [src/boldtailor/denoising.py:385](../../src/boldtailor/denoising.py#L385)

<a id="definition-8"></a>

## `_count_selection`

[Source, line 170](../../src/boldtailor/denoising.py#L170)

```python
_count_selection(data, labels, settings, pool)
```

pcstop's count, then Boldtailor's significance gate on it.

Usage in package/tests:

- [src/boldtailor/denoising.py:386](../../src/boldtailor/denoising.py#L386)

<a id="definition-9"></a>

## `_public_tables`

[Source, line 199](../../src/boldtailor/denoising.py#L199)

```python
_public_tables(count)
```

Aggregate scores and per-fold scores keyed by the held-out run label.

Usage in package/tests:

- [src/boldtailor/denoising.py:396](../../src/boldtailor/denoising.py#L396)

<a id="definition-10"></a>

## `_diagnostics`

[Source, line 206](../../src/boldtailor/denoising.py#L206)

```python
_diagnostics(components, labels, pool_size)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:414](../../src/boldtailor/denoising.py#L414)

<a id="definition-11"></a>

## `_folds`

[Source, line 217](../../src/boldtailor/denoising.py#L217)

```python
_folds(setup)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:415](../../src/boldtailor/denoising.py#L415)

<a id="definition-12"></a>

## `_baseline_confounds`

[Source, line 233](../../src/boldtailor/denoising.py#L233)

```python
_baseline_confounds(data, labels)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:371](../../src/boldtailor/denoising.py#L371)

<a id="definition-13"></a>

## `_excluded_counts`

[Source, line 244](../../src/boldtailor/denoising.py#L244)

```python
_excluded_counts(scores)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:338](../../src/boldtailor/denoising.py#L338)

<a id="definition-14"></a>

## `_floats`

[Source, line 251](../../src/boldtailor/denoising.py#L251)

```python
_floats(values)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:336](../../src/boldtailor/denoising.py#L336)
- [src/boldtailor/denoising.py:337](../../src/boldtailor/denoising.py#L337)
- [src/boldtailor/denoising.py:302](../../src/boldtailor/denoising.py#L302)

<a id="definition-15"></a>

## `_rho_summary`

[Source, line 255](../../src/boldtailor/denoising.py#L255)

```python
_rho_summary(gate, labels)
```

Per run: min, median, and max lag-1 coefficient over tested features.

Usage in package/tests:

- [src/boldtailor/denoising.py:284](../../src/boldtailor/denoising.py#L284)

<a id="definition-16"></a>

## `_noise_model_record`

[Source, line 270](../../src/boldtailor/denoising.py#L270)

```python
_noise_model_record(gate, labels)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:291](../../src/boldtailor/denoising.py#L291)

<a id="definition-17"></a>

## `_gate_record`

[Source, line 288](../../src/boldtailor/denoising.py#L288)

```python
_gate_record(gate, labels)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:367](../../src/boldtailor/denoising.py#L367)

<a id="definition-18"></a>

## `_pool_records`

[Source, line 311](../../src/boldtailor/denoising.py#L311)

```python
_pool_records(settings, statistic, masks, scored)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:365](../../src/boldtailor/denoising.py#L365)

<a id="definition-19"></a>

## `_count_records`

[Source, line 329](../../src/boldtailor/denoising.py#L329)

```python
_count_records(settings, count)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:366](../../src/boldtailor/denoising.py#L366)

<a id="definition-20"></a>

## `_selection_activity`

[Source, line 349](../../src/boldtailor/denoising.py#L349)

```python
_selection_activity(context, pool, selected, prefixes)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:397](../../src/boldtailor/denoising.py#L397)

<a id="definition-21"></a>

## `_provenance`

[Source, line 376](../../src/boldtailor/denoising.py#L376)

```python
_provenance(operation, record, activity)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:579](../../src/boldtailor/denoising.py#L579)
- [src/boldtailor/denoising.py:421](../../src/boldtailor/denoising.py#L421)

<a id="definition-22"></a>

## `_select`

[Source, line 384](../../src/boldtailor/denoising.py#L384)

```python
_select(operation, data, labels, settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:470](../../src/boldtailor/denoising.py#L470)

<a id="definition-23"></a>

## `_assemble`

[Source, line 390](../../src/boldtailor/denoising.py#L390)

```python
_assemble(operation, context, pool, selected)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:387](../../src/boldtailor/denoising.py#L387)

<a id="definition-24"></a>

## `select_denoising`

[Source, line 425](../../src/boldtailor/denoising.py#L425)

```python
select_denoising(data: AnalysisData, *, task_model: TaskModel=TaskModel(), library: HrfLibrary | None=None, counts: tuple[int, ...]=tuple(range(11)), pool_r2_threshold: float | str='auto', pcstop: float=1.05, significance_gate: bool=True, gate_alpha: float=0.05, gate_binomial_alpha: float=0.05, gate_noise_model: str='ar1', feature_signature: str | None=None, run_labels: Sequence[str] | None=None) -> DenoisingResult
```

Choose a temporal-PC count as GLMsingle does; return full-data PCs.

Requires at least three runs. Every feature is a candidate (no mask).
HRFs are selected once on all runs with baseline confounds and frozen.
The pool is the features whose ON-OFF R² falls below
``pool_r2_threshold`` (``"auto"``: GLMsingle's Gaussian-mixture tail
threshold; a degenerate fit raises suggesting a fixed value); features
above it are scored (the 100 best when none passes). Counts are scored by
leave-one-run-out held-out task prediction, pooled per feature across
folds, with the median over features, and chosen by GLMsingle's
``pcstop`` rule (``>= 1``). Positive counts that any fold cannot support
are unavailable; zero always remains. Scores are selection statistics,
not independent performance estimates.

``significance_gate`` (a Boldtailor addition, not part of GLMsingle)
then keeps the pcstop count only if adding those PCs passes per-feature
in-sample F-tests (``p < gate_alpha``) in more scoring features than
chance, by a one-sided binomial test at ``gate_binomial_alpha``;
otherwise zero PCs are chosen. ``gate_noise_model="ar1"`` (default)
prewhitens each run's data and design with its own lag-1 coefficient per
feature, following Nilearn's AR(1) convention; ``"ols"`` does not.
``significance_gate=False`` reproduces GLMsingle's pcstop-only choice.

Usage in package/tests:

- [tests/test_denoising.py:54](../../tests/test_denoising.py#L54)
- [tests/test_denoising.py:778](../../tests/test_denoising.py#L778)
- [tests/test_denoising_gate.py:687](../../tests/test_denoising_gate.py#L687)
- [tests/test_denoising_validation.py:46](../../tests/test_denoising_validation.py#L46)

<a id="definition-25"></a>

## `_already_applied`

[Source, line 476](../../src/boldtailor/denoising.py#L476)

```python
_already_applied(data, result)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:489](../../src/boldtailor/denoising.py#L489)

<a id="definition-26"></a>

## `_check_application`

[Source, line 485](../../src/boldtailor/denoising.py#L485)

```python
_check_application(data, result, signature)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:575](../../src/boldtailor/denoising.py#L575)

<a id="definition-27"></a>

## `_augmented_confounds`

[Source, line 505](../../src/boldtailor/denoising.py#L505)

```python
_augmented_confounds(data, result)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:578](../../src/boldtailor/denoising.py#L578)

<a id="definition-28"></a>

## `_augmentation_activity`

[Source, line 515](../../src/boldtailor/denoising.py#L515)

```python
_augmentation_activity(result)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:579](../../src/boldtailor/denoising.py#L579)

<a id="definition-29"></a>

## `_annotated`

[Source, line 528](../../src/boldtailor/denoising.py#L528)

```python
_annotated(ref, mark)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising.py:537](../../src/boldtailor/denoising.py#L537)
- [src/boldtailor/denoising.py:539](../../src/boldtailor/denoising.py#L539)

<a id="definition-30"></a>

## `_annotated_run`

[Source, line 534](../../src/boldtailor/denoising.py#L534)

```python
_annotated_run(sources, mark)
```

Mark the confounds source (else the signal); file identity is kept.

Usage in package/tests:

- [src/boldtailor/denoising.py:553](../../src/boldtailor/denoising.py#L553)

<a id="definition-31"></a>

## `_annotated_record`

[Source, line 542](../../src/boldtailor/denoising.py#L542)

```python
_annotated_record(data, result)
```

Parent provenance whose sources identify the augmented confounds.

Downstream analysis ids derive from source metadata, so distinct
augmentations of the same files must not share it.

Usage in package/tests:

- [src/boldtailor/denoising.py:576](../../src/boldtailor/denoising.py#L576)

<a id="definition-32"></a>

## `with_denoising`

[Source, line 557](../../src/boldtailor/denoising.py#L557)

```python
with_denoising(data: AnalysisData, result: DenoisingResult, *, feature_signature: str | None=None) -> AnalysisData
```

Append ``denoise_pc_000``, ... to the baseline confounds of each run.

``data`` must be the analysis ``result`` was selected on: same runs in the
same order, time grids, signals (feature order), baseline confounds, and
events, and a matching ``feature_signature``. Applying a result twice or
colliding with existing confound names raises. Signals, events, timing,
and source file identities are preserved. Each run's confounds source
(the signal source when there are no confounds) gains a
``denoising_augmentation`` annotation, so downstream analysis ids differ
between augmentations, and provenance gains one activity. A zero-count
result adds no columns but is still recorded.

Usage in package/tests:

- [tests/test_denoising.py:64](../../tests/test_denoising.py#L64)
- [tests/test_denoising.py:324](../../tests/test_denoising.py#L324)
- [tests/test_denoising.py:342](../../tests/test_denoising.py#L342)
- [tests/test_denoising.py:611](../../tests/test_denoising.py#L611)
