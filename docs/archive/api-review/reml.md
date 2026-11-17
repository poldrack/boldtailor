# `boldtailor.reml`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/reml.py)

Scalar restricted maximum likelihood for one random-effect variance.

## Definitions

- [`RemlGroup`](#definition-1)
- [`RemlGroup.__post_init__`](#definition-2)
- [`RemlFit`](#definition-3)
- [`RemlFit.__post_init__`](#definition-4)
- [`_RhoFeatureSubset`](#definition-5)
- [`_project_out_fixed_effects_by_group`](#definition-6)
- [`_pooled_energy_exceeds_floor`](#definition-7)
- [`_prepare_estimable_features_for_rho`](#definition-8)
- [`_prepare_feature_subsets_by_rho`](#definition-9)
- [`_classify_variance_ratio_optimum`](#definition-10)
- [`_fit_subset_variance_components`](#definition-11)
- [`_allocate_fit_buffers`](#definition-12)
- [`_copy_fit_values_into`](#definition-13)
- [`_build_reml_fit`](#definition-14)
- [`_require_common_restricted_dof`](#definition-15)
- [`_mark_unresolved_in_place`](#definition-16)
- [`_prepare_reml_fit`](#definition-17)
- [`_fit_subsets_into_buffers`](#definition-18)
- [`fit_reml`](#definition-19)
- [`RemlSelection`](#definition-20)
- [`RemlSelection.__post_init__`](#definition-21)
- [`_initialize_candidate_selection`](#definition-22)
- [`_update_selection_from_fit`](#definition-23)
- [`_fit_candidate_and_update_selection`](#definition-24)
- [`_compute_runner_up_criterion_gap`](#definition-25)
- [`_evaluate_candidates_and_update_fit_buffers`](#definition-26)
- [`select_random_effects_design`](#definition-27)

<a id="definition-1"></a>

## `RemlGroup`

[Source, line 44](../../src/boldtailor/reml.py#L44)

```python
@dataclass(frozen=True)
class RemlGroup
```

Signals ``(scans, features)`` and their unpenalised fixed design ``C``.

Declared fields:

```python
signals: np.ndarray
fixed_design: np.ndarray
```

Usage in module:

- [src/boldtailor/reml.py:250](../../src/boldtailor/reml.py#L250)
- [src/boldtailor/reml.py:372](../../src/boldtailor/reml.py#L372)

<a id="definition-2"></a>

## `RemlGroup.__post_init__`

[Source, line 50](../../src/boldtailor/reml.py#L50)

```python
RemlGroup.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `RemlFit`

[Source, line 55](../../src/boldtailor/reml.py#L55)

```python
@dataclass(frozen=True)
class RemlFit
```

Per-feature REML variance estimates; ``noise_variance`` is marginal.

``unrefined`` marks a positive ratio taken from the search grid without a
verified stationary point; ``at_upper`` marks those at the grid's upper edge.

``no_residual_signal`` means residual energy is within projection rounding.
``numerically_unresolved_residual`` means it exceeds rounding but remains
within fixed-span uncertainty. Both skip fitting and have NaN variance
estimates, ratio and criterion. ``variance_zero`` instead denotes a fitted
zero random-effects variance with resolved residual signal.

Declared fields:

```python
ratio: np.ndarray
noise_variance: np.ndarray
effect_variance: np.ndarray
criterion: np.ndarray
status: tuple[str, ...]
at_upper: np.ndarray
unrefined: np.ndarray
rho: np.ndarray
restricted_dof: int
```

Usage in module:

- [src/boldtailor/reml.py:197](../../src/boldtailor/reml.py#L197)
- [src/boldtailor/reml.py:199](../../src/boldtailor/reml.py#L199)
- [src/boldtailor/reml.py:254](../../src/boldtailor/reml.py#L254)
- [src/boldtailor/reml.py:277](../../src/boldtailor/reml.py#L277)

<a id="definition-4"></a>

## `RemlFit.__post_init__`

[Source, line 78](../../src/boldtailor/reml.py#L78)

```python
RemlFit.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `_RhoFeatureSubset`

[Source, line 89](../../src/boldtailor/reml.py#L89)

```python
@dataclass(frozen=True)
class _RhoFeatureSubset
```

Features with resolved residual signal sharing one noise correlation.

``fit_feature_indices`` identifies features retained for fitting.
``unresolved_feature_indices`` identifies residuals above projection
rounding but within the uncertainty of the computed fixed span.

Declared fields:

```python
fit_feature_indices: np.ndarray
projections: list
dof: int
unresolved_feature_indices: np.ndarray
```

Usage in module:

- [src/boldtailor/reml.py:134](../../src/boldtailor/reml.py#L134)

<a id="definition-6"></a>

## `_project_out_fixed_effects_by_group`

[Source, line 103](../../src/boldtailor/reml.py#L103)

```python
_project_out_fixed_effects_by_group(groups, feature_indices, rho)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in module:

- [src/boldtailor/reml.py:118](../../src/boldtailor/reml.py#L118)

<a id="definition-7"></a>

## `_pooled_energy_exceeds_floor`

[Source, line 110](../../src/boldtailor/reml.py#L110)

```python
_pooled_energy_exceeds_floor(projections, floor)
```

Features whose pooled projected energy exceeds the pooled named floor.

Usage in module:

- [src/boldtailor/reml.py:119](../../src/boldtailor/reml.py#L119)
- [src/boldtailor/reml.py:124](../../src/boldtailor/reml.py#L124)

<a id="definition-8"></a>

## `_prepare_estimable_features_for_rho`

[Source, line 116](../../src/boldtailor/reml.py#L116)

```python
_prepare_estimable_features_for_rho(groups, feature_indices, rho)
```

Prepare resolved residuals for one rho; this does not establish identifiability.

Usage in module:

- [src/boldtailor/reml.py:149](../../src/boldtailor/reml.py#L149)

<a id="definition-9"></a>

## `_prepare_feature_subsets_by_rho`

[Source, line 142](../../src/boldtailor/reml.py#L142)

```python
_prepare_feature_subsets_by_rho(groups, rho)
```

Batch features by identical rho to reuse whitening and fixed projections.

This is a computational grouping: every feature has its own variance
estimates. The same prepared subsets are reused across candidate designs.

Usage in module:

- [src/boldtailor/reml.py:232](../../src/boldtailor/reml.py#L232)

<a id="definition-10"></a>

## `_classify_variance_ratio_optimum`

[Source, line 156](../../src/boldtailor/reml.py#L156)

```python
_classify_variance_ratio_optimum(optimum)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in module:

- [src/boldtailor/reml.py:172](../../src/boldtailor/reml.py#L172)

<a id="definition-11"></a>

## `_fit_subset_variance_components`

[Source, line 161](../../src/boldtailor/reml.py#L161)

```python
_fit_subset_variance_components(subset, random_design)
```

Fit variance components in the order of subset.fit_feature_indices.

Usage in module:

- [src/boldtailor/reml.py:245](../../src/boldtailor/reml.py#L245)
- [src/boldtailor/reml.py:329](../../src/boldtailor/reml.py#L329)

<a id="definition-12"></a>

## `_allocate_fit_buffers`

[Source, line 178](../../src/boldtailor/reml.py#L178)

```python
_allocate_fit_buffers(n_features)
```

Allocate NaN estimates; fitting overwrites features with resolved residuals.

Usage in module:

- [src/boldtailor/reml.py:233](../../src/boldtailor/reml.py#L233)

<a id="definition-13"></a>

## `_copy_fit_values_into`

[Source, line 190](../../src/boldtailor/reml.py#L190)

```python
_copy_fit_values_into(destination, feature_indices, values, keep=slice(None))
```

Copy values into destination feature positions; leave sources intact.

Usage in module:

- [src/boldtailor/reml.py:242](../../src/boldtailor/reml.py#L242)
- [src/boldtailor/reml.py:320](../../src/boldtailor/reml.py#L320)

<a id="definition-14"></a>

## `_build_reml_fit`

[Source, line 197](../../src/boldtailor/reml.py#L197)

```python
_build_reml_fit(fit_buffers, rho, restricted_dof) -> RemlFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in module:

- [src/boldtailor/reml.py:260](../../src/boldtailor/reml.py#L260)
- [src/boldtailor/reml.py:412](../../src/boldtailor/reml.py#L412)

<a id="definition-15"></a>

## `_require_common_restricted_dof`

[Source, line 212](../../src/boldtailor/reml.py#L212)

```python
_require_common_restricted_dof(subsets) -> int
```

The restricted dof shared by every rho subset; rank must not flip with rho.

Usage in module:

- [src/boldtailor/reml.py:235](../../src/boldtailor/reml.py#L235)

<a id="definition-16"></a>

## `_mark_unresolved_in_place`

[Source, line 220](../../src/boldtailor/reml.py#L220)

```python
_mark_unresolved_in_place(fit_buffers, subsets)
```

Mark features whose residual signal is numerically unresolved.

Usage in module:

- [src/boldtailor/reml.py:234](../../src/boldtailor/reml.py#L234)

<a id="definition-17"></a>

## `_prepare_reml_fit`

[Source, line 229](../../src/boldtailor/reml.py#L229)

```python
_prepare_reml_fit(groups, rho)
```

Return feature correlations, prepared subsets, shared dof and fit buffers.

Usage in module:

- [src/boldtailor/reml.py:258](../../src/boldtailor/reml.py#L258)
- [src/boldtailor/reml.py:405](../../src/boldtailor/reml.py#L405)

<a id="definition-18"></a>

## `_fit_subsets_into_buffers`

[Source, line 238](../../src/boldtailor/reml.py#L238)

```python
_fit_subsets_into_buffers(fit_buffers, subsets, random_design)
```

Fit each subset and write estimates into the supplied buffers in place.

Usage in module:

- [src/boldtailor/reml.py:259](../../src/boldtailor/reml.py#L259)
- [src/boldtailor/reml.py:406](../../src/boldtailor/reml.py#L406)

<a id="definition-19"></a>

## `fit_reml`

[Source, line 249](../../src/boldtailor/reml.py#L249)

```python
fit_reml(groups: Sequence[RemlGroup], random_design: Iterable[np.ndarray | None], *, rho: float | np.ndarray=0.0) -> RemlFit
```

Estimate the variance ratio of the random effects, feature by feature.

<a id="definition-20"></a>

## `RemlSelection`

[Source, line 267](../../src/boldtailor/reml.py#L267)

```python
@dataclass(frozen=True)
class RemlSelection
```

Per-feature best random design among candidates, by restricted likelihood.

``any_at_upper`` marks features for which some candidate's criterion was
truncated at the upper edge of its search range. That criterion is an
upper bound on the candidate's optimum, so the comparison is provisional.
``any_unrefined`` also includes interior grid fallbacks of any candidate.

Declared fields:

```python
winner: np.ndarray
fit: RemlFit
gap: np.ndarray
n_candidates: int
any_at_upper: np.ndarray
any_unrefined: np.ndarray
```

Usage in module:

- [src/boldtailor/reml.py:376](../../src/boldtailor/reml.py#L376)
- [src/boldtailor/reml.py:410](../../src/boldtailor/reml.py#L410)

<a id="definition-21"></a>

## `RemlSelection.__post_init__`

[Source, line 283](../../src/boldtailor/reml.py#L283)

```python
RemlSelection.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-22"></a>

## `_initialize_candidate_selection`

[Source, line 289](../../src/boldtailor/reml.py#L289)

```python
_initialize_candidate_selection(n_features)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in module:

- [src/boldtailor/reml.py:356](../../src/boldtailor/reml.py#L356)

<a id="definition-23"></a>

## `_update_selection_from_fit`

[Source, line 299](../../src/boldtailor/reml.py#L299)

```python
_update_selection_from_fit(selection_state, fit_buffers, feature_indices, values, candidate_index)
```

Update candidate rankings and winning-fit buffers using eligible fits.

Usage in module:

- [src/boldtailor/reml.py:330](../../src/boldtailor/reml.py#L330)

<a id="definition-24"></a>

## `_fit_candidate_and_update_selection`

[Source, line 323](../../src/boldtailor/reml.py#L323)

```python
_fit_candidate_and_update_selection(selection_state, fit_buffers, subsets, random_design, candidate_index)
```

Fit one candidate and update selection state and fit buffers in place.

Usage in module:

- [src/boldtailor/reml.py:362](../../src/boldtailor/reml.py#L362)

<a id="definition-25"></a>

## `_compute_runner_up_criterion_gap`

[Source, line 339](../../src/boldtailor/reml.py#L339)

```python
_compute_runner_up_criterion_gap(selection_state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in module:

- [src/boldtailor/reml.py:413](../../src/boldtailor/reml.py#L413)

<a id="definition-26"></a>

## `_evaluate_candidates_and_update_fit_buffers`

[Source, line 351](../../src/boldtailor/reml.py#L351)

```python
_evaluate_candidates_and_update_fit_buffers(groups, candidates, subsets, fit_buffers)
```

Update winning-fit buffers; return selection state and candidate count.

Usage in module:

- [src/boldtailor/reml.py:407](../../src/boldtailor/reml.py#L407)

<a id="definition-27"></a>

## `select_random_effects_design`

[Source, line 371](../../src/boldtailor/reml.py#L371)

```python
select_random_effects_design(groups: Sequence[RemlGroup], candidates: Iterable[Iterable[np.ndarray | None]], *, rho: float | np.ndarray=0.0) -> RemlSelection
```

Select random-effects designs by REML, independently for each feature.

Each candidate supplies one Z matrix per group, in group order; None means
no random-effect columns for that group. The signals, fixed designs C,
and supplied rho are held constant across candidates. For example, a
candidate can contain trial-response designs made with one HRF, while
another candidate uses a different HRF for the same trials.

For every candidate, estimate the shared-across-groups variance ratio and
noise variance, then compare its optimized restricted-likelihood criterion
(lower is better). This selects the candidate covariance structure, not
random columns or random observations. It uses the supplied data directly;
it is not cross-validation and does not return fitted coefficients b_r.

winner contains zero-based candidate indices. Only fits with status "ok"
compete; zero-variance or unidentifiable fits do not become winners. A
winner of -1 means no supported candidate; inspect fit.status to distinguish
zero random-effects variance, absent residual signal, and numerically
unresolved residual signal. Exact ties favor the earlier candidate. fit
holds the winning estimates, or the no-random-effects baseline when no
candidate wins. Features skipped for absent or unresolved residuals retain
NaN estimates. gap is the runner-up criterion minus the best criterion,
or NaN when fewer than two candidates qualify.
any_at_upper and any_unrefined flag limitations in any candidate's search.

Candidates are consumed once, with only the best and runner-up retained.
