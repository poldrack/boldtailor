# `boldtailor.reml`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/reml.py)

Scalar restricted maximum likelihood for one random-effect variance.

## Definitions

- [`RemlGroup`](#definition-1)
- [`RemlGroup.__post_init__`](#definition-2)
- [`RemlFit`](#definition-3)
- [`RemlFit.__post_init__`](#definition-4)
- [`_Subset`](#definition-5)
- [`_project`](#definition-6)
- [`_above`](#definition-7)
- [`_live_subset`](#definition-8)
- [`_subsets`](#definition-9)
- [`_status`](#definition-10)
- [`_fit_values`](#definition-11)
- [`_empty_result`](#definition-12)
- [`_copy_fit_values_into`](#definition-13)
- [`_assemble`](#definition-14)
- [`_single_dof`](#definition-15)
- [`_mark_unresolved_in_place`](#definition-16)
- [`_start`](#definition-17)
- [`_fit_into`](#definition-18)
- [`fit_reml`](#definition-19)
- [`RemlSelection`](#definition-20)
- [`RemlSelection.__post_init__`](#definition-21)
- [`_empty_race`](#definition-22)
- [`_update_selection_from_fit`](#definition-23)
- [`_fit_and_update_selection`](#definition-24)
- [`_gap`](#definition-25)
- [`_run_race`](#definition-26)
- [`select_random_effects_design`](#definition-27)

<a id="definition-1"></a>

## `RemlGroup`

[Source, line 16](../../src/boldtailor/reml.py#L16)

```python
@dataclass(frozen=True)
class RemlGroup
```

Signals ``(scans, features)`` and their unpenalised fixed design ``C``.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
signals: np.ndarray
fixed_design: np.ndarray
```

Usage in package/tests:

- [tests/test_reml.py:77](../../tests/test_reml.py#L77)
- [tests/test_reml.py:87](../../tests/test_reml.py#L87)
- [tests/test_reml.py:25](../../tests/test_reml.py#L25)
- [tests/test_reml.py:206](../../tests/test_reml.py#L206)

<a id="definition-2"></a>

## `RemlGroup.__post_init__`

[Source, line 22](../../src/boldtailor/reml.py#L22)

```python
RemlGroup.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `RemlFit`

[Source, line 27](../../src/boldtailor/reml.py#L27)

```python
@dataclass(frozen=True)
class RemlFit
```

Per-feature REML variance estimates; ``noise_variance`` is marginal.

``unrefined`` marks a positive ratio taken from the search grid without a
verified stationary point; ``at_upper`` marks those at the grid's upper edge.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

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

Usage in package/tests:

- [src/boldtailor/reml.py:140](../../src/boldtailor/reml.py#L140)

<a id="definition-4"></a>

## `RemlFit.__post_init__`

[Source, line 44](../../src/boldtailor/reml.py#L44)

```python
RemlFit.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `_Subset`

[Source, line 55](../../src/boldtailor/reml.py#L55)

```python
@dataclass(frozen=True)
class _Subset
```

Live columns of one rho value and their fixed projections per group.

``unresolved`` holds the columns whose energy is above projection rounding
but within the uncertainty of the computed fixed span.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
columns: np.ndarray
projections: list
dof: int
unresolved: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/reml.py:89](../../src/boldtailor/reml.py#L89)

<a id="definition-6"></a>

## `_project`

[Source, line 68](../../src/boldtailor/reml.py#L68)

```python
_project(groups, columns, rho)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:80](../../src/boldtailor/reml.py#L80)

<a id="definition-7"></a>

## `_above`

[Source, line 72](../../src/boldtailor/reml.py#L72)

```python
_above(projections, floor)
```

Features whose pooled projected energy exceeds the pooled named floor.

Usage in package/tests:

- [src/boldtailor/reml.py:81](../../src/boldtailor/reml.py#L81)
- [src/boldtailor/reml.py:82](../../src/boldtailor/reml.py#L82)

<a id="definition-8"></a>

## `_live_subset`

[Source, line 78](../../src/boldtailor/reml.py#L78)

```python
_live_subset(groups, columns, rho)
```

Project one rho subset; when it mixes in dead features, re-project the live.

Usage in package/tests:

- [src/boldtailor/reml.py:94](../../src/boldtailor/reml.py#L94)

<a id="definition-9"></a>

## `_subsets`

[Source, line 92](../../src/boldtailor/reml.py#L92)

```python
_subsets(groups, rho)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:171](../../src/boldtailor/reml.py#L171)

<a id="definition-10"></a>

## `_status`

[Source, line 99](../../src/boldtailor/reml.py#L99)

```python
_status(optimum)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:115](../../src/boldtailor/reml.py#L115)

<a id="definition-11"></a>

## `_fit_values`

[Source, line 104](../../src/boldtailor/reml.py#L104)

```python
_fit_values(subset, random_design)
```

Fit one candidate on one subset; arrays are indexed like ``subset.columns``.

Usage in package/tests:

- [src/boldtailor/reml.py:251](../../src/boldtailor/reml.py#L251)
- [src/boldtailor/reml.py:180](../../src/boldtailor/reml.py#L180)

<a id="definition-12"></a>

## `_empty_result`

[Source, line 121](../../src/boldtailor/reml.py#L121)

```python
_empty_result(n_features)
```

Every feature starts as no_signal; fits overwrite the live ones.

Usage in package/tests:

- [src/boldtailor/reml.py:172](../../src/boldtailor/reml.py#L172)

<a id="definition-13"></a>

## `_copy_fit_values_into`

[Source, line 133](../../src/boldtailor/reml.py#L133)

```python
_copy_fit_values_into(result, columns, values, keep=slice(None))
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:245](../../src/boldtailor/reml.py#L245)
- [src/boldtailor/reml.py:180](../../src/boldtailor/reml.py#L180)

<a id="definition-14"></a>

## `_assemble`

[Source, line 138](../../src/boldtailor/reml.py#L138)

```python
_assemble(result, rho, restricted_dof) -> RemlFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:194](../../src/boldtailor/reml.py#L194)
- [src/boldtailor/reml.py:288](../../src/boldtailor/reml.py#L288)

<a id="definition-15"></a>

## `_single_dof`

[Source, line 153](../../src/boldtailor/reml.py#L153)

```python
_single_dof(subsets) -> int
```

The restricted dof shared by every rho subset; rank must not flip with rho.

Usage in package/tests:

- [src/boldtailor/reml.py:174](../../src/boldtailor/reml.py#L174)

<a id="definition-16"></a>

## `_mark_unresolved_in_place`

[Source, line 161](../../src/boldtailor/reml.py#L161)

```python
_mark_unresolved_in_place(result, subsets)
```

Unresolved features claim neither a zero ratio nor an absent signal.

Usage in package/tests:

- [src/boldtailor/reml.py:173](../../src/boldtailor/reml.py#L173)

<a id="definition-17"></a>

## `_start`

[Source, line 168](../../src/boldtailor/reml.py#L168)

```python
_start(groups, rho)
```

Per-feature rho, its subsets, their dof and the result to fill in.

Usage in package/tests:

- [src/boldtailor/reml.py:192](../../src/boldtailor/reml.py#L192)
- [src/boldtailor/reml.py:283](../../src/boldtailor/reml.py#L283)

<a id="definition-18"></a>

## `_fit_into`

[Source, line 177](../../src/boldtailor/reml.py#L177)

```python
_fit_into(result, subsets, random_design)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:193](../../src/boldtailor/reml.py#L193)
- [src/boldtailor/reml.py:284](../../src/boldtailor/reml.py#L284)

<a id="definition-19"></a>

## `fit_reml`

[Source, line 183](../../src/boldtailor/reml.py#L183)

```python
fit_reml(groups: Sequence[RemlGroup], random_design: Iterable[np.ndarray | None], *, rho: float | np.ndarray=0.0) -> RemlFit
```

Estimate the variance ratio of the random effects, feature by feature.

Usage in package/tests:

- [tests/test_reml.py:29](../../tests/test_reml.py#L29)
- [tests/test_reml.py:190](../../tests/test_reml.py#L190)
- [tests/test_reml.py:204](../../tests/test_reml.py#L204)
- [tests/test_reml.py:206](../../tests/test_reml.py#L206)

<a id="definition-20"></a>

## `RemlSelection`

[Source, line 201](../../src/boldtailor/reml.py#L201)

```python
@dataclass(frozen=True)
class RemlSelection
```

Per-feature best random design among candidates, by restricted likelihood.

``any_at_upper`` marks features for which some candidate's criterion was
truncated at the upper edge of its search range. That criterion is an
upper bound on the candidate's optimum, so the comparison is provisional.
``any_unrefined`` also includes interior grid fallbacks of any candidate.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
winner: np.ndarray
fit: RemlFit
gap: np.ndarray
n_candidates: int
any_at_upper: np.ndarray
any_unrefined: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/reml.py:286](../../src/boldtailor/reml.py#L286)

<a id="definition-21"></a>

## `RemlSelection.__post_init__`

[Source, line 217](../../src/boldtailor/reml.py#L217)

```python
RemlSelection.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-22"></a>

## `_empty_race`

[Source, line 223](../../src/boldtailor/reml.py#L223)

```python
_empty_race(n_features)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:264](../../src/boldtailor/reml.py#L264)

<a id="definition-23"></a>

## `_update_selection_from_fit`

[Source, line 233](../../src/boldtailor/reml.py#L233)

```python
_update_selection_from_fit(race, result, columns, values, candidate)
```

Update the running best and runner-up with one candidate's ok features.

Usage in package/tests:

- [src/boldtailor/reml.py:252](../../src/boldtailor/reml.py#L252)

<a id="definition-24"></a>

## `_fit_and_update_selection`

[Source, line 248](../../src/boldtailor/reml.py#L248)

```python
_fit_and_update_selection(race, result, subsets, random_design, candidate)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:268](../../src/boldtailor/reml.py#L268)

<a id="definition-25"></a>

## `_gap`

[Source, line 255](../../src/boldtailor/reml.py#L255)

```python
_gap(race)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/reml.py:289](../../src/boldtailor/reml.py#L289)

<a id="definition-26"></a>

## `_run_race`

[Source, line 262](../../src/boldtailor/reml.py#L262)

```python
_run_race(groups, candidates, subsets, result)
```

Fit candidates one at a time as they are drawn; return the race and count.

Usage in package/tests:

- [src/boldtailor/reml.py:285](../../src/boldtailor/reml.py#L285)

<a id="definition-27"></a>

## `select_random_effects_design`

[Source, line 275](../../src/boldtailor/reml.py#L275)

```python
select_random_effects_design(groups: Sequence[RemlGroup], candidates: Iterable[Iterable[np.ndarray | None]], *, rho: float | np.ndarray=0.0) -> RemlSelection
```

Pick, per feature, the candidate random design with the best criterion.

Usage in package/tests:

- [tests/test_reml.py:275](../../tests/test_reml.py#L275)
- [tests/test_reml_adversarial.py:201](../../tests/test_reml_adversarial.py#L201)
- [tests/test_reml_adversarial.py:212](../../tests/test_reml_adversarial.py#L212)
- [tests/test_reml_adversarial.py:268](../../tests/test_reml_adversarial.py#L268)
