# `boldtailor._conventional`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_conventional.py)

Internal support; inspect the definitions and their call sites below.

## Definitions

- [`RunFit`](#definition-1)
- [`ConventionalFit`](#definition-2)
- [`_GLMFit`](#definition-3)
- [`fit_designs`](#definition-4)
- [`fit_r2_designs`](#definition-5)
- [`_fit_run`](#definition-6)
- [`_fit_glm`](#definition-7)
- [`_full_rank_basis`](#definition-8)
- [`_in_fitted_basis`](#definition-9)
- [`_combine_contrasts`](#definition-10)
- [`_fixed_effects`](#definition-11)
- [`_preflight_contrasts`](#definition-12)
- [`_resolve_run_contrasts`](#definition-13)
- [`_resolve_contrast`](#definition-14)
- [`_nilearn_t_contrast`](#definition-15)
- [`_contrast_vector`](#definition-16)
- [`_weight_vector`](#definition-17)
- [`_validate_estimable`](#definition-18)
- [`_warn_if_rank_deficient`](#definition-19)
- [`_validate_designs`](#definition-20)
- [`_validate_residual_dof`](#definition-21)
- [`_prediction`](#definition-22)
- [`_sums_of_squares`](#definition-23)
- [`_r2_from_sums`](#definition-24)

<a id="definition-1"></a>

## `RunFit`

[Source, line 18](../../src/boldtailor/_conventional.py#L18)

```python
@dataclass(frozen=True)
class RunFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
contrasts: Mapping[str, object]
r2: np.ndarray
residual_sum: np.ndarray
total_sum: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/_conventional.py:99](../../src/boldtailor/_conventional.py#L99)

<a id="definition-2"></a>

## `ConventionalFit`

[Source, line 26](../../src/boldtailor/_conventional.py#L26)

```python
@dataclass(frozen=True)
class ConventionalFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
run_fits: tuple[RunFit, ...]
contrasts: Mapping[str, object]
aggregate_r2: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/_conventional.py:65](../../src/boldtailor/_conventional.py#L65)

<a id="definition-3"></a>

## `_GLMFit`

[Source, line 33](../../src/boldtailor/_conventional.py#L33)

```python
@dataclass(frozen=True)
class _GLMFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
labels: np.ndarray
regression_results: dict
residual_sum: np.ndarray
total_sum: np.ndarray
basis: np.ndarray | None
```

Usage in package/tests:

- [src/boldtailor/_conventional.py:117](../../src/boldtailor/_conventional.py#L117)

<a id="definition-4"></a>

## `fit_designs`

[Source, line 41](../../src/boldtailor/_conventional.py#L41)

```python
fit_designs(signals: Sequence[np.ndarray], designs: Sequence[pd.DataFrame], contrasts: Mapping[str, ContrastValue], noise_model: str) -> ConventionalFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:109](../../src/boldtailor/_hrf_glm.py#L109)
- [src/boldtailor/fit.py:115](../../src/boldtailor/fit.py#L115)
- [src/boldtailor/prepared_fit.py:63](../../src/boldtailor/prepared_fit.py#L63)

<a id="definition-5"></a>

## `fit_r2_designs`

[Source, line 68](../../src/boldtailor/_conventional.py#L68)

```python
fit_r2_designs(signals: Sequence[np.ndarray], designs: Sequence[pd.DataFrame], noise_model: str) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-6"></a>

## `_fit_run`

[Source, line 83](../../src/boldtailor/_conventional.py#L83)

```python
_fit_run(signals: np.ndarray, design: pd.DataFrame, contrasts: Mapping[str, np.ndarray], noise_model: str) -> RunFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:50](../../src/boldtailor/_conventional.py#L50)

<a id="definition-7"></a>

## `_fit_glm`

[Source, line 107](../../src/boldtailor/_conventional.py#L107)

```python
_fit_glm(signals: np.ndarray, design: np.ndarray, noise_model: str) -> _GLMFit
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:90](../../src/boldtailor/_conventional.py#L90)
- [src/boldtailor/_conventional.py:75](../../src/boldtailor/_conventional.py#L75)

<a id="definition-8"></a>

## `_full_rank_basis`

[Source, line 120](../../src/boldtailor/_conventional.py#L120)

```python
_full_rank_basis(design: np.ndarray) -> np.ndarray | None
```

Orthonormal row-space basis when the design is rank deficient, else None.

Nilearn divides the dispersion by ``n - columns`` but reports
``n - rank`` degrees of freedom, so rank-deficient designs are fitted as
``design @ basis`` instead.

Usage in package/tests:

- [src/boldtailor/_conventional.py:112](../../src/boldtailor/_conventional.py#L112)

<a id="definition-9"></a>

## `_in_fitted_basis`

[Source, line 135](../../src/boldtailor/_conventional.py#L135)

```python
_in_fitted_basis(vector: np.ndarray, basis: np.ndarray | None) -> np.ndarray
```

Map an estimable contrast onto the columns actually fitted.

Usage in package/tests:

- [src/boldtailor/_conventional.py:95](../../src/boldtailor/_conventional.py#L95)

<a id="definition-10"></a>

## `_combine_contrasts`

[Source, line 140](../../src/boldtailor/_conventional.py#L140)

```python
_combine_contrasts(run_fits: tuple[RunFit, ...], names: tuple[str, ...]) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:58](../../src/boldtailor/_conventional.py#L58)

<a id="definition-11"></a>

## `_fixed_effects`

[Source, line 150](../../src/boldtailor/_conventional.py#L150)

```python
_fixed_effects(contrasts: list[object]) -> object
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:145](../../src/boldtailor/_conventional.py#L145)

<a id="definition-12"></a>

## `_preflight_contrasts`

[Source, line 157](../../src/boldtailor/_conventional.py#L157)

```python
_preflight_contrasts(designs: Sequence[pd.DataFrame], contrasts: Mapping[str, ContrastValue]) -> tuple[Mapping[str, np.ndarray], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:47](../../src/boldtailor/_conventional.py#L47)
- [src/boldtailor/_hrf_glm_design.py:32](../../src/boldtailor/_hrf_glm_design.py#L32)

<a id="definition-13"></a>

## `_resolve_run_contrasts`

[Source, line 167](../../src/boldtailor/_conventional.py#L167)

```python
_resolve_run_contrasts(design: pd.DataFrame, contrasts: Mapping[str, ContrastValue], run: int) -> dict[str, np.ndarray]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:162](../../src/boldtailor/_conventional.py#L162)

<a id="definition-14"></a>

## `_resolve_contrast`

[Source, line 179](../../src/boldtailor/_conventional.py#L179)

```python
_resolve_contrast(value: ContrastValue, columns: pd.Index, design: np.ndarray, name: str, run: int) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:174](../../src/boldtailor/_conventional.py#L174)

<a id="definition-15"></a>

## `_nilearn_t_contrast`

[Source, line 197](../../src/boldtailor/_conventional.py#L197)

```python
_nilearn_t_contrast(labels: np.ndarray, regression_results: dict, vector: np.ndarray) -> object
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:92](../../src/boldtailor/_conventional.py#L92)

<a id="definition-16"></a>

## `_contrast_vector`

[Source, line 218](../../src/boldtailor/_conventional.py#L218)

```python
_contrast_vector(value: ContrastValue, columns: pd.Index, name: str, run: int) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:186](../../src/boldtailor/_conventional.py#L186)

<a id="definition-17"></a>

## `_weight_vector`

[Source, line 234](../../src/boldtailor/_conventional.py#L234)

```python
_weight_vector(weights: Mapping[str, float], columns: pd.Index, name: str, run: int) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:231](../../src/boldtailor/_conventional.py#L231)

<a id="definition-18"></a>

## `_validate_estimable`

[Source, line 254](../../src/boldtailor/_conventional.py#L254)

```python
_validate_estimable(vector: np.ndarray, design: np.ndarray, name: str, run: int) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:190](../../src/boldtailor/_conventional.py#L190)

<a id="definition-19"></a>

## `_warn_if_rank_deficient`

[Source, line 265](../../src/boldtailor/_conventional.py#L265)

```python
_warn_if_rank_deficient(design: np.ndarray, run: int) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:278](../../src/boldtailor/_conventional.py#L278)
- [src/boldtailor/_conventional.py:192](../../src/boldtailor/_conventional.py#L192)

<a id="definition-20"></a>

## `_validate_designs`

[Source, line 275](../../src/boldtailor/_conventional.py#L275)

```python
_validate_designs(designs: Sequence[pd.DataFrame]) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:48](../../src/boldtailor/_conventional.py#L48)
- [src/boldtailor/_conventional.py:73](../../src/boldtailor/_conventional.py#L73)
- [src/boldtailor/_hrf_glm_design.py:33](../../src/boldtailor/_hrf_glm_design.py#L33)

<a id="definition-21"></a>

## `_validate_residual_dof`

[Source, line 282](../../src/boldtailor/_conventional.py#L282)

```python
_validate_residual_dof(design: np.ndarray, run: int) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:279](../../src/boldtailor/_conventional.py#L279)

<a id="definition-22"></a>

## `_prediction`

[Source, line 291](../../src/boldtailor/_conventional.py#L291)

```python
_prediction(labels: np.ndarray, regression_results: dict, design: np.ndarray, shape: tuple[int, int]) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:115](../../src/boldtailor/_conventional.py#L115)

<a id="definition-23"></a>

## `_sums_of_squares`

[Source, line 303](../../src/boldtailor/_conventional.py#L303)

```python
_sums_of_squares(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:116](../../src/boldtailor/_conventional.py#L116)

<a id="definition-24"></a>

## `_r2_from_sums`

[Source, line 315](../../src/boldtailor/_conventional.py#L315)

```python
_r2_from_sums(residual_sum: np.ndarray, total_sum: np.ndarray) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_conventional.py:80](../../src/boldtailor/_conventional.py#L80)
- [src/boldtailor/_conventional.py:65](../../src/boldtailor/_conventional.py#L65)
- [src/boldtailor/_conventional.py:101](../../src/boldtailor/_conventional.py#L101)
