# `boldtailor.denoising_results`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/denoising_results.py)

Owned results of task-guided denoising selection.

## Module contract

Owned results of task-guided denoising selection.

Scores and fold diagnostics are selection statistics used to choose one PC
count; they are not independent performance estimates. HRFs, the noise pool,
and its components come from all supplied runs (as in GLMsingle) and are
meant for final fitting.

## Definitions

- [`component_names`](#definition-1)
- [`PcaDiagnostics`](#definition-2)
- [`PcaDiagnostics.__post_init__`](#definition-3)
- [`PcaDiagnostics.table`](#definition-4)
- [`SignificanceGate`](#definition-5)
- [`SignificanceGate.__post_init__`](#definition-6)
- [`SignificanceGate.n_excluded`](#definition-7)
- [`DenoisingFold`](#definition-8)
- [`DenoisingFold.__post_init__`](#definition-9)
- [`DenoisingResult`](#definition-10)
- [`DenoisingResult.__post_init__`](#definition-11)
- [`DenoisingResult._validate`](#definition-12)
- [`DenoisingResult._validate_gate`](#definition-13)
- [`DenoisingResult.candidate_scores`](#definition-14)
- [`DenoisingResult.fold_scores`](#definition-15)
- [`DenoisingResult.perf`](#definition-16)
- [`DenoisingResult.curve`](#definition-17)
- [`DenoisingResult.component_names`](#definition-18)
- [`DenoisingResult.selection_cv_r2`](#definition-19)
- [`DenoisingResult.initial_hrf_indices`](#definition-20)

<a id="definition-1"></a>

## `component_names`

[Source, line 30](../../src/boldtailor/denoising_results.py#L30)

```python
component_names(count: int) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/denoising_results.py:240](../../src/boldtailor/denoising_results.py#L240)

<a id="definition-2"></a>

## `PcaDiagnostics`

[Source, line 35](../../src/boldtailor/denoising_results.py#L35)

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class PcaDiagnostics
```

Per-run normalized pool PCA: effective ranks and all singular values.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
run_labels: tuple[str, ...]
pool_size: int
ranks: tuple[int, ...]
singular_values: tuple[np.ndarray, ...]
rank_tolerances: tuple[float, ...]
retained_columns: tuple[int, ...]
```

Usage in package/tests:

- [src/boldtailor/denoising.py:207](../../src/boldtailor/denoising.py#L207)

<a id="definition-3"></a>

## `PcaDiagnostics.__post_init__`

[Source, line 45](../../src/boldtailor/denoising_results.py#L45)

```python
PcaDiagnostics.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `PcaDiagnostics.table`

[Source, line 49](../../src/boldtailor/denoising_results.py#L49)

```python
PcaDiagnostics.table(self) -> pd.DataFrame
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `SignificanceGate`

[Source, line 62](../../src/boldtailor/denoising_results.py#L62)

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class SignificanceGate
```

Boldtailor's F-test gate on the pcstop count (not part of GLMsingle).

For each tested scoring feature, ``f_statistic``/``p_value`` compare
in-sample fits on all runs without and with the first ``pcstop_count``
PCs of every run (``df1``, ``df2`` its degrees of freedom). With
``noise_model="ar1"`` both fits are AR(1) prewhitened per run, and
``ar_coefficients`` (runs x features) holds the binned lag-1
coefficients used; with ``"ols"`` it is all NaN. ``m`` of the ``n`` tested features have ``p < alpha``; the
count is kept when the one-sided binomial p-value ``binomial_p`` is below
``binomial_alpha``. ``excluded`` features (rank-deficient designs,
nonpositive degrees of freedom, or a zero target in any run) are not in
``n``; ``exclusions`` lists ``(hrf_index, n_features, reason)``.
``decision`` is ``"kept"``, ``"rejected"``, ``"skipped_zero_count"``,
or ``"disabled"``; untested values are NaN.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
enabled: bool
alpha: float
binomial_alpha: float
noise_model: str
pcstop_count: int
n_components: int
decision: str
f_statistic: np.ndarray
p_value: np.ndarray
df1: np.ndarray
df2: np.ndarray
tested: np.ndarray
excluded: np.ndarray
ar_coefficients: np.ndarray
exclusions: tuple[tuple[int, int, str], ...]
m: int
n: int
binomial_p: float
```

Usage in package/tests:

- [src/boldtailor/_denoising_gate.py:290](../../src/boldtailor/_denoising_gate.py#L290)

<a id="definition-6"></a>

## `SignificanceGate.__post_init__`

[Source, line 98](../../src/boldtailor/denoising_results.py#L98)

```python
SignificanceGate.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `SignificanceGate.n_excluded`

[Source, line 104](../../src/boldtailor/denoising_results.py#L104)

```python
@property
SignificanceGate.n_excluded(self) -> int
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `DenoisingFold`

[Source, line 109](../../src/boldtailor/denoising_results.py#L109)

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class DenoisingFold
```

One count-selection fold: its runs and the held-out targets.

HRFs, the pool, its PCs, and the scoring features are full-data and
shared by every fold. ``zero_target`` marks numerically zero held-out
targets; ``target_energy`` is the held-out target energy (NaN for
unscored features).

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
validation_run: str
training_runs: tuple[str, ...]
zero_target: np.ndarray
target_energy: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/denoising.py:220](../../src/boldtailor/denoising.py#L220)

<a id="definition-9"></a>

## `DenoisingFold.__post_init__`

[Source, line 123](../../src/boldtailor/denoising_results.py#L123)

```python
DenoisingFold.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `DenoisingResult`

[Source, line 130](../../src/boldtailor/denoising_results.py#L130)

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class DenoisingResult
```

Chosen PC count, full-data pool and run PCs, and diagnostics.

``onoff_r2`` is GLMsingle's ON-OFF R² that defined ``noise_pool`` (below
``noise_pool_threshold``) and ``scoring_mask`` (above it, or the 100
best features when ``scoring_fallback``). ``scored`` is the scoring mask
minus features with a zero held-out target. ``pool_r2_threshold`` is the
setting (``"auto"`` or a float) and ``noise_pool_mixture`` the mixture
fit when automatic. ``initial_selection`` holds the frozen full-data
HRFs. ``perf``/``curve`` are the per-count median performance and its
gain over zero PCs (NaN when unavailable). ``pcstop_count`` is the
count GLMsingle's pcstop rule chose and ``n_components`` the count kept
by Boldtailor's ``significance_gate`` (equal when the gate is disabled).

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
n_components: int
pcstop_count: int
significance_gate: SignificanceGate
counts: tuple[int, ...]
pool_r2_threshold: float | str
pcstop: float
noise_pool_threshold: float
noise_pool_mixture: MixtureThreshold | None
noise_pool: np.ndarray
scoring_mask: np.ndarray
scoring_fallback: bool
scored: np.ndarray
onoff_r2: np.ndarray
initial_selection: HrfSelectionResult
run_components: tuple[np.ndarray, ...]
components: PcaDiagnostics
folds: tuple[DenoisingFold, ...]
_candidate_scores: pd.DataFrame
_fold_scores: pd.DataFrame
run_labels: tuple[str, ...]
feature_signature: str | None
source_identity: tuple[RunIdentity, ...]
provenance: ProvenanceRecord
```

Usage in package/tests:

- [src/boldtailor/denoising.py:398](../../src/boldtailor/denoising.py#L398)

<a id="definition-11"></a>

## `DenoisingResult.__post_init__`

[Source, line 169](../../src/boldtailor/denoising_results.py#L169)

```python
DenoisingResult.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-12"></a>

## `DenoisingResult._validate`

[Source, line 181](../../src/boldtailor/denoising_results.py#L181)

```python
DenoisingResult._validate(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-13"></a>

## `DenoisingResult._validate_gate`

[Source, line 207](../../src/boldtailor/denoising_results.py#L207)

```python
DenoisingResult._validate_gate(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-14"></a>

## `DenoisingResult.candidate_scores`

[Source, line 221](../../src/boldtailor/denoising_results.py#L221)

```python
@property
DenoisingResult.candidate_scores(self) -> pd.DataFrame
```

Per count: eligibility, median performance, curve, and reasons.

<a id="definition-15"></a>

## `DenoisingResult.fold_scores`

[Source, line 226](../../src/boldtailor/denoising_results.py#L226)

```python
@property
DenoisingResult.fold_scores(self) -> pd.DataFrame
```

Per fold and count: eligibility, reason, fold median R², sizes.

<a id="definition-16"></a>

## `DenoisingResult.perf`

[Source, line 231](../../src/boldtailor/denoising_results.py#L231)

```python
@property
DenoisingResult.perf(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-17"></a>

## `DenoisingResult.curve`

[Source, line 235](../../src/boldtailor/denoising_results.py#L235)

```python
@property
DenoisingResult.curve(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-18"></a>

## `DenoisingResult.component_names`

[Source, line 239](../../src/boldtailor/denoising_results.py#L239)

```python
@property
DenoisingResult.component_names(self) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-19"></a>

## `DenoisingResult.selection_cv_r2`

[Source, line 243](../../src/boldtailor/denoising_results.py#L243)

```python
@property
DenoisingResult.selection_cv_r2(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-20"></a>

## `DenoisingResult.initial_hrf_indices`

[Source, line 247](../../src/boldtailor/denoising_results.py#L247)

```python
@property
DenoisingResult.initial_hrf_indices(self) -> np.ndarray
```

No source docstring. Read the linked implementation and call sites for the precise contract.
