# Scalar REML Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One tested scalar REML implementation inside `boldtailor`, used by every analysis in `experiments/`, replacing three experiment-local implementations.

**Architecture:** Sufficient statistics from one SVD per group (`_reml_terms`), a closed-form profiled criterion (`_reml_criterion`), one vectorised 1-D search (`_reml_search`), and AR(1) handled entirely by whitening (`ar1`, `_ar1_moments`). Two public modules, `boldtailor.reml` and `boldtailor.ar1`, take plain arrays. Experiments keep only design building and thin adapters.

**Tech Stack:** NumPy, SciPy (runtime); pytest, `hypothesis`, `statsmodels` (dev only).

**Spec:** `docs/superpowers/specs/2026-11-14-reml-module-design.md` — read it first; the model, the criterion formula and the selection semantics are defined there and not repeated here.

## Global Constraints

- `uv run` for every command; `uv add --dev` for dependencies. Never bare `python3`.
- Every `__init__.py` stays empty.
- Short functions; one responsibility per module.
- RED-GREEN-Refactor: the failing test is committed **before** the implementation, as a separate commit. Never weaken a test to make it pass; a failing statistical or property test means the implementation (or the plan) is wrong — stop and report.
- Test functions and fixtures, no test classes.
- Arrays: signals `(scans, features)`, designs `(scans, columns)`, float64. Public result arrays are read-only via `boldtailor._arrays.readonly_array`.
- Experiments import only `boldtailor.reml` and `boldtailor.ar1`, never the private modules.
- `tests/reml_oracles.py` imports nothing from `boldtailor`.
- Default `uv run pytest -q` must stay fast: the REML tests added to it total under 60 s.
- Work on branch `feature/reml-module`. Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Deviations from the spec's API sketch

The spec leaves names to the builder. Two refinements, used throughout:

- The noise code is its own public module `boldtailor.ar1` (so experiments can whiten and estimate rho without touching private modules).
- `RemlGroup` holds only `signals` and `fixed_design`. The random-effects design is always passed separately as `random_design`, a sequence with one matrix per group; a selection *candidate* is exactly such a sequence. So `fit_reml(groups, random_design, *, rho=0.0)`.
- Naming: `fixed_design` is the matrix `C` of unpenalised (nuisance) regressors; `random_design` is the matrix `Z` whose coefficients are random effects. Neither name refers to random numbers. Use these two names for every package parameter and field; the oracle helpers in `tests/reml_oracles.py` may keep the short `fixed`/`random`.

## Review Focus

Conditions the spec implies that a user will hit; each has a named test in the task shown.

1. Random designs in very different units (RT in seconds next to unit-peak responses, or columns scaled by 1e±6): the optimum must not sit on a grid edge by accident → Task 7, `test_scaling_random_design_is_equivariant` with `k` drawn from `[1e-6, 1e6]`.
2. Dead or constant features mixed with good ones (masked vertices): only those features are `no_signal`; no NaN, warning or changed result leaks into the others → Task 5, `test_dead_features_do_not_affect_others`.
3. Groups of unequal length, unequal fixed-column counts, and a random column that is all zero in one group only (a run with no missing trials) → Task 5, `test_ragged_groups_match_dense_oracle`.
4. NaN or inf anywhere in signals or designs raises `ValueError` naming the offending input rather than returning NaN results → Task 5, `test_non_finite_input_raises`.
5. Empty candidate iterable, a candidate with the wrong number of groups, or a scan-count mismatch raises `ValueError`; a generator is consumed exactly once → Task 6, `test_invalid_candidates_raise` and `test_candidates_consumed_once`.

## File structure

| File | Responsibility |
| --- | --- |
| `src/boldtailor/_ar1_moments.py` | Lag-one residual ratio; its expectation under AR(1) after projection; inversion; pooling |
| `src/boldtailor/ar1.py` | Public: `whiten`, `estimate_rho`, `RhoEstimate` |
| `src/boldtailor/_reml_terms.py` | `FixedProjection`, `RemlTerms`, `project_fixed`, `reml_terms`; the only place rank tolerances live |
| `src/boldtailor/_reml_criterion.py` | `criterion`, `profiled_variance` |
| `src/boldtailor/_reml_search.py` | `RatioOptimum`, `minimize_ratio` |
| `src/boldtailor/reml.py` | Public: `RemlGroup`, `RemlFit`, `RemlSelection`, `fit_reml`, `select_random_effects_design` |
| `tests/reml_oracles.py` | Data simulator, dense criterion/fit, `MixedLM` fit, null-LRT reference. No boldtailor imports |
| `tests/test_ar1.py`, `test_reml_terms.py`, `test_reml_criterion.py`, `test_reml_search.py`, `test_reml.py`, `test_reml_selection.py` | Unit tests per module |
| `tests/test_reml_oracle.py`, `test_reml_properties.py`, `test_reml_statistics.py` | Oracle agreement, hypothesis properties, Monte Carlo |
| `experiments/hrf_estimation/reml_selection.py` | New adapter: runs + HRF library → package call → `HrfEstimate` |

---

### Task 1: Branch, dev dependencies, AR(1) whitening and rho moments

**Files:**
- Modify: `pyproject.toml` (dev group), `uv.lock`, `tests/conftest.py`
- Create: `src/boldtailor/_ar1_moments.py`, `src/boldtailor/ar1.py`, `tests/reml_oracles.py`, `tests/test_ar1.py`

**Interfaces — Produces:**
```python
# boldtailor/ar1.py
def whiten(values: np.ndarray, rho: float) -> np.ndarray
    # row 0 unchanged; row t>=1 is (v[t] - rho*v[t-1]) / sqrt(1 - rho**2). 1-D or 2-D input.
    # ValueError unless rho is finite and |rho| < 1.
@dataclass(frozen=True)
class RhoEstimate:
    values: np.ndarray   # (groups, features) corrected; NaN where undefined
    raw: np.ndarray      # (groups, features) uncorrected lag-one ratio; NaN where undefined
    pooled: float        # median of finite `values`, clipped to [-0.8, 0.8]; 0.0 if none
def estimate_rho(signals: Sequence[np.ndarray], reference: Sequence[np.ndarray]) -> RhoEstimate

# boldtailor/_ar1_moments.py
def residual_lag_ratio(signals, design) -> np.ndarray      # (features,)
    # e = OLS residual; sum(e[1:]*e[:-1]) / sum(e[:-1]**2);
    # NaN where the denominator <= 64*eps*sum(signals**2)
def expected_lag_ratio(design, rhos) -> np.ndarray         # tr(M L M R(rho)) / tr(M D M R(rho))
def correct_lag_ratio(values, design) -> np.ndarray        # invert on np.linspace(-0.95, 0.95, 2001)
def pool_rho(values) -> float

# tests/reml_oracles.py
def ar1_correlation(n: int, rho: float) -> np.ndarray      # dense rho**|i-j|
```
`M = I − X X⁺`, `L` has 0.5 on the two first off-diagonals, `D` is the identity with the last diagonal entry zero. `correct_lag_ratio` interpolates when the expected curve is strictly increasing and otherwise takes the nearest grid point (the existing behaviour in `experiments/hrf_estimation/reml_extensions.py:137`).

- [ ] **Step 1:** `git switch -c feature/reml-module`; `uv add --dev statsmodels hypothesis`; in `tests/conftest.py` register a `slow` marker and a `--run-slow` option exactly as `notebook`/`--run-notebooks` are registered there (skip unless the option is given). Run `uv run pytest -q` — expected: same pass count as before. Commit `chore: add statsmodels and hypothesis dev dependencies and slow marker`.
- [ ] **Step 2: Write failing tests** in `tests/test_ar1.py`:
  - `test_whitened_ar1_covariance_is_identity` (hypothesis: `n` in 3–40, `rho` in (−0.95, 0.95)): build `W = whiten(np.eye(n), rho)`; `assert_allclose(W @ ar1_correlation(n, rho) @ W.T, np.eye(n), atol=1e-10)`.
  - `test_zero_rho_is_identity`; `test_whiten_does_not_modify_input`; `test_invalid_rho_raises` for `1.0`, `-1.0`, `nan`, `inf`.
  - `test_expected_lag_ratio_matches_dense_traces`: random `(40, 6)` design, `rhos = [-0.5, 0, 0.3, 0.7]`, compare with `np.trace(M@L@M@R)/np.trace(M@D@M@R)` built from dense matrices, `rtol=1e-10`.
  - `test_correction_inverts_expectation`: `correct_lag_ratio(expected_lag_ratio(X, rhos), X)` returns `rhos` within `1e-3` for `rhos` in `[-0.8, 0.8]`.
  - `test_degenerate_feature_is_nan`: a signal column lying exactly in the design's span gives NaN in `raw` and `values` and is ignored by `pooled`.
  - `test_pooled_is_clipped_median` and `test_no_valid_values_pools_to_zero`.
- [ ] **Step 3:** `uv run pytest tests/test_ar1.py -q` — expected: collection error, `No module named 'boldtailor.ar1'`. Commit `test: specify AR(1) whitening and rho moment correction`.
- [ ] **Step 4:** Implement both modules and `ar1_correlation`.
- [ ] **Step 5:** `uv run pytest tests/test_ar1.py -q` — expected: all pass. Commit `feat: add normalised AR(1) whitening and projection-corrected rho`.

---

### Task 2: Sufficient statistics

**Files:**
- Create: `src/boldtailor/_reml_terms.py`, `tests/test_reml_terms.py`
- Modify: `tests/reml_oracles.py`

**Interfaces — Consumes:** `boldtailor.ar1.whiten`.
**Produces:**
```python
@dataclass(frozen=True)
class FixedProjection:
    basis: np.ndarray       # (scans, rank) orthonormal basis of the whitened fixed design
    signals: np.ndarray     # (scans, features) whitened signals with the fixed design projected out
    energy: np.ndarray      # (features,) sum of squares of `signals`
    raw_energy: np.ndarray  # (features,) sum of squares of the whitened signals before projection
    dof: int                # scans - rank
    rho: float
def project_fixed(signals, fixed_design, rho: float = 0.0) -> FixedProjection
    # ValueError("positive restricted degrees of freedom required") when scans <= rank

@dataclass(frozen=True)
class RemlTerms:
    singular2: np.ndarray     # (k,) squared singular values of the projected random design
    coordinates2: np.ndarray  # (k, features)
    residual_ss: np.ndarray   # (features,) energy - coordinates2.sum(0), floored at 0
    dof: int
    energy: np.ndarray
    raw_energy: np.ndarray
def reml_terms(projection: FixedProjection, random_design: np.ndarray | None) -> RemlTerms

# tests/reml_oracles.py
def simulate(rng, *, scans=(60, 60), n_fixed=4, n_random=3, ratio=1.0, sigma2=1.0,
             rho=0.0, n_features=1) -> tuple[list[signals], list[fixed], list[random]]
    # one entry of `scans` per group; fixed designs include an intercept column;
    # the same designs are shared by all features, coefficients and noise are independent per feature
```
Rank rule (the only tolerance in the package): keep singular values `s > max(shape) * eps * norm2`, where `norm2` is the spectral norm of the whitened random design **before** projection. This is what makes "random design inside the span of the fixed design" produce `k = 0` instead of roundoff-level directions. `random_design=None` or zero columns gives `k = 0`.

- [ ] **Step 1: Write failing tests** in `tests/test_reml_terms.py`:
  - `test_energy_decomposes`: `energy == residual_ss + coordinates2.sum(0)` to `rtol=1e-10`.
  - `test_projected_signals_are_orthogonal_to_fixed`: `fixed_whitened.T @ projection.signals` is zero to `1e-9`.
  - `test_dof_counts_rank_not_columns`: duplicate a fixed column; `dof` is unchanged.
  - `test_no_restricted_dof_raises`: 5 scans, 5 independent fixed columns.
  - `test_random_inside_fixed_span_has_no_directions`: `random = fixed[:, :2] @ A` → `len(singular2) == 0`.
  - `test_duplicated_random_column_drops_rank`: `k == q - 1`.
  - `test_missing_random_design_has_no_directions`: `random_design=None` and `np.empty((n, 0))`.
  - `test_whitening_applied_to_all_inputs`: `project_fixed(y, C, 0.4)` equals `project_fixed(whiten(y,.4), whiten(C,.4), 0.0)` in `signals`, `energy`, `dof`.
- [ ] **Step 2:** Run — expected: import error. Commit `test: specify REML sufficient statistics`.
- [ ] **Step 3:** Implement `simulate` and the module.
- [ ] **Step 4:** `uv run pytest tests/test_reml_terms.py -q` — expected: pass. Commit `feat: add REML sufficient statistics`.

---

### Task 3: Criterion

**Files:**
- Create: `src/boldtailor/_reml_criterion.py`, `tests/test_reml_criterion.py`
- Modify: `tests/reml_oracles.py`

**Interfaces — Consumes:** `RemlTerms`.
**Produces:**
```python
def profiled_variance(ratios, terms: Sequence[RemlTerms]) -> np.ndarray   # Q(λ)/N
def criterion(ratios, terms: Sequence[RemlTerms]) -> np.ndarray           # N log(Q/N) + Σ log(1 + λ s²)
    # `ratios` has shape (..., features); the result has the same shape.

# tests/reml_oracles.py
def dense_criterion(signals, fixed, random, ratio: float, rho: float = 0.0) -> np.ndarray  # (features,)
    # per group: K = scipy.linalg.null_space(C.T); V = K.T @ (ar1_correlation(n, rho) + ratio * Z @ Z.T) @ K;
    # q += y'K V^-1 K'y; logdet += slogdet(V); n += K.shape[1]; return n*log(q/n) + logdet
    # Works on UNWHITENED data.
```

- [ ] **Step 1: Write failing tests:**
  - `test_criterion_differences_match_dense_iid` and `..._ar1` (`rho=0.5`): for ratios `[0, 0.01, 1, 10, 1e3]`, `criterion(r) − criterion(0)` equals the dense difference, `rtol=1e-9, atol=1e-9`. Three groups of unequal length, three features.
  - `test_profiled_variance_matches_dense_quadratic_form` at the same ratios.
  - `test_criterion_broadcasts_over_leading_axes`: `ratios` of shape `(7, features)` equals seven separate calls.
  - `test_no_random_directions_is_constant_in_ratio`.
- [ ] **Step 2:** Run — expected: import error. Commit `test: specify profiled REML criterion against dense covariance`.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run — expected: pass. Commit `feat: add profiled REML criterion`.

---

### Task 4: Search

**Files:**
- Create: `src/boldtailor/_reml_search.py`, `tests/test_reml_search.py`
- Modify: `tests/reml_oracles.py`

**Interfaces — Consumes:** `criterion`, `RemlTerms`.
**Produces:**
```python
@dataclass(frozen=True)
class RatioOptimum:
    ratio: np.ndarray      # (features,)
    criterion: np.ndarray
    at_zero: np.ndarray    # bool: criterion(0) <= criterion at the best positive ratio
    at_upper: np.ndarray   # bool: best positive ratio is the last grid point, and not at_zero
def minimize_ratio(terms, *, grid_size: int = 121, iterations: int = 60) -> RatioOptimum

# tests/reml_oracles.py
def dense_fit(signals, fixed, random, rho=0.0) -> tuple[np.ndarray, np.ndarray]   # ratio, sigma2 per feature
    # scipy.optimize.minimize_scalar(bounded) on log(ratio) over [-30, 30] of dense_criterion,
    # then compare with ratio = 0; sigma2 from the dense quadratic form / n.
```
Grid: `log λ` evenly spaced from `log(1e-8 / s2max)` to `log(1e8 / s2min)` over all groups' `singular2`; golden-section refinement in the bracket around the best grid point; keep the better of grid point and refined point; then compare with `λ = 0`. With no random directions: ratio 0, `at_zero` True. Ties between zero and positive go to zero.

- [ ] **Step 1: Write failing tests:**
  - `test_optimum_not_worse_than_any_grid_point_or_zero` (hypothesis over seeds, true ratio in `{0, 0.1, 10}`): evaluate `criterion` on a 400-point log grid plus 0; returned criterion ≤ min + `1e-9`.
  - `test_matches_dense_fit`: ratio within `rtol=1e-5` of `dense_fit` for true ratios `[0.05, 1, 20]`, iid and `rho=0.5`.
  - `test_signal_orthogonal_to_random_design_is_at_zero`.
  - `test_noise_free_signal_hits_upper_edge`: `y = Z b` exactly (after projection) → `at_upper` True, `at_zero` False.
  - `test_no_directions_returns_zero`.
- [ ] **Step 2:** Run — expected: import error. Commit `test: specify variance-ratio search`.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run — expected: pass. Commit `feat: add vectorised variance-ratio search`.

---

### Task 5: `fit_reml` and the `MixedLM` oracle

**Files:**
- Create: `src/boldtailor/reml.py`, `tests/test_reml.py`, `tests/test_reml_oracle.py`
- Modify: `tests/reml_oracles.py`

**Interfaces — Consumes:** Tasks 1–4.
**Produces:**
```python
@dataclass(frozen=True)
class RemlGroup:
    signals: np.ndarray   # (scans, features)
    fixed_design: np.ndarray   # (scans, p): C, unpenalised nuisance regressors

@dataclass(frozen=True)
class RemlFit:
    ratio: np.ndarray            # λ̂ = τ̂²/σ̂², (features,)
    noise_variance: np.ndarray   # σ̂² (marginal)
    effect_variance: np.ndarray  # τ̂² = ratio * noise_variance
    criterion: np.ndarray
    status: tuple[str, ...]      # "ok" | "variance_zero" | "no_signal"
    at_upper: np.ndarray         # bool
    rho: np.ndarray              # (features,)
    restricted_dof: int

def fit_reml(groups: Sequence[RemlGroup], random_design: Sequence[np.ndarray | None], *,
             rho: float | np.ndarray = 0.0) -> RemlFit

# tests/reml_oracles.py
def mixedlm_fit(signals, fixed, random) -> tuple[float, float]   # ratio, sigma2; single feature
```
`no_signal`: pooled `energy <= 64 * eps * raw_energy`; such features get NaN `noise_variance`, `effect_variance`, `criterion` and ratio 0. `variance_zero`: `at_zero`. Per-feature `rho`: partition feature indices by `np.unique(rho)`, fit each subset with its scalar rho, scatter results; `restricted_dof` is the same for every subset. Validation raises `ValueError` for: no groups; `len(random_design) != len(groups)`; scan-count mismatch within a group; feature-count mismatch across groups; non-finite input (message names `signals`, `fixed_design` or `random_design` and the group index); `rho` not scalar or `(features,)`, or `|rho| >= 1`.

`mixedlm_fit` (the construction is not obvious, so it is fixed here):
```python
from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec
groups = np.repeat(np.arange(len(signals)), [len(y) for y in signals])
names = [[f"z{g}_{j}" for j in range(z.shape[1])] for g, z in enumerate(random)]
vc = VCSpec(["z"], [names], [list(random)])
result = MixedLM(np.concatenate(signals), scipy.linalg.block_diag(*fixed), groups, exog_vc=vc).fit(reml=True)
return result.vcomp[0] / result.scale, result.scale
```

- [ ] **Step 1: Write failing tests** in `tests/test_reml.py`:
  - `test_fit_matches_layers`: `ratio`/`criterion` equal `minimize_ratio` on hand-built terms; `effect_variance == ratio * noise_variance`.
  - `test_results_are_read_only`: writing to `fit.ratio` raises.
  - `test_float32_input_is_converted`.
  - `test_per_feature_rho_equals_separate_fits`: rho `[0.0, 0.3, 0.3, 0.6]` over four features equals three scalar-rho fits on the matching columns.
  - `test_dead_features_do_not_affect_others` *(Review Focus 2)*: features 1 and 3 are all-zero and constant respectively (intercept in `fixed`); their status is `no_signal`; features 0 and 2 are bit-identical to a fit without the dead columns; `warnings.simplefilter("error")` is active.
  - `test_ragged_groups_match_dense_oracle` *(Review Focus 3)*: scans `(40, 73, 55)`, fixed columns `(3, 5, 4)`, one random column all zero in group 1 only; ratio within `1e-5` of `dense_fit`.
  - `test_non_finite_input_raises` *(Review Focus 4)*: NaN in each of signals/fixed_design/random_design, and inf; message matches the input name.
  - `test_invalid_shapes_raise` and `test_invalid_rho_raises` for each validation rule above.
  - `test_variance_zero_status` and `test_no_restricted_dof_raises`.
- [ ] **Step 2: Write failing oracle tests** in `tests/test_reml_oracle.py`, parametrised over: groups 1, 3, 12; scans equal and unequal; `n_random` 1, 3, 6; true ratio 0.05, 1, 20; plus one rank-deficient random design (duplicated column) and AR(1) cases (`rho=0.5`, data simulated with AR(1) noise, `MixedLM` given `whiten`-ed arrays):
  - `test_matches_mixedlm`: `ratio` and `noise_variance` within `rtol=2e-4` of `mixedlm_fit`.
  - `test_matches_dense_fit`: same cases, `rtol=1e-5`, AR(1) cases on **unwhitened** data with `rho` passed to both.
- [ ] **Step 3:** Run — expected: import error. Commit `test: specify fit_reml against MixedLM and dense oracles`.
- [ ] **Step 4:** Implement `mixedlm_fit` and `reml.py` (`fit_reml` only). Keep validation, rho partitioning and result assembly in separate small functions.
- [ ] **Step 5:** `uv run pytest tests/test_reml.py tests/test_reml_oracle.py -q` — expected: pass. If a `MixedLM` case fails to converge, that case is recorded with `pytest.param(..., marks=pytest.mark.xfail(strict=True, reason=...))` only after confirming against `dense_fit` that boldtailor is right; never loosen the tolerance. Commit `feat: add fit_reml`.

---

### Task 6: `select_random_effects_design`

**Files:**
- Modify: `src/boldtailor/reml.py`
- Create: `tests/test_reml_selection.py`

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class RemlSelection:
    winner: np.ndarray    # int64 (features,), index into the candidates; -1 if none
    fit: RemlFit          # the winner's fit per feature; for winner == -1 the λ = 0 fit (or no_signal)
    gap: np.ndarray       # runner-up criterion minus winner criterion among "ok" candidates; NaN if < 2
    n_candidates: int

def select_random_effects_design(groups: Sequence[RemlGroup],
                         candidates: Iterable[Sequence[np.ndarray | None]], *,
                         rho: float | np.ndarray = 0.0) -> RemlSelection
```
`project_fixed` is called once per group per distinct rho, before the candidate loop. Candidates are consumed in one pass with a running best/second-best per feature; a strictly lower criterion is required to displace the current winner (ties → earlier candidate). Only candidates whose status for that feature is `ok` compete.

- [ ] **Step 1: Write failing tests:**
  - `test_planted_design_wins`: data simulated from candidate 2 of 5 at ratio 5; every feature selects 2; `fit` equals `fit_reml` on candidate 2.
  - `test_ties_go_to_earlier_candidate`: the same design supplied twice → winner is the first; `gap == 0`.
  - `test_no_supported_candidate_gives_sentinel`: signals orthogonal to every candidate → `winner == -1`, `gap` NaN, status `variance_zero`, criterion equals the `λ = 0` criterion.
  - `test_no_signal_feature_gives_sentinel`.
  - `test_single_ok_candidate_has_nan_gap`.
  - `test_candidates_consumed_once` *(Review Focus 5)*: pass a generator that counts `next` calls and raises if iterated twice, with per-feature rho of three distinct values.
  - `test_invalid_candidates_raise` *(Review Focus 5)*: empty iterable; candidate with wrong group count; scan mismatch — each `ValueError`.
  - `test_per_feature_rho_equals_separate_selections`.
  - `test_fixed_projection_built_once_per_rho`: monkeypatch `boldtailor.reml.project_fixed` with a counting wrapper; 4 candidates, 2 groups, 2 distinct rho → exactly 4 calls.
- [ ] **Step 2:** Run — expected: `ImportError: cannot import name 'select_random_effects_design'`. Commit `test: specify REML candidate selection semantics`.
- [ ] **Step 3:** Implement. Refactor `fit_reml` to share the projection/partition helpers.
- [ ] **Step 4:** `uv run pytest tests/test_reml.py tests/test_reml_selection.py tests/test_reml_oracle.py -q` — expected: pass. Commit `feat: add REML candidate selection`.

---

### Task 7: Property-based tests

**Files:**
- Create: `tests/test_reml_properties.py`

Tests only, against the public API. Use `@settings(derandomize=True, deadline=None, max_examples=50)`. Strategies draw a seed and small integers (groups 1–4, scans 12–40, fixed columns 1–4, random columns 1–4, features 1–3, true ratio from `{0, 0.1, 1, 10}`) and call `simulate`. A failure here is an implementation defect: fix the owning module with its own RED test added to that module's test file, not by narrowing the strategy.

- [ ] **Step 1: Write the tests** (comparisons `rtol=1e-6` on ratio and variances unless stated; statuses must be equal):
  - `test_fixed_reparameterisation_is_invariant`: `fixed @ A` for a random well-conditioned invertible `A`.
  - `test_random_rotation_is_invariant`: `random @ Q`, `Q` orthogonal.
  - `test_group_permutation_is_invariant`; `test_feature_permutation_is_equivariant`.
  - `test_scaling_signals_is_equivariant`: `k` in `[1e-6, 1e6]`; ratio unchanged, `noise_variance` × `k²`.
  - `test_scaling_random_design_is_equivariant` *(Review Focus 1)*: `k` in `[1e-6, 1e6]`; ratio and `effect_variance` × `1/k²`, `noise_variance` and `criterion` unchanged (`criterion` to `atol=1e-6`); `at_upper` unchanged.
  - `test_adding_fixed_span_to_signals_changes_nothing`: `signals + fixed @ B`.
  - `test_appending_random_column_inside_fixed_span_changes_nothing`.
  - `test_zero_rho_matches_default`.
  - `test_selection_is_invariant_to_candidate_order`: permuting candidates permutes `winner` accordingly when the gap exceeds `1e-8`.
- [ ] **Step 2:** `uv run pytest tests/test_reml_properties.py -q` — expected: pass. If any fail, stop adding tests, fix as described above, re-run.
- [ ] **Step 3:** Commit `test: add REML invariance and edge-case properties`.

---

### Task 8: Statistical behaviour

**Files:**
- Create: `tests/test_reml_statistics.py`
- Modify: `tests/reml_oracles.py`

Tests only. Monte Carlo replicates are **features**: one `simulate(..., n_features=R)` call and one `fit_reml` call per scenario, so the light suite stays fast. Every test takes `replicates` from a fixture: 2,000 by default, 50,000 under `@pytest.mark.slow` variants (same test body, parametrised). Seeds are fixed. `se` below is the Monte Carlo standard error computed from the replicates.

**Produces (oracle):**
```python
def null_lrt_reference(fixed, random, n_draws, seed) -> np.ndarray
    # Finite-sample null of the restricted LRT for one variance component (Crainiceanu & Ruppert 2004).
    # mu = eigenvalues > tol of K_g' Z_g Z_g' K_g pooled over groups (np.linalg.eigvalsh; K_g = null_space(C_g.T));
    # N = total columns of K_g. Per draw: w ~ N(0,1)^len(mu), r ~ chi2(N - len(mu));
    # f(lam) = N*log((r + sum(w**2/(1+lam*mu)))/N) + sum(log(1+lam*mu));
    # statistic = f(0) - min(f(0), min over lam>0 by scipy.optimize.minimize_scalar on log lam in [-30, 30]).
```

- [ ] **Step 1: Write the tests:**
  - `test_noise_variance_unbiased_at_true_ratio`: 3 groups × 80 scans, 25 fixed columns, 4 random, ratio 1, σ² 2. `profiled_variance` at the true ratio (build terms through `project_fixed`/`reml_terms`): `|mean − 2| < 3·se`. Same statistic rescaled by `N / total_scans` (the ML divisor): `|mean − 2| > 10·se`.
  - `test_estimates_are_consistent`: ratio 1, σ² 1, groups in `(2, 8, 32)`: `|median(ratio) − 1|` and `|mean(noise_variance) − 1|` are non-increasing in groups beyond `2·se`, and at 32 groups are below `0.05` and `0.02`.
  - `test_null_lrt_matches_finite_sample_reference`: ratio 0, 2 groups × 60 scans, 6 fixed, 3 random. Statistic = `criterion(0) − fit.criterion` (zero when `variance_zero`). Reference: `null_lrt_reference(..., n_draws=200_000)`. Assert `|P(stat ≤ 1e-9) − P_ref| < 4·sqrt(p(1−p)/R)` and, for the reference 0.90 and 0.95 quantiles, exceedance rate within `4·sqrt(p(1−p)/R)` of 0.10 and 0.05. Also assert the reference `P(stat = 0)` differs from 0.5 by more than 0.02 for this design (documents why the asymptotic mixture is not used).
  - `test_null_lrt_matches_dense_fits` (`slow` only): the statistic from `dense_fit` on the same simulated data agrees per replicate to `1e-6` on 500 replicates.
  - `test_rho_correction_removes_projection_bias`: 200 scans, reference design with 30 columns, AR(1) noise at ρ ∈ {0, 0.3, 0.6}, 6 groups: `|pooled − ρ| < 0.02`; for ρ = 0.6, `|median(raw) − ρ| > 0.06`.
  - `test_ar1_fit_recovers_parameters`: AR(1) noise ρ = 0.5, ratio 1, σ² 1, 16 groups × 100 scans; with `rho=0.5`: `|median(ratio) − 1| < 0.05`, `|mean(noise_variance) − 1| < 0.02`; with `rho=0.0` on the same data the absolute error of `mean(noise_variance)` or of `median(ratio)` is at least three times larger.
  - `test_selection_rate_rises_with_snr`: 5 candidate designs with pairwise column correlation ≈ 0.8 (shared component plus independent part), data from candidate 0, 4 groups × 60 scans, ratios `(0.02, 0.2, 2)`: correct-selection rate (winner `-1` counts as incorrect) increases between successive ratios by more than `2·se`, and exceeds 0.95 at ratio 2.
- [ ] **Step 2:** `uv run pytest tests/test_reml_statistics.py -q --durations=10` — expected: pass, total under 45 s. Then `uv run pytest tests/test_reml_statistics.py -q --run-slow` once — expected: pass. A threshold that fails is reported to the user with the measured value and Monte Carlo error; it is not adjusted silently.
- [ ] **Step 3:** Commit `test: add Monte Carlo checks of REML statistical behaviour`.

---

### Task 9: Migrate `experiments/hrf_estimation`

**Files:**
- Create: `experiments/hrf_estimation/reml_selection.py`, `experiments/hrf_estimation/test_reml_selection.py`
- Delete: `reml.py`, `reml_scalar.py`, `reml_extensions.py`, `reml_extended_fit.py`, `regularized_fit.py`, `test_reml.py`, `test_reml_scalar.py`, `test_reml_extensions.py` (all in `experiments/hrf_estimation/`)
- Modify: `pilot.py:86-93`, `regularized_noise.py`, `test_regularized_noise.py`, `test_rt_ablation.py`, `results.py`, `test_results_notebook.py`, `session_comparison.py`, `all_methods_results.py`
- Modify (one-line "superseded by docs/superpowers/specs/2026-11-14-reml-module-design.md" note at the top): `reml_model.md`, `reml_validation.md`, `reml_scalar_validation.md`

**Interfaces — Consumes:** `boldtailor.reml`, `boldtailor.ar1`.
**Produces:**
```python
# experiments/hrf_estimation/reml_selection.py
def random_design(run, candidate, rt_mode="amplitude") -> np.ndarray      # column_stack([task, junk]) from build_design
def reference_design(run, rt_mode="amplitude") -> np.ndarray              # nuisance + fast/canonical/slow task+junk (moved from regularized_noise.py:14)
def estimate_noise(runs, rt_mode="amplitude") -> dict
    # keys unchanged: rho, corrected_median, raw_median, n_valid, clipped, training_run_ids, reference
def select_hrf(runs, library, design, rho=0.0) -> HrfEstimate
    # design(run, candidate) -> (scans, q). kernels[i] is None where winner == -1.
    # diagnostics: variance_ratio, sigma2, criterion, at_zero, at_upper, candidate_gap, rho, library_fingerprint
def fit_library_reml(runs, library, settings) -> HrfEstimate              # iid; settings["rt_mode"]
    # adds rt_mode, n_random_effects_per_run (column count of the design built), random_effects
def fit_ar1_reml(runs, library, settings) -> HrfEstimate                  # estimate_noise, then select_hrf; adds diagnostics["noise"]
```
`regularized_noise.fit_grouped(runs, library, rho, verbose=False)` becomes `select_hrf(runs, library, random_design, rho)` with its existing rho validation; `run_rhos` returns `estimate_rho(...).values`; `whitening_diagnostics` uses `boldtailor.ar1.whiten`. `pilot.py` accepts methods `reml` and `remlar1` only; any other `reml*` name raises `ValueError("unknown method")`.

- [ ] **Step 1: Write failing tests** in `test_reml_selection.py` (reuse the `planted_runs` fixture from `test_library_cv.py`):
  - `test_planted_hrf_is_selected_iid` and `..._ar1`.
  - `test_estimate_matches_package_selection`: `select_hrf` kernels equal `library.candidates[i]` for `i` from a direct `select_random_effects_design` call.
  - `test_no_rt_mode_has_four_random_effects` (moved from `test_rt_ablation.py:46-48`).
  - `test_noise_estimate_keys_and_bounds`.
  - `test_removed_variants_are_rejected`: `pilot` dispatch with `"remlcombined"` raises.
  Retarget `test_rt_ablation.py` and `test_regularized_noise.py` imports to `reml_selection`; replace their dense-covariance assertions on deleted internals (`prepare_terms`, `criterion_grid`, `prepare_common`, `component_terms`, `fit_scalar_components`) with assertions on `select_hrf` output against `tests.reml_oracles.dense_fit` for the winning candidate.
- [ ] **Step 2:** `uv run pytest experiments/hrf_estimation -q` — expected: failures only in the files just edited (import errors). Commit `test: specify hrf_estimation REML adapters on the package module`.
- [ ] **Step 3:** Implement the adapter; edit `pilot.py` and `regularized_noise.py`; delete the eight files.
- [ ] **Step 4:** Remove the multi-component sections:
  - `results.py`: every cell from the `REML_METHODS = [...]` cell through the "Fit diagnostics and comparison status" cells that read `reml-final-session*`, `reports/reml_validation_summary.tsv` or `reports/reml_validation_diagnostics.tsv`, and the markdown that introduces them. Delete `test_notebook_includes_verified_reml_validation` from `test_results_notebook.py`.
  - `session_comparison.py`: item 3 of the overview list, the "Additional REML variants compared in sessions 10 and 11" section, the `reml`/`reml_methods` frames read from `two_session_reml_comparison.tsv` and the figures built from them. Adjust its notebook test if it asserts on those figures.
  - `all_methods_results.py:184-186`: reword the two bullets to say the multi-component variants were removed.
  Leave everything under `reports/` and `outputs/` untouched. Do not execute notebooks.
- [ ] **Step 5:** `uv run pytest experiments/hrf_estimation -q` — expected: all pass. `grep -rnE "remlcomponents|remlshared|remlcombined|reml_extensions|reml_scalar|regularized_fit" experiments --include='*.py'` — expected: only the rejection test. Commit `refactor: move hrf_estimation REML onto boldtailor.reml and drop multi-component variants`.

---

### Task 10: Migrate `experiments/hrf_simulations`

**Files:**
- Modify: `experiments/hrf_simulations/fitting.py`, `calibration.py:9-12,23-56,157-167`, `runner.py:40-54`, `test_fitting.py`, `test_calibration.py`, `README.md` ("What is being fitted" section)

**Interfaces — Consumes:** `boldtailor.reml.select_random_effects_design`, `RemlGroup`; `boldtailor.ar1.estimate_rho`; `experiments.hrf_estimation.regularized_noise.shrink_rho` (unchanged).
**Produces (signatures unchanged for `runner.py`):**
```python
def estimate_rho(runs, cache=None) -> float      # boldtailor.ar1.estimate_rho(signals, cached reference designs).pooled
def fit(runs, library, model, convention, noise_mode, oracle_rho=None, cache=None,
        settings=None, design_builder=task_design) -> HrfEstimate
```
`fit` keeps `DesignCache`, `random_scales` and the diagnostics keys `rho, oracle, model, convention, noise_mode, training_run_ids, library_fingerprint, criterion, variance_ratio, sigma2, at_zero, at_upper, random_scales` (each wrapped in a one-element list as today, except the first seven). Candidates are a generator yielding, per candidate, `[column_stack([p.task, p.junk]) / scales for p in parts]`; `random_scales` for the winner is recomputed from its cached parts. `_terms`, `_candidate_fit`, `_reference` and the settings `grid_size`, `iterations`, `variance_bound`, `scalar_grid_size` are removed. `calibration._reference_fit` gets the corrected per-vertex rho from `estimate_rho([run.signals], [design]).values[0]`. `runner.implementation_fingerprint` lists `reml_selection.py` in place of `reml.py`, `reml_scalar.py`, `reml_extensions.py`.

- [ ] **Step 1: Update tests first** in `test_fitting.py`:
  - Delete `test_iid_adapter_matches_original_solver` (the original is gone; Task 5 oracles replace it).
  - Add `test_iid_and_ar1_paths_agree_at_zero_rho`: `fit(..., "iid")` and `fit(..., "ar1", oracle_rho=0.0)` return the same kernel and equal `criterion`, `variance_ratio`, `sigma2` to `rtol=1e-10`. This is the property the spec's anti-goal demands and the old code violated.
  - Add `test_unit_rms_scaling_leaves_selection_and_fit_variance_consistent`: with `regressor_scaling="unit_rms"`, `variance_ratio` is finite and the kernel matches the native-scaling kernel on a high-SNR planted case.
  - Keep every other test as is.
- [ ] **Step 2:** `uv run pytest experiments/hrf_simulations -q` — expected: the new tests fail (old code differs between paths) or error on import. Commit `test: require identical iid and AR(1) solver paths in simulations`.
- [ ] **Step 3:** Implement the changes; update the README paragraph that says the original solvers "are reused unchanged" to describe the package module and the normalised whitening.
- [ ] **Step 4:** `uv run pytest experiments/hrf_simulations -q` and `uv run python -m experiments.hrf_simulations.cli smoke --output "$(mktemp -d)/smoke"` — expected: tests pass; smoke prints `Synthetic smoke complete`. Commit `refactor: run simulations on boldtailor.reml`.

---

### Task 11: Documentation and full verification

**Files:**
- Modify: `docs/development.md` (code-organisation table row; new "REML conventions" subsection under Numerical conventions), `docs/api.md` (entries for `boldtailor.reml` and `boldtailor.ar1`), `experiments/hrf_estimation/README.md`

- [ ] **Step 1:** Write the docs. The conventions subsection states: the model; that `σ²` is the marginal noise variance and whitening is normalised; that criteria are comparable across random designs only at fixed `C` and `ρ`; status values and the `-1` winner; the single rank tolerance; which oracles validate it and how to run `--run-slow`.
- [ ] **Step 2:** `uv run pytest -q` (expected: pass; note REML test time from `--durations=15`, must total under 60 s), `uv run pytest -q experiments/hrf_estimation experiments/hrf_simulations` (expected: pass), `uv run pytest -q tests/test_reml_statistics.py --run-slow` (expected: pass), `uv run black --check src tests experiments/hrf_estimation/reml_selection.py experiments/hrf_simulations/fitting.py`, and whichever existing test guards doc/API consistency (`uv run pytest -q tests/test_docs_mention_cli.py tests/test_distribution.py`).
- [ ] **Step 3:** Commit `docs: document the REML module and its conventions`. Report to the user: test counts, slow-suite timing, any Monte Carlo threshold that needed discussion, and the list of reports that now need rerunning (`hrf_simulations` reports; `hrf_estimation` REML/AR(1) outputs).
