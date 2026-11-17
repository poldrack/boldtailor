# Readable Trial Estimation and CV Implementation Plan

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the user's selected native execution method. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make trial estimation and candidate scoring readable without changing the statistical objective, coefficient units, or validation isolation.

**Architecture:** Replace positional preparation tuples and coordinated generators with small explicit prepared-fit objects. Keep normalized ridge and raw-basis fractional ridge visibly distinct, sharing only their nuisance projection and scale-aware design checks. Evaluate candidates on demand and reuse the fractional decomposition for its fixed OLS validation target.

**Tech Stack:** Python >=3.12, NumPy, SciPy, Nilearn, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-28-scientific-readability-design.md`

**Status:** Implemented, verified, and independently reviewed with no required findings.

## Global Constraints

- Use `uv run` for Python, pytest, and formatting; keep every `__init__.py` empty.
- Write, observe, and commit failing tests before implementation.
- Preserve the committed branch's numerical behavior; do not apply stash `c4cfbec`.
- Preserve raw fractional coefficient norms, normalized shared-alpha penalties, and unpenalized nuisance coefficients.
- Preserve training-only HRF selection, within-run encoding, fixed OLS targets for fractional CV, candidate-specific targets for shared-alpha CV, and the legacy absolute encoding mode.
- Preserve feature/run order, undefined-feature masks, error context, tolerances, tie handling, and pooled SSE/SST definitions.
- Keep generator entry points used by validation scripts as thin adapters; remove their coordination from CV itself.
- No new dependencies, generic estimator framework, or public result-container redesign in this stage.

## Review Focus

1. Evaluate candidates out of order and repeatedly without retaining old beta arrays (Tasks 1–3).
2. Preserve identifiability decisions for differently scaled columns and nearly collinear designs (Tasks 1–2).
3. Keep constant and excluded features in their original positions with the same NaN behavior (Tasks 1–3).
4. Reuse prepared state for the fixed OLS target without admitting held-out data to training (Task 3).
5. Keep independent reference calculations independent when moving shared test helpers (Task 4).

## Files and interfaces

| File | Responsibility after this stage |
| --- | --- |
| `_single_trial_fit.py` | Named projected-design state, normalized prepared solver, conventional trial fit and adapter |
| `_fractional_ridge.py` | Raw-basis prepared solver, norm-to-alpha solution, full fractional fit and adapter |
| `_ridge_cv.py` | Explicit per-run/group candidate evaluation and visible fold scoring |
| `tests/test_single_trial.py` | Normalized oracle, scaling, conditioning, memory ownership |
| `tests/test_fractional_ridge.py` | Raw fractional oracle, arbitrary evaluation, undefined features |
| `tests/test_ridge_cv.py`, `tests/test_fractional_cv.py` | Whole-CV equivalence and decomposition reuse |
| `tests/test_within_run_cv_isolation.py` | Held-out isolation |
| `tests/oracles.py`, `tests/conftest.py`, `tests/__init__.py` | Shared reference calculations, fixtures, empty package marker |
| `docs/development.md`, validation note | Explain solver flow, dimensions, and scientific equivalence evidence |

New internal interfaces:

```text
# In _single_trial_fit.py:
prepare_trial_betas(x, nuisance, signals) -> PreparedTrialBetas
PreparedTrialBetas.betas_at(alpha: float) -> np.ndarray

# In _fractional_ridge.py:
prepare_fraction_betas(x, nuisance, signals) -> PreparedFractionBetas
PreparedFractionBetas.betas_at(fraction: float) -> np.ndarray
PreparedFractionBetas.solve(fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]

# In _ridge_cv.py:
prepare_run_beta_path(data, run_index, prepared, ids, label, *, fractional=False) -> RunBetaPath
RunBetaPath.betas_at(value: float) -> np.ndarray
```

Here `betas_at` always returns `(n_trials, n_features)` arrays. Fractional
`solve` accepts a validated per-feature fraction vector and also returns its
implied-alpha vector. Each evaluation returns a fresh output array.

## Task 1: Name projected-design state and normalize candidate evaluation

**Files:** `_single_trial_fit.py`, `_fractional_ridge.py` (consume named state),
`tests/test_single_trial.py`.

- [x] Add a regression using the existing `problem` fixture. Its independent
  augmented least-squares calculation also pins nonuniform column scaling:

```python
def test_prepared_trial_betas_are_order_independent(problem):
    import gc
    import weakref
    from boldtailor import _single_trial_fit as solver

    data, x, nuisance = problem
    x = x * [1.0, 2.0, 3.0, 4.0]
    n, y = nuisance[0], data.signals[0]
    prepared = solver.prepare_trial_betas(x, n, y)
    scale = np.linalg.norm(x - n @ np.linalg.lstsq(n, x, rcond=None)[0], axis=0)
    for alpha in (2.0, 0.0, 0.1, 2.0):
        penalty = np.column_stack([
            np.diag(np.sqrt(alpha) * scale),
            np.zeros((x.shape[1], n.shape[1])),
        ])
        design = np.vstack([np.column_stack([x, n]), penalty])
        target = np.vstack([y, np.zeros((x.shape[1], y.shape[1]))])
        expected = np.linalg.lstsq(design, target, rcond=None)[0][:x.shape[1]]
        actual = prepared.betas_at(alpha)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
        reference = weakref.ref(actual)
        del actual
        gc.collect()
        assert reference() is None
```

Parameterize the existing conditioning, constant-feature, and solver-rejection
tests over the old adapter and new prepared interface. Keep their expected
values and tolerances. Run `uv run pytest tests/test_single_trial.py -q -W error`;
observe missing-interface failures and commit tests.

- [x] Add a `ProjectedTrialDesign` dataclass with fields `nuisance_basis`,
  `column_scale`, `left_vectors`, `singular_values`, `right_vectors`, and
  `diagnostics`. Have `_project_design` return this object instead of six tuple
  entries, retaining the existing projection, rank thresholds, and positive-dof
  checks. Update every caller, including fractional preparation, by field name.

  Add `PreparedTrialBetas` storing the projected design, projected signal
  coordinates for varying features, feature mask, and output shape. Its solver:

```python
def betas_at(self, alpha):
    alpha = validate_alpha(alpha)
    d = self.design
    s = d.singular_values
    attenuation = 1 / s if alpha == 0 else s / (s * s + alpha)
    coefficients = d.right_vectors.T @ (attenuation[:, None] * self.coordinates)
    betas = np.full(self.shape, np.nan)
    betas[:, self.varying] = coefficients / d.column_scale[:, None]
    return betas
```

  `prepare_trial_betas` performs the existing projection once, storing
  `coordinates = left_vectors.T @ projected_signals[:, varying]`. Use the
  normalized SVD for alpha zero only if the existing OLS and conditioning
  regressions pass unchanged; investigate numerical differences rather than
  loosening tolerances. Convert `trial_beta_path` into a validated-grid adapter
  that prepares once and yields `(alpha, prepared.betas_at(alpha))`.
  Keep full-fit diagnostics and nuisance recovery visible in `fit_trial_run`;
  use the same prepared solver for its beta calculation to remove duplicate
  formula paths. Do not change zero SSE handling for constant features in this
  baseline; the preserved scientific remediation is a separate change.

- [x] Run normalized, fractional, selected-HRF, and CV regressions. The existing
  factorization-count test still observes `_project_design`; preserve it.
  Format changed Python, run `git diff --check`, and commit the implementation.

## Task 2: Explicit raw-basis fractional preparation

**Files:** `_fractional_ridge.py`, `tests/test_fractional_ridge.py`.

- [x] Add this test using the existing `regression` fixture and independent
  `oracle` (its import changes in Task 4):

```python
def test_prepared_fraction_betas_reuse_state_in_any_order(regression, monkeypatch):
    from boldtailor import _fractional_ridge as solver

    x, n, y = regression
    prepared = solver.prepare_fraction_betas(x, n, y)
    expected = {
        fraction: np.column_stack([
            oracle(x, n, y[:, feature], fraction)[0]
            for feature in range(y.shape[1])
        ])
        for fraction in (0.2, 1.0, 0.8)
    }

    def refactorization(*args, **kwargs):
        raise AssertionError("candidate evaluation must reuse prepared SVD")

    monkeypatch.setattr(np.linalg, "svd", refactorization)
    for fraction in (0.2, 1.0, 0.8, 0.2):
        np.testing.assert_allclose(prepared.betas_at(fraction), expected[fraction], atol=1e-9)
```

  Parameterize the existing undefined-feature, ill-conditioning, and invalid-grid
  tests to exercise the prepared object too. Add the same weak-reference output
  ownership check as Task 1. Run RED and commit tests before implementation.

- [x] Replace `_prepare`'s positional tuple with `PreparedFractionBetas` fields:
  `trial_design`, `nuisance`, `signals`, `projected_signals`, `singular_values`,
  `right_vectors`, `ols_coordinates`, `valid`, `diagnostics`.
  Reuse `_project_design` for scale-aware validation, then perform the raw SVD
  explicitly. Document why those bases differ. Remove the all-ones scale vector
  and its divisions; keep `_alphas`' existing bracket/bisection algorithm.

```python
def betas_at(self, fraction):
    fractions = fraction_map(fraction, self.signals.shape[1])
    return self.solve(fractions)[0]
```

  Move `_solve`'s current calculation into `solve`, using named fields and
  `betas[:, valid] = right_vectors.T @ (ols_coordinates[:, valid] * attenuation)`.
  Keep implied alphas, validity tolerances, and NaN exclusions unchanged.
  `fraction_beta_path` becomes a validated-grid adapter to this object;
  `fit_fraction_run` uses its fields for unchanged nuisance and SSE calculations.

- [x] Run `uv run pytest tests/test_fractional_ridge.py tests/test_fractional_cv.py
  tests/test_fractional_ridge_ablation.py tests/test_fractional_ablation_simulation.py
  -q -W error`, format, check the diff, and commit after GREEN.

## Task 3: Make fold scoring independent of iterator order

**Files:** `_ridge_cv.py`, `tests/test_ridge_cv.py`, `tests/test_fractional_cv.py`.

- [x] Add tests of `prepare_run_beta_path` using canonical designs from the
  `ridge_problem` fixture. For normalized mode compare to `augmented_beta`;
  for fractional mode compare each feature to the augmented/root-finding oracle.
  Evaluate grids in forward, reverse, and repeated order, keeping the constant
  feature at NaN. Add a grouped-HRF case using the fixture's library and prepared
  runs; preserve original feature order and exclude ID `-1`.

  Add an integration check that fixed OLS validation targets require no second
  run preparation. Count calls to the new builder while executing fractional
  CV, and verify the full output against the existing independent CV reference:

```python
def test_fraction_cv_prepares_each_run_once_per_fold(ridge_problem, monkeypatch):
    from boldtailor import _ridge_cv
    from boldtailor.fractional_ridge import score_fraction_candidates

    data, predictors, _ = ridge_problem
    calls = []
    original = _ridge_cv.prepare_run_beta_path

    def observe(data, run_index, *args, **kwargs):
        calls.append(run_index)
        return original(data, run_index, *args, **kwargs)

    monkeypatch.setattr(_ridge_cv, "prepare_run_beta_path", observe)
    score_fraction_candidates(data, predictors, fractions=[1.0, 0.5])
    assert calls == list(range(data.n_runs)) * data.n_runs
```

  Keep the existing oracle, fixed-target-grid, and held-out isolation tests as
  the scientific gates; call counting alone is insufficient. Run RED and commit.

- [x] Replace `_run_beta_path` with `RunBetaPath` storing shape and groups of
  `(feature_indices, prepared_solver)`. Prepare each group's solver using Task 1
  or Task 2 according to `fractional`. Its only evaluation method is:

```python
def betas_at(self, value):
    betas = np.full(self.shape, np.nan)
    for features, solver in self.groups:
        betas[:, features] = solver.betas_at(value)
    return betas
```

  In `_score_fold`, prepare all runs once. For fractional mode, obtain
  `fixed_target = paths[test].betas_at(1.0)` once before the candidate loop.
  For each candidate, evaluate training paths and use the fixed validation
  target; do not compute a candidate-regularized validation result that is
  immediately discarded. For normalized mode, evaluate all runs at that alpha.
  Retain fold/run context in exceptions from both preparation and evaluation.
  Keep candidate aggregation, predictor centering, and provenance unchanged.

- [x] Run both CV suites, `test_within_run_cv_isolation.py`, and all NSD ridge/
  fractional workflow tests. Confirm serial/parallel and block-size equivalence.
  Run the complete default suite and formatter before committing.

## Task 4: Give shared test references a stable home

**Files:** create empty `tests/__init__.py`, create `tests/oracles.py`; update
`tests/conftest.py`, `test_ridge_cv.py`, `test_selected_hrf_fit.py`,
`test_fractional_ridge.py`, `test_fractional_cv.py`, and
`test_within_run_cv_isolation.py`.

- [x] Move `ridge_problem` and `selected_fixture` into `tests/conftest.py`, with
  their existing fixture decorators and required imports. Move `oracle` from
  `test_fractional_ridge.py` into `tests/oracles.py` as `fractional_beta_oracle`;
  move `subset` from `test_ridge_cv.py` there as `subset_runs`. Keep their bodies
  unchanged so the reference calculation remains independent.
  Replace cross-test-module imports with:

```python
from tests.oracles import fractional_beta_oracle as oracle
from tests.oracles import subset_runs as subset
```

  Remove imported fixtures; pytest supplies them from conftest. This is a pure
  test relocation, not product implementation. No artificial failing test is
  needed to test the test files. Keep the package initializer empty.

- [x] Run the affected numerical/CV suites, then full default pytest. Confirm
  no `from test_...` imports remain in `tests/`. Do not remove `pythonpath` from
  pytest config until example imports have moved into the package in Stage 5.
  Document array dimensions, raw versus normalized bases, and the candidate
  evaluation sequence in `docs/development.md`. Record RED/GREEN commands and
  full-suite results in `docs/validation/scientific-readability-numerics-2026-09-28.md`.
  Commit the relocation and documentation separately from scientific code.

## Completion

Require the complete default suite, formatting, and installed-wheel smoke.
Request one independent whole-stage review. Report changed control flow and
removed duplication with examples, and any numerical discrepancies explicitly.
Do not silently change tolerances or mark the broader roadmap complete.
