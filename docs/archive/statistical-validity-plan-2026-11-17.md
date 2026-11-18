# Implementation Plan: Statistical Validity Fixes & Property-Based Test Expansion

Following `AGENTS.md` and `docs/testing.md`, each phase below follows **RED–GREEN–Refactor** (commit failing tests before implementation), adds a short 1–2 sentence purpose docstring to every new test function, and uses `uv run pytest` for verification.

---

## Phase 1: Fix Statistical & Numerical Validity Issues

### 1.1 Scale-Invariant Rank & Estimability in Conventional GLM
- **Files**: `src/boldtailor/_conventional.py`, `tests/test_multirun.py` (or a new `tests/test_conventional_properties.py`)
- **RED (Tests first)**:
  - Add a Hypothesis property-based test verifying that scaling confound columns across extreme decades (e.g., $10^{-6}$ to $10^{8}$) leaves design rank, `residual_dof`, contrast $t$-statistics, $R^2$, and `task_delta_r2` invariant.
  - Add a unit test with a full-rank design whose confound column has scale `1e9` and task column has scale `1e-4`, asserting that `_full_rank_basis` returns `None` (full rank) and `_validate_estimable` accepts the task contrast.
- **GREEN (Implementation)**:
  - In `src/boldtailor/_conventional.py`:
    - Update `_full_rank_basis` to scale nonzero columns to unit norm (matching `boldtailor._numerics.conditioned_basis` / `_unit_columns`) before SVD and rank thresholding, while still returning the orthonormal row-space basis of the **original** unscaled design (or computing the basis in original coordinates once rank is determined, so contrast mapping `_in_fitted_basis` remains exact).
    - Update `_validate_estimable` and `_validate_residual_dof` to judge rank/estimability on column-normalized designs (e.g., via `conditioned_basis` or scaled QR/SVD) so column units never distort rank or estimability.
- **Verify**: `uv run pytest tests/test_fit.py tests/test_multirun.py tests/test_hrf_glm.py tests/test_fit_diagnostics.py -q -W error`

### 1.2 Denoising Significance Gate: AR(1) Bias Correction & Null Calibration
- **Files**: `src/boldtailor/denoising.py`, `tests/test_denoising_gate.py`
- **RED (Tests first)**:
  - Add a **null-calibration test** under independent **AR(1) noise** ($\rho \in \{0.4, 0.7\}$) for both the noise pool and scoring features (e.g., across multiple seeds / short runs), checking per-feature $F$-test $p$-value uniformity / false-positive rate around nominal $\alpha$.
  - Add a test demonstrating the spatial-correlation behavior of the binomial gate (documenting or testing the chosen mitigation).
- **GREEN (Implementation)**:
  - Evaluate replacing Nilearn's un-corrected `_yule_walker` in `ar1_coefficients` (`src/boldtailor/denoising.py`) with `boldtailor.ar1.estimate_rho` (using each run's full nuisance+task+PC design as the reference) so projection-induced downward bias in $\hat{\rho}$ is removed. *(Note: if Nilearn `run_glm` parity is required for a separate reason, make bias-corrected `estimate_rho` the default or document the trade-off explicitly.)*
  - Document the feature-independence assumption of `stats.binomtest` in `apply_gate` / `SignificanceGate` provenance (or, if desired as a design change, add an effective-sample-size / permutation option—decide before coding).
- **Verify**: `uv run pytest tests/test_denoising*.py -q -W error`

### 1.3 Pooled Within-Run Correlation Degrees of Freedom
- **Files**: `src/boldtailor/diagnostics.py`, `tests/test_diagnostics.py`
- **RED (Tests first)**:
  - Add a test in `tests/test_diagnostics.py` where run 1 has 2 valid trials and run 2 has 1 valid trial (`total_count == 3`, but residual $\text{df} = 3 - 2 = 1$ after within-run centering), asserting `correlate_rt(...)["all"]` is `NaN` rather than $\pm 1.0$.
- **GREEN (Implementation)**:
  - In `src/boldtailor/diagnostics.py`, track the number of contributing runs (or residual degrees of freedom $\sum_r \max(n_r - 1, 0)$) and require residual $\text{df} \ge 2$ before computing pooled Pearson $r$.
- **Verify**: `uv run pytest tests/test_diagnostics.py -q -W error`

### 1.4 Guard Zero-Norm `ols` in `fraction_alphas`
- **Files**: `src/boldtailor/_fractional_ridge.py`, `tests/test_fractional_ridge.py`
- **RED (Tests first)**:
  - Add a unit test calling `fraction_alphas(s, ols, fractions)` directly with an all-zero `ols` column, asserting it returns `0.0` (or raises `ValueError`—pick one consistent convention) without producing `NaN` intermediate weights.
- **GREEN (Implementation)**:
  - In `fraction_alphas` (`src/boldtailor/_fractional_ridge.py`), mask columns with `np.linalg.norm(ols[:, shrink], axis=0) == 0` so they return `0.0` cleanly without `0 / 0` division.
- **Verify**: `uv run pytest tests/test_fractional_ridge.py -q -W error`

---

## Phase 2: Expand Hypothesis Property-Based Testing (PBT)

Add dedicated property-based tests (keeping runtime bounded via `@settings(max_examples=...)` and checking `--durations=20`). Group into focused property test files or alongside existing module tests:

1. **Conventional GLM & $\Delta R^2$** (`tests/test_conventional_properties.py` or `tests/test_multirun.py`):
   - Invariance of $R^2$, contrast $t$-statistics, and `task_delta_r2` to arbitrary invertible linear transformations and column scaling of nuisance regressors.
   - Monotonicity: `full_r2 >= nuisance_r2` (up to `NESTED_OLS_TOLERANCE`) across arbitrary nested designs.
   - Run-order permutation invariance of `aggregate_r2` and combined fixed-effects contrasts.

2. **Mixture Threshold** (`tests/test_mixture_threshold.py`):
   - Shift and positive-scale equivariance: `threshold(a * x + b) ≈ a * threshold(x) + b` for $a > 0$ (within grid resolution).
   - Permutation invariance of `threshold`, `means`, `sds`, and `weights`.
   - `robust_range` invariance to extreme upper/lower outliers.

3. **Denoising CV & Pool PCA** (`tests/test_denoising_pool.py`, `tests/test_denoising_cv.py`):
   - `onoff_r2` and `run_components` invariance to positive signal scaling and invertible nuisance transformations.
   - `choose_count`: `pcstop=1.0` always selects the global argmax of available counts; `max(curve) <= 0` always selects `0`.

4. **HRF CV Selection** (`tests/test_hrf_cv.py`):
   - `loro_scores` invariance to signal scaling, run-order permutations, and orthogonal nuisance transformations.
   - `choose_eligible` tie-breaking (lowest eligible ID within tolerance) and feature-order independence.

5. **Single-Trial, Normalized Ridge & Fractional Ridge** (`tests/test_single_trial.py`, `tests/test_fractional_ridge.py`):
   - Normalized ridge: $\|\hat{\beta}(\alpha_1)\|_2 \le \|\hat{\beta}(\alpha_2)\|_2$ for $\alpha_1 > \alpha_2 \ge 0$; equivariance to signal scaling; invariance to nuisance column scaling/rotation.
   - Fractional ridge: exact norm ratio $\|\hat{\beta}(f)\|_2 / \|\hat{\beta}_{\text{OLS}}\|_2 = f$ across arbitrary ill-conditioned designs and target scales; monotonicity in $f \in (0, 1]$.

6. **Trial Encoding** (`tests/test_trial_encoding.py`):
   - `within_run` mode: exact invariance to arbitrary per-run constant shifts in `betas` and `predictors`.
   - Equivariance of predictions and invariance of $R^2$ to affine scaling of predictors.
   - Exact equivalence to dummy-variable OLS across random unbalanced run sizes and missing-trial NaN masks.

7. **Diagnostics & Reliability** (`tests/test_diagnostics.py`, `tests/test_reliability.py`):
   - `one_sample_t`: invariance of `t` and `p_uncorrected` to positive scaling of `run_betas` and arbitrary splitting/regrouping of trials across runs.
   - `correlate_rt`: invariance to run-specific additive shifts and positive scaling of `beta` and `rt`, and Cauchy–Schwarz bound $|r| \le 1$.

---

## Phase 3: Final Verification

1. **Test documentation check** (ensures all new test functions have purpose docstrings):
   ```bash
   uv run pytest tests/test_test_documentation.py -q
   ```
2. **Integration suite**:
   ```bash
   uv run pytest -m integration -q -W error
   ```
3. **Full default suite with timing review**:
   ```bash
   uv run pytest -q -W error --durations=20
   ```
