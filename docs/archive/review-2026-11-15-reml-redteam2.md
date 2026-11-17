# Second red-team review and fixes for `boldtailor.reml`

Date: 2026-11-15  
Reviewed HEAD: `725dbc4`  
Fixes: commits `5bb1712` (failing tests) and `2e4a3c5` (implementation) on `main`

An independent red-team review of `boldtailor.reml` and its supporting modules (`_reml_inputs.py`, `_reml_terms.py`, `_reml_criterion.py`, `_reml_search.py`, and `_numerics.py`) confirmed that the statistical model, profiled restricted likelihood, and prior review fixes are sound, and identified three remaining issues that were fixed and verified.

## Summary of findings and fixes

| ID | Category | Finding | Resolution |
| --- | --- | --- | --- |
| S1 | Statistical / Search | Exact zero-slope grid points were flagged `unrefined=True` | Fixed in `_reml_search._best_positive` |
| S2 | Numerical | `_spectral_norm` and `project_out` overflowed/underflowed on extreme scales | Fixed via peak scaling in `_reml_terms._spectral_norm` and `_numerics.project_out` |
| A2 | Architectural | `_fit_values` pooled terms twice; `_live_subset` recomputed `conditioned_basis` | Fixed via `minimize_ratio(PooledTerms)` and `_reml_terms.reproject_signals` |

## Details

### S1. Exact zero-slope grid points flagged `unrefined`
- **Location:** `src/boldtailor/_reml_search.py` (`_crossing_roots`, `_best_positive`).
- **Issue:** `_crossing_roots` detects crossings with `(slopes[:-1] < 0) & (slopes[1:] > 0)`. When a stationary point falls exactly on a grid point (`slope == 0.0`), neither adjacent cell brackets a strict sign change, so no root was accepted and `_best_positive` marked the grid point `unrefined=True` despite it being an exact stationary point.
- **Fix:** `_best_positive` now checks `slopes == 0.0` at the winning grid index (`grid_stationary`) and treats it as `refined`.

### S2. Scale overflow and underflow in `_spectral_norm` and `project_out`
- **Location:** `src/boldtailor/_reml_terms.py` (`_spectral_norm`) and `src/boldtailor/_numerics.py` (`project_out`).
- **Issue:** Unlike `_unit_columns`, `_spectral_norm` formed `Z.T @ Z` directly (overflowing to `LinAlgError` at `1e200` and underflowing to `0.0` at `1e-200`), and `project_out` squared `signals` and `coordinates` directly before applying the relative factor (overflowing `uncertainty_floor` to `inf` for large signals with high amplification, e.g. `1e150` with `amplification = 1e6`).
- **Fix:** `_spectral_norm` now normalizes by `peak = np.abs(matrix).max()` before forming `_smaller_gram`, and `project_out` normalizes `signals` and `coordinates` by per-feature peak magnitude before squaring.

### A2. Redundant pooling and basis recomputation in `reml.py`
- **Location:** `src/boldtailor/reml.py` (`_fit_values`, `_live_subset`), `src/boldtailor/_reml_search.py` (`minimize_ratio`), `src/boldtailor/_reml_terms.py` (`reproject_signals`).
- **Issue:** `_fit_values` called `minimize_ratio(terms)` and then `profiled_variance(optimum.ratio, terms)`, pooling `terms` twice per subset and candidate. `_live_subset` re-ran `project_fixed` when a `rho` subset mixed live and dead features, recomputing `conditioned_basis(whiten(fixed_design, rho))` from scratch.
- **Fix:** `minimize_ratio` now accepts `PooledTerms` directly so `_fit_values` pools once and reuses `pooled` for `pooled_quadratic`. Added `reproject_signals` in `_reml_terms.py` so `_live_subset` reuses the already-computed fixed-design basis.

## Open informational / deferred items
- **Documented search and selection caveats:** Two stationary points inside one grid cell are not detected without adaptive subdivision; near-confounded designs are not flagged; `RemlSelection` exposes `any_at_upper` but not `any_unrefined` for losing interior unrefined candidates.
- **Package-wide numerical consolidation (R6):** `_single_trial_fit.py`, `_hrf_cv.py`, and `_conventional.py` still retain their own basis/rank helpers outside `_numerics.py`.

## Verification
- Failing tests committed before implementation (`5bb1712`) in `tests/test_reml_search.py`, `tests/test_reml_terms.py`, `tests/test_numerics.py`, and `tests/test_reml.py`.
- Implementation committed (`2e4a3c5`).
- `uv run pytest -q tests/test_reml*.py tests/test_ar1*.py tests/test_numerics.py`: **690 passed, 9 skipped, 2 xfailed**.
- `uv run pytest -q docs/reml-review-2026-11-15-reproducers/`: **5 passed**.
