# Response to the follow-up assessment of the REML review response

Date: 2026-11-15  
Follow-up: [`review-2026-11-15-reml-followup.md`](review-2026-11-15-reml-followup.md), of commit `87affe6`  
Fixes: commits `a59b32c` to `3ed1d2c` on branch `reml-review-followup` (tests were committed before each implementation)

Both follow-up findings are accepted and fixed. F1 was a regression introduced by the first R3 fix. The two saved probes in `docs/reml-review-2026-11-15-reproducers/test_response_projection.py` failed on `87affe6` as reported and pass now. The open items from the assessment table (wide-design oracle and benchmark, peak memory, upper-bound selection) are also addressed below. R6 remains partial.

## Summary

| Item | Outcome | Commits |
| --- | --- | --- |
| F1: conditioning-weighted floor rejects exactly computed residuals | Fixed: fixed-design columns are normalised before the basis; residuals that cannot be certified get a new `unresolved` status instead of `no_signal` | `a59b32c`, `f900618` |
| F2: stationary-point guarantee not enforced | Fixed: grid fallbacks are flagged `unrefined`; the guarantee is stated only for optima that are not flagged | `715c8f5`, `02284ea` |
| R4: wide-design oracle comparison and benchmark | Added | `7ecd4e1` |
| R5: peak-memory measurement | Measured and documented | `3ed1d2c` |
| Upper-bound selection | `RemlSelection.any_at_upper` marks features with a truncated candidate | `715c8f5`, `02284ea` |
| R6 | Unchanged: still limited to REML and AR(1) | — |

## F1. The floor rejected exactly computed residuals

**The reviewer's analysis is correct.** The first fix weighted the floor by `s_max / s_i` of the fixed design as given. That ratio grows with unequal column units even when the span is computed exactly, and a bound on the error was being used as if it were a measurement of it. The first response's claim that the floor "restores invariance over the resolvable range" was too strong, as was the matching sentence in `docs/development.md`; both the code and that sentence are corrected.

The problem was not confined to the contrived case. On a design with an intercept, an unnormalised cubic drift and small-unit motion columns, the amplification was `2.3e8`, and unit-variance noise on a mean of `1e4` sat only 137 times above the floor.

**What was done.**

- **Column normalisation.** `_numerics.conditioned_basis` scales each column of a projected-out design to unit norm before the SVD. The span is unchanged, so the fitted model is unchanged; REML and AR(1) report no fixed-effect coefficients whose units could be affected. The amplification then reflects collinearity between columns and not their units. On the drift design it falls from `2.3e8` to 87, and on the reviewer's case to 1.
- **Rank semantics.** Rank is judged on the normalised columns, and only exactly zero columns are dropped for their size. Restricted dof, and therefore the fit, no longer depends on the units of a fixed-design column. This is a deliberate behaviour change: a column of rounding residue now counts as a direction and costs a degree of freedom, where it was previously dropped; a legitimate regressor in very small units is no longer dropped silently, which was the worse error. Both are documented.
- **Two floors.** `project_out` returns a rounding floor, `8 · n · eps · ‖y‖`, and an uncertainty floor, the same multiple of the norm with each basis coordinate amplified by `s_max / s_i` of the normalised design.
- **New status.** Energy at or below the rounding floor is `no_signal`. Energy between the floors is `unresolved`, with every estimate NaN, `ratio` included. Above the uncertainty floor the feature is fitted. `unresolved` can now arise only when fixed-design columns are nearly collinear and the signal loads on the direction they barely determine. In a selection such a feature has winner `-1` and a NaN gap.
- **AR(1).** The lag ratio is NaN at or below the uncertainty floor. `ar1` has no status channel, so the two cases are not distinguished there; this is documented.

**Recalibration.** With normalised columns:

- When ill-conditioning comes only from column units (condition numbers to `1e12`, ranks from 1 to n − 1, n from 10 to 1000, signal on any column), exact-span signals leave at most about 1.2 times `n · eps · ‖y‖`. The rounding floor alone is therefore enough for these designs.
- When columns are genuinely nearly collinear (gaps of `1e-3` to `1e-9`), exact-span signals leave up to `1.7e5` times that scale and under 0.04 times the amplified one. The weight is still needed, and only there.

The factor stays at 8. `tests/test_numerics.py` pins both cases with a fourfold margin in norm.

**Tests.** The two reviewer probes, moved into `tests/`; shift invariance on the drift design at scales 1, `1e4` and `1e8`, for REML and AR(1); the fit and the lag ratio unchanged when fixed columns are rescaled by factors from `1e-20` to `1e15`; a column in units of `1e-30` still costing a degree of freedom; an `unresolved` case (a residual of `1e-8` beside an order-one component along the weak direction of a collinear design) through both `fit_reml` and `select_random_design`; the same residual without that component staying resolved; and unresolved features leaving the other features' results bit-identical.

**What remains.** The uncertainty floor is still a bound. An adversarial, exactly computable near-collinear design can produce a residual that is real yet below it; the difference from before is that it is now reported as `unresolved` and not as `no_signal`. A measured, per-column a-posteriori error estimate would be sharper for such exact cases. I did not implement it, because for dense designs it reduces to the same weighting.

## F2. Stationarity guarantee

**The reviewer is correct** that the code could return a grid point with status `ok` while the documentation promised a stationary point without qualification.

**What was done.** I took the first of the two options offered: check and expose.

- `RatioOptimum` and `RemlFit` gain a per-feature boolean, `unrefined`, true when a positive ratio is a grid point for which no slope root was accepted. `at_upper` is the special case at the upper edge.
- The documented guarantee is now conditional: a positive ratio that is not `unrefined` is a stationary point to near machine precision. An unrefined fit keeps status `ok` and need not be stationary.
- Status and selection eligibility are unchanged for unrefined fits.

**Tests.** A coarse grid (`grid_size=3`) on a spread spectrum produces an interior fallback whose slope is about 2, and it is flagged; the default grid refines the same problem to a stationary point; upper-edge fits are flagged; boundary and unidentifiable optima are not; and across the oracle and property-style collections every positive, unflagged optimum has `|slope| ≤ 1e-9 · dof`.

**Evidence, with its limit.** At the default grid, 1,600 adversarial spectra produced no interior fallback. That supports the flag being rare, not its absence, and the documentation says so. Adaptive subdivision is still not implemented, and the limitation of two stationary points inside one cell stands as documented.

## Items from the assessment table

**R4, wide designs.** A wide-design fit (80 random columns on 30 and 34 scans, at `rho` 0 and 0.4) now matches the dense oracle. Timing one fit with 150 scans, against the originally reviewed code:

| Random columns | Reviewed code | Now |
| --- | --- | --- |
| 30 | 0.001 s | 0.002 s |
| 600 | 0.024 s | 0.011 s |
| 3,000 | 1.11 s | 0.032 s |
| 12,000 | 83.5 s | 0.108 s |

The estimated ratios agree in every row.

**R5, peak memory.** Measured with `tracemalloc` on 100 pooled directions by 30,000 features (22.9 MiB of pooled statistics held throughout):

| Code and budget | Peak extra allocation in the search | Time |
| --- | --- | --- |
| Reviewed code | 168 MiB | 1.7 s |
| Now, 1 MiB | 31 MiB | 2.3 s |
| Now, 8 MiB (default) | 78 MiB | 2.1 s |
| Now, 64 MiB | 221 MiB | 2.2 s |

The peak is about ten times the budget, because roughly ten temporaries of that size are alive at once. The documentation previously said "a few"; it now says about ten and gives these numbers. The budget is a per-array bound, as the assessment describes it, and not a bound on the working set. I did not change it to budget the working set.

**Upper-bound selection.** The assessment's point is fair: `at_upper` on the winning fit does not reveal a losing candidate whose truncated criterion might have won at its true optimum. `RemlSelection.any_at_upper` now marks every feature for which any candidate, winner or not, was truncated, and the documentation calls those selections provisional. The truncation itself is not resolved, and truncated candidates still compete.

**R6.** No further consolidation. `_single_trial_fit`, `_fractional_ridge`, `_hrf_cv` and `_conventional` keep their own helpers, and they do not get the column normalisation.

## Contract changes in this round

- `RemlFit.status` can be `unresolved`; `RemlFit` has a new `unrefined` field; `RemlSelection` has a new `any_at_upper` field.
- `RhoReference.amplification` is now defined on unit-norm columns.
- Restricted dof and all results are invariant to fixed-design column units, with the rank consequences described under F1.
- `experiments/hrf_simulations/fitting.py` was left unchanged. It skips fit diagnostics for `no_signal` only, so an `unresolved` feature records NaN diagnostics there.

## Verification

- REML, AR(1) and numerics tests: 715 passed, 9 skipped, 2 expected failures.
- All five saved reviewer probes (three original, two follow-up) pass.
- The four REML and AR(1) experiment test files that were run: 30 passed.
- Against the originally reviewed code on a well-conditioned selection (3,000 features, 8 candidates, 4 groups of 150 scans): identical winners, ratios equal to `1e-13` relative.
- Full default suite: 2372 passed, 1 failed, 9 skipped, 2 expected failures. The failure is the same `test_notebook_kernel_is_not_a_runtime_requirement` (`ipykernel` in `pyproject.toml`), untouched by this work.
