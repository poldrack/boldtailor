# Response to the red-team review of `boldtailor.reml`

Date: 2026-11-15  
Review: [`review-2026-11-15-reml.md`](review-2026-11-15-reml.md), of commit `587929c`  
Fixes: commits `1245b49` to `665a2e2` on `main` (13 commits; tests were committed before each implementation)

All three reproduced failures (R1–R3) and both resource findings (R4, R5) are fixed. R6 is addressed for REML and AR(1) only; the other modules are deferred. The related concern about upper-bound fits was left unchanged, for the reasons given below. Each section states what was done, where the fix departs from the review's proposal, and what remains open.

## Summary

| ID | Finding | Outcome | Commits |
| --- | --- | --- | --- |
| R1 | Unidentifiable variance components reported as `ok` | Fixed: new `unidentifiable` status, excluded from selection; zero wins within rounding noise | `76ed4b6`, `37947a9` |
| R2 | Search misses a better positive ratio | Fixed: every slope crossing is refined; grid spacing capped. Not a certified global search | `1245b49`, `ac946e3`, `c800525` |
| R3 | Fixed-effect offset flips a fit to `no_signal` | Fixed jointly with AR(1): conditioning-aware projection floor | `375fe3e`, `43b09eb`, `8b1e907`, `1926758` |
| R4 | Spectral norm scales with the number of random columns | Fixed: smaller Gram matrix | `5e87def`, `92e4bdd` |
| R5 | Documented memory bound not enforced | Fixed: feature batching; documentation states what is covered | `5e87def`, `92e4bdd`, `665a2e2` |
| R6 | Numerical policy duplicated across the package | Partly addressed: shared `_numerics` module used by REML and AR(1); four other modules deferred | `1926758` |
| — | Upper-bound fits admitted to selection | No change | — |

## Verification of the review

The three saved probes in `docs/reml-review-2026-11-15-reproducers/test_redteam.py` were run before any change and failed as the review reports. They pass on the fixed code. Expanded versions live in `tests/test_reml_adversarial.py`.

One correction to the review's account of R2: the grid did not fail to bracket the missed minimum. The slope already changed sign between the grid points `5.0e-7` and `1.0e-6`, on either side of the minimum at `7.13e-7`. The minimum was lost because only the bracket around the grid argmin was refined, and both of those grid values were worse than the value at zero. This changed which fix was needed (see R2).

## R1. Unidentifiable variance components

**What was done.**

- `minimize_ratio` now checks for exact confounding before searching (`_confounded` in `_reml_search.py`). A fit is unidentifiable when the pooled number of retained random directions equals the pooled restricted degrees of freedom and all pooled squared singular values are equal within `4 · dof · eps` relative.
- The first condition accounts for the zero eigenvalues the review asked for: if any restricted degree of freedom has no retained direction, the random covariance is not a multiple of the identity.
- The check is on the pooled spectrum, so it requires one proportionality constant across groups. Groups that are each isotropic at different scales remain identifiable.
- Such fits get status `unidentifiable`, with NaN `ratio`, `noise_variance` and `effect_variance`. `criterion` is the constant value of the flat profile, which equals the value at `λ = 0`.
- In a selection only `ok` candidates compete, so an unidentifiable candidate cannot win. If no other candidate is supported, the winner is `-1` and the fit is the `λ = 0` fit, as before.
- The all-zero random covariance keeps its existing behaviour (`variance_zero`).
- As a separate numerical safeguard, a positive ratio must now beat zero by more than the criterion's rounding noise, `64 · eps · (|F| + dof)`, the slack already used elsewhere in the search.

**Departures from the proposal.**

- No combined-variance diagnostic is exposed. The review offered this as one possible contract; nothing in the package needs it.
- Near-confounded designs are not flagged. They are weak-information cases, not non-identifiable ones, and are reported as `ok` or `variance_zero`. This is documented, but there is no conditioning policy beyond the exact check and the rounding slack. With `Z = I` perturbed by `1e-6` the estimated ratios are still erratic, and the code does not warn about that.

**Tests.** Identity covariance over eight seeds; confounding after fixed effects are projected out; AR(1) analogues at two `rho` values; two groups with a common scale (unidentifiable) and with differing scales (identifiable, estimates near the simulated values); an unsaturated isotropic design (identifiable); a near-confounded design (not flagged); candidate order in both directions; a selection with only an unidentifiable candidate; and a unit test that removing the zero-comparison slack makes a rounding-level improvement win.

## R2. Missed positive-ratio solution

**What was done.**

- The criterion and its slope in `log λ` are evaluated together on the grid. Bisection refines the root inside every cell where the slope turns from negative to positive, and the lowest root competes with the best grid point.
- The grid now gains points with spectral spread, so its spacing never exceeds `log(1e16) / 120 ≈ 0.31` in `log λ`. Previously 121 points covered any span, giving a spacing of `0.69` on the review's counterexample.
- `at_upper` is now set only when the best grid point is the last one and no refined root replaced it.
- The dense test oracle no longer relies on one `minimize_scalar` call. It scans `log λ` and runs a bounded search around every local minimum of the scan.

**Departures from the proposal.**

- No adaptive subdivision of unresolved intervals (item 3). A minimum and a maximum that both fall strictly inside one grid cell leave no sign change and are still missed. The module docstring and `docs/development.md` state this limit.
- No new status for an unrefined grid fallback or unresolved search (item 5). After the change I found no case that would produce one.
- No certified global method (item 6). The actual guarantee is documented instead.

**Tests.** The review's counterexample at three levels (criterion against a dense evaluation, status and ratio, and selection winner), a unit test on the same problem expressed as sufficient statistics, and a grid-spacing test. A further 240 random problems (wide, moderate, clustered and repeated spectra, with heavy-tailed signals) are compared against an independent fine scan. Those 240 passed before the fix as well, so they are guards, not evidence of the bug: in a search over several thousand random spectra the old code failed only at truncated upper-edge cases, and in a family of spans around the counterexample only the reviewer's exact span failed.

## R3. Fixed-effect offsets and `no_signal`

**What was done.**

- The rule `energy > 64 · eps · raw_energy` is gone from both `reml._live` and `_ar1_moments.basis_lag_ratio`. Both now use one floor from the new `boldtailor._numerics.project_out`.
- The floor is in norm units: a residual is treated as rounding when its norm is at most `8 · n · eps` times an error scale. That scale is the norm of the signal with its coordinate on each basis vector amplified by `s_max / s_i` of the projected-out design.
- The error model is that a basis computed from `C` spans the range of `C + E` with `‖E‖ ~ n · eps · ‖C‖`, so signal along a weak direction of `C` leaves proportionally more rounding.
- The factor 8 was set from calibration, not chosen freely. Signals lying exactly in the span left at most 1.0 times the error scale across n = 10 to 1000, ranks from 1 to n − 1, condition numbers up to `1e12`, amplitudes `1e-3` to `1e6` and AR(1) whitening. `tests/test_numerics.py` pins this with a fourfold margin.

**Departure from the plan agreed before implementation.** The first version used the plain scale `n · eps · ‖y‖`. Calibration showed that model is wrong for ill-conditioned, nearly saturated fixed designs, where exact-span residue reached about `1e5` times that scale. The conditioning weight was added for that reason, which is also the review's own caveat that a replacement "must account for dimensions, conditioning, and the actual projection algorithm".

**Consequences.**

- `RhoReference` gains a required `amplification` field, since the AR(1) path needs the conditioning of its basis.
- Some features that previously received NaN `rho` or `no_signal` now receive estimates.
- The old threshold (about `1.2e-7` in amplitude) happened to sit at float32 epsilon, so it masked near-constant voxels read from float32 files. The new float64 floor does not. Masking such features is left to callers and this is documented.

**Not adopted.** Stable centering, re-projection and orthogonal-transform residuals. The floor change alone restores invariance over the resolvable range, and these would not help once the offset has destroyed the residual in storage.

**Tests.** Fit invariance under `y + C a` at scales `1e4` to `1e10`, for `rho` 0 and 0.4 and fixed ranks 1 and 3, with tolerance `100 · n · eps · scale`; selection invariance at the same scales; exact-span signals stay `no_signal` across ranks, `rho` and scale; a shift of `1e18`, which destroys the residual, gives `no_signal`; an ill-conditioned saturated design with the signal along its weakest column gives `no_signal`. The AR(1) lag ratio has the corresponding invariance, exact-span, zero-signal and ill-conditioned tests.

**Remaining limit.** A fixed-design column that falls below the rank tolerance is dropped from the basis, and signal along it is not removed. This was observed during calibration, predates this work, and is unchanged.

## R4. Spectral norm

**What was done.** `_spectral_norm` eigendecomposes the smaller of `Z'Z` and `ZZ'`. The rank cutoff still uses the norm of the unprojected `Z`, so the rejection of random directions inside the fixed span is unchanged.

**Tests.** Agreement with a direct SVD for tall, square, wide, rank-deficient and zero matrices; a check that the decomposed matrix has the smaller dimension; and a wide design inside the fixed span that must still yield no directions.

**Not done.** No benchmark of wide designs, and no full-fit comparison against the dense oracle for a wide design.

## R5. Memory bound

**What was done.**

- The search runs in feature batches. The batch size is chosen so that each search temporary (`(grid rows, k, features)` or `(k, crossings)`) and each retained `(grid, features)` table stays within the 8 MiB working budget. Crossing refinement is chunked the same way.
- One feature is the smallest batch, so a temporary is never below `8 k` bytes. This is stated as the minimum unavoidable allocation.
- The criterion at zero is computed without a `(k, features)` temporary.
- The documentation now says what the bound covers: the search only. The pooled statistics (`(k, features)` per fit) and the projected signals are held in full, and no claim is made about total process memory.

**Departure from the proposal.** Best-grid values are not streamed. The grid tables are retained per batch and bounded by the batch size instead.

**Tests.** Recorded sizes of every evaluation's temporary stay within the budget when one grid row over all features would exceed it; the review's shape (100 directions by 30,000 features) stays within the default budget; a budget below one feature falls back to single features; results are identical across batch sizes, including on the multi-basin problem. The grid-row chunking test was replaced, because row chunking is the contract R5 showed to be insufficient.

**On the 8 MiB value.** It was inherited from the earlier implementation without a recorded rationale. Timing `select_random_design` at budgets from 0.25 MiB to unbatched, on four problems (3,000 to 100,000 features, 12 to 120 pooled directions), showed differences of a few percent from 1 MiB upward and a 5–25% penalty below that. The value was kept and the finding is recorded in a comment at the constant. Peak memory was not measured.

## R6. Duplicated numerical policy

**What was done.** `boldtailor._numerics` holds the rank tolerance, the orthonormal column basis (with its conditioning), residual projection and the projection floor. `_reml_terms`, `reml`, `_ar1_moments` and `ar1` use it, so the R3 fix applies to REML and AR(1) through one definition.

**Deferred.** `_single_trial_fit`, `_fractional_ridge`, `_hrf_cv` and `_conventional` still carry their own basis helpers and tolerances. Migrating them is a separate, behaviour-preserving pass. `experiments/hrf_estimation/library_cv.py` also keeps its own copy of the `64 · eps` rule.

## Upper-bound fits in selection

No change. `at_upper` is carried on `RemlSelection.fit`, the documentation already says such a criterion is a truncation and not a comparable optimum, and the review reports no counterexample. The one related change is the narrower definition of `at_upper` described under R2.

## Implementation order

The review's order was followed with one swap: R2 was fixed before R1, because the R1 safeguard (comparison with zero) sits on top of the search result.

## Verification

- REML, AR(1) and numerics tests: 655 passed, 9 skipped, 2 expected failures (the two existing MixedLM cases).
- Full default suite: 2312 passed, 1 failed. The failure is `tests/test_distribution.py::test_notebook_kernel_is_not_a_runtime_requirement`, caused by `ipykernel` being listed in `pyproject.toml`; these changes do not touch it.
- The four REML and AR(1) experiment test files that were run: 30 passed.
- Against the reviewed code on a well-conditioned problem (3,000 features, 8 candidates, 4 groups of 150 scans): identical winners, ratios equal to `2e-15` relative, and run time 2.16 s against 1.73 s. The slowdown comes from evaluating the slope on the grid and refining every crossing, not from batching.

## Documentation

`docs/api.md` (Scalar REML) describes the four statuses and the new `RhoReference` field. `docs/development.md` (REML conventions) describes the identifiability check, the search and its limit, the memory bound and the projection floor. The REML design spec carries a dated note pointing to both.
