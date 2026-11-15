# Scalar REML module for boldtailor — design

Date: 2026-11-14. Status: approved by the user 2026-11-14.

## Intent

REML is currently implemented three times in `experiments/` (two scalar
solvers with different search conventions, one multi-component solver), with
the nuisance-projection step copied in three places and no test against an
external implementation. The project needs **one** scalar REML implementation,
inside the `boldtailor` package, that every analysis imports, and whose
statistical behaviour is demonstrated by tests rather than asserted in prose.

Priorities, in order: (1) statistical correctness that is tested, (2) logic a
reader can follow and verify, (3) clean, modular code.

## Goals

- A package module that fits the scalar random-effects model below by REML,
  for iid or AR(1) noise, vectorised over features.
- A generic "choose the candidate random design with the best restricted
  likelihood" selector built on it.
- Tests against `statsmodels` `MixedLM(reml=True)` and against a dense
  covariance calculation; seeded Monte Carlo tests of statistical behaviour;
  property-based tests (`hypothesis`) for invariances and edge cases.
- `experiments/hrf_estimation` and `experiments/hrf_simulations` import the
  package module; their own REML code and its duplicates are deleted.

## Non-goals

- Multi-component models (separate condition/RT/junk variances, shared
  across-run component). The user decided scalar is sufficient. The variants
  `remlcomponents`, `remlshared`, `remlcombined` are removed, not ported.
- Estimating rho by REML, or comparing criteria across different rho values.
- Event, HRF, or BIDS knowledge inside the module. It takes arrays.
- Preserving numerical identity with existing reports. The user will rerun them.
- Wiring REML selection into `boldtailor run` / `boldtailor.workflow`.

## Anti-goals

- Two solvers for the same model (the current state).
- An iid path and an AR(1) path that differ in anything except the noise model.
- A test suite that only checks the code against itself.

## Constraints

- Project rules: uv; `uv run` for everything; empty `__init__.py`; short
  functions; pytest with test functions and fixtures; RED-GREEN-Refactor with
  failing tests committed before implementation; never weaken a test.
- Core is anatomy-agnostic: arrays in, results in memory. Signals are
  `(scans, features)`, designs `(scans, columns)`, float64, as in
  `docs/development.md` §Numerical conventions. Public result arrays are
  read-only via `_arrays.readonly_array`.
- Custom numerics are acceptable only with a library oracle test (standing
  project preference). `statsmodels` and `hypothesis` are dev-only dependencies.
- Default `uv run pytest -q` runs package tests only and must stay fast.

## Approaches considered

1. **Custom spectral scalar solver on plain arrays (chosen).** Per-group
   sufficient statistics from one SVD, a closed-form profiled criterion, one
   vectorised 1-D search. Measured on an NSD-sized problem (12 runs × 226
   scans, 30 nuisance and 5 random columns): 55 µs per feature per candidate
   when vectorised over 1,750 features.
2. **`statsmodels` `MixedLM` as the implementation.** Most library-first, and
   it can express the model exactly (one variance component, run-specific
   fixed effects). Rejected on speed: 323 ms per feature per candidate on the
   same problem, i.e. about 84 core-hours for one 533-candidate × 1,750-vertex
   fit that the spectral solver does in under a minute. Kept as the oracle.
3. **Dense GLS REML with a SciPy optimiser.** Easiest to read; O(n³) per
   evaluation and not vectorised. Kept as a second, independent oracle.

The user chose approach 1 after seeing the benchmark.

## The statistical model

For group (run) `g = 1..G` and one feature:

    y_g = C_g a_g + Z_g b_g + e_g
    b_g ~ N(0, τ² I),  e_g ~ N(0, σ² R_g(ρ)),  independent across groups

`C_g` is the fixed (nuisance) design with unpenalised group-specific
coefficients `a_g`. `Z_g` is the random design. `R_g` is the identity (iid) or
the stationary AR(1) correlation matrix with known `ρ`. One ratio
`λ = τ²/σ²` and one `σ²` are shared across groups; `σ²` is always the
**marginal** noise variance.

**Whitening.** `W_g` is the AR(1) whitener normalised so `W R W' = I`
(first row `[1, 0, …]`, later rows `[−ρ, 1]/√(1−ρ²)`; identity at `ρ = 0`).
This differs from the existing `whiten`, which leaves variance `(1−ρ²)` and so
silently makes `σ²` the innovation variance and rescales `λ` in the AR(1)
path. With the normalised whitener `λ` and `σ²` mean the same thing in both
paths. `y`, `C` and `Z` are all whitened.

**Restricted likelihood.** After whitening, project `y_g` and `Z_g` onto the
orthogonal complement of `C_g`; let `d_g = n_g − rank(C_g)`, thin SVD of the
projected `Z_g = U S V'`, `c = U' y_proj`, `r_g = ‖y_proj‖² − ‖c‖²`. With
`N = Σ d_g` and `Q(λ) = Σ_g [ r_g + Σ_i c_gi² / (1 + λ s_gi²) ]`,

    −2 ℓ_R(λ) = N log(Q(λ)/N) + Σ_g Σ_i log(1 + λ s_gi²) + const,   σ̂²(λ) = Q(λ)/N

This is the exact restricted likelihood (the likelihood of `K'y` for `K`
spanning `null(C')`), profiled over `σ²`. The constant depends only on `C` and
`ρ`, so criteria are comparable across candidate `Z` at fixed `C` and `ρ`, and
not across `ρ`.

## Module layout

Private modules hold one idea each; one public module is the entry point.
Names follow the package pattern (`fractional_ridge` / `_fractional_ridge`).

| Module | Responsibility |
| --- | --- |
| `_ar1_moments.py` | Lag-one residual autocorrelation (`e'Le / e'De`); the projection-bias correction (`tr(MLMR)/tr(MDMR)` inverted on a grid); pooling (median, clip to ±0.8). No REML knowledge; the public `whiten` lives in `ar1.py`. |
| `_reml_terms.py` | `FixedProjection` (per group: orthonormal basis of whitened `C`, projected signals, `d_g`, raw energy), built **once** per problem; `RemlTerms` (per group: `s²`, `c²`, `r`, `d`); `reml_terms(projection, random_design)`. Rank tolerances live here only. |
| `_reml_criterion.py` | `criterion(ratios, terms)`, `criterion_slope` and `profiled_variance(ratios, terms)`: pure functions of the groups' pooled sufficient statistics (`pool_terms`), vectorised over features. |
| `_reml_search.py` | `minimize_ratio(terms)`: log grid over `λ·s²`, bisection on the analytic slope inside the neighbouring grid points, explicit `λ = 0` candidate; returns ratio, criterion, boundary flags. The only optimiser. |
| `ar1.py` (public) | `whiten`, `prepare_reference`, `RhoReference`, `estimate_rho`, `RhoEstimate`; the lag-ratio moments live in `_ar1_moments.py`. |
| `_reml_inputs.py` | Validation of the public REML inputs (groups, random designs and candidates, `rho`). |
| `reml.py` (public) | `RemlGroup`, `fit_reml`, `select_random_design`, result dataclasses. |

Public API sketch (names are the builder's to refine; shapes and semantics are not):

```python
RemlGroup(signals, fixed_design)             # (scans, features), (scans, p)

fit_reml(groups, random_design, *, rho=0.0) -> RemlFit
# random_design: one (scans, q) matrix per group — the design Z whose coefficients are
# random effects. fixed_design is C, the unpenalised nuisance regressors.
# rho: a scalar, or one value per feature. Per-feature values are handled by
# partitioning features on the distinct rho values (callers bin rho beforehand).
# RemlFit: ratio, noise_variance, effect_variance, criterion, status  (one per feature),
#          restricted_dof, rho

select_random_design(groups, candidates, *, rho=0.0) -> RemlSelection
# `candidates` is an iterable; each item is one `random_design` (a sequence with one
# matrix per group). Consumed lazily, one candidate in memory.
# Each candidate is whitened once per distinct rho, so the iterable is consumed once
# even with per-feature rho; FixedProjection is built once per distinct rho.
# RemlSelection: winner index, the winner's RemlFit fields, and the criterion gap to
#                the runner-up, per feature. Ties go to the earlier candidate.

estimate_rho(groups_of (signals, reference_design)) -> RhoEstimate
# per-group, per-feature corrected values (NaN where undefined), plus the pooled value.
```

`status` per feature is one of `ok`, `variance_zero` (optimum at `λ = 0`, so
the random design is not supported by the data), `no_signal` (no energy left
after projecting out `C`), with a separate boolean for hitting the upper grid
edge.

Added after the 2026-11-15 review (`docs/review-2026-11-15-reml.md`): a fourth
status, `unidentifiable`, for a random covariance that is a multiple of the
identity on the whole restricted space. The criterion is then constant in `λ`;
the ratio and variances are NaN and the candidate is not eligible in a
selection. `no_signal` is decided by a conditioning-aware projection rounding
floor instead of `64 · eps · ‖y‖²`. See the REML conventions in
`docs/development.md` for both.

Added after the follow-up review (`docs/review-2026-11-15-reml-followup.md`):
a fifth status, `unresolved`, for energy above projection rounding but within
the uncertainty of the computed fixed span; fixed-design columns are
normalised so that neither rank nor any status depends on their units; an
`unrefined` flag on fits marks a positive ratio that is a grid point and not
a verified stationary point; and `any_at_upper` on a selection marks features
for which some candidate's criterion was truncated.

Selection semantics per feature: the winner is the candidate with the lowest
criterion among candidates whose status is `ok`. At `λ = 0` every candidate has
the same criterion, and any candidate with `λ̂ > 0` is strictly better, so a
feature has no winner exactly when every candidate is `variance_zero` or the
feature is `no_signal`. Then the winner index is `-1`, the gap is NaN, and the
status says which case applied. The gap is also NaN when only one candidate is `ok`.

What moves out of the solver and stays with callers: building `Z` from events
and HRFs, the choice of AR(1) reference design, regressor scaling
(`unit_rms`), rho shrinkage across vertices (`shrink_rho`), result containers
such as `HrfEstimate`.

## Testing

All in `tests/`, one file per private module plus the files below. Oracles go
in a new `tests/reml_oracles.py` and share no code with the implementation.

**Oracle agreement** (`test_reml_oracle.py`)
- Dense oracle: `K = null_space(C')`, `V = K'(R + λ Z Z')K`, criterion from
  `slogdet` and `solve` on **unwhitened** data. Criterion differences across
  `λ` match to ~1e-9 relative, for iid and AR(1). This validates the algebra
  and the whitening independently.
- `MixedLM(reml=True)` with one variance component and block-diagonal fixed
  effects: `λ̂`, `σ̂²` agree to the optimiser's tolerance (~1e-4 relative),
  across a parametrised set: 1–12 groups, unequal group lengths, `q` = 1–6,
  true `λ` from 0 to large, rank-deficient `Z`. AR(1) cases feed `MixedLM` the
  whitened arrays.

**Statistical behaviour** (`test_reml_statistics.py`, seeded Monte Carlo;
tolerances stated as multiples of Monte Carlo standard error)
- `σ̂²` at the **true** `λ` is exactly unbiased (`E[Q(λ)]/N = σ²`) even with
  many fixed-effect columns; the Monte Carlo mean is within 3 standard errors.
  The same statistic computed with `N` replaced by the total scan count (the ML
  divisor) is shown to fail that check. This is the reason to use REML.
- `σ̂²` and `λ̂` at the estimated optimum are consistent, not unbiased (`λ̂` is
  truncated at 0). Tests state an explicit absolute bias allowance for a fixed
  large design and assert that the absolute bias decreases as groups are added.
  No "coverage" claim is tested; the module returns no interval.
- Null calibration at `τ² = 0`: the likelihood-ratio statistic for `λ = 0`
  (restricted LRT) follows the finite-sample null distribution for one variance
  component (Crainiceanu & Ruppert 2004), not the asymptotic ½χ²₀ + ½χ²₁
  mixture, which is not exact here. The reference distribution is produced on
  the oracle side and shares nothing with the implementation: eigenvalues from
  `eigvalsh` of the dense projected `Z Z'`, maximised with
  `scipy.optimize.minimize_scalar`. Compared quantities: `P(λ̂ = 0)` and upper
  tail rates, within binomial Monte Carlo error. The slow version also
  maximises the dense criterion on the simulated data itself.
- Rho: the correction inverts a ratio of expectations and the pooled value is a
  clipped median, so it is not exactly unbiased. For `ρ ∈ {0, 0.3, 0.6}` after
  projecting a realistic nuisance design, the corrected estimate's absolute
  bias is below a stated allowance (0.02 unless the builder shows that is
  unattainable and documents why), and the uncorrected estimate's bias is
  several times larger.
- AR(1) fit: with AR(1) noise and the true `ρ`, `λ̂` and `σ̂²` are recovered,
  and the iid fit on the same data is measurably worse.
- Selection: a planted random design among correlated decoys is selected at a
  rate that rises with SNR and approaches 1.

A light version (few hundred replicates, well under a minute in total) runs in
the default suite; a heavier version sits behind a new `slow` marker with a
`--run-slow` option, registered the way `notebook` is in `tests/conftest.py`.

**Property-based** (`test_reml_properties.py`, `hypothesis`, deterministic
via `derandomize=True`)
- Invariance: invertible reparameterisation of `C`; orthogonal rotation of
  `Z` columns; permutation of groups and of features.
- Equivariance: `y → k·y` leaves `λ̂` fixed and scales `σ̂²` by `k²`;
  `Z → k·Z` scales `λ̂` and `τ̂²` by `1/k²` and leaves `σ̂²` and the
  criterion unchanged.
- Optimality: the returned criterion is ≤ the criterion at every grid point
  and at `λ = 0`.
- Edge cases: duplicated or zero `Z` columns; `Z` inside the span of `C`;
  `y` inside the span of `C` (`no_signal`); `q = 0`; one group; `n ≤ rank(C)`
  raises; non-finite input raises; `|ρ| ≥ 1` raises.
- Whitening: `W R W' = I` against a dense `R`; `ρ = 0` is the identity.

## Migration

- `experiments/hrf_estimation`: delete `reml.py`, `reml_scalar.py`,
  `reml_extensions.py`, `reml_extended_fit.py` and their tests; `pilot.py`
  keeps methods `reml` and `remlar1` only, implemented as thin adapters
  (build designs → package call → `HrfEstimate`). `regularized_fit.py` and
  `regularized_noise.py` call the package. Markdown notes describing the
  removed variants get a one-line "superseded" pointer to this document.
- `estimate_noise` (pooled rho for `remlar1`, with a reference design that
  depends on `rt_mode`) stays in `experiments/hrf_estimation` as an adapter:
  it builds the reference design and calls the package's `estimate_rho`.
- `rt_ablation.py` keeps `rt_mode="none"` (a design-building choice) and
  computes `n_random_effects_per_run` from the design it built.
  `test_rt_ablation.py` and `test_regularized_noise.py` are retargeted at the
  adapters and the package; assertions about deleted solver internals go.
- `hrf_simulations/test_fitting.py::test_iid_adapter_matches_original_solver`
  is deleted with the original solver; the package oracle tests replace it.
- `experiments/hrf_simulations`: `fitting.py` keeps only `DesignCache`, the
  design-building loop and the adapter; `_terms`, `estimate_rho` and both
  solver calls go. `calibration.py` uses the package's rho correction.
  `runner.implementation_fingerprint` drops the deleted file names.
- Experiment tests that asserted solver internals are deleted with the code
  they tested; tests of experiment behaviour (planted-HRF recovery, split
  handling, checkpointing) stay and must pass unchanged in intent.
- Docs: a model-and-conventions section in `docs/development.md`, an entry in
  the code-organisation table, and API entries in `docs/api.md`.

## Left to the builder

Exact names; grid size and refinement iterations (must satisfy the optimality
property and oracle tolerances); how `FixedProjection` stores its basis;
Monte Carlo replicate counts; hypothesis strategies; whether the selector also
returns the full candidate × feature criterion table behind an option.

## Decisions from review

- The Sherlock-only `find-links` entry was removed from `pyproject.toml` by the
  user; plain `uv add --dev` and `uv lock` work locally.
- The sections and assertions in `experiments/hrf_estimation/results.py` and
  `session_comparison.py` (and their notebook tests) that read
  `remlcomponents`, `remlshared` and `remlcombined` outputs or assert against
  `reports/reml_validation_summary.tsv` are deleted. Committed report files
  stay as historical records.

## Decisions during implementation

- The search refines the best log-grid point by bisection on the analytic
  derivative of the criterion in `log λ` inside the neighbouring grid points,
  accepted unless worse than the grid point beyond floating-point rounding, then
  compares with `λ = 0` (ties to zero). Without a sign change in that bracket the
  grid point stands. The contract is the guarantees, not iteration counts.
  (Final fix wave, item (b): the original golden-section stage was dropped once
  the groups' sufficient statistics were pooled once per call; every existing
  optimality, oracle and property test passed unchanged, and a single-feature
  12-run selection became about 9x faster.)
- (Final fix wave, item (a).) Input validation moved from `reml.py` to the
  private `_reml_inputs.py`; `reml.py` still calls `project_fixed` through its
  own namespace. Candidate errors name the candidate index, and a candidate may
  be any iterable of per-group designs (one candidate is materialised at a
  time, never the candidate stream). All `ρ` subsets must share one restricted
  dof, else `ValueError`.
- (Final fix wave, item (c).) `boldtailor.ar1` gained a public prepared
  reference: `prepare_reference(design) -> RhoReference` (frozen, read-only:
  the orthonormal basis and the expected lag ratio on the grid), and
  `estimate_rho` accepts, per group, a design array or a `RhoReference` with
  identical results; it validates its input and names the group. The
  simulations cache the prepared reference per run under the `DesignCache`
  `"reference"` key; rho estimation stays out of `runner.py`.
- `project_fixed` runs once per group per distinct `ρ`, plus once more for a
  `ρ` subset mixing features with and without signal, so live features are
  bit-identical whether or not dead features are present.
- The `statsmodels` `MixedLM(reml=True)` oracle uses BFGS. Two cases are strict
  xfails where MixedLM does not reach the optimum, confirmed against the dense
  optimiser in `tests/reml_oracles.py`.
