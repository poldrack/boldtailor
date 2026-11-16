# Targeted red-team review of `boldtailor.reml`

Date: 2026-11-15  
Reviewed commit: `587929c088062384b80765e47b9664a47422dc75`  
Scope: statistical logic, numerical reliability, architectural cleanliness, and relevant patterns inherited from the larger package.

This document records review findings and proposed solutions. It does not describe implemented fixes. Three statistical failures were reproduced through the public API. Two resource-scaling concerns and a package-wide architectural concern were identified separately. The numerical examples below are adversarial cases, not estimates of failure frequency in typical fMRI analyses.

## Findings and priorities

| ID | Priority | Finding | Evidence |
| --- | --- | --- | --- |
| R1 | High | Unidentifiable variance components receive `ok` status and can win selection | Public-API reproduction; exact covariance argument |
| R2 | Medium | The grid search can return `variance_zero` despite a better positive ratio | Public-API reproduction; independent dense covariance calculation |
| R3 | Medium | A fixed-effect offset can change a valid fit to `no_signal` | Public-API reproduction; same effect observed in AR(1) lag estimation |
| R4 | Medium | Spectral-norm calculation scales unnecessarily with the number of random columns | Direct inspection of the Gram-matrix eigendecomposition |
| R5 | Medium | Grid chunking does not enforce the documented temporary-memory limit | Reproduced chunk-size calculation; inspection of refinement |
| R6 | Architectural | Projection, rank, and degeneracy policies are scattered across the package | Comparison of REML, AR(1), single-trial, fractional-ridge, and HRF-CV implementations |

## What appears sound

The model is explicitly documented as

```text
y_g = C_g a_g + Z_g b_g + e_g
b_g ~ N(0, tau² I)
e_g ~ N(0, sigma² R_g(rho))
```

Random effects and errors are independent across groups; one noise variance and one random-effect variance are shared across groups. Rho is supplied, not estimated by the REML optimizer. The normalized AR(1) whitener is consistent with interpreting sigma² as marginal noise variance.

After whitening and eliminating the fixed-effect span, the implemented profiled criterion is

```text
lambda = tau² / sigma²
Q(lambda) = residual_ss + sum_i coordinates_i² / (1 + lambda s_i²)
F(lambda) = N log(Q(lambda) / N) + sum_i log(1 + lambda s_i²)
```

This algebra is consistent with the stated model. Dropping the additive constant is appropriate for comparisons at fixed fixed-design matrices and rho. Criteria should not be compared across changes in those quantities.

The division into input validation, sufficient statistics, criterion, search, and public selection is sensible. The dense-covariance, MixedLM, simulation, and property tests are useful existing safeguards. The review does not recommend replacing the entire implementation.

## R1. Non-identifiable variance components are reported as supported

### Location and mechanism

See [`minimize_ratio`](../src/boldtailor/_reml_search.py), particularly the comparison `at_zero_value <= value`; [`_fit_values`](../src/boldtailor/reml.py), which maps every positive result to `ok`; and `_compete`, which treats that status as selection eligibility.

For one group with no fixed effects, iid noise, and `Z = I`, the covariance is

```text
sigma² I + tau² Z Z' = (sigma² + tau²) I.
```

Only the sum is identifiable. Profiling out sigma² makes the criterion mathematically constant in lambda. This is not evidence that tau² is zero or positive: the data cannot distinguish the allocations.

Floating-point cancellation in `N log(Q/N) + logdet` produces tiny differences along the flat profile. A positive grid point can appear better than zero, and the API then presents an arbitrary split as an ordinary successful estimate.

### Reproduction and impact

With `y = default_rng(0).normal(size=(10, 1))`, `C` of shape `(10, 0)`, and `Z = eye(10)`, the reviewed code returned:

```text
ratio = 10000
status = "ok"
at_upper = False
select_random_effects_design(...).winner = 0
```

Independent evaluations of the exact flat criterion at ratios 0, 1, and 10000 agreed to rounding, approximately `-5.859064715414505`. Other seeds produced other large positive ratios.

This matters for saturated random designs and any design whose restricted random covariance is proportional to the restricted noise covariance. It is a model-identifiability issue, not merely an optimizer convergence issue.

### Proposed solution

1. Detect covariance-component dependence before fitting. In whitened restricted coordinates, the covariance is `sigma² I_N + tau² B`. Exact confounding occurs when the complete block-diagonal `B` is a common scalar multiple of `I_N`.
2. The existing spectra can support this check without building a dense covariance. Include the zero eigenvalues implied by each group's restricted degrees of freedom minus its retained random rank. Nonzero eigenvalues alone are insufficient. The proportionality constant must be the same across groups; individually isotropic groups with different scales can still identify the two components jointly.
3. Distinguish the all-zero random covariance from a nonzero covariance proportional to identity. Both lack information about some parameter, but their interpretation differs. Preserve or explicitly revise the existing zero-design behavior rather than changing it accidentally.
4. Expose a clear `unidentifiable` status, or reject the model with a descriptive error. If returning a fit, do not present an arbitrary variance split as identified. NaN component estimates plus a defined combined-variance diagnostic are one possible contract.
5. Define selection semantics explicitly. An unidentifiable candidate must not acquire apparent support because of floating-point noise. Whether it is represented as a null-equivalent candidate or excluded should be documented.
6. Use a roundoff-aware comparison with the zero boundary as an additional numerical safeguard, not as a substitute for identifiability detection. Near-proportional covariance requires a documented conditioning policy rather than an arbitrary large tolerance.

### Regression requirements

Test identity covariance, the same construction with fixed effects projected out, AR(1) analogues, and multiple groups with both common and differing proportionality constants. Test candidate-order behavior and ensure a genuine identifiable positive effect remains selectable. Include near-confounding cases to separate exact non-identifiability from weak information.

## R2. The search can miss a better positive-ratio solution

### Location and mechanism

[`_best_positive`](../src/boldtailor/_reml_search.py) evaluates 121 equally spaced log-ratio values, selects the best grid point, and refines only its neighboring bracket. The grid spans `1e-8 / max(s²)` to `1e8 / min(s²)`. Increasing spectral spread therefore increases spacing without increasing resolution.

A narrow improvement can fall between grid points and never receive refinement. Comparing the result against zero does not repair this: it can turn the missed improvement into a false `variance_zero` classification. Slope bisection is useful inside a valid bracket, but does not establish that all relevant basins were considered.

### Reproduction and impact

The saved reproducer uses ten observations, no fixed effects, and eight orthogonal random columns with squared singular values `logspace(-10, 10, 8)`. Its full signal vector is embedded in the test file.

```text
Reported ratio:       0
Reported criterion:   112.42466973789602
Reported status:      variance_zero
Selection winner:     -1

Alternative ratio:    7.126268271680137e-7
Dense criterion:      112.40094347046998
```

The alternative is evaluated independently using `V = I + lambda Z Z'`, a linear solve, and `slogdet`. No global-optimum claim is required to establish the failure: exhibiting one feasible point with a lower objective is enough.

The design is deliberately ill-conditioned; it establishes a correctness gap in accepted inputs, not its prevalence in ordinary designs.

### Proposed solution

1. Make search resolution depend on log-span or spectral transitions rather than fixing the number of grid points for every problem.
2. Evaluate slope information across the search and refine competing negative-to-positive crossings and candidate minima, rather than refining only the single grid winner.
3. Adaptively subdivide unresolved intervals. Merely checking endpoint signs can still miss multiple stationary points within an interval; a denser grid or all-crossings policy is an improvement, not an unconditional global proof.
4. Compare all refined candidates with zero and the search boundaries using numerically justified tolerances.
5. Distinguish an accepted stationary solution from an unrefined grid fallback or unresolved search. Do not map every positive returned number to an unqualified `ok` result.
6. Document the actual optimization guarantee. If a strict global guarantee is required, use a method with certified bounds or root isolation over the stated domain. If practical adaptive search is retained, document its limits and expose failure diagnostics.

The dense oracle also needs care: its single `minimize_scalar(..., method="bounded")` call is a local optimization method, not an independent guarantee of global optimality. See the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html). Keep independent objective construction and strengthen the oracle search separately.

### Regression requirements

Retain the supplied counterexample. Add broad spectral ranges, clustered eigenvalues, nearly flat profiles, multiple candidate minima, boundary minima, and cases where zero genuinely wins. Compare against independently evaluated feasible points and an independently searched objective. Verify both fit values and downstream winner/status behavior.

## R3. Fixed-effect shifts can change the signal-validity decision

### Location and mechanism

[`_live`](../src/boldtailor/reml.py) currently requires

```text
projected_energy > 64 * machine_epsilon * raw_energy.
```

REML depends on the signal after removing the fixed-effect span. Adding `C a` should not change that statistical problem. The threshold nevertheless grows with the added nuisance signal, allowing a numerically resolvable residual to be classified as absent.

At the amplitude level the rule is proportional to `sqrt(eps)`, not `eps`. It is therefore much more conservative than a straightforward first-order floating-point projection-error scale, although a replacement must account for dimensions, conditioning, and the actual projection algorithm.

### Unsupported tolerance choice

The specific choice `_NO_SIGNAL_FACTOR = 64.0` has no documented derivation or calibration in the reviewed implementation plan, introducing commit, or relevant tests. The [implementation plan](superpowers/plans/2026-11-14-reml-module.md) prescribes `64 * eps * raw_energy`; commit `c0259c9ca116667a031f050c484a6d4284e76b26` introduces it without further rationale. This establishes a documentation/evidence gap, not proof that no rationale ever existed.

Suppressing floating-point residue from an exact fixed-span signal is a valid purpose. However, neither the multiplier 64 nor the use of epsilon against **squared** norms is supported by an error analysis here. For float64, the implemented condition requires a residual-to-original norm ratio greater than `sqrt(64 * eps)`, approximately `1.19e-7`. A constant that looks small at the energy level therefore imposes a substantially larger relative threshold at the amplitude level.

This is a flaw in the numerical validity policy, with the offset counterexample below demonstrating a concrete consequence. It is not a statistical significance threshold and must not be interpreted as evidence that a residual signal is scientifically absent. Changing 64 to another arbitrary constant would not establish correctness. A replacement needs an explicit error model or empirical calibration over justified operating conditions, along with tests of both exact degeneracy and small but resolvable residuals. Nor should epsilon simply be squared without accounting for the dimensions, conditioning, and projection algorithm.

### Reproduction and impact

A 30-observation example with an intercept and two random columns gives:

```text
Original:             status=ok, ratio≈3.61182022, noise_variance≈1.45799735
After adding 1e8:     status=no_signal, ratio=0, noise_variance=NaN
```

The offset is in the fixed-effect span and the remaining variation is still representable in float64. This is an extreme nuisance-to-residual scale case, not a claim about typical fMRI intensity offsets.

The same policy is shared with [`basis_lag_ratio`](../src/boldtailor/_ar1_moments.py). On this signal, the raw lag estimate changes from approximately `-0.05379941` to NaN after the offset. Consequently, a REML-only fix would leave related noise estimation inconsistent.

### Proposed solution

Develop a documented projection-error policy shared by the relevant numerical kernels. Assess degeneracy against an error bound for the computed residual, preferably in norm units; square a norm threshold only when comparing squared quantities. Do not simply replace the existing constant without a numerical justification.

Stable centering when an intercept is present, re-projection, or orthogonal-transform residual calculations may improve robustness. They must preserve the general fixed-design semantics and must not silently add an intercept. Exact invariance cannot be promised once adding an offset destroys representable residual information; that limitation should be distinguished from the current overly aggressive threshold.

### Regression requirements

Test `y -> y + C a` over a range of resolvable scales, iid and AR(1) settings, and several fixed-design ranks. Verify status, ratio, variance, and selection invariance within justified tolerances. Retain tests for exact fixed-span signals and zero signals. Add corresponding AR(1) lag-estimation tests and explicit tests for genuinely unresolved cases.

## R4. The spectral-norm helper assumes a small number of random columns

[`_spectral_norm`](../src/boldtailor/_reml_terms.py) forms `Z.T @ Z` and computes all of its eigenvalues to obtain the largest one. For `Z` of shape `(n, q)`, the Gram matrix needs O(q²) storage and its dense eigendecomposition O(q³) work. Forming it also costs O(n q²). The public API does not restrict q to be small.

The subsequent projected-design decomposition is a thin SVD, making the preliminary full eigendecomposition especially wasteful when `q >> n`. This is a structural scaling concern; no out-of-memory experiment was run.

### Proposed solution and validation

Use a rectangular spectral-norm calculation, or form the smaller of `Z.T @ Z` and `Z @ Z.T`. Preserve the intended use of the unprojected norm for the rank cutoff: substituting the projected norm without analysis would change how fixed-span random directions are rejected.

Compare the replacement against a direct SVD across tall, square, wide, rank-deficient, and fixed-span designs. Check complete REML outputs for equivalence and benchmark representative wide cases. The goal is to remove unnecessary dependence on q²/q³, not to alter the statistical rank policy incidentally.

## R5. The documented memory bound is not enforced

[`_chunk_rows`](../src/boldtailor/_reml_search.py) chunks grid rows, but always returns at least one row. If one row of shape `(pooled_directions, features)` exceeds the budget, the limit cannot hold.

For 100 pooled directions and 30,000 features:

```text
8 * 100 * 30000 bytes = 22.888 MiB
_chunk_rows(...) = 1
```

This exceeds the documented 8 MiB per-temporary claim before accounting for simultaneous arrays. `_grid_values` retains grid results for concatenation, and slope refinement operates across all features with several direction-by-feature intermediates. Grid-row chunking alone therefore does not bound peak memory.

### Proposed solution and validation

Add feature-axis batching to both grid evaluation and refinement. Budget the working set, not only one array. Consider streaming best-grid indices and values instead of retaining the entire grid-by-feature table. If even one feature exceeds the budget, explicitly document the minimum unavoidable allocation or use direction-axis accumulation where practical.

Clarify which allocations are covered: solver temporaries, retained projections, inputs, outputs, and total process RSS are different quantities. Avoid advertising a total bound when only one category is controlled.

Test batch-size invariance, including a case where one full feature batch exceeds the target. Use allocation-shape checks for deterministic regressions and a separate representative memory benchmark. Allow justified floating-point tolerance if accumulation order changes; require stable statuses and winners away from genuine ties.

## R6. Package-wide numerical policy is duplicated

Related basis, rank, projection, and degeneracy logic occurs in:

- [`_reml_terms.py`](../src/boldtailor/_reml_terms.py): fixed-design basis, random-design rank cutoff, projections.
- [`_ar1_moments.py`](../src/boldtailor/_ar1_moments.py): another SVD basis helper and raw-energy degeneracy floor.
- [`_single_trial_fit.py`](../src/boldtailor/_single_trial_fit.py): nuisance basis, residual-column support, and rank validation.
- [`_fractional_ridge.py`](../src/boldtailor/_fractional_ridge.py): residualization and a different amplitude-based signal threshold.
- [`_hrf_cv.py`](../src/boldtailor/_hrf_cv.py): another basis helper and projected-column support checks.

This is architectural debt shared with the larger package, rather than a defect confined to the new REML module. Similar-looking operations can receive different fixes, empty-input handling, and tolerances. R3 provides a concrete example of a problematic rule shared by REML and AR(1).

### Proposed direction

Extract a small internal numerical module for clearly common operations: rank tolerance, orthonormal column bases, residual projection, and justified projection-error estimates. Make policy arguments explicit where semantics differ. Retain model-specific requirements at callers: an OLS design requiring full column rank is not the same as a random-effects design that legitimately accepts redundant columns.

Do not build a generic modeling framework or force every fitting path to use one validity rule. Start with characterization tests and the helpers needed to fix confirmed failures, then migrate callers incrementally. Keep every `__init__.py` empty.

## Related contract concern: upper-bound fits in selection

The current documentation already warns that an `at_upper` criterion is truncated and is not a comparable optimum. Nevertheless, `_fit_values` gives such fits `ok` status and `_compete` admits them to ordinary selection.

This was not established as an additional numerical counterexample in the review, and it follows the current documented selection behavior. It is a contract risk to address during the status/search work: either resolve the boundary, return an explicitly provisional selection, or clearly carry boundary uncertainty through the selection result. Simply excluding all upper-bound candidates is not automatically correct, because a genuine zero-noise boundary may be scientifically relevant.

## Saved reproductions and verification

The standalone [pytest reproducer](reml-review-2026-11-15-reproducers/test_redteam.py) contains three tests covering R1–R3. These are expected failures on the reviewed implementation. They are outside the default test directory so that saving review evidence does not silently change the normal suite.

Run from the repository root:

```bash
uv run pytest -q docs/reml-review-2026-11-15-reproducers/test_redteam.py
```

The targeted existing suite was run with:

```bash
uv run pytest -q tests/test_reml.py tests/test_reml_criterion.py tests/test_reml_terms.py tests/test_reml_search.py tests/test_reml_selection.py tests/test_reml_oracle.py tests/test_reml_properties.py tests/test_reml_statistics.py tests/test_ar1.py
```

Observed results during review:

- Existing targeted suite: **256 passed, 9 skipped, 2 xfailed**.
- New adversarial probes: **3 failed**, each for its intended assertion.
- R3's shared AR(1) behavior and R5's oversized grid row were separately reproduced.
- R4 and R6 are based on code inspection; resource-exhaustion and full-package regression runs were not performed.

Some local verification commands used `UV_CACHE_DIR=/tmp/boldtailor-review-cache uv run --no-sync` because the default uv cache was not writable in the execution sandbox. This used the existing project environment.

## Implementation order and acceptance criteria

1. **Agree the result/status contract for identifiability and unresolved optimization.** Add explicit cases rather than overloading `variance_zero` or `no_signal`.
2. **Commit failing regression tests before implementation**, following the project RED–GREEN–Refactor requirement. The saved probes are starting evidence; expand them to cover the full chosen contract.
3. **Fix R1 and R2 first.** They directly affect variance interpretation and candidate selection. Validate against independent covariance calculations and adversarial spectra.
4. **Fix R3 jointly with the AR(1) degeneracy rule.** Establish the numerical error policy before extracting shared helpers.
5. **Address R4 and R5 with equivalence tests and representative benchmarks.** State the actual resource guarantees.
6. **Consolidate common numerical helpers incrementally.** Preserve distinct statistical requirements and run affected package tests after each migration.

Use uv for package management and `uv run` for local commands. Prefer short functions, pytest functions, and fixtures. Do not weaken tests to accommodate an incomplete fix. No implementation changes were made as part of this review or its documentation.
