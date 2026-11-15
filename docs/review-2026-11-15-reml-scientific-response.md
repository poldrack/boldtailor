# Response to the scientific-validity re-review

Date: 2026-11-15  
Branch: `fix/reml-scientific-validity`  
Review: [scientific re-review](review-2026-11-15-reml-scientific-rereview.md)

The two demonstrated defects are fixed. Selection qualifications now reach
both experiment results and their scientific summary tables. Tests were
committed failing before each production fix; no implementation was placed in
an `__init__.py`.

## Changes

**B1, residual energy.** `reml_terms` computes the norm of the residual outside
the retained random span directly, instead of subtracting two large squared
norms. When random rank fills the restricted space, the complement has dimension
zero and its energy is exactly zero. The saturated and small-positive-noise
counterexamples now agree with independent diagonal-covariance likelihoods.
Row-order invariance is restored for the saved example. This improves the
error scale from squared-energy cancellation to amplitude rounding; it does
not recover information below floating-point resolution.

**B2, AR(1) identification.** `invert_lag_ratio` returns NaN for nonfinite,
flat, numerically unresolved, and nonmonotone correction curves. It interpolates
both increasing and decreasing identified curves. Endpoint clipping is retained.
The guard on adjacent differences is `64 eps max(1, max(abs(expected)))` in
dimensionless lag-ratio units. Undefined values do not enter pooling; zero
when no finite values remain is still a documented fallback. This conservatively
rejects the whole nonmonotone curve instead of choosing a possibly unique local
branch for particular observations.

*Revised after review of this branch.* Rejecting the whole curve also
discarded usable estimates when the curve bends only near an end of the rho
grid, and the pooled value then fell back to 0.0 (a true rho of 0.4 reported
as 0.0 with half the volumes censored). The curve is now required to be
monotone only across the pooling range, [−0.8, 0.8]; the branch spanning it
is inverted for every value and values beyond it are clipped to its ends.
Flat curves and curves that turn inside the pooling range still give NaN, so
the demonstrated B2 failure stays fixed. See the REML conventions in
`docs/development.md`.

**M1, uncertainty reporting.** `RemlSelection.any_unrefined` now aggregates
interior grid fallbacks as well as upper-bound fits from every candidate.
Both experiment adapters preserve `unrefined`, `any_at_upper`, `any_unrefined`
and `provisional` (equal to `any_unrefined`). The selected kernel remains
available; the qualification is not silently converted into a failed or null
effect. HRF score summaries and simulation score tables preserve `provisional`.
Simulation reports include `n_qualified`, `n_provisional`, and
`provisional_rate`. Historical results lacking the field remain unknown:
they are not relabeled as resolved, and an entirely unqualified group has
zero known qualifications and a NaN provisional rate.

## Scientific validation added

- Five saved scientific probes promoted to the default test suite.
- Generated known orthogonal residual energies across random ranks, including
  saturation and small residuals beside a large random-effect component.
- Independent covariance likelihood, quadratic-form, and finite-difference
  score checks over generated spectra and variance ratios.
- Search results checked against independent feasible-point objectives.
- Generated exact-confounding classifications across group scales and ranks.
- Generated AR(1) moment polynomials checked against dense restricted traces,
  and identifiable inversion checked against known rho.
- Nonzero/negative-rho time-reversal checks at high random rank.
- Fit and candidate-order properties now include the new qualification flags.
- The repeated-spectrum generator now exercises two distinct repeated values.
- Matched-reference estimated-rho recovery and relative null-tail checks at
  rho ±0.5, with null and positive effects.
- Null library-size checks demonstrating that a selected candidate is not a
  significance test.
- Sensitivity checks for incomplete task references, unequal run variances,
  heterogeneous group rho, and a non-isotropic coefficient covariance.
- End-to-end preservation of scientific qualifications in experiment results
  and summaries, including unknown metadata in historical results.

The correlated-effect sensitivity case changes both covariance shape and
average scale. Its conclusion is that fitting the wrong coefficient covariance
changes the scalar estimand; it does not isolate correlation from overall
variance. The independent reviewer identified this interpretation limit and
found no Critical or Important issues in the reviewed numerical changes.

## Verification

| Check | Result |
| --- | --- |
| Final REML/AR(1)/numerical/property suites and saved reviewer probes | 755 passed, 9 skipped, 2 xfailed |
| Six relevant experiment files, including scoring and reporting | 40 passed |
| Full default suite | 2430 passed, 1 failed, 9 skipped, 2 xfailed |
| Additional sensitivity cases added after the full run | Passed; included in final 755-test result |
| Fresh read-only review | No Critical or Important findings |

The full-suite failure is the unchanged
`tests/test_distribution.py::test_notebook_kernel_is_not_a_runtime_requirement`:
`ipykernel` remains a runtime requirement. It is unrelated to this scientific
scope and was not modified. The run also emitted a platform core-count warning.

The new regression tests, property tests, and Monte Carlo tests are respectively
`tests/test_reml_scientific_regressions.py`,
`tests/test_reml_scientific_properties.py`, and
`tests/test_reml_plugin_calibration.py`. The existing expected failures and
skipped slow statistical cases are not claimed as passes.

## Scientific limits retained

The search is practical, not a certified global optimizer. Weakly identified
variance splits remain weakly identified. The plug-in calibration checks pool
over 1,500 replicate features with matched reference spans; they do not certify
single-vertex or small-pool calibration. The oracle empirical quantile measures
relative null-tail behavior, not independent proof of a nominal significance
test. Library-size scenarios use random candidates and do not cover every
correlated HRF library. Sensitivity tests demonstrate modeling limits rather
than remove them: shared noise variance, shared rho, iid random coefficients,
and an appropriate reference span remain assumptions. No biological claim
about a selected HRF follows solely from these numerical checks.
