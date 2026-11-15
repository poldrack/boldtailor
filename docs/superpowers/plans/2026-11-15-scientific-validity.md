# Scientific validity implementation plan

**Goal:** Correct the two reproduced scientific defects, preserve selection
uncertainty, and add independent scientific property and simulation checks.
**Spec:** `docs/review-2026-11-15-reml-scientific-rereview.md`.
**Architecture:** Keep the scalar REML model. Compute residual sums of squares
from residual coordinates rather than subtracting energies. Make AR(1)
inversion undefined for non-identifying correction curves. Propagate search
qualifications through the existing result objects and experiment adapters.
**Tech stack:** NumPy, SciPy, pytest, Hypothesis; all commands through uv.

## Constraints and review focus

Keep every `__init__.py` empty. Commit failing tests before implementation.
Test small positive orthogonal noise as well as saturated random designs;
ambiguous/flat AR(1) curves as well as invertible curves; losing provisional
candidates as well as winners; negative rho and high random rank; plug-in rho
and library-size effects rather than claiming calibrated significance.

## Tasks

- [x] 1. Promote saved regressions to `tests/test_reml_scientific_regressions.py`,
  run RED, commit tests. Fix `_reml_terms.reml_terms` by computing a residual
  norm and enforcing a structural zero at full restricted rank. Verify the
  independent diagonal likelihood and old REML suites, then commit.
- [x] 2. Add undefined-curve and monotone inversion tests to `test_ar1.py`.
  Replace the old nonmonotone nearest-grid contract with NaN for a curve that
  cannot uniquely identify rho; retain monotone interpolation/clipping.
  Run RED, commit tests, implement and verify, commit.
- [x] 3. Add scientific qualification tests for selections and experiment
  diagnostics, including `any_unrefined`. Run RED and commit tests. Preserve
  flags in core selection and both adapters, verify and commit.
- [x] 4. Add generated independent tests of spectra, residual energy,
  covariance likelihood/derivative, confounding, AR(1) traces/inversion, and
  extend public invariance checks to nonzero rho and high rank. Add bounded
  Monte Carlo checks for estimated rho and selection calibration. Commit
  tests before any corrections they expose. Document model sensitivity.
- [x] 5. Run the targeted, experiment, saved-probe, and full default suites;
  obtain a fresh review; document results and remaining scientific limits.

## Execution ledger

Authorization: user explicitly requested implementation of the review.
Use a new branch in the shared workspace, preserving unrelated untracked files.
The existing review established the root causes; no speculative model changes
are needed. Tasks 3 and 4 depend on the core status/inversion contract in 1–2.
Characterization/property tests that already pass need no production change.


Completed on branch `fix/reml-scientific-validity`.

- Task 1: four REML regressions RED in e23cbbb, GREEN with 8a6c2c3.
- Task 2: seven inversion tests RED in ac8c740, GREEN with 8d149cc.
- Task 3: six qualification tests RED in 80e437f, GREEN with cc25c5f;
  three summary checks RED in 28bd6d2, GREEN with a2c1eb6.
- Task 4: independent generated oracles and nine bounded calibration/sensitivity
  scenarios pass; these characterization tests required no production changes.
- Task 5: final scientific suites 755 passed, 9 skipped, 2 xfailed; six
  experiment files 40 passed. Full default run 2430 passed, one unchanged
  distribution failure, 9 skipped, 2 xfailed. Three subsequently added
  sensitivity cases are included in the final 755-pass run.
- Fresh read-only review: no Critical/Important issues. Its minor qualification
  is recorded in the response: correlated-effect sensitivity changes both
  covariance shape and average scale and does not isolate correlation alone.

Rulings:
- Reject an entire nonmonotone correction curve rather than choose a local
  inverse branch. This is conservative and can withhold some locally unique
  estimates; it avoids manufacturing a rho from a non-identifying reference.
- Corrected one new test's mistaken assumption that noiseless simulation data
  must imply a truncated fit: the fitted task model can differ from the
  generating model. The actual fit diagnostic is the proper summary oracle.
- A runner test initially saw a manifest mismatch because formatting changed
  source during its run; with a stable checkout all 40 experiment tests pass.
- Leave the known ipykernel distribution failure unchanged: it is unrelated to
  scientific validity and the user excluded operational-only work.
