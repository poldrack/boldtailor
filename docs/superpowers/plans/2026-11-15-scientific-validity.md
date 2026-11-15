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

- [ ] 1. Promote saved regressions to `tests/test_reml_scientific_regressions.py`,
  run RED, commit tests. Fix `_reml_terms.reml_terms` by computing a residual
  norm and enforcing a structural zero at full restricted rank. Verify the
  independent diagonal likelihood and old REML suites, then commit.
- [ ] 2. Add undefined-curve and monotone inversion tests to `test_ar1.py`.
  Replace the old nonmonotone nearest-grid contract with NaN for a curve that
  cannot uniquely identify rho; retain monotone interpolation/clipping.
  Run RED, commit tests, implement and verify, commit.
- [ ] 3. Add scientific qualification tests for selections and experiment
  diagnostics, including `any_unrefined`. Run RED and commit tests. Preserve
  flags in core selection and both adapters, verify and commit.
- [ ] 4. Add generated independent tests of spectra, residual energy,
  covariance likelihood/derivative, confounding, AR(1) traces/inversion, and
  extend public invariance checks to nonzero rho and high rank. Add bounded
  Monte Carlo checks for estimated rho and selection calibration. Commit
  tests before any corrections they expose. Document model sensitivity.
- [ ] 5. Run the targeted, experiment, saved-probe, and full default suites;
  obtain a fresh review; document results and remaining scientific limits.

## Execution ledger

Authorization: user explicitly requested implementation of the review.
Use a new branch in the shared workspace, preserving unrelated untracked files.
The existing review established the root causes; no speculative model changes
are needed. Tasks 3 and 4 depend on the core status/inversion contract in 1–2.
Characterization/property tests that already pass need no production change.
