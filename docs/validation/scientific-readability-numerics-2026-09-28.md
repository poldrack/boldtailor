# Scientific readability: numerical preparation and CV

Implementation base: `03f00eb`; scientific baseline: `57acff5`.
Work remains on `refactor/scientific-readability`. Main and stash
`c4cfbec5fccd5397ea650c550f5b485ea588c2cd` are unchanged.

## Changes and scientific checks

Normalized trial fitting now names its projected-design state and uses one
SVD coefficient formula for OLS and ridge. Full-fit nuisance recovery and sums
of squares remain explicit. Fractional preparation names its raw-basis state;
it retains the separate normalized rank check and the existing 60-step
log-alpha bisection. The all-ones scaling vector and its divisions are gone.

CV prepares each run/HRF group once per fold and calls `betas_at(value)`.
Fractional validation targets reuse that same preparation at fraction one;
the old separate target preparation and discarded penalized validation betas
are removed. Normalized CV retains candidate-specific validation targets.
Generator adapters remain available to existing validation scripts.

| Step | Observed RED | Test commit | Observed GREEN |
| --- | --- | --- | --- |
| Normalized prepared solver | 7 missing-interface failures, 32 passed | `461e76a` | 83 normalized/fractional/selected-HRF/CV tests passed in 32.71 s |
| Fractional prepared solver | 10 missing-interface failures, 13 passed | `086dc9d` | 52 fractional/CV/ablation/simulation tests passed in 21.24 s |
| Explicit CV evaluation | 5 missing-interface failures, 26 deselected | `e07624f` | Full default suite: 782 passed in 168.20 s |

Commands used `uv run pytest ... -q -W error` (or the existing environment via
`uv run --no-cache --no-sync pytest`). Task 1 covered `test_single_trial.py`,
`test_fractional_ridge.py`, `test_selected_hrf_fit.py`, `test_ridge_cv.py`, and
`test_fractional_cv.py`. Task 2 covered `test_fractional_ridge.py`,
`test_fractional_cv.py`, `test_fractional_ridge_ablation.py`, and
`test_fractional_ablation_simulation.py`. Task 3 RED selected the new prepared
run tests in the two CV files; GREEN used `uv run pytest -q -W error`.

Existing independent augmented least-squares and root-finding oracles keep
all original tolerances. Added cases cover repeated/reversed candidate order,
ill-conditioning, NaN positions, excluded HRFs, and releasing returned arrays.
The fractional candidate test forbids SVD calls after preparation. The CV
preparation-count test also compares full scores, SSE, SST, and assignments
against the independent reference. Existing tests cover within-run/absolute
encoding, held-out isolation, fixed targets, and serial/parallel/block-size
NSD equivalence. No scientific discrepancy required a tolerance change.

Shared fixtures moved to `tests/conftest.py`; independent references moved to
`tests/oracles.py`, with an empty `tests/__init__.py`. An AST comparison against
`20257a6` confirmed all four moved definitions are identical apart from the two
function renames. No imports from other test modules remain.

## Verification

- Full default suite after production changes: 782 passed, warnings as errors.
- Final suite after test relocation: 782 passed in 167.87 s, warnings as errors.
- Black: 110 files passed (standard CI scope plus the changed ablation script).
- `git diff --check`: passed.
- `uv build --wheel --quiet`: exit 0.
- Isolated installed-wheel smoke via `tests/check_installed_package.py`: exit 0,
  with installed-module location, no ipykernel, and independent OLS comparison.
  This ran after the final production-code change; subsequent work moves tests
  and updates documentation only.

Remote CI has not run; no remote is configured. The locked, yanked Nilearn
0.14.0 integer-image limitation remains as documented in foundation validation.
The preserved stash contains separate scientific changes and was not applied.
The full architectural roadmap is not yet complete.

## Execution decisions

- Continued in the requested refactoring branch/current checkout, with exact-path
  staging. This retains less filesystem isolation than a separate worktree.
- Recorded observed checks directly instead of repeating them through the skill's
  ledger script, following the instruction to avoid redundant test runs.
- Extended the existing normalized oracle test instead of duplicating its
  calculation. Existing tolerances and the independent reference are preserved.
- Validated fractional inputs inside `solve`. Invalid per-feature maps and invalid
  candidate grids have separate tests: repeated per-feature fractions are valid,
  while repeated grid entries are not. This adds small per-candidate validation work.
- Migrated the ablation script's positional unpack to named fields after its
  existing committed tests exposed the missed caller. The experimental equations
  and their tests are unchanged.
- Pure test relocation used existing regression coverage and AST comparison;
  no artificial failing production test was introduced for moving test helpers.

## Independent review

An independent reviewer examined `03f00eb..7151ddd`, the surrounding encoding
code, original baseline solver, tests, and execution decisions. Verdict: ready
to merge, with no critical, important, or minor findings requiring changes.
The refactoring branch remains unmerged while later roadmap work continues.

The reviewer also compared the original and new normalized solver at alpha
0, 0.1, and 2 with increasingly collinear columns. Relative coefficient
changes stayed below 5.7e-16, and both versions rejected the most degenerate
case. At an extreme 1e-12 column perturbation, OLS residual SSE differed by
up to 6.45e-5 because enormous opposing coefficients amplify cancellation;
regularized SSE differences stayed below 1.5e-14. The reviewer judged this
expected numerical sensitivity, not changed science or identifiability.
These observations do not promise bitwise-identical results for arbitrary
ill-conditioned inputs.

Matters explicitly set aside by the reviewer are resolved as follows:

- Constant-feature SSE conventions remain the committed baseline's zeros for
  normalized fits; changing them is a separate scientific change. This keeps
  the known convention until that change is explicitly integrated.
- Stashed scientific changes remain unapplied and preserved; this stage does
  not claim the broader stashed remediation is present.
- Ownership, fit bookkeeping, provenance/publication, and notebooks remain
  later roadmap work; the next ownership increment has its own proposed plan.
- Remote CI remains unexecuted without a remote. Local verification is recorded,
  and no claim of clean Linux-runner success is made.
