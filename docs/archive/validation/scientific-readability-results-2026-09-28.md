# Scientific readability: shared results and diagnostics

> **Dated validation record:** Results and test counts apply to the recorded
> revision/settings, not necessarily this checkout. See the
> [documentation index](../README.md) for current API and methods guidance.

Implementation base: `d4b6783`. Work is on `refactor/scientific-readability`.
The user approved the results/diagnostics plan and documented API changes.
Main remains at `57acff5`; stash
`c4cfbec5fccd5397ea650c550f5b485ea588c2cd` is preserved.

## Candidate scores

`CandidateScores` replaces two duplicate containers. The explicit
`regularization` kind distinguishes normalized from fractional ridge;
`grid` labels rows without reordering them. Public scoring functions still
validate/sort candidates. Selection-result fields and serialized
`alphas`/`fractions` keys are unchanged. NSD block assembly and output callers
use the new schema.

RED: the schema and CV suites produced **10 failures, 23 passes** because
`CandidateScores` and `grid` were absent. Tests committed as `146f4ba`
before implementation `2849939`.
GREEN: `uv run pytest tests/test_result_schemas.py tests/test_ridge_cv.py
tests/test_fractional_cv.py tests/test_within_run_cv_isolation.py
examples/NSD/test_ridge_workflow.py examples/NSD/test_fractional_workflow.py
-q -W error`: **56 passed** in 59.95 seconds. Independent CV oracles,
leakage guards, and serial/process NSD checks passed.

## Trial result composition

Both trial fitting entry points return `SingleTrialResult`. Common numerical,
metadata, and penalty fields remain directly accessible. Its `design` contains
a `SharedTrialDesign` (SPM or shared custom HRF) or `SelectedTrialDesign`
(assignments, grouped matrices, selection provenance). Each container owns its
inputs; public design tables/mappings are defensive copies. Removed the duplicate
`HrfSingleTrialResult`. Conventional GLM result APIs are unchanged.

RED: schema/ownership and trial suites produced **11 failures, 42 passes** on
missing containers and access paths. Tests committed as `5970398` before
implementation `3dbb021`.
GREEN: `uv run pytest tests/test_result_schemas.py tests/test_single_trial.py
tests/test_selected_hrf_fit.py tests/test_fractional_ridge.py tests/test_hrf_glm.py
examples/NSD -q -W error`: **241 passed** in 110.50 seconds. This includes
independent coefficient/R² oracles, both penalty families, selected HRFs,
NaNs, nested event metadata, custom shared HRFs, and all NSD tests.

## Shared fit diagnostics

Conventional and prepared fits now share nested-OLS tolerance/validation,
rank warnings, result dimensions, and contrast metadata. Input labels and error
messages are preserved. The no-op drift-order overwrite was replaced with a
copy of the same fingerprint payload. Grouped-HRF NaN validation remains
separate. Logging policies and fit lifecycle are unchanged.

RED: **4 failures, 30 passes** on missing shared interfaces. Tests committed
as `de516e0` before implementation `2b126fc`.
GREEN: `uv run --no-cache --no-sync pytest tests/test_fit_diagnostics.py
tests/test_fit.py tests/test_prepared_fit.py tests/test_logging.py
tests/test_hrf_glm.py -q -W error`: **117 passed** in 1.32 seconds. This
retains public parent/dimension, fingerprint, event, and contrast coverage
and checks the -1e-12/-1.1e-12 nested-OLS boundary.

Existing numerical expectations and tolerances were not relaxed. Existing
test access-path changes follow the approved API migration; new ownership
tests verify actual mutation isolation.

## Final verification and review

- `uv run pytest -q -W error`: **801 passed** in 165.64 seconds.
- `uv run --no-cache --no-sync black --check src tests examples/NSD
  examples/stop_signal_demo.py`: **113 files** passed.
- `git diff --check`: passed.
- `uv build --wheel --quiet`: exit 0.
- `uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl
  python tests/check_installed_package.py`: exit 0. Installed-package location,
  no ipykernel dependency, and independent OLS comparison passed.
- All project `__init__.py` files are empty.

An independent reviewer inspected `d4b6783..2b126fc` and surrounding
consumers. No Critical, Important, or Minor findings. It confirmed candidate
grid/kind alignment through NSD assembly, field-by-field trial-result mapping,
ownership, diagnostic labels/thresholds, and identical contrast fingerprint
payloads. It did not repeat the tests. Its approval was conditional on the
full suite and wheel smoke; both subsequently passed.

The user additionally requested a complete documentation audit. See
[the audit record](documentation-audit-2026-09-28.md). Subsequent Python changes
are docstrings only and notebook changes are Markdown only, verified against
the tested implementation's AST/code cells.

## Execution decisions and limits

- Continued in the requested branch/current checkout with exact-path staging;
  this provides less filesystem isolation than another worktree.
- Recorded observed successful test commands directly instead of repeating
  them solely through the skill ledger script, following the instruction to
  avoid redundant tests. The test evidence is unchanged.
- User documentation and executable consumers were migrated; historical plans
  and reviews retain historical API names.
- Lifecycle, logging, provenance simplification, publication, and imaging remain
  roadmap work. This increment completes the approved results/diagnostics plan.
- Remote CI has not run. The locked Nilearn integer-image limitation remains
  outstanding as recorded in foundation validation.


The reviewer explicitly set aside lifecycle/logging consolidation and
publication/imaging extraction. Both remain required roadmap work and were
unchanged by this increment. Deferring them means the larger architectural
refactor is still incomplete; this review approves only the result/diagnostic
increment.
