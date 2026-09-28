# Scientific readability refactor roadmap

> **Development plan:** Includes proposed work as well as implemented changes.
> See the [documentation index](../../README.md) for current API and methods guidance.

**Approved design:** [scientific-readability-design](../specs/2026-09-28-scientific-readability-design.md).

This roadmap keeps the full architectural scope visible. Each stage produces
working software and receives a concrete implementation plan before execution.
Stages 1 and 2 and the ownership/result/diagnostic portions of Stage 3 are implemented; the
[numerical/CV plan](2026-09-28-scientific-readability-numerics.md) records the second
stage. Later stages retain separate implementation plans. This is deliberate:
decisions about the numerical interfaces should inform the later fit and
workflow plans, rather than locking all of them into a large speculative rewrite.

## Shared rules

- Work on `refactor/scientific-readability`; main's baseline is `57acff5`.
- Keep stash `c4cfbec5fccd5397ea650c550f5b485ea588c2cd` intact. It contains
  independent scientific changes as well as documentation and notebook edits.
  Never apply it wholesale as part of this refactor.
- Use `uv`; run Python, pytest, and formatting through `uv run`.
- Commit failing tests before implementation. Existing committed failing tests
  satisfy the RED prerequisite for fixing the behavior they already exercise.
- Keep every `__init__.py` completely empty, including new subpackages.
- Prefer visible mathematical steps and explicit data over generic frameworks.
- Public API changes are authorized when they simplify the project. Ship
  migration examples with each change; preserve scientific interpretation.
- Do not change ridge objectives, fixed-effects weighting, HRF kernels, timing
  semantics, or coefficient units as incidental cleanup.

## Stage 1: A passing and portable baseline

Execute [the foundation plan](2026-09-28-scientific-readability-foundation.md).
Resolve the committed notebook/test mismatch, make default pytest collect NSD
tests, remove the notebook kernel from runtime dependencies, defer version
lookups, and add minimal CI plus an installed-wheel smoke check.

Exit: default full-suite tests pass with warnings as errors; the installed
library works outside the checkout without notebook dependencies. Document
any clean-runner failures rather than claiming CI passed without running it.

## Stage 2: Readable numerical preparation and candidate evaluation

Primary files: `_single_trial_fit.py`, `_fractional_ridge.py`, `_ridge_cv.py`,
`ridge_selection.py`, `fractional_ridge.py`, and their numerical/CV tests.

Separate nuisance projection, scale-aware identifiability checks, raw or
normalized decomposition, coefficient evaluation, and scoring. Replace the
coordinated `next()` protocol with an explicit object that evaluates a requested
candidate from stored run/group decompositions. Obtain fractional CV's OLS
validation target from that same state. Keep memory proportional to one
candidate's beta arrays plus decompositions, not to the whole grid.

Move shared numerical oracles to `tests/oracles.py` with an empty
`tests/__init__.py`; move reusable fixtures to conftest files instead of importing
test modules. Preserve independent augmented-lstsq and root-finding references.

Exit: candidate order/repeated evaluation is irrelevant; raw and normalized
ridge remain scientifically distinct; ill-scaled design decisions, beta units,
NaN masks, CV losses, selected penalties, and leakage safeguards remain covered.
Avoid an unconditional one-SVD target: rank validation may legitimately require
a scale-invariant decomposition separate from the raw regularization basis.

## Stage 3: Fits and results with less bookkeeping

The [ownership increment](2026-09-28-scientific-readability-ownership.md) is
implemented: ordinary read-only arrays, direct internal prepared-design access,
and local fractional-result ownership. The implemented
[result/diagnostic plan](2026-09-28-scientific-readability-results.md) consolidates
candidate scores and trial results while sharing fit diagnostics. Lifecycle
sharing is sequenced with Stage 4's logging/provenance changes because they
modify the same event and exception boundaries; it remains required work.

Primary files: `fit.py`, `prepared_fit.py`, `single_trial.py`, `_hrf_glm.py`,
`_selected_hrf_fit.py`, `prepared.py`, `data.py`, `_arrays.py`, `results.py`,
`single_trial_results.py`, `hrf_results.py`, `hrf_glm_results.py`, `ridge_results.py`.

Keep statistical steps visible in fit entry points; shared lifecycle operations
are implemented with Stage 4. Consolidate nested-OLS validation, rank diagnostics, and
contrast serialization. Internal access to a prepared run must not copy every
run's tables. Use owned arrays with normal read-only flags and explicitly
document their limits. Remove cross-module frozen-object mutation.

Unify candidate score containers with an explicit regularization kind and grid.
Use composition for genuinely shared trial results; avoid deep inheritance or
one class full of mutually exclusive fields. Recheck the baseline before
removing `_SelectionBoundaries`: that mixin exists in the preserved work, not
in the committed branch, so it is not a current refactor target.

Exit: all fit entry points retain scientific outputs and coherent lifecycle
records; table ownership is documented and avoids quadratic copying. API
migrations accompany result-schema changes.

## Stage 4: Proportionate provenance and publication

The [fit lifecycle, logging, and provenance plan](2026-09-28-scientific-readability-lifecycle.md)
is prepared for review. It covers shared lifecycle boundaries, categorical
failure logs, and removal of redundant provenance construction. Publication
remains a separate increment.

Primary files: `provenance.py`, `logging.py`, `publication.py`,
`bids_provenance.py`, and fit callers.

Share lifecycle operations through a small context manager while keeping
statistical work explicit in each caller. Remove redundant records and unused
event-history construction. Apply one
structured logging policy at the boundary; raw exceptions remain available to
callers, not exported logs. Preserve useful model, split, ordering, and version
metadata. Label metadata-based comparison checks accurately.

Publication supports local, cooperating scientific workflows: basic path
containment, protected inputs, explicit overwrite, one writer lock, staging
on the destination filesystem, per-file replacement, and recoverable-failure
rollback. Remove fd anchoring and recovery subsystems only after matching that
contract with tests and documentation. Never call a sequence of file replaces
an atomic set. If rollback itself fails, preserve backups and report their
location; do not silently delete the user's only recoverable data.

Keep the draft BIDS projection unless a separate requirements decision removes
it. Basic absolute-path exclusion in exported source references is still useful;
environment-substring redaction and generic path-shaped-string bans are not a
substitute for a clear output schema.

Exit: injected publish failures preserve prior files; output guarantees and
limitations match implementation; privacy behavior is consistent across fits.

## Stage 5: Package imaging workflows and simplify notebooks

Promote reusable NIfTI functions from `examples/stop_signal_demo.py` into
`src/boldtailor/nifti.py`. Promote the 21 non-test Python modules currently in
`examples/NSD/` into `src/boldtailor/nsd/`, retaining their focused responsibilities
initially. Keep the new subpackage initializer empty. Make plotting dependencies
an explicit imaging extra and use direct dependencies for library imports.

Move NSD tests into `tests/nsd/`, adjust imports, and test installed-package use.
Move notebook-instrumented scientific assertions into direct library tests;
retain one synthetic-data notebook execution smoke test. Preserve scientific
artifact values, spatial axes, feature order, source protection, and real-process
serial/parallel equivalence. Make BIDS paths environment-supplied with actionable
errors. Strip stored notebook outputs and personal paths from shared examples.

Exit: scientists can read a notebook's analysis sequence without dynamic import
or plotting audit code; reusable workflows can be imported from an installed
wheel; default pytest still runs the whole suite.

## Final branch review

Compare the branch with `57acff5`, distinguishing API changes from scientific
changes. Require numerical/CV regressions, installed-wheel checks, formatting,
and default full-suite pytest. Report eliminated duplication, confusing control
flow, and unnecessary copying with examples, rather than a line-count quota.
Maintain a migration guide and a short orientation to these process documents.
Do not delete unrelated branches or restored user files as repository hygiene.
