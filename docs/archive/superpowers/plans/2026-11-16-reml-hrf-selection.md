# REML HRF Selection Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task after the proposed design has been reviewed. Steps use checkbox syntax for tracking.

**Goal:** Make REML available alongside cross-validation for HRF selection in the installed package, including downstream fitting and the BIDS workflow.

**Architecture:** Keep `select_hrfs` as the public entry point and dispatch to separate CV and REML implementations. Share the HRF assignment, library, identity checks, and downstream fitting interface; keep statistical diagnostics specific to the selection method. Reuse the existing REML engine without changing its objective or optimizer.

**Tech Stack:** Python, NumPy, pandas, existing Nilearn convolution helpers, pytest, Hypothesis, and uv. No new dependencies.

**Spec:** Approved by the user on 2026-11-16. Implementation and verification are recorded in the execution ledger.

## Global constraints

- Scope is `src/boldtailor`, package tests, and documentation. Do not import, modify, or depend on older experiment code.
- Keep cross-validation as the default and preserve its numerical results.
- Write short functions with separate responsibilities. Every `__init__.py` remains empty.
- Use `uv run` for local commands and uv for package management.
- Write pytest tests first, observe the intended RED failure, and commit tests before implementation. Every test function needs a short purpose docstring.
- Never weaken a test or change the scientific problem to make a test pass.
- Use owned, read-only result arrays and copied table accessors, following current conventions.
- Do not interpret REML criteria or criterion gaps as R², p-values, or independent predictive performance.

## Proposed design

### Current package boundaries

`hrf_selection.select_hrfs` performs leave-one-run-out prediction and returns
`HrfSelectionResult`. That result requires three CV R² arrays. The assignment
validator and both selected-HRF fit paths require this result class and check
its library, spatial signature, and assignment fingerprint.

`reml.select_random_effects_design` already selects random-effect designs and
returns winning estimates, candidate indices, criterion gaps, and numerical
qualification flags. It requires a common fixed design and noise correlation
across candidates. Its candidate indices must be mapped explicitly to library
IDs. Its zero-variance and unidentifiable candidates do not win.

Other affected paths include independent split evaluation, HRF selection inside
ridge folds, workflow feature blocks, and writers that currently assume CV maps.
The low-level REML engine should remain usable independently.

### Alternatives and recommendation

1. Add a separate public `select_hrfs_reml` returning a separate result. This
   limits initial edits but requires every consumer to support two assignment
   interfaces.
2. Add a method argument to `select_hrfs`, retain one assignment result, and
   attach method-specific diagnostics. **Recommended:** users can change the
   selection method while retaining the same fitting and spatial checks.
3. Put REML values into existing CV score fields. Reject this approach because
   the quantities have different meanings and would be mislabeled in outputs.

The approved implementation follows option 2.

### Statistical model: decision to review before implementation

For run r and candidate h, use

```text
y_r = C_r beta_r + Z_r(h) b_r + epsilon_r
b_r ~ N(0, tau² I)
epsilon_r ~ N(0, sigma² R(rho))
```

`Z_r(h)` contains one unit-peak event-response column per trial, built using
the package's existing convolution and event-duration conventions. Do not
normalize these columns to unit length. `tau²` and `sigma²` are fitted per
feature and shared across runs, as in the existing engine. Trial coefficients
are independent across runs under this model; repeated stimulus IDs do not
create shared coefficients.

**Proposed fixed design:** run confounds, an intercept, and the requested
`TaskModel` columns convolved with one fixed reference HRF (canonical SPM by
default). Include missing-value indicators in that same reference basis.
Construct this once per run and reuse it unchanged across candidates.
This allows nonzero mean task responses while preserving comparable restricted
likelihoods. It selects a covariance shape for trial deviations conditional
on the specified mean model. It does not allow each candidate to change the
fixed mean-response shape.

This mean-model choice is the main scientific decision in the plan. A
canonical reference may influence selection when the actual mean HRF differs.
Test that sensitivity explicitly. If the intended model instead requires a
candidate-specific fixed mean, this engine's REML comparison is not sufficient;
that requires a separate statistical design rather than an adapter change.
Do not silently rebuild the fixed task columns for every candidate or omit
the mean response to avoid this issue.

Initially require a supplied scalar or per-feature `rho`; default to 0.0 and
document that this means independent noise. Automatic rho estimation is a
follow-up, because its reference, pooling, fold isolation, and dependence on
feature blocks require their own specification. A supplied estimate derived
from test data is the caller's leakage responsibility, stated in the API docs.

REML selection can use one run when the restricted model is estimable. CV
continues to require two. Wide or rank-deficient random designs are valid for
REML when their covariance is identifiable; do not impose single-trial OLS
full-rank rules on them. Unsupported event timing remains an input error.

### Public API

Create `boldtailor.hrf_options.RemlHrfOptions`, a frozen, owned value object:

```python
RemlHrfOptions(rho=0.0, reference_hrf_id=0)
```

Validate finite scalar/per-feature rho with `abs(rho) < 1`, its feature count
at the call site, and a reference ID present in the supplied library. Reject
booleans as numeric settings. Copy array inputs. Provide a helper to subset
per-feature options for workflow blocks, without changing run-level meaning.

Add keyword arguments to the existing signatures, retaining other arguments:

```python
select_hrfs(data, *, method="cv", reml_options=None, ...)
evaluate_hrf_split(data, *, method="cv", reml_options=None, ...)
score_fraction_candidates(data, predictors, *, hrf_method="cv", reml_options=None, ...)
score_ridge_candidates(data, predictors, *, hrf_method="cv", reml_options=None, ...)
```

Allowed methods are `cv` and `reml`. `method="reml"` with no options uses
`RemlHrfOptions()`. Reject REML options with CV, unknown method names, and a
nondefault HRF method when a ridge scorer has no HRF library. The existing
`candidate_batch_size` stays validated; REML streams one candidate at a time
and does not change its statistical result with this setting.

### Results and identities

Extend `HrfSelectionResult` with `method="cv"` and
`reml: RemlHrfDiagnostics | None = None`. Retain the existing CV field names
for compatibility, but allow their values to be `None`. CV results have all
three arrays and no REML diagnostics. REML results have all three CV fields
set to `None` and require REML diagnostics. Validate these combinations.

Define `RemlHrfDiagnostics` in `hrf_results.py` containing:

- `selection: RemlSelection`, including the winning `RemlFit` and all existing flags;
- `candidate_ids`, mapping engine positions to library IDs;
- `canonical_fit: RemlFit | None`, evaluated under exactly the same fixed design and rho;
- `delta_criterion`, canonical criterion minus selected criterion, NaN where either comparison is unavailable;
- `reference_hrf_id`.

Evaluate canonical once more with `fit_reml` if needed to retain its full
diagnostics; this avoids changing the streaming engine. The criterion gap
between eligible candidates remains the engine's `selection.gap`.
Retain canonical fit status alongside its criterion: a zero-variance canonical
fit can supply a baseline criterion but cannot win under current engine rules.

Common fields remain `hrf_indices`, library, labels, task model, feature
signature, provenance, eligibility, and parameter-bound diagnostics. Map no
winner to `-1`, with no silent canonical fallback. Retain provisional winners
and their flags; do not silently treat an unresolved search as a reliable fit.

Eligibility records must distinguish structural input validity from featurewise
REML status and downstream OLS estimability. Do not label REML eligibility as
CV eligibility. Unknown/invalid event inputs should raise an informative error,
not be swallowed as an ineligible candidate.

Record method, reference HRF, task model, fixed-design fingerprints, rho values
or an array digest plus saved values, covariance assumptions, normalization,
library identity, assignment identity, and numerical qualifications. Preserve
existing CV identities where possible; consumers reading legacy provenance
should interpret a missing method as CV. New REML identities must differ from
CV even when assignments happen to match.

### Prediction and nested selection

Split evaluation selects using training signals only. It then estimates the
existing shared task amplitudes on training runs with the selected HRFs and
predicts test runs. This remains an evaluation of mean task prediction, not
prediction of unobserved random trial deviations. Keep the existing `test_r2`
fields and explain that distinction.

Use method-specific run minima: CV requires two training runs; REML requires
one. Both need at least one disjoint test run. Model preparation or eligibility
may examine held-out timing/confounds under the existing split policy, but
held-out BOLD must not affect selection, rho, or training amplitudes. If a
selected design cannot support the prediction model, mark its evaluation
undefined with a reason; do not change the training winner using test data.

Propagate method and options to every inner and outer HRF selection in ridge
tuning and final refits. REML selection must be recomputed within each fold,
not reused from an all-run assignment. With a library, canonical/CV ridge
paths retain their existing minima; REML needs at least two total runs for
leave-one-run-out encoding. Existing predictor-rank requirements still apply.

### Workflow and scope of the first release

Add `WorkflowSettings.hrf_method="cv"`, `reml_rho=None`, and
`reml_reference_hrf_id=None`; REML resolves omitted values to 0.0 and 0.
The CLI initially accepts scalar rho only:

```text
--hrf-method cv|reml
--reml-rho FLOAT
--reml-reference-hrf-id INTEGER
```

Reject REML-only flags in CV mode. Include resolved settings in dry runs,
metadata, and the report's equivalent command. Preserve the existing default
CV output directory name. REML directory names append a method/rho/reference
suffix so automatic output paths cannot collide with CV or distinct REML
settings; explicit shared paths still use overwrite protection.

Keep CV output map names unchanged. REML outputs include IDs, criterion,
canonical criterion, delta criterion, runner-up gap, ratio, both variances,
rho, winning search flags, and any-candidate search flags. Save status codes
with a documented mapping and candidate-index mapping in metadata. No REML
selection map or report label should say CV R². Independent evaluation and
ridge encoding R² retain their existing names because they remain prediction
scores. Reports show counts of unsupported and provisional selections.

For odd/even reliability, REML needs at least one run in each half; CV keeps
two. Nested optimized ridge evaluation needs at least two runs in each half
with REML, versus three with CV. Retain all other workflow input checks.

Task-guided denoising keeps its current internal CV selection for this release.
Document this explicitly. Users may apply denoising and then select HRFs with
REML, but this does not turn the denoising procedure itself into REML tuning.
No changes to the REML optimizer, BLUP/trial-beta estimation, or automatic rho
estimation are included.

## Review focus

1. Candidate-specific mean or missingness regressors accidentally change the
   fixed span: test identical fixed matrices across candidates (Task 2).
2. A losing candidate has an unresolved optimum: preserve any-candidate flags
   through fitting, maps, and reports (Tasks 3 and 6).
3. Wide trial designs are valid for covariance selection but invalid for OLS
   betas: distinguish selection success from downstream fit failure (Task 3).
4. Nested tuning silently falls back to CV or sees test signals: instrument
   every fold and perturb held-out signals (Tasks 4 and 5).
5. Feature blocking changes a per-feature rho or candidate mapping: compare
   blocked, unblocked, serial, and parallel outputs (Tasks 3 and 6).

## Implementation sequence

For each task, the RED phase ends with a tests-only commit before product code
is written. Run the task's tests and the integration suite before each commit.
For a RED commit, record expected new failures and verify that existing tests
have no unexpected failures. After implementation, both must pass. Do not
change tests to accommodate incomplete behavior.

### Task 1: Define selection options and result contracts

**Files:** create `src/boldtailor/hrf_options.py`; modify
`src/boldtailor/hrf_results.py`; add `tests/test_hrf_options.py`; extend
`tests/test_result_schemas.py` and `tests/test_public_names.py`.

**Interfaces:** produce `RemlHrfOptions`, `RemlHrfDiagnostics`, and the extended
`HrfSelectionResult` described above. Preserve existing CV construction.

- [x] Write tests for unchanged CV fields, REML fields being absent for CV,
      `cv_r2 is None` for REML, rejected mixed diagnostics, immutable arrays,
      copied options, invalid rho/reference settings, and candidate ID mapping.
- [x] Run `uv run pytest tests/test_hrf_options.py tests/test_result_schemas.py tests/test_public_names.py -q -W error`; confirm failures identify missing contracts. Commit tests.
- [x] Implement the contracts, validation, and feature-subsetting helper.
- [x] Rerun those tests and integration; commit the implementation separately.

### Task 2: Build candidate-independent fixed designs and trial designs

**Files:** create `src/boldtailor/_hrf_reml.py`; reuse
`_task_design.py`, `_single_trial_design.py`, and `_hrf_design.py`; add
`tests/test_hrf_reml_design.py`.

**Interfaces:** `_hrf_reml.prepare_reml_groups(data, library, task_model,
options, labels) -> tuple[RemlGroup, ...]` and
`_hrf_reml.iter_trial_designs(data, library, candidate_ids, labels)` yielding
one tuple of random-design arrays per candidate. These own model preparation;
they do not select winners.

- [x] Write tests that fixed matrices include confounds, intercept, reference
      task columns, and reference missing indicators exactly once; assert they
      do not change when random candidate h changes.
- [x] Test sub-TR onsets, unequal durations, repeated event IDs, categorical
      modulators, ragged run lengths, absent levels, invalid event support,
      and matching peak normalization against existing trial designs.
- [x] Test that wide random designs reach REML instead of being rejected by
      OLS rank validation. Confirm the fixed span leaves residual degrees of freedom.
- [x] Run `uv run pytest tests/test_hrf_reml_design.py -q -W error`; observe RED and commit tests.
- [x] Implement preparation with existing helpers, short functions, and lazy
      candidate iteration. Do not materialize candidates × runs × scans × trials.
- [x] Run the new tests plus `tests/test_task_design.py tests/test_single_trial_design.py tests/test_hrf_design.py` and integration; commit.

### Task 3: Connect REML selection to the common assignment interface

**Files:** modify `src/boldtailor/hrf_selection.py`, `_hrf_reml.py`,
`_hrf_assignment.py`, and, where necessary, `_hrf_glm.py` and
`_selected_hrf_fit.py`; add `tests/test_hrf_reml_selection.py`; extend
`tests/test_hrf_selection.py`, `tests/test_hrf_glm.py`, and
`tests/test_selected_hrf_fit.py`.

**Interfaces:** implement `select_hrfs(..., method="cv", reml_options=None)`;
`_hrf_reml.select_reml_hrfs(...)` adapts the engine result into shared result
values and method-specific provenance. Keep lifecycle handling in the public entry point.

- [x] Test default versus explicit CV equality, one-run REML, two-run CV
      minimum, and errors for invalid method/option combinations.
- [x] Compare REML IDs, fit arrays, statuses, and gaps with direct engine calls
      on the same designs. Include a library with filtered/nonconsecutive IDs,
      exact ties, no supported winner, a single eligible candidate, and an
      unrefined losing candidate. Tie order follows stable library order.
- [x] Test that changes to method, reference, or rho change recorded identity;
      library, spatial, and assignment mismatches still fail validation.
- [x] Test REML-selected conventional and trial fits on estimable designs;
      undefined selections remain NaN. A wide-design OLS fit must give its
      existing clear estimability error without invalidating REML selection.
- [x] Run the new tests to observe RED; commit tests before implementation.
- [x] Implement dispatch, canonical comparison, winner mapping, diagnostics,
      and provenance without changing `reml.py` search behavior.
- [x] Run the affected test files, `tests/test_reml*.py`, and integration; commit.

### Task 4: Support independent split evaluation

**Files:** modify `src/boldtailor/hrf_selection.py` and its internal split
helpers; extend `tests/test_hrf_reml_selection.py` and `tests/test_hrf_cv.py`.

**Interfaces:** implement method/options on `evaluate_hrf_split`; retain
`HrfEvaluationResult` and mean-response prediction semantics.

- [x] Test one-training/one-test REML, unchanged CV minimum, disjointness,
      unsupported test designs, and canonical/selected prediction fields.
- [x] Perturb test BOLD and assert identical training HRF IDs, REML estimates,
      training amplitudes, and training identity. Test R² should change.
- [x] Compare predictions with a manual training-only amplitude fit; do not
      assert that REML selection must outperform CV on every simulation.
- [x] Observe RED and commit tests; implement train-only dispatch with separate
      selection eligibility and prediction eligibility.
- [x] Run `uv run pytest tests/test_hrf_reml_selection.py tests/test_hrf_cv.py tests/test_hrf_selection.py -q -W error` plus integration; commit.

### Task 5: Propagate selection into nested ridge tuning

**Files:** modify `src/boldtailor/fractional_ridge.py`, `ridge_selection.py`,
`_ridge_cv.py`, and `workflow/beta_series.py`; extend
`tests/test_fractional_cv.py`, `tests/test_ridge_cv.py`,
`tests/test_within_run_cv_isolation.py`, and `tests/workflow/test_beta_series.py`.

**Interfaces:** implement `hrf_method` and `reml_options` on both public scorers
and pass them through every inner selection, outer selection, and final refit.

- [x] Instrument fold selectors and assert the requested method/options and
      training run IDs on every call. Test method-specific run minima.
- [x] Test that outer test perturbations cannot change tuning or training
      selection, and that a whole-data assignment is never reused inside folds.
- [x] Test unchanged fixed-OLS fractional targets, candidate-regularized
      shared-alpha targets, and unchanged default CV results.
- [x] Observe RED and commit tests; implement option plumbing and fold metadata.
- [x] Run the four affected test files plus `tests/test_fractional_ridge.py
      tests/test_ridge_selection.py` and integration; commit.

### Task 6: Expose method selection in the workflow and saved results

**Files:** modify `src/boldtailor/cli.py` and `workflow/settings.py`,
`inputs.py`, `run.py`, `analysis.py`, `beta_series.py`, `outputs.py`,
`summaries.py`, `plots.py`, and `report.py`; extend matching tests under
`tests/workflow/` and `tests/test_parallel.py`.

**Interfaces:** settings and CLI flags specified above; method-aware selection
maps and report tables; existing CV outputs retained. Centralize map schemas
so block assembly and export cannot disagree about field order.

- [x] Test settings round trips, legacy settings defaulting to CV, invalid
      combinations, dry-run plans, equivalent commands, and distinct automatic
      output paths for method/rho/reference changes.
- [x] Test method-specific run minima for reliability and nested ridge, while
      retaining task-variable and predictor-rank input checks.
- [x] Test every REML output field, status mapping, unsupported/provisional
      counts, and absence of mislabeled CV selection maps. Round-trip saved
      diagnostics including NaNs through the existing strict JSON/image writers.
- [x] Test unequal feature blocks and per-feature API rho against unblocked
      results; compare serial and parallel output. The CLI remains scalar-rho only.
- [x] Add a small synthetic BIDS REML workflow test marked `integration` that
      reaches selection, downstream fitting, publication, and report rendering.
- [x] Observe RED and commit tests; implement settings-to-report propagation.
- [x] Run `uv run pytest tests/workflow tests/test_parallel.py -q -W error`
      plus integration; commit.

### Task 7: Verify statistical properties and document the public method

**Files:** add `tests/test_hrf_reml_properties.py` and
`tests/test_hrf_reml_statistics.py`; update `docs/api.md`, `docs/user-guide.md`,
`docs/practical-guide.md`, `docs/README.md`, `README.md`, and
`docs/result-migration.md`. Update the HRF row of `docs/testing.md` for new tests.

**Interfaces:** generated tests exercise the public adapter, complementing
existing engine properties; documentation uses only installed package APIs.

- [x] Add Hypothesis properties for feature/run permutation, signal scaling,
      fixed-span shifts, candidate-order mapping for decisive winners, and
      block/batch equivalence. Include nonzero rho, NaNs in excluded results,
      missing indicators, and retained search qualifications.
- [x] Compare adapter selection against independently constructed dense
      restricted-likelihood calculations on small, well-conditioned cases.
      Do not use the same adapter to construct the oracle designs.
- [x] Add seeded matched-model recovery tests across SNR and null data.
      Measure waveform recovery rather than only exact HRF-ID matches.
      Predeclare statistical tolerances from Monte Carlo uncertainty; avoid
      claims that null selection constitutes significance or that REML always
      selects the generating curve.
- [x] Add reference-mean misspecification cases and report their effect on
      selection. This validates the chosen model's limits rather than hiding
      them behind a favorable matched-model simulation.
- [x] Commit tests before any corrective implementation; run the new tests
      and existing engine properties. If a new property fails, diagnose the
      adapter or engine and retain the intended scientific assertion.
- [x] Document `method="reml"`, fixed-reference mean semantics, supplied rho,
      one-run selection, optional CV fields, provisional fits, independent
      evaluation, downstream OLS limits, and unchanged denoising internals.
- [x] Execute the new documented examples on synthetic inputs. Run integration
      and `uv run pytest -q -W error --durations=20` before merge, because the
      result contract has broad consumers and test-selection documentation changed.
- [x] Run `uv run pytest tests/test_reml_statistics.py --run-slow -q -W error`
      for the statistical acceptance check. Record all suites actually run;
      no experiment or notebook execution is needed for this package-only change.

## Acceptance criteria

- A user can choose `method="reml"`, obtain an identified HRF assignment, and
  pass it directly to conventional or single-trial fitting when those designs
  are estimable.
- Default CV behavior, values, and output map names are preserved.
- Fixed designs and rho do not change between candidate comparisons.
- Split evaluation and nested ridge selection do not use held-out BOLD for
  training choices, and never silently change selection method.
- Saved results identify their method, model assumptions, and all numerical
  qualifications; REML criteria are never labeled as prediction scores.
- The package tests cover engine agreement, independent oracles, invariance,
  failure cases, and a complete workflow. No experiment imports are introduced.

## Draft review and implementation boundary

The approved scope uses a fixed reference mean and initially supplied rho.
Automatic rho estimation and candidate-specific fixed means would each change
the scientific scope and should be specified explicitly if required.

## Implementation verification

Completed on branch `feature/reml-hrf-selection`. Tests were committed before
implementation for Tasks 1–6. Task 7 added acceptance tests for the completed
adapter; they passed without corrective implementation.

- Full default suite: 2,567 passed, 9 skipped, 2 expected failures.
- Integration suite: 23 passed.
- Workflow and parallel tests: 291 passed.
- Engine and public property checks: 17 passed.
- Opt-in REML statistical suite (`--run-slow`): 17 passed.
- Synthetic guide examples: selection, trial fitting, and split evaluation passed.
- Package `__init__.py` files remain empty.
- Fresh read-only review: no critical or important findings; one minor
  provenance follow-up below. No historical experiment suite was run.

### Execution decisions and deferred work

- Ruling: Native managed worktree with original environment and explicit PYTHONPATH — avoid reinstalling or modifying the user's environment — incorrect path resolution would test the wrong checkout; verify imported module path.
- Task 2 Ruling: Build fixed task columns by aggregating existing unit-peak trial regressors and expanded task amplitudes — Nilearn emits null-duration warnings for valid impulses under -W error — avoids warnings while keeping exact trial-response normalization; risk is expansion-order mismatch, covered by independent weight assertions.
- Task 3 Ruling: Library IDs are contiguous and the adapter does not filter structurally valid candidates — test explicit mapping in diagnostics and engine permutation/tie tests rather than inventing a public filtered-library feature — risk is future filtering, for which the mapping remains explicit.
- Task 5 Ruling: Add task_model to ridge scorer keywords and propagate the selection model for REML workflow folds — otherwise inner REML fixed means differ from outer/final means — CV workflow defaults remain task-only as before; risk is a newly exposed modeling option, covered by fold provenance and engine comparisons.
- Task 7 Ruling: Statistical acceptance tests characterize the already implemented adapter and passed on first execution; no corrective implementation was needed — preserve these tests without inventing a RED failure — their coverage is complementary to earlier tests-first implementation gates.
- Final: minor (deferred): Fingerprint random-design inputs in addition to fixed designs for fuller provenance; no concrete model collision or incorrect result established.
- Final: Ruling: Unchanged REML optimizer — retain existing engine and statistical acceptance coverage; expanding optimizer scope is unnecessary — risk is an engine defect outside adapter tests.
- Final: Ruling: Automatic rho estimation — remain supplied-only as approved, with training-only estimates required for independent evaluation — risk is callers supplying a contaminated or poor estimate.
- Final: Ruling: BLUP/trial-beta estimation — keep existing downstream OLS/ridge fitting and estimability checks — risk is confusing covariance selection with trial-effect estimation, addressed in docs.
- Final: Ruling: Denoising — retain its internal CV procedure as approved — risk is assuming the HRF method flag also changes denoising.
- Final: Ruling: Historical experiments and notebooks — exclude from implementation and validation as requested — risk is untested compatibility in those external callers.
