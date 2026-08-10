# Task 1 Review: Nuisance-Only Design Compilation

## Verdict

- **Spec Compliance:** Pass
- **Task Quality:** Pass
- **Ready:** Yes (based on the reviewed implementation and supplied execution
  evidence; tests were intentionally not rerun for this read-only audit).

## Findings

- **Critical:** None.
- **Important:** None.
- **Minor:** None.

## Evidence

- The linear commit sequence is test-only RED (`992038a`), production
  (`6c2299b`), then a tests-only formatting commit (`fac0498`).  At the RED
  commit, `tests/test_design.py` imports `compile_nuisance_designs`, while
  `src/boldtailor/design.py` in that same tree contains only
  `compile_designs`; the reported collection failure is therefore genuine.
- `compile_nuisance_designs` compiles every run independently
  ([design.py:30](../../../../../src/boldtailor/design.py#L30)); its helper
  selects only configured confounds ([design.py:65](../../../../../src/boldtailor/design.py#L65)) and calls Nilearn with `events=None` and `hrf_model=None`
  ([design.py:105](../../../../../src/boldtailor/design.py#L105)).  It passes the
  exact model drift, high-pass, drift-order, minimum-onset, and oversampling
  options ([design.py:109](../../../../../src/boldtailor/design.py#L109)).
- The nuisance metadata is intentionally event-free: zero exclusions and the
  frame-time-relative minimum-onset cutoff ([design.py:68](../../../../../src/boldtailor/design.py#L68)).
- Shared duplicate-column and finite-matrix validation preserves the original
  event compiler checks ([design.py:50](../../../../../src/boldtailor/design.py#L50),
  [design.py:122](../../../../../src/boldtailor/design.py#L122)); selected
  confound errors remain run-contextualized ([design.py:151](../../../../../src/boldtailor/design.py#L151)), and Nilearn failures are contextualized for the
  nuisance path ([design.py:116](../../../../../src/boldtailor/design.py#L116)).
- Tests assert event exclusion, selected-confound filtering, cosine drift and
  metadata ([test_design.py:57](../../../../../tests/test_design.py#L57)), exact
  no-drift Nilearn parity ([test_design.py:86](../../../../../tests/test_design.py#L86)),
  and missing/non-finite selected-confound behavior for both compilers
  ([test_design.py:119](../../../../../tests/test_design.py#L119)).
- The reviewed range changes only the requested production/test files plus the
  task report; it contains no preprocessing, chunking, dataset, or
  `__init__.py` change. `git diff --check` produced no output. The supplied
  report records 10 focused tests, 223 full-suite tests, and Black verification.

## Audit limitation

`uv run git` could not start in this sandbox because its cache contains an
inaccessible `.git` path. Read-only Git inspection was used only for history,
scope, and diff verification; no tests were rerun, per the review request.
