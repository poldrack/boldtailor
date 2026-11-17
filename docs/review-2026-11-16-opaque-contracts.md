# Survey of opaque contracts and implicit behavior

Reviewed 2026-11-16 against `ea3b243` on `refactor/fit-operation-lifecycle`,
including the recent logging/provenance separation. That refactor is not yet on
main. This survey made no production-code changes.

## Scope and method

Searched the runtime package for custom context managers, context-local/global
state, reflective dispatch, mutable result assembly, caches, rebuild closures,
exception conversion, and implicit argument handling. Also scanned examples and
scripts for custom context-manager definitions. Read relevant implementations
and callers; ran four small behavioral probes using the worktree's uv environment
and temporary files. This is a targeted design survey, not an exhaustive
correctness audit. Findings distinguish observed behavior from maintenance risks.

There are three custom context managers in the runtime package:
`fit_operation`, `bind_context`, and `_input_errors`. No additional mandatory
method-call completion protocol was found. `_input_errors` transparently converts
input exceptions and has no body-side protocol. File handles, writer locks,
parallel worker scopes, and context restoration are generally appropriate uses
of `with`.

## 1. Deprecated keyword handling depends on object identity

**Priority: high; confirmed inconsistent behavior.**

Sources: [`_deprecation.py:19`](../src/boldtailor/_deprecation.py#L19),
[`single_trial.py:34`](../src/boldtailor/single_trial.py#L34).

`renamed_keyword` promises to reject simultaneous old/new keyword arguments,
but tests whether the new value `is not default`. `fit_single_trials` supplies
`"spm"` as an actual default rather than retaining whether it was omitted.
Consequently, explicitly supplying both names can silently let the old name win.
Worse, equal string values behave differently depending on their identity.

Probe of the helper used by the public function:

- Old value `"old-value"`, new value the same `"spm"` object as the default:
  returns `"old-value"` with a deprecation warning.
- Same old value, new value `"".join(["s", "p", "m"])`:
  raises `TypeError`, despite being equal to `"spm"`.

**Recommendation:** use `UNSET` for the new parameter as well. Detect whether
both were supplied before resolving defaults. Value equality or object identity
cannot reliably recover whether a caller supplied an argument.

## 2. Publication's transaction boundary excludes stale-file cleanup

**Priority: high; confirmed partial-success failure behavior, inferred concurrency risk.**

Sources: [`workflow/outputs.py:1123`](../src/boldtailor/workflow/outputs.py#L1123),
[`workflow/outputs.py:95`](../src/boldtailor/workflow/outputs.py#L95),
[`publication.py:65`](../src/boldtailor/publication.py#L65).

`publish_workflow` reads the previous manifest, calls the locked/rollback-capable
`publish_artifact_set`, and then deletes stale files after that call has returned
and released its lock. The apparent publication operation therefore has two
separate failure boundaries. Injecting an error into `_remove_stale` produced a
raised `PermissionError` while both the new artifact and report already existed.

The unlocked cleanup also permits another writer to publish between the commit
and cleanup; the earlier cleanup could then delete a file the later writer
requires. This race follows from lock placement; it was not exercised with
concurrent processes in this survey.

Related opacity: `_listed_artifacts` treats absent, unreadable, malformed, and
wrongly shaped manifests as the same empty list. A corrupt manifest can silently
prevent cleanup, leaving a mixture of older and newer outputs.

**Recommendation:** include old-manifest reading and stale removals in the same
locked transaction, with rollback for deletions. Treat missing manifests
separately from corrupt/unreadable ones. Do not suggest reader-atomic publication:
the low-level API explicitly documents that readers can see partial replacement.

## 3. Workflow stages have unstated input-state requirements

**Priority: medium-high; maintenance risk, not a failure in the current stage order.**

Sources: [`workflow/run.py:53`](../src/boldtailor/workflow/run.py#L53),
[`workflow/run.py:223`](../src/boldtailor/workflow/run.py#L223),
[`workflow/run.py:426`](../src/boldtailor/workflow/run.py#L426).

Stages accept one broad mutable `_State`, populate fields, and implicitly depend
on earlier stages having populated other fields. Their signatures do not identify
required inputs or produced results. Most fields default to `None` or empty
containers, allowing incomplete states to look valid.

A particularly misleading failure mode: `_glm_stage` passes `state.selections`
to the fit labeled `OptimizedGLM`. If selection has not run, this is `None`, which
`analysis.fit_glms` interprets as canonical fitting. A stage-order mistake could
therefore fit the canonical model under the optimized label rather than fail.
The current `_STAGES` order is correct; enabled-stage validation does not express
this readiness condition in the stage's interface.

**Recommendation:** pass required results explicitly and return named stage
results. As an incremental safeguard, require non-None selections for the
optimized stage. Avoid replacing this with a more elaborate implicit state machine.

## 4. The final ridge fit relies on the last iteration of a loop

**Priority: medium; maintenance risk with a small, clear fix.**

Source: [`workflow/beta_series.py:502`](../src/boldtailor/workflow/beta_series.py#L502).

The odd, even, and all-run tuning loop repeatedly overwrites `alpha`. After the
loop, `_final_fit` uses that variable. It is the all-run choice only because
`"all"` is currently appended last. Reordering the loop would quietly change
which penalty is used for the final fit while the provenance-link call still
names the all-run tuning decision. `alpha` also holds a fraction array in
fractional mode, making its meaning harder to follow.

**Recommendation:** explicitly extract `final_selection` from
`result["tuning"]["all"]` and derive a clearly named final penalty. Alternatively,
separate outer evaluation from all-run tuning/refitting.

## 5. Linking provenance mutates the supplied result and is not repeatable

**Priority: medium; confirmed side effect and hidden record-layout dependency.**

Source: [`workflow/beta_series.py:642`](../src/boldtailor/workflow/beta_series.py#L642).

`link_final_provenance(result, decision)` returns nothing and replaces each
nested `block["record"]` in place. Each call adds an activity and generates a
new execution ID. In a probe, two identical calls grew the activity history
from one to two to three activities and changed the ID on both calls.
For identified inputs, the second call also hashes an already-linked analysis
as though it were the original fitting analysis.

The helper also expects the last decision activity to contain `encoding_mode`.
`tuning_provenance` has the same positional dependence on the final scoring
activity. Appending an otherwise legitimate activity could break these consumers.

**Recommendation:** construct and return a new linked record/result; make
repeated application explicitly rejected or idempotent if supported. Pass the
encoding configuration directly or retrieve a named/typed decision activity,
rather than assuming `activities[-1]` is the desired record.

## 6. `None` in logging context means “inherit,” not “clear”

**Priority: medium-low; confirmed surprising semantics, not another completion protocol.**

Source: [`logging.py:24`](../src/boldtailor/logging.py#L24).

`bind_context` drops None-valued updates and merges the rest into the enclosing
context. Thus `bind_context(analysis_id=None)` inside a scope with an analysis ID
retains that ID. The same convention applies to `emit_event` overrides.
`inherit=False` clears everything, but there is no selective-clear argument.

The developer guide describes additive inheritance, so the behavior is not
wholly undocumented; the function itself has no docstring making the None
semantics explicit. The corrected fit context already uses `inherit=False`,
so this does not undo the recent fix.

**Recommendation:** document this prominently, or use an omission sentinel so
explicit None can clear a field. Prefer an explicit clear API if compatibility
requires keeping existing None semantics. Retain the context manager itself:
scoped restoration is a sensible use of it.

## 7. Some result objects contain executable reconstruction recipes

**Priority: lower; documented design tradeoff that remains easy to overlook.**

Sources: [`single_trial_results.py:28`](../src/boldtailor/single_trial_results.py#L28),
[`hrf_glm_results.py:17`](../src/boldtailor/hrf_glm_results.py#L17),
[`_hrf_glm_design.py:53`](../src/boldtailor/_hrf_glm_design.py#L53),
[`parallel.py`](../src/boldtailor/parallel.py).

Selected-HRF results contain closures that retain timing/event/nuisance inputs
or prepared runs. `group_design()` recompiles a design; `design.matrix()`
reconstructs it using cached trial matrices. These objects are therefore not
simply portable bundles of already computed arrays: access can perform substantial
work, captured inputs stay alive, and standard pickle cannot serialize the
closures. The package documents these limitations, including cloudpickle support.

**Recommendation:** keep lazy reconstruction if its memory benefits matter, but
consider explicit `rebuild_*` names, an optional materialization step, and a
serializable recipe object in place of closures. This is not evidence of wrong
numerical results or a reason to eagerly store every design by default.

## 8. Report construction implicitly exports the local namespace

**Priority: low; readability/maintenance issue.**

Source: [`workflow/report.py:321`](../src/boldtailor/workflow/report.py#L321).

`data = dict(locals())` turns parameter/local variable names into the string-key
schema consumed by `_bodies`. A local rename can become a runtime dictionary-key
failure; unrelated locals automatically enter the container. The dependencies
are not expressed by a named data type or explicit construction.

**Recommendation:** build an explicit mapping or a small report-input dataclass.
This is a contained cleanup, not a reason to redesign the reporting system.

## Other patterns considered

- `redirect_stdout` in `workflow/analysis.py:100` suppresses output for the whole
  process while a fit runs. Its purpose is explained, and process workers limit
  exposure, but a serial call from a multithreaded application could suppress
  unrelated output. Treat this as a lower-priority integration caveat.
- Defensive copies, readonly-array flags, and `object.__setattr__` during frozen
  dataclass initialization mostly implement the documented ownership contract.
  They are not analogous to a hidden mandatory completion call.
- Data-independent design caches and explicit eligibility checks have legitimate
  performance/scientific purposes. They should not be removed merely for using
  mutable cache internals.
- The existing `fit` return-type dispatch is an API choice already discussed;
  it is distinct from a hidden context-manager protocol.

## Suggested order

Fix keyword/default detection and publication boundaries first. Then make the
final ridge choice explicit and remove provenance-linking mutation. Address
stage input/output contracts as a focused workflow refactor. Logging-context
semantics, result reconstruction APIs, and report input cleanup can follow.

No fixes were applied in this survey. The four probes exercised only temporary
or in-memory objects; real datasets and saved workflow outputs were untouched.
