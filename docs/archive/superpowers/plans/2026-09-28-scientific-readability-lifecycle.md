# Fit Lifecycle, Logging, and Provenance Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` with the preserved native execution method. Implement inline, with one independent review after the complete increment. Steps use checkbox (`- [x]`) syntax.

**Goal:** Make fitting code easier to read by sharing lifecycle bookkeeping, reporting failures consistently, and removing redundant provenance construction.

**Architecture:** A small private context manager surrounds each fit's explicit scientific operations. It owns execution IDs, start/failure/completion events, and bounded history. Result construction and provenance validation stay inside the failure boundary. Logging exports fixed error categories; callers receive the original exceptions. Provenance retains its scientific fields and identity formulas while avoiding serialization round trips.

**Tech Stack:** Python >=3.12, NumPy, pandas, pytest, uv; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-28-scientific-readability-design.md`

**Status:** Implemented; final verification and independent review are recorded in `docs/validation/scientific-readability-lifecycle-2026-09-28.md`. Base `3188f7f`; 801 tests passed before this increment.

## Global constraints

- Keep main at `57acff5`, the preserved stash, and unrelated untracked files unchanged. Continue on `refactor/scientific-readability`.
- Use `uv run` for Python, pytest, and formatting. Every `__init__.py` stays empty.
- Write, observe, and commit failing tests before implementation. Changes to existing logging assertions reflect the explicitly changed log contract, not weakened scientific requirements.
- Preserve numerical formulas, independent oracles, tolerances, coefficient units, HRF choices, regularization objectives, feature/run ordering, NaNs, and source/analysis fingerprint payloads.
- Preserve detailed exceptions and their identity/tracebacks for callers. Do not inspect environment values or stringify an exception to construct a log record.
- Keep the eight-event provenance bound. Keep existing successful activity metadata and source descriptors; newly added lifecycle events are documented metadata changes.
- Keep source URI containment and current JSON-safe metadata validation. This increment does not redesign arbitrary user annotations, the provenance schema, BIDS projection, or file publication.
- HRF-selection and CV routines retain their existing selection records. This increment unifies fitting and comparison lifecycles; it does not add per-candidate or per-feature progress logging.
- Update current documentation alongside implementation. Historical records remain labeled snapshots.

## Concrete contract

### Failure logs

`emit_event(..., error=...)` accepts an exception or legacy string but never exports its text, repr, traceback, or class name. New failure records use `error_code`:

| Input | Code |
| --- | --- |
| `ValueError` or `TypeError` | `invalid_input` |
| `ArithmeticError` or `numpy.linalg.LinAlgError` | `numerical_failure` |
| `OSError` | `io_failure` |
| Other exception or legacy string | `operation_failed` |

No `error` field is emitted in new records. `error=None` adds no category. Custom exception subclasses inherit the listed category without exposing their names. Existing serialized historical records are not rewritten. Structured IDs/event names remain caller-supplied metadata; categorical errors are not a promise that arbitrary annotations are safe to share.

### Lifecycle boundary

All eight fit/comparison paths use one boundary:

| Entry point | Event prefix |
| --- | --- |
| Conventional `fit` | `fit` |
| Prepared `fit_prepared` | `fit` |
| Selected-HRF conventional fit | `fit` |
| Conventional and selected-HRF task comparison | `task_delta_r2` |
| Prepared task comparison | `task_delta_r2_prepared` |
| Shared-HRF trial fit | `single_trial` |
| Selected-HRF trial fit | `selected_hrf_single_trial` (new) |

Each attempt emits one start followed by one completion or one failure. Validation, design compilation, estimation, metadata, and result construction are inside the boundary. Start records identify execution and available source metadata; analysis identity may not yet exist and is omitted. Completion includes the final analysis identity when available. Failures include the identity if already established. This intentionally removes the old requirement that ordinary/prepared start records already contain an analysis ID.

A nested operation starts a fresh ID scope; absent source/analysis/run IDs must not inherit unrelated outer values. Exiting restores the prior context on success and failure. Ordinary `bind_context` retains its existing additive default.

Completion events are prepared for the returned provenance, but emitted only after provenance and result construction succeed. A failed constructor must never leave a misleading completion log. An unused reserved sequence number after failure is acceptable; sequence numbers need not be contiguous. No numerical arrays are recopied merely to attach completion history.

### Provenance simplification

Replace `extend_provenance`'s `to_dict()` → `from_dict()` round trip with direct dataclass construction/replacement. Preserve extension fields, warning order/deduplication, activities, sources, and analysis IDs. Keep validation at construction.

For both input constructors, compute source metadata identity directly using the existing fingerprint algorithm. Construct one final record instead of an identity-only record, a discarded validation record, and a final record. Stage the completion event before final construction and emit it only after the owned input object exists. A metadata/result-construction failure produces failure, not completion.

## Files and responsibilities

| File | Change |
| --- | --- |
| `src/boldtailor/logging.py` | Fixed error categories; separate private event creation/emission; optional fresh context binding |
| New `src/boldtailor/_fit_lifecycle.py` | One small fit-operation context and provenance preparation |
| `fit.py`, `prepared_fit.py`, `_hrf_glm.py` | Explicit science inside shared lifecycle; remove parallel wrappers and sanitizers |
| `single_trial.py`, `_selected_hrf_fit.py` | Shared lifecycle, including validation and construction; selected-trial events |
| `provenance.py` | Direct record extension and reuse of source fingerprint computation |
| `data.py`, `prepared.py` | One record construction per successful input normalization |
| `tests/test_logging.py`, `tests/test_prepared_fit.py` | Updated category/start-ID expectations; preserved exception and privacy checks |
| New `tests/test_fit_lifecycle.py`, shared fixtures in `tests/conftest.py` | Lifecycle outcomes, constructor failures, nested context restoration |
| `tests/test_provenance.py`, `tests/test_data.py`, `tests/test_prepared.py` | Identity/extension preservation and normalization construction behavior |
| `docs/api.md`, `docs/development.md`, new `docs/lifecycle-migration.md` | Current logging and metadata contract |

## Review focus

1. Late provenance/result-construction failures must produce failure without completion; exception identity must survive (Tasks 2–3).
2. Nested fits, including anonymous inputs, must not inherit another analysis/run identity and must restore the outer context (Tasks 2–3).
3. Errors containing private text, custom class names, or a broken `__str__` must not leak or replace the original exception (Tasks 1–3).
4. Direct provenance extension must preserve additive fields, deterministic identities, source order, and owned nested metadata (Task 4).
5. Normalization metadata failures must be caught before completion; reducing record construction must retain validation and quality warnings (Task 4).

## Task 1: One error policy at the logging boundary

- [x] Add the following tests to `tests/test_logging.py`; keep the existing logger-configuration and context tests.

```python
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ValueError("private /home/person/data"), "invalid_input"),
        (TypeError("private-value"), "invalid_input"),
        (FloatingPointError("private-value"), "numerical_failure"),
        (np.linalg.LinAlgError("private-value"), "numerical_failure"),
        (OSError("private-value"), "io_failure"),
        (RuntimeError("private-value"), "operation_failed"),
        ("legacy private-value", "operation_failed"),
    ],
)
def test_failure_events_export_only_fixed_categories(caplog, error, code):
    caplog.set_level(logging.INFO, logger="boldtailor")
    event = emit_event("fit_failed", stage="fit", level=logging.ERROR, error=error)
    assert event["error_code"] == code
    assert "error" not in event
    assert "private" not in caplog.text


def test_failure_logging_does_not_stringify_exceptions(caplog):
    class UnprintableError(ValueError):
        def __str__(self):
            raise AssertionError("exception text must not be inspected")

    event = emit_event("fit_failed", stage="fit", error=UnprintableError())
    assert event["error_code"] == "invalid_input"
    assert "UnprintableError" not in caplog.text
```

- [x] Update existing assertions on newly emitted `record["error"]` to `error_code`, retaining assertions about the raw raised exception, paths, values, context restoration, and absence of data in logs. Do not alter historical `ProvenanceRecord.from_dict` round-trip expectations.
- [x] Run `uv run pytest tests/test_logging.py tests/test_prepared_fit.py -q -W error`; observe missing category/unsafe formatting failures and commit tests.
- [x] Add `_error_code(error)` using the table above. `_event_payload` stores the category instead of text; `_history_entry` retains `error_code`. Change current normalization/fit callers from `error=str(error)` and `error=_sanitize_error(error)` to `error=error`. Remove prepared-fit regex/environment sanitizers and their now-unused imports.

```python
def _error_code(error):
    if isinstance(error, (ArithmeticError, np.linalg.LinAlgError)):
        return "numerical_failure"
    if isinstance(error, (ValueError, TypeError)):
        return "invalid_input"
    if isinstance(error, OSError):
        return "io_failure"
    return "operation_failed"
```

- [x] Add `docs/lifecycle-migration.md`: new `error_code` values, removal of exported text, original exceptions unchanged, historical records unchanged. Run the same targeted suites, format, and commit after GREEN.

## Task 2: A small shared fit-operation context

**Interfaces:** Private `fit_operation(name, parent)` yields an operation with `execution_id`, `analysis_id`, and `provenance(activity, *, analysis_id, warnings=())`. Parent is a `ProvenanceRecord`; its source fingerprint and event history initialize the operation. The returned provenance is constructed once and already includes the prospective completion event.

- [x] Create `tests/test_fit_lifecycle.py` with a fixture returning `ProvenanceRecord(execution_id=str(uuid4()))`. Import only public record types and the new private lifecycle module, not another test module.

```python
def test_late_constructor_failure_never_logs_completion(caplog, parent_record):
    from boldtailor._fit_lifecycle import fit_operation

    caplog.set_level(logging.INFO, logger="boldtailor")
    failure = ValueError("private constructor detail")
    with pytest.raises(ValueError) as caught:
        with fit_operation("fit", parent_record) as operation:
            operation.provenance({"name": "fit"}, analysis_id=None)
            raise failure
    assert caught.value is failure
    records = [json.loads(r.getMessage()) for r in caplog.records]
    assert [r["event"] for r in records] == ["fit_started", "fit_failed"]
    assert records[-1]["error_code"] == "invalid_input"
    assert "private" not in caplog.text


def test_completion_matches_returned_provenance(caplog, parent_record):
    from boldtailor._fit_lifecycle import fit_operation

    caplog.set_level(logging.INFO, logger="boldtailor")
    with fit_operation("fit", parent_record) as operation:
        record = operation.provenance({"name": "fit"}, analysis_id="a" * 64)
        assert [json.loads(r.getMessage())["event"] for r in caplog.records] == [
            "fit_started"
        ]
    completed = json.loads(caplog.records[-1].getMessage())
    assert dict(record.events[-1]) == completed
    assert completed["execution_id"] == record.execution_id
    assert completed["analysis_id"] == record.analysis_fingerprint
```

- [x] Add parametrized success/failure tests under `bind_context(execution_id="outer", data_id="outer-data", analysis_id="outer-analysis", run_index=9)`: an anonymous child omits the latter three fields; an event after child exit restores all four outer values. Add a child-inside-child test with distinct UUIDs and no cross-contamination. Verify history retains its last eight events.
- [x] Add a provenance-construction failure test by injecting a raising `extend_provenance`; assert original exception and only start/failure. A context exited without preparing provenance raises `RuntimeError` and logs failure. Preparing provenance twice likewise raises rather than emitting duplicate completions.

```python
@pytest.mark.parametrize("fail", [False, True])
def test_anonymous_child_restores_outer_context(caplog, parent_record, fail):
    from boldtailor._fit_lifecycle import fit_operation
    from boldtailor.logging import bind_context, emit_event

    caplog.set_level(logging.INFO, logger="boldtailor")
    outer = dict(execution_id="outer", data_id="outer-data",
                 analysis_id="outer-analysis", run_index=9)
    failure = ValueError("private child detail")
    with bind_context(**outer):
        try:
            with fit_operation("fit", parent_record) as operation:
                if fail:
                    raise failure
                operation.provenance({"name": "fit"}, analysis_id=None)
        except ValueError as caught:
            assert fail and caught is failure
        restored = emit_event("after_child", stage="fit")
    child_records = [json.loads(r.getMessage()) for r in caplog.records[:-1]]
    assert [r["event"] for r in child_records] == [
        "fit_started", "fit_failed" if fail else "fit_completed"
    ]
    for record in child_records:
        assert record["execution_id"] != "outer"
        assert not {"data_id", "analysis_id", "run_index"} & record.keys()
    assert all(restored[key] == value for key, value in outer.items())


@pytest.mark.parametrize("mode", ["missing", "duplicate", "construction"])
def test_invalid_finalization_logs_failure(caplog, parent_record, monkeypatch, mode):
    import boldtailor._fit_lifecycle as lifecycle

    caplog.set_level(logging.INFO, logger="boldtailor")
    failure = RuntimeError("private provenance detail")

    def reject_provenance(*args, **kwargs):
        raise failure

    if mode == "construction":
        monkeypatch.setattr(lifecycle, "extend_provenance", reject_provenance)
    with pytest.raises(RuntimeError) as caught:
        with lifecycle.fit_operation("fit", parent_record) as operation:
            if mode != "missing":
                operation.provenance({"name": "fit"}, analysis_id=None)
            if mode == "duplicate":
                operation.provenance({"name": "fit"}, analysis_id=None)
    if mode == "construction":
        assert caught.value is failure
    records = [json.loads(r.getMessage()) for r in caplog.records]
    assert [r["event"] for r in records] == ["fit_started", "fit_failed"]
    assert records[-1]["error_code"] == "operation_failed"
```

- [x] Run `uv run pytest tests/test_fit_lifecycle.py -q -W error`, observe absent-module failures, commit tests.
- [x] In `logging.py`, extract `_emit_record(payload)` from `emit_event`; retain one JSON serialization path. Extract `_make_event(...)` with the existing event fields and defaults so a completion can be prepared without emission. Add `inherit=True` to `bind_context`; when false, start from an empty mapping rather than the outer context.
- [x] Implement `_fit_lifecycle.py` using a dataclass and context manager, not callbacks for scientific work or result-type introspection. The control flow is:

```python
@contextmanager
def fit_operation(name, parent):
    operation = FitOperation(name, parent)
    with bind_context(
        execution_id=operation.execution_id,
        data_id=parent.metadata_fingerprint,
        inherit=False,
    ):
        operation.history = append_event_history(
            parent.events, emit_event(f"{name}_started", stage="fit")
        )
        try:
            yield operation
            if operation.completed is None:
                raise RuntimeError("fit operation has no completed provenance")
        except Exception as error:
            emit_event(
                f"{name}_failed", stage="fit", level=logging.ERROR,
                error=error, analysis_id=operation.analysis_id,
            )
            raise
        else:
            _emit_record(operation.completed)
```

`FitOperation` stores `name`, `parent`, fresh `execution_id`, `analysis_id=None`, `history=()`, and `completed=None`. Its `provenance` method rejects a second call, sets `analysis_id`, prepares a completion using `_make_event`, calls `extend_provenance` with `append_event_history(history, completion)`, then stores the pending completion and returns the record. Set `completed` only after successful record construction. The caller immediately constructs/returns its result inside the context. No extra array copies and no mutable result attachment hook.

- [x] Run `uv run pytest tests/test_fit_lifecycle.py tests/test_logging.py -q -W error`; format and commit GREEN.

## Task 3: Migrate all fit and comparison paths

- [x] Add integration coverage beside the existing fixtures in `tests/test_fit.py`, `tests/test_prepared_fit.py`, `tests/test_hrf_glm.py`, `tests/test_single_trial.py`, and `tests/test_selected_hrf_fit.py`. Use each file's actual fitting fixture; do not import those test modules from one another.
- [x] For each of the eight tabled entry points, assert one start and completion on success, matching final execution/source/analysis identities, bounded history, and omission of an unknown start analysis ID. Compare scientific outputs using the existing oracles without tolerance changes.
- [x] Exercise late construction failure at these real boundaries: `fit.make_result`, `prepared_fit.make_result`, `_hrf_glm._assemble`, `single_trial._assemble_result`, and `_selected_hrf_fit.SingleTrialResult`. Each injected constructor raises the same sentinel exception; check exception identity and absence of a completion. For each comparison, inject failure in its shared provenance extension after its numerical comparison exists. Use this assertion pattern with a local fixture/call for the owning API:

```python
failure = ValueError("private late result failure")

def reject_result(*args, **kwargs):
    raise failure

monkeypatch.setattr(fit_module, "make_result", reject_result)
caplog.clear()
with pytest.raises(ValueError) as caught:
    fit(data, model)
assert caught.value is failure
records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "boldtailor"]
assert [r["event"] for r in records] == ["fit_started", "fit_failed"]
assert "private" not in caplog.text
```

- [x] Also exercise early failures: incompatible feature signature, invalid trial penalty/run labels, invalid prepared contrasts/model metadata, and mismatched comparison parent. These must start/fail even when no analysis identity was computed. Commit observed RED tests before migration.
- [x] Put scientific work inside `with fit_operation(...)`. Example conventional flow:

```python
with fit_operation("fit", data.provenance) as operation:
    model_provenance = _model_provenance(model)
    compiled, numerical = _fit_analysis(data, model)
    provenance = operation.provenance(
        _fit_activity(data, model_provenance.activity, compiled),
        analysis_id=_analysis_id(data.provenance.metadata_fingerprint,
                                 model_provenance.fingerprint),
        warnings=model_provenance.warnings,
    )
    return make_result(
        numerical.contrasts,
        tuple(design.matrix for design in compiled),
        tuple(_design_provenance(design) for design in compiled),
        tuple(run.r2 for run in numerical.run_fits),
        numerical.aggregate_r2,
        provenance,
    )
```

Keep `fit`'s selected-HRF dispatch outside the ordinary-fit context so dispatch creates only one pair of events. Move ordinary `feature_signature` rejection inside its context. Dispatch does not precompute fallible model metadata outside the selected path: move that calculation into the selected operation or pass the model and calculate there.

- [x] Flatten prepared lifecycle wrappers into `fit_prepared` and `task_delta_r2_prepared`; retain `_prepare_fit_spec`, identity, design, diagnostic, and activity helpers. Remove `_fit_with_lifecycle`, `_comparison_with_lifecycle`, validation-only log helpers, `_with_events`, and the provenance forwarding wrapper. Eliminate discarded history updates.
- [x] For selected GLMs, retain `_prepare`, `_group_fits`, `_assemble`, and `_ols_comparison`. Convert `_comparison_provenance` to an activity builder and compute the same comparison fingerprint from `context.analysis_id` plus that activity. Put assembly and final comparison replacement inside the context.
- [x] For shared trials, move regularization/label validation and `_model_metadata` inside the context. For selected trials, change `_provenance` into an activity builder; keep its design digest and selection payload identical, and wrap `fit_groups` with the new selected-trial lifecycle. Pass final provenance to `SingleTrialResult` once.
- [x] Existing comparison construction may use its parent provenance temporarily to compute diagnostic summaries; retain this where needed. Replace only the final comparison's provenance inside the boundary. Do not introduce a generic result mutation protocol.
- [x] Run `uv run pytest tests/test_fit_lifecycle.py tests/test_logging.py tests/test_fit.py tests/test_prepared_fit.py tests/test_hrf_glm.py tests/test_single_trial.py tests/test_selected_hrf_fit.py tests/test_fractional_ridge.py tests/test_ridge_cv.py tests/test_fractional_cv.py -q -W error`. Document changed start IDs and selected-trial events; format and commit GREEN.

## Task 4: Remove redundant provenance records and serialization

- [x] Extend `tests/test_provenance.py` with a record-extension test that computes its expected serialized result before temporarily replacing `ProvenanceRecord.to_dict` with a raising function. Call `extend_provenance` and require matching fields without serializing the parent. Include an additive extension field, nested activity input mutated afterward, source ordering, and existing quality warning. Restore the method before checking canonical JSON. This tests the removed round-trip cost as well as semantic equivalence.

```python
def test_extension_preserves_fields_without_parent_serialization(monkeypatch):
    from copy import deepcopy
    from boldtailor.provenance import extend_provenance

    payload = _record_payload(extra_note={"labels": ["original"]})
    parent = ProvenanceRecord.from_dict(payload)
    activity = {"name": "fit", "settings": {"alpha": 0.1}}
    expected = parent.to_dict()
    execution_id = "123e4567-e89b-12d3-a456-426614174001"
    expected.update(
        execution_id=execution_id,
        activities=[*expected["activities"], deepcopy(activity)],
        events=[], warnings=expected["warnings"], analysis_fingerprint="a" * 64,
    )

    def reject_serialization(self):
        raise AssertionError("extension must not serialize its parent")

    with monkeypatch.context() as patch:
        patch.setattr(ProvenanceRecord, "to_dict", reject_serialization)
        extended = extend_provenance(
            parent, execution_id=execution_id, activity=activity,
            analysis_id="a" * 64, events=[], warnings=(),
        )
    assert extended.to_dict() == expected
    assert extended.metadata_fingerprint == parent.metadata_fingerprint
    activity["settings"]["alpha"] = 99
    payload["extra_note"]["labels"].append("changed")
    assert extended.to_dict() == expected
```

- [x] In data/prepared tests, instrument the module's `ProvenanceRecord` constructor with a counting wrapper around the real class. Assert one construction for successful normalization with complete and anonymous sources, unchanged source fingerprint/quality warning, and exact completion event parity. Also test rejected non-JSON metadata and injected final input-object construction failures: no completed log, original exception re-raised. Commit RED tests.
- [x] Implement direct record extension:

```python
return replace(
    record,
    execution_id=execution_id,
    activities=(*record.activities, activity),
    events=tuple(events),
    warnings=(*record.warnings, *warnings),
    _extra={**record._extra, "analysis_fingerprint": analysis_id},
)
```

Add `replace` to the dataclass import. Preserve all constructor validation. Reuse the existing `_metadata_fingerprint(sources)` function directly in data/prepared normalization; delete `_data_id`/`_source_fingerprint` wrappers that allocate records just to read their fingerprints.

- [x] In each input constructor, after normalization succeeds, use `_make_event` to prepare completion, create final provenance with that bounded history, then construct the owned input object within the existing try boundary. Catch `Exception` consistently and pass the exception object to `emit_event`. Emit the prepared completion only after construction succeeds, then return the constructed input. Remove the discarded record validation call; final record construction performs that validation. Use a fresh context for each normalization attempt.
- [x] Avoid repeated source hashing inside `ProvenanceRecord.__post_init__`: compute the metadata fingerprint once after source ownership and pass its availability into `_augment_warnings`. Keep canonical bytes/hash definitions unchanged.
- [x] Run `uv run pytest tests/test_provenance.py tests/test_data.py tests/test_prepared.py tests/test_logging.py tests/test_prepared_fit.py tests/test_bids_provenance.py -q -W error`. Format and commit GREEN.

## Completion and documentation

Update `docs/lifecycle-migration.md`, API/developer guides, documentation index, and roadmap with actual implemented behavior. Explicitly describe metadata-based identities (not content checks), new category fields, start-ID availability, selected-trial events, and the removal of environment-substring sanitization. Keep publication guarantees unchanged. Record evidence in `docs/validation/scientific-readability-lifecycle-2026-09-28.md`.

Run default `uv run pytest -q -W error`, standard Black scope, `git diff --check`, wheel build, and isolated installed-wheel smoke verification. Recheck current documentation links and examples affected by lifecycle fields. Request one fresh independent reviewer of the whole increment and resolve important findings with RED-first tests. Keep this branch unmerged.

Remaining roadmap work after this increment: publication simplification and packaged imaging/notebook cleanup, followed by whole-branch review. Generic annotation/schema redesign is not required for this increment's lifecycle and record-construction cleanup.
