"""Logging scopes and provenance construction have independent contracts."""

import json
import logging
from dataclasses import replace
from uuid import uuid4

import pytest

from boldtailor.provenance import ProvenanceRecord


@pytest.fixture
def parent_record():
    return ProvenanceRecord(execution_id=str(uuid4()))


def records(caplog):
    return [json.loads(record.getMessage()) for record in caplog.records]


def test_normal_exit_logs_completion_without_building_provenance(caplog, parent_record):
    from boldtailor._fit_lifecycle import fit_operation

    caplog.set_level(logging.INFO, logger="boldtailor")
    with fit_operation("fit", parent_record) as operation:
        operation.analysis_id = "a" * 64
    events = records(caplog)
    assert [event["event"] for event in events] == ["fit_started", "fit_completed"]
    assert events[-1]["analysis_id"] == "a" * 64
    assert events[-1]["execution_id"] == operation.execution_id


def test_late_constructor_failure_never_logs_completion(caplog, parent_record):
    from boldtailor._fit_lifecycle import build_fit_provenance, fit_operation

    caplog.set_level(logging.INFO, logger="boldtailor")
    failure = ValueError("private constructor detail")
    with pytest.raises(ValueError) as caught:
        with fit_operation("fit", parent_record) as operation:
            operation.analysis_id = "a" * 64
            build_fit_provenance(operation, {"name": "fit"})
            raise failure
    assert caught.value is failure
    events = records(caplog)
    assert [event["event"] for event in events] == ["fit_started", "fit_failed"]
    assert events[-1]["error_code"] == "invalid_input"
    assert events[-1]["analysis_id"] == "a" * 64
    assert "private" not in caplog.text


def test_provenance_builder_does_not_mutate_operation_or_log(caplog, parent_record):
    from boldtailor._fit_lifecycle import build_fit_provenance, fit_operation

    caplog.set_level(logging.INFO, logger="boldtailor")
    with fit_operation("fit", parent_record) as operation:
        operation.analysis_id = "a" * 64
        before = vars(operation).copy()
        first = build_fit_provenance(operation, {"name": "fit"})
        second = build_fit_provenance(operation, {"name": "fit"})
        assert vars(operation) == before
        assert [event["event"] for event in records(caplog)] == ["fit_started"]
    completed = records(caplog)[-1]
    for record in (first, second):
        assert record.execution_id == completed["execution_id"]
        assert record.analysis_fingerprint == completed["analysis_id"]
        snapshot = dict(record.events[-1])
        assert snapshot["event"] == completed["event"] == "fit_completed"
        assert snapshot["sequence"] < completed["sequence"]
    assert [event["event"] for event in records(caplog)] == ["fit_started", "fit_completed"]


@pytest.mark.parametrize("fail", [False, True])
def test_anonymous_child_restores_outer_context(caplog, parent_record, fail):
    from boldtailor._fit_lifecycle import fit_operation
    from boldtailor.logging import bind_context, emit_event

    caplog.set_level(logging.INFO, logger="boldtailor")
    outer = dict(execution_id="outer", data_id="outer-data", analysis_id="outer-analysis", run_index=9)
    failure = ValueError("private child detail")
    with bind_context(**outer):
        try:
            with fit_operation("fit", parent_record):
                if fail:
                    raise failure
        except ValueError as caught:
            assert fail and caught is failure
        restored = emit_event("after_child", stage="fit")
    events = records(caplog)[:-1]
    assert [event["event"] for event in events] == [
        "fit_started", "fit_failed" if fail else "fit_completed",
    ]
    for event in events:
        assert event["execution_id"] != "outer"
        assert not {"data_id", "analysis_id", "run_index"} & event.keys()
    assert all(restored[key] == value for key, value in outer.items())


def test_provenance_construction_failure_logs_failure(caplog, parent_record, monkeypatch):
    import boldtailor._fit_lifecycle as lifecycle

    caplog.set_level(logging.INFO, logger="boldtailor")
    failure = RuntimeError("private provenance detail")

    def reject_provenance(*args, **kwargs):
        raise failure

    monkeypatch.setattr(lifecycle, "extend_provenance", reject_provenance)
    with pytest.raises(RuntimeError) as caught:
        with lifecycle.fit_operation("fit", parent_record) as operation:
            lifecycle.build_fit_provenance(operation, {"name": "fit"})
    assert caught.value is failure
    events = records(caplog)
    assert [event["event"] for event in events] == ["fit_started", "fit_failed"]
    assert events[-1]["error_code"] == "operation_failed"


def test_builder_can_run_after_scope_without_inheriting_unrelated_context(caplog, parent_record):
    from boldtailor._fit_lifecycle import build_fit_provenance, fit_operation
    from boldtailor.logging import bind_context, emit_event

    caplog.set_level(logging.INFO, logger="boldtailor")
    with fit_operation("fit", parent_record) as operation:
        operation.analysis_id = "a" * 64
    before = len(caplog.records)
    with bind_context(execution_id="unrelated", run_index=9):
        record = build_fit_provenance(operation, {"name": "fit"})
        assert len(caplog.records) == before
        restored = emit_event("restored", stage="test")
    assert record.events[-1]["execution_id"] == operation.execution_id
    assert record.events[-1]["analysis_id"] == "a" * 64
    assert "run_index" not in record.events[-1]
    assert restored["execution_id"] == "unrelated" and restored["run_index"] == 9


def test_nested_operations_restore_context_and_bound_history(caplog, parent_record):
    from boldtailor._fit_lifecycle import build_fit_provenance, fit_operation
    from boldtailor.logging import emit_event

    caplog.set_level(logging.INFO, logger="boldtailor")
    parent = replace(parent_record, events=tuple({"event": f"old_{i}"} for i in range(8)))
    with fit_operation("outer", parent) as outer:
        with fit_operation("inner", parent) as inner:
            inner.analysis_id = "b" * 64
            inner_record = build_fit_provenance(inner, {"name": "inner"})
        restored = emit_event("restored", stage="fit")
        outer.analysis_id = "a" * 64
        outer_record = build_fit_provenance(outer, {"name": "outer"})
    assert restored["execution_id"] == outer.execution_id
    assert "analysis_id" not in restored
    assert inner.execution_id != outer.execution_id
    for record, name in [(inner_record, "inner"), (outer_record, "outer")]:
        assert [event["event"] for event in record.events] == [
            *(f"old_{i}" for i in range(2, 8)), f"{name}_started", f"{name}_completed",
        ]
        assert record.events[-1]["execution_id"] == record.execution_id
