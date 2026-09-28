import json
import logging
from uuid import uuid4

import pytest

from boldtailor.provenance import ProvenanceRecord


@pytest.fixture
def parent_record():
    return ProvenanceRecord(execution_id=str(uuid4()))


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


def test_nested_operations_restore_context_and_bound_history(caplog, parent_record):
    from dataclasses import replace
    from boldtailor._fit_lifecycle import fit_operation
    from boldtailor.logging import emit_event

    caplog.set_level(logging.INFO, logger="boldtailor")
    parent = replace(parent_record, events=tuple({"event": f"old_{i}"} for i in range(8)))
    with fit_operation("outer", parent) as outer:
        with fit_operation("inner", parent) as inner:
            inner_record = inner.provenance({"name": "inner"}, analysis_id="b" * 64)
        restored = emit_event("restored", stage="fit")
        outer_record = outer.provenance({"name": "outer"}, analysis_id="a" * 64)
    assert restored["execution_id"] == outer.execution_id
    assert "analysis_id" not in restored
    assert inner.execution_id != outer.execution_id
    for record, name in [(inner_record, "inner"), (outer_record, "outer")]:
        assert [e["event"] for e in record.events] == [
            *(f"old_{i}" for i in range(2, 8)), f"{name}_started", f"{name}_completed"
        ]
        assert record.events[-1]["execution_id"] == record.execution_id
