import json
import logging
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.logging import bind_context, emit_event
from boldtailor.model import ModelSpec
from boldtailor.provenance import RunSources, SourceRef


def _complete_sources() -> list[RunSources]:
    return [
        RunSources(
            signal=SourceRef(
                role="signal",
                uri="sub-01/func/sub-01_task-localizer_run-01_bold.tsv",
                media_type="text/tab-separated-values",
                byte_size=2048,
                modified_at="2026-08-08T12:00:00Z",
            ),
            events=SourceRef(
                role="events",
                uri="sub-01/func/sub-01_task-localizer_run-01_events.tsv",
                media_type="text/tab-separated-values",
                byte_size=512,
                modified_at="2026-08-08T12:01:00Z",
            ),
            confounds=SourceRef(
                role="confounds",
                uri="sub-01/func/sub-01_task-localizer_run-01_confounds.tsv",
                media_type="text/tab-separated-values",
                byte_size=1024,
                modified_at="2026-08-08T12:02:00Z",
            ),
        )
    ]


def _structured_records(caplog) -> list[dict[str, object]]:
    return [json.loads(record.getMessage()) for record in caplog.records]


def test_bind_context_nests_optional_ids_and_resets_after_success(caplog):
    caplog.set_level(logging.INFO, logger="boldtailor")

    emit_event("outside_before", stage="test")
    with bind_context(execution_id="123e4567-e89b-12d3-a456-426614174000"):
        emit_event("outer", stage="test")
        with bind_context(
            data_id="stable-data-id",
            analysis_id="analysis-001",
            run_index=2,
        ):
            emit_event("inner", stage="test")
        emit_event("after_inner", stage="test")
    emit_event("outside_after", stage="test")

    records = _structured_records(caplog)

    assert [record["event"] for record in records] == [
        "outside_before",
        "outer",
        "inner",
        "after_inner",
        "outside_after",
    ]
    assert [record["sequence"] for record in records] == sorted(
        record["sequence"] for record in records
    )
    assert all(record["timestamp"].endswith("Z") for record in records)
    assert records[0].get("execution_id") is None
    assert records[1]["execution_id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert records[2]["execution_id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert records[2]["data_id"] == "stable-data-id"
    assert records[2]["analysis_id"] == "analysis-001"
    assert records[2]["run_index"] == 2
    assert records[3]["execution_id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert records[3].get("data_id") is None
    assert records[4].get("execution_id") is None


def test_from_arrays_logs_structured_records_without_mutating_loggers(caplog):
    caplog.set_level(logging.INFO, logger="boldtailor")
    logger = logging.getLogger("boldtailor")
    root = logging.getLogger()
    root_state = (root.level, tuple(root.handlers))
    logger_state = (logger.level, tuple(logger.handlers), logger.propagate)
    events = pd.DataFrame(
        {
            "onset": [0.0, 4.0],
            "duration": [1.0, 1.0],
            "trial_type": ["raw-secret-event", "house"],
        }
    )
    confounds = pd.DataFrame(
        {"noise_label": [["raw-secret-confound"] for _ in range(10)]}
    )
    signals = np.full((10, 2), 987654.5)

    data = from_arrays(
        signals,
        events,
        tr=2.0,
        confounds=confounds,
        sources=_complete_sources(),
    )

    records = _structured_records(caplog)
    combined = "\n".join(record.getMessage() for record in caplog.records)

    assert [record["event"] for record in records] == [
        "normalization_started",
        "normalization_completed",
    ]
    assert [record["sequence"] for record in records] == sorted(
        record["sequence"] for record in records
    )
    assert {record["level"] for record in records} == {"INFO"}
    assert {record["stage"] for record in records} == {"data"}
    assert all(record["timestamp"].endswith("Z") for record in records)
    assert all(
        record["execution_id"] == data.provenance.execution_id for record in records
    )
    assert data.provenance.metadata_fingerprint is not None
    assert records[1]["data_id"] == data.provenance.metadata_fingerprint
    assert (root.level, tuple(root.handlers)) == root_state
    assert (logger.level, tuple(logger.handlers), logger.propagate) == logger_state
    assert "raw-secret-event" not in combined
    assert "raw-secret-confound" not in combined
    assert "987654.5" not in combined
    assert str(Path.home()) not in combined
    assert Path.cwd().as_posix() not in combined
    assert Path.home().name not in combined
    assert socket.gethostname() not in combined


def test_from_arrays_logs_failure_and_resets_context_after_exception(caplog):
    caplog.set_level(logging.INFO, logger="boldtailor")
    events = pd.DataFrame({"onset": [0.0], "duration": [-1.0]})

    try:
        from_arrays(
            np.ones((10, 2)),
            events,
            tr=2.0,
            sources=_complete_sources(),
        )
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("from_arrays should reject negative event durations")

    emit_event("after_failure", stage="test")
    records = _structured_records(caplog)

    assert [record["event"] for record in records] == [
        "normalization_started",
        "normalization_failed",
        "after_failure",
    ]
    assert records[0]["execution_id"] == records[1]["execution_id"]
    assert records[1]["data_id"] is not None
    assert records[1]["level"] == "ERROR"
    assert records[1]["stage"] == "data"
    assert records[1]["error"] == "run 0 event durations must be non-negative"
    assert records[2].get("execution_id") is None
    assert records[2].get("data_id") is None


def test_fit_logs_structured_records_with_correlated_ids(caplog):
    caplog.set_level(logging.INFO, logger="boldtailor")
    events = pd.DataFrame(
        {
            "onset": [0.0, 8.0, 16.0, 24.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["face", "house", "face", "house"],
        }
    )
    frame_times = np.arange(20) * 2.0
    design = np.column_stack(
        [
            np.sin(frame_times / 6.0),
            np.cos(frame_times / 7.0),
        ]
    )
    signals = design + 0.1
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model=None,
        noise_model="ols",
    )
    data = from_arrays(signals, events, tr=2.0, sources=_complete_sources())

    result = fit(data, model)
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]

    assert [record["event"] for record in fit_records] == [
        "fit_started",
        "fit_completed",
    ]
    assert fit_records[0]["execution_id"] == result.provenance.execution_id
    assert fit_records[1]["execution_id"] == result.provenance.execution_id
    assert fit_records[0]["data_id"] == data.provenance.metadata_fingerprint
    assert fit_records[1]["data_id"] == data.provenance.metadata_fingerprint
    assert fit_records[0]["analysis_id"] == result.provenance.analysis_fingerprint
    assert fit_records[1]["analysis_id"] == result.provenance.analysis_fingerprint


def test_fit_logs_failure_and_resets_context_after_exception(caplog):
    caplog.set_level(logging.INFO, logger="boldtailor")
    events = pd.DataFrame(
        {
            "onset": [0.0, 8.0, 16.0, 24.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["face", "house", "face", "house"],
        }
    )
    signals = np.ones((20, 2))
    model = ModelSpec(
        contrasts={"missing": {"missing": 1.0}},
        drift_model=None,
        noise_model="ols",
    )
    data = from_arrays(signals, events, tr=2.0, sources=_complete_sources())

    with pytest.raises(ValueError, match="contrast 'missing'"):
        fit(data, model)

    emit_event("after_fit_failure", stage="test")
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]

    assert [record["event"] for record in fit_records] == [
        "fit_started",
        "fit_failed",
    ]
    assert fit_records[0]["execution_id"] == fit_records[1]["execution_id"]
    assert fit_records[0]["data_id"] == data.provenance.metadata_fingerprint
    assert fit_records[1]["level"] == "ERROR"
    assert fit_records[1]["error"].startswith("run 0 contrast 'missing'")
    assert records[-1].get("execution_id") is None
    assert records[-1].get("data_id") is None
    assert records[-1].get("analysis_id") is None
