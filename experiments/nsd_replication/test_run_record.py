import argparse
import json
from datetime import datetime, timezone

import pytest

from experiments.nsd_replication.config import ExperimentConfig, select_sessions
from experiments.nsd_replication.run_record import (
    complete_run_record,
    start_run_record,
)

STARTED = datetime(2026, 10, 7, 12, 30, 5, 123456, tzinfo=timezone.utc)


@pytest.fixture
def config(tmp_path):
    full = ExperimentConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        freesurfer_dir=tmp_path,
        subjects=("sub-07",),
        sessions=("ses-a", "ses-b", "ses-c"),
    )
    return select_sessions(full, ["ses-c", "ses-a"])


@pytest.fixture
def args(tmp_path):
    return argparse.Namespace(
        command="fit", source="ppdata", config=tmp_path / "c.toml"
    )


def _read(path):
    return json.loads(path.read_text())


def test_record_written_under_run_records(config, args):
    path = start_run_record(config, args, ("b1", "b2"), now=STARTED)
    assert path.parent == config.output_dir / "run_records"
    assert path.name == "20261007T123005123456Z_fit_ppdata.json"


def test_record_lists_sessions_run_and_configured(config, args):
    record = _read(start_run_record(config, args, ("b1", "b2"), now=STARTED))
    assert record["sessions_run"] == ["ses-a", "ses-c"]
    assert record["config_sessions"] == ["ses-a", "ses-b", "ses-c"]
    assert record["session_subset"] is True


def test_record_describes_the_invocation(config, args):
    record = _read(start_run_record(config, args, ("b1", "b2"), now=STARTED))
    assert record["command"] == "fit" and record["source"] == "ppdata"
    assert record["config"] == str(args.config.resolve())
    assert record["levels"] == ["b1", "b2"]
    assert record["subjects"] == ["sub-07"]
    assert record["started_at"] == "2026-10-07T12:30:05.123456+00:00"
    assert {"boldtailor_version", "git_commit"} <= set(record)
    assert "completed_at" not in record


def test_full_session_set_is_not_a_subset(config, args):
    full = ExperimentConfig(
        bids_dir=config.bids_dir,
        output_dir=config.output_dir,
        freesurfer_dir=config.freesurfer_dir,
        subjects=config.subjects,
        sessions=config.configured_sessions,
    )
    record = _read(start_run_record(full, args, ("b1",), now=STARTED))
    assert record["session_subset"] is False


def test_completion_adds_completed_at(config, args):
    path = start_run_record(config, args, ("b1",), now=STARTED)
    complete_run_record(path)
    record = _read(path)
    assert datetime.fromisoformat(record["completed_at"]) >= STARTED
    assert record["sessions_run"] == ["ses-a", "ses-c"]
