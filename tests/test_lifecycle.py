import json
import logging
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.data import from_arrays
from boldtailor.fit import fit, task_delta_r2
from boldtailor.logging import emit_event
from boldtailor.model import ModelSpec
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared, task_delta_r2_prepared
from boldtailor.single_trial import fit_selected_hrfs, fit_single_trials
from boldtailor._software import software_environment

ENTRY_POINTS = [
    "fit",
    "task_delta_r2",
    "single_trial",
    "selected_hrf_single_trial",
    "prepared_fit",
    "prepared_task_delta_r2",
    "selected_glm_fit",
    "selected_glm_task_delta_r2",
]
# Entry points whose lifecycle events are not named after the entry point.
EVENT_PREFIX = {
    "prepared_fit": "fit",
    "prepared_task_delta_r2": "task_delta_r2_prepared",
    "selected_glm_fit": "fit",
    "selected_glm_task_delta_r2": "task_delta_r2",
}


def _events(caplog, name):
    prefix = EVENT_PREFIX.get(name, name)
    records = [json.loads(r.getMessage()) for r in caplog.records]
    wanted = {f"{prefix}_{kind}" for kind in ("started", "completed", "failed")}
    return [r for r in records if r["event"] in wanted]


@pytest.fixture
def glm_problem(complete_sources):
    events = pd.DataFrame(
        {
            "onset": [0.0, 8.0, 16.0, 24.0, 32.0, 40.0],
            "duration": np.ones(6),
            "trial_type": ["face", "house", "face", "house", "face", "house"],
        }
    )
    frame_times = np.arange(30) * 2.0
    signals = np.column_stack(
        (
            np.sin(frame_times / 6.0) + np.cos(frame_times / 11.0),
            np.cos(frame_times / 7.0) - np.sin(frame_times / 13.0),
        )
    )
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model=None,
        noise_model="ols",
    )
    data = from_arrays(signals, events, tr=2.0, sources=complete_sources(1))
    return data, model, fit(data, model)


@pytest.fixture
def prepared_problem(complete_sources):
    rng = np.random.default_rng(20260817)
    designs = tuple(
        pd.DataFrame(
            {
                "face": rng.normal(size=n),
                "house": rng.normal(size=n),
                "motion": rng.normal(size=n),
                "constant": np.ones(n),
            }
        )
        for n in (40, 56)
    )
    roles = {
        "face": "task",
        "house": "task",
        "motion": "nuisance",
        "constant": "intercept",
    }
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=tuple(rng.normal(size=(len(d), 2)) for d in designs),
        design_matrices=designs,
        tr=2.0,
        column_roles=(roles, roles),
        sources=complete_sources(2),
    )
    kwargs = dict(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        noise_model="ar1",
        model_metadata={"origin": "fitlins"},
    )
    return prepared, kwargs, fit_prepared(prepared, **kwargs)


@pytest.fixture
def selected_glm_problem(selected_fixture, complete_sources):
    data, selection = selected_fixture
    glm_data = from_arrays(
        data.signals,
        [e[["onset", "duration"]].assign(trial_type="stimulus") for e in data.events],
        frame_times=data.frame_times,
        confounds=data.confounds,
        sources=complete_sources(data.n_runs),
    )
    model = ModelSpec(
        contrasts={"stimulus": {"stimulus": 1}},
        confounds=("motion",),
        noise_model="ols",
    )
    kwargs = dict(hrf_selection=selection, feature_signature="ordered-axis")
    return glm_data, model, kwargs, fit(glm_data, model, **kwargs)


def _glm_calls(glm_problem):
    data, model, full = glm_problem
    ar1 = replace(model, noise_model="ar1")
    return {
        "fit": (
            lambda: fit(data, model),
            lambda: fit(data, model, feature_signature="unexpected"),
        ),
        "task_delta_r2": (
            lambda: task_delta_r2(data, model, full),
            lambda: task_delta_r2(data, ar1, full),
        ),
    }


def _single_trial_calls(selected_fixture):
    data, selection = selected_fixture
    selected = dict(selection=selection, feature_signature="ordered-axis")
    return {
        "single_trial": (
            lambda: fit_single_trials(data),
            lambda: fit_single_trials(data, ridge_alpha=-1),
        ),
        "selected_hrf_single_trial": (
            lambda: fit_selected_hrfs(data, **selected),
            lambda: fit_selected_hrfs(data, run_labels=[], **selected),
        ),
    }


def _prepared_calls(prepared_problem):
    prepared, kwargs, full = prepared_problem
    bad = dict(kwargs, model_metadata={"bad": object()})
    changed = dict(kwargs, model_metadata={"changed": True})
    return {
        "prepared_fit": (
            lambda: fit_prepared(prepared, **kwargs),
            lambda: fit_prepared(prepared, **bad),
        ),
        "prepared_task_delta_r2": (
            lambda: task_delta_r2_prepared(prepared, full, **kwargs),
            lambda: task_delta_r2_prepared(prepared, full, **changed),
        ),
    }


def _selected_glm_calls(selected_glm_problem):
    data, model, kwargs, full = selected_glm_problem
    wrong = dict(kwargs, feature_signature="wrong")
    ar1 = replace(model, noise_model="ar1")
    return {
        "selected_glm_fit": (
            lambda: fit(data, model, **kwargs),
            lambda: fit(data, model, **wrong),
        ),
        "selected_glm_task_delta_r2": (
            lambda: task_delta_r2(data, model, full),
            lambda: task_delta_r2(data, ar1, full),
        ),
    }


@pytest.fixture
def entry_calls(glm_problem, selected_fixture, prepared_problem, selected_glm_problem):
    return {
        **_glm_calls(glm_problem),
        **_single_trial_calls(selected_fixture),
        **_prepared_calls(prepared_problem),
        **_selected_glm_calls(selected_glm_problem),
    }


@pytest.fixture
def run_entry_point(entry_calls):
    """Run a valid call and return its result object."""
    return lambda name: entry_calls[name][0]()


@pytest.fixture
def run_bad_argument(entry_calls):
    """Run a call that is rejected by argument validation."""
    return lambda name: entry_calls[name][1]()


@pytest.fixture
def break_late(monkeypatch):
    @contextmanager
    def patch():
        failure = RuntimeError("private late detail")

        def reject(*args, **kwargs):
            raise failure

        monkeypatch.setattr("boldtailor._fit_lifecycle.extend_provenance", reject)
        yield failure

    return patch


def _assert_context_reset():
    after = emit_event("after_entry_point", stage="test")
    assert not {"execution_id", "data_id", "analysis_id", "run_index"} & after.keys()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_success_logs_started_then_completed_with_one_execution_id(
    name, run_entry_point, caplog
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    run_entry_point(name)
    events = _events(caplog, name)
    prefix = EVENT_PREFIX.get(name, name)
    assert [e["event"] for e in events] == [f"{prefix}_started", f"{prefix}_completed"]
    assert len({e["execution_id"] for e in events}) == 1
    assert "analysis_id" not in events[0]
    assert len({e.get("data_id") for e in events}) == 1
    _assert_context_reset()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_completion_event_is_the_returned_provenance_record(
    name, run_entry_point, caplog
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    provenance = run_entry_point(name).provenance
    completed = _events(caplog, name)[-1]
    assert dict(provenance.events[-1]) == completed
    assert completed["execution_id"] == provenance.execution_id
    assert completed.get("analysis_id") == provenance.analysis_fingerprint
    assert completed.get("data_id") == provenance.metadata_fingerprint
    assert len(provenance.events) <= 8


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_late_failure_logs_failed_and_never_completed(
    name, run_entry_point, break_late, caplog
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    with break_late() as failure, pytest.raises(RuntimeError) as caught:
        run_entry_point(name)
    assert caught.value is failure
    assert "private" not in caplog.text
    events = _events(caplog, name)
    prefix = EVENT_PREFIX.get(name, name)
    assert [e["event"] for e in events] == [f"{prefix}_started", f"{prefix}_failed"]
    assert len({e["execution_id"] for e in events}) == 1
    assert "analysis_id" not in events[0]
    assert events[0].get("data_id") == events[-1].get("data_id")
    assert events[-1]["level"] == "ERROR"
    assert events[-1]["error_code"] == "operation_failed"
    _assert_context_reset()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_early_argument_failure_logs_failed_and_never_completed(
    name, run_bad_argument, caplog
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    with pytest.raises(ValueError):
        run_bad_argument(name)
    events = _events(caplog, name)
    prefix = EVENT_PREFIX.get(name, name)
    assert [e["event"] for e in events] == [f"{prefix}_started", f"{prefix}_failed"]
    assert len({e["execution_id"] for e in events}) == 1
    assert "analysis_id" not in events[0]
    assert events[0].get("data_id") == events[-1].get("data_id")
    assert events[-1]["level"] == "ERROR"
    assert events[-1]["error_code"] == "invalid_input"
    assert "private" not in caplog.text
    _assert_context_reset()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_operation_records_the_software_environment(name, run_entry_point):
    result = run_entry_point(name)
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["software"] == software_environment()
