import json
import logging
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.data import from_arrays
from boldtailor.fit import fit, task_delta_r2
from boldtailor.fractional_ridge import score_fraction_candidates
from boldtailor.hrf_selection import evaluate_hrf_split, select_hrfs
from boldtailor.logging import emit_event
from boldtailor.model import ModelSpec
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.ridge_selection import score_ridge_candidates
from boldtailor.prepared_fit import fit_prepared, task_delta_r2_prepared
from boldtailor.single_trial import fit_selected_hrfs, fit_single_trials
from boldtailor.provenance import software_environment

ENTRY_POINTS = [
    "fit",
    "task_delta_r2",
    "single_trial",
    "selected_hrf_single_trial",
    "prepared_fit",
    "prepared_task_delta_r2",
    "selected_glm_fit",
    "selected_glm_task_delta_r2",
    "hrf_selection",
    "hrf_independent_evaluation",
    "encoding_guided_ridge_cv",
    "encoding_guided_fractional_ridge_cv",
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
    ar1 = model.replace(noise_model="ar1")
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
    selected = dict(hrf_selection=selection, feature_signature="ordered-axis")
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


def _selection_calls(selected_fixture):
    data, selection = selected_fixture
    library = selection.library
    split = dict(library=library, train_runs=[0, 1], test_runs=[2])
    return {
        "hrf_selection": (
            lambda: select_hrfs(data, library=library),
            lambda: select_hrfs(data, library="not a library"),
        ),
        "hrf_independent_evaluation": (
            lambda: evaluate_hrf_split(data, **split),
            lambda: evaluate_hrf_split(data, **dict(split, test_runs=[1])),
        ),
    }


def _prepared_calls(prepared_problem):
    prepared, kwargs, full = prepared_problem
    bad = dict(kwargs, model_metadata={"bad": object()})
    record = full.provenance
    unprepared = replace(full, _provenance=record.replace(activities=()))
    return {
        "prepared_fit": (
            lambda: fit_prepared(prepared, **kwargs),
            lambda: fit_prepared(prepared, **bad),
        ),
        "prepared_task_delta_r2": (
            lambda: task_delta_r2_prepared(prepared, full),
            lambda: task_delta_r2_prepared(prepared, unprepared),
        ),
    }


def _selected_glm_calls(selected_glm_problem):
    data, model, kwargs, full = selected_glm_problem
    wrong = dict(kwargs, feature_signature="wrong")
    ar1 = model.replace(noise_model="ar1")
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


def _ridge_scoring_calls(ridge_problem):
    data, predictors, _ = ridge_problem
    return {
        "encoding_guided_ridge_cv": (
            lambda: score_ridge_candidates(data, predictors, alphas=[0.0, 1.0]),
            lambda: score_ridge_candidates(data, predictors, alphas=[-1.0]),
        ),
        "encoding_guided_fractional_ridge_cv": (
            lambda: score_fraction_candidates(data, predictors, fractions=[0.5, 1.0]),
            lambda: score_fraction_candidates(data, predictors, fractions=[2.0]),
        ),
    }


@pytest.fixture
def entry_calls(
    glm_problem, selected_fixture, prepared_problem, selected_glm_problem, ridge_problem
):
    return {
        **_ridge_scoring_calls(ridge_problem),
        **_glm_calls(glm_problem),
        **_single_trial_calls(selected_fixture),
        **_selection_calls(selected_fixture),
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
    assert (
        not {"execution_id", "data_fingerprint", "analysis_fingerprint", "run_index"}
        & after.keys()
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_success_logs_started_then_completed_with_one_execution_id(
    name, run_entry_point, caplog
):
    """Verify success logs started then completed with one execution id."""
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    run_entry_point(name)
    events = _events(caplog, name)
    prefix = EVENT_PREFIX.get(name, name)
    assert [e["event"] for e in events] == [f"{prefix}_started", f"{prefix}_completed"]
    assert len({e["execution_id"] for e in events}) == 1
    assert "analysis_fingerprint" not in events[0]
    assert len({e.get("data_fingerprint") for e in events}) == 1
    _assert_context_reset()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_completion_log_and_provenance_share_operation_identity(
    name, run_entry_point, caplog
):
    """Verify completion log and provenance share operation identity."""
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    provenance = run_entry_point(name).provenance
    completed = _events(caplog, name)[-1]
    snapshot = dict(provenance.events[-1])
    # Provenance is built independently before the result constructor and scope exit.
    assert snapshot["sequence"] < completed["sequence"]
    timing = {"timestamp", "sequence"}
    assert {k: v for k, v in snapshot.items() if k not in timing} == {
        k: v for k, v in completed.items() if k not in timing
    }
    assert completed["execution_id"] == provenance.execution_id
    assert completed.get("analysis_fingerprint") == provenance.analysis_fingerprint
    assert completed.get("data_fingerprint") == provenance.data_fingerprint
    assert len(provenance.events) <= 8


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_late_failure_logs_failed_and_never_completed(
    name, run_entry_point, break_late, caplog
):
    """Verify late failure logs failed and never completed."""
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
    assert "analysis_fingerprint" not in events[0]
    assert events[0].get("data_fingerprint") == events[-1].get("data_fingerprint")
    assert events[-1]["level"] == "ERROR"
    assert events[-1]["error_code"] == "operation_failed"
    _assert_context_reset()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_early_argument_failure_logs_failed_and_never_completed(
    name, run_bad_argument, caplog
):
    """Verify early argument failure logs failed and never completed."""
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    with pytest.raises(ValueError):
        run_bad_argument(name)
    events = _events(caplog, name)
    prefix = EVENT_PREFIX.get(name, name)
    assert [e["event"] for e in events] == [f"{prefix}_started", f"{prefix}_failed"]
    assert len({e["execution_id"] for e in events}) == 1
    assert "analysis_fingerprint" not in events[0]
    assert events[0].get("data_fingerprint") == events[-1].get("data_fingerprint")
    assert events[-1]["level"] == "ERROR"
    assert events[-1]["error_code"] == "invalid_input"
    assert "private" not in caplog.text
    _assert_context_reset()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_operation_records_the_software_environment(name, run_entry_point):
    """Verify every operation records the software environment."""
    result = run_entry_point(name)
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["software"] == software_environment()


@pytest.fixture
def sourced_calls(selected_fixture, selected_glm_problem, complete_sources):
    data, selection = selected_fixture
    sourced = from_arrays(
        data.signals,
        data.events,
        frame_times=data.frame_times,
        confounds=data.confounds,
        sources=complete_sources(data.n_runs),
    )
    library = selection.library
    glm_data, model, kwargs, _ = selected_glm_problem
    split = dict(library=library, train_runs=[0, 1], test_runs=[2])

    def choose(target):
        return select_hrfs(target, library=library, feature_signature="ordered-axis")

    def selected_fit():
        return fit_selected_hrfs(
            sourced, hrf_selection=choose(sourced), feature_signature="ordered-axis"
        )

    def glm_fit():
        glm_kwargs = dict(kwargs, hrf_selection=choose(glm_data))
        return fit(glm_data, model, **glm_kwargs)

    return {
        "hrf_independent_evaluation": lambda: evaluate_hrf_split(sourced, **split),
        "selected_hrf_single_trial": selected_fit,
        "selected_glm_fit": glm_fit,
    }


@pytest.mark.parametrize(
    "name",
    ["hrf_independent_evaluation", "selected_hrf_single_trial", "selected_glm_fit"],
)
def test_analysis_ids_do_not_depend_on_the_software_environment(
    name, sourced_calls, monkeypatch
):
    """Verify analysis ids do not depend on the software environment."""
    first = sourced_calls[name]().provenance
    assert first.analysis_fingerprint is not None
    changed = {k: f"changed-{v}" for k, v in software_environment().items()}
    for module in ("_fit_lifecycle", "data", "prepared"):
        monkeypatch.setattr(
            f"boldtailor.{module}.software_environment", lambda: changed
        )
    second = sourced_calls[name]().provenance
    assert first.to_dict()["activities"][-1]["software"] != (
        second.to_dict()["activities"][-1]["software"]
    )
    assert first.analysis_fingerprint == second.analysis_fingerprint
