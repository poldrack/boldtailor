"""Profiled fitting must improve an off-grid HRF without accessing test data."""

import importlib
from dataclasses import replace

import numpy as np
import pytest

from boldtailor.hrf_library import HrfCandidate
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.test_library_cv import planted_runs


@pytest.fixture
def offgrid(planted_runs):
    runs, library = planted_runs
    truth = HrfCandidate(99, "double_gamma", (4.1, 12, 0.8, 1.5, 6, 0, 36))
    result = []
    for run in runs[:2]:
        parts = build_design(run, truth)
        y = parts.task @ np.array([2, 5, 1.5]) + parts.junk @ np.array([1, 3]) + 20
        result.append(replace(run, signals=y[:, None], feature_ids=np.array([4])))
    return tuple(result), library, truth


def test_profiled_objective_matches_explicit_least_squares(offgrid):
    runs, _, truth = offgrid
    fn = importlib.import_module("experiments.hrf_estimation.continuous").profile_loss
    assert fn(runs, 0, truth.parameters) == pytest.approx(0, abs=1e-20)
    wrong = (6, 16, 1, 1, 6, 0, 36)
    sse = energy = 0
    for run in runs:
        p = build_design(run, HrfCandidate(0, "double_gamma", wrong))
        x = np.column_stack([p.nuisance, p.junk, p.task])
        y = run.signals[:, 0]
        sse += np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2)
        c = p.nuisance
        energy += np.sum((y - c @ np.linalg.lstsq(c, y, rcond=None)[0]) ** 2)
    assert fn(runs, 0, wrong) == pytest.approx(sse / energy)


@pytest.mark.parametrize("optimizer", ["L-BFGS-B", "Powell"])
def test_offgrid_optimization_improves_initial_shape(offgrid, optimizer):
    runs, library, truth = offgrid
    api = importlib.import_module("experiments.hrf_estimation.continuous")
    estimate = api.fit_continuous(
        runs,
        {
            "library": library,
            "lambda": 0,
            "optimizer": optimizer,
            "maxiter": 150,
            "maxfun": 2500,
            "max_starts": 2,
        },
    )
    assert estimate.kernels[0] is not None
    loss = api.profile_loss(runs, 0, estimate.kernels[0].parameters)
    initial = min(
        api.profile_loss(runs, 0, (*c.parameters[:5], 0, 36))
        for c in library.candidates
    )
    assert loss < initial * 0.05
    assert estimate.training_run_ids == tuple(r.run_id for r in runs)
    assert estimate.kernels[0].parameters[-2:] == (0, 36)
    assert (
        estimate.diagnostics["objective"][0]
        <= estimate.diagnostics["initial_objective"][0]
    )


def test_zero_signal_and_failed_budget_are_explicit(offgrid):
    runs, library, _ = offgrid
    api = importlib.import_module("experiments.hrf_estimation.continuous")
    zeros = tuple(replace(r, signals=np.ones_like(r.signals)) for r in runs)
    result = api.fit_continuous(zeros, {"library": library, "lambda": 0})
    assert result.kernels == (None,)
    assert result.status == ("no_signal",)
    result = api.fit_continuous(
        runs, {"library": library, "lambda": 0, "maxiter": 1, "maxfun": 2}
    )
    assert result.status[0] != "ok"


def test_continuous_reports_nonidentifiable_task_effects(offgrid):
    runs, library, _ = offgrid
    changed = tuple(replace(r, events=r.events.assign(response_time=1.0)) for r in runs)
    fit = importlib.import_module(
        "experiments.hrf_estimation.continuous"
    ).fit_continuous
    result = fit(
        changed,
        {"library": library, "lambda": 0, "maxiter": 5, "maxfun": 50, "max_starts": 1},
    )
    assert result.status == ("rank_deficient",)
    assert result.diagnostics["min_task_rank"] == [2]


def test_continuous_records_parameter_boundary_proximity(offgrid):
    from boldtailor.hrf_library import HrfLibrary

    runs, _, _ = offgrid
    parameters = (2.0, 12.0, 0.8, 1.5, 6.0, 0.0, 36.0)
    library = HrfLibrary.from_parameters([parameters])
    truth = HrfCandidate(1, "double_gamma", parameters)
    changed = []
    for run in runs:
        parts = build_design(run, truth)
        y = parts.task @ np.array([2, 5, 1.5]) + parts.junk @ np.array([1, 3]) + 20
        changed.append(replace(run, signals=y[:, None]))
    fit = importlib.import_module(
        "experiments.hrf_estimation.continuous"
    ).fit_continuous
    result = fit(
        tuple(changed),
        {"library": library, "lambda": 0, "maxiter": 5, "maxfun": 50, "max_starts": 1},
    )
    assert result.diagnostics["near_lower_bound"][0][0]
    assert result.diagnostics["normalized_bound_distance"][0][0] == pytest.approx(
        0, abs=1e-6
    )
