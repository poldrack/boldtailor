"""Persist enough truth and held-out scores to interpret simulation recovery."""

import importlib
import json
from dataclasses import replace

import numpy as np

from experiments.hrf_estimation.test_library_cv import planted_runs


def test_grid_crosses_noise_rt_truth_and_replicates():
    api = importlib.import_module("experiments.hrf_estimation.sensitivity_runner")
    cases = api.simulation_cases()
    assert len(cases) == 288
    assert {c["rho"] for c in cases} == {0, .4, .7}
    assert {c["snr"] for c in cases} == {.5, 2}
    assert {c["truth"] for c in cases} == {"canonical", "fast", "offgrid", "empirical"}
    assert any(c["generating_rt"] != c["fitted_rt"] for c in cases)


def test_case_saves_held_out_scores_recovery_and_verified_restart(planted_runs, tmp_path):
    api = importlib.import_module("experiments.hrf_estimation.sensitivity_runner")
    runs, library = planted_runs
    case = dict(truth="fast", generating_rt="amplitude", fitted_rt="amplitude",
                rho=0., snr=2., seed=17)
    settings = {"lambda": 0., "max_starts": 1, "maxiter": 10, "maxfun": 100}
    result = api.evaluate_case(runs, library, case, tmp_path, settings,
                               methods=("canonical", "librarycv"))
    assert set(result.method) == {"canonical", "librarycv"}
    assert np.isfinite(result.r2_task_partial).all()
    assert np.isfinite(result.curve_rmse).all()
    assert (tmp_path / "COMPLETE.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["training_run_ids"]).isdisjoint(manifest["test_run_ids"])
    again = api.evaluate_case(runs, library, case, tmp_path, settings,
                              methods=("canonical", "librarycv"))
    np.testing.assert_allclose(result.r2_task_partial, again.r2_task_partial)
