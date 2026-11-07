"""Noise and RT sensitivities must have known, reproducible ground truth."""

import importlib
import numpy as np
import pytest

from experiments.hrf_estimation.test_library_cv import planted_runs
from experiments.hrf_estimation.design import build_design


def test_stationary_noise_has_requested_correlation_and_scale():
    api = importlib.import_module("experiments.hrf_estimation.sensitivity")
    noise = api.ar1_noise(100000, .7, np.random.default_rng(14))
    assert np.std(noise) == pytest.approx(1, abs=.025)
    assert np.corrcoef(noise[1:], noise[:-1])[0, 1] == pytest.approx(.7, abs=.02)
    with pytest.raises(ValueError):
        api.ar1_noise(20, 1.1, np.random.default_rng(1))


@pytest.mark.parametrize("mode", ["amplitude", "duration"])
def test_simulation_matches_true_design_and_is_reproducible(planted_runs, mode):
    runs, library = planted_runs
    api = importlib.import_module("experiments.hrf_estimation.sensitivity")
    truth = library.candidates[1]
    clean = api.simulate_runs(runs, truth, mode, rho=.4, snr=np.inf, seed=5)
    noisy = api.simulate_runs(runs, truth, mode, rho=.4, snr=2, seed=5)
    again = api.simulate_runs(runs, truth, mode, rho=.4, snr=2, seed=5)
    for template, exact, observed, repeat in zip(runs, clean, noisy, again):
        parts = build_design(exact, truth, mode)
        matrix = np.column_stack([parts.nuisance, parts.junk, parts.task])
        error = exact.signals - matrix @ np.linalg.lstsq(matrix, exact.signals, rcond=None)[0]
        np.testing.assert_allclose(error, 0, atol=1e-10)
        np.testing.assert_array_equal(observed.signals, repeat.signals)
        assert not np.allclose(exact.signals, observed.signals)
        assert observed.feature_ids.tolist() == [int(template.feature_ids[0])]
        assert not np.shares_memory(observed.signals, template.signals)


def test_parametric_library_is_exact_full_library_subset():
    api = importlib.import_module("experiments.hrf_estimation.sensitivity")
    from boldtailor.hrf_library import default_hrf_library
    library = default_hrf_library(8)
    subset = api.parametric_library(library)
    assert all(c.kind != "glmsingle" for c in subset.candidates)
    assert [(c.kind, c.parameters) for c in subset.candidates] == [
        (c.kind, c.parameters) for c in library.candidates if c.kind != "glmsingle"]


def test_synthetic_templates_include_motion_junk_and_valid_timing():
    api = importlib.import_module("experiments.hrf_estimation.sensitivity")
    runs = api.synthetic_templates(seed=9)
    assert len(runs) == 4
    for run in runs:
        assert run.signals.shape == (226, 1)
        assert run.confounds.shape == (226, 32)
        assert run.events.missing_data.sum() > 0
        assert (run.events.is_correct == 0).sum() > 0
        assert run.events.onset.max() < run.frame_times[-1]
    again = api.synthetic_templates(seed=9)
    np.testing.assert_array_equal(runs[0].confounds, again[0].confounds)


def test_full_vtc_jobs_match_scopes_and_keep_tuning_nested():
    api = importlib.import_module("experiments.hrf_estimation.full_vtc")
    jobs = api.comparison_jobs()
    assert len(jobs) == 16
    assert all("continuouspowell" not in j["methods"] for j in jobs)
    assert {j["session"] for j in jobs} == {11, 12}
    assert {j["rt_mode"] for j in jobs} == {"amplitude", "duration"}
    assert {j["library"] for j in jobs} == {"full", "parametric"}
    settings = api.estimator_settings("amplitude", 4)
    assert "lambda" not in settings
    assert settings["lambda_grid"] == [0, .001, .01, .1, 1]
    assert settings["max_starts"] == 4
    assert settings["maxfun"] == 5000
