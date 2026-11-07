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
