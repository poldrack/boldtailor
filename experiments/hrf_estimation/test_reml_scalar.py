"""The fast AR(1) solver must preserve the restricted likelihood and bounds."""
from dataclasses import replace
import importlib
import numpy as np
import pytest
from experiments.hrf_estimation.test_library_cv import planted_runs
from experiments.hrf_estimation.reml_extensions import prepare_common, component_terms, component_objective
from experiments.hrf_estimation.reml_extended_fit import fit_components


@pytest.fixture
def terms(planted_runs):
    runs, library = planted_runs
    rng = np.random.default_rng(101)
    runs = tuple(replace(r, signals=r.signals+rng.normal(0,.4,r.signals.shape)) for r in runs)
    return component_terms(runs, library.candidates[1], prepare_common(runs, .4))


def api():
    return importlib.import_module("experiments.hrf_estimation.reml_scalar")


def test_spectral_objective_matches_generic_for_all_vertices(terms):
    spectral = api().prepare_scalar(terms)
    for ratio in [0., 1e-8, .1, 10., 1e6]:
        values, sigma = api().scalar_objective(np.full(2, ratio), spectral)
        for f in range(2):
            expected, _, expected_sigma = component_objective([ratio], terms, f)
            assert values[f] == pytest.approx(expected, abs=2e-6)
            assert sigma[f] == pytest.approx(expected_sigma, rel=1e-7)


def test_scalar_optimum_matches_generic(terms):
    actual = api().fit_scalar_components(terms, {})
    for f, fit in enumerate(actual):
        expected = fit_components(terms, f, {})
        assert fit["criterion"] == pytest.approx(expected["criterion"], abs=1e-6)
        np.testing.assert_allclose(fit["ratios"], expected["ratios"], rtol=2e-4)
        assert fit["converged"] and fit["valid"]


def test_scalar_boundaries_and_invalid_signal():
    terms = dict(gram=np.eye(2), coordinates=np.array([[0., 4., 0.], [0., 0., 0.]]),
                 groups=np.zeros(2,dtype=int),names=["task"],energy=np.array([20.,20.,0.]),
                 raw_energy=np.array([20.,20.,1.]),n=20)
    fits = api().fit_scalar_components(terms, {"variance_bound":.001})
    assert fits[0]["ratios"] == [0.] and fits[0]["at_zero"]
    assert fits[1]["ratios"][0] == pytest.approx(.01)
    assert fits[1]["at_upper"]
    assert not fits[2]["valid"]
    terms["gram"][:]=0
    terms["coordinates"][:]=0
    inactive=api().fit_scalar_components(terms,{})
    assert inactive[0]["inactive"] == [True] and inactive[0]["at_zero"]


def test_ar1_dispatch_avoids_generic_optimizer(planted_runs, monkeypatch):
    from experiments.hrf_estimation import reml_extended_fit as extended
    runs, library = planted_runs
    def forbidden(*args, **kwargs):
        raise AssertionError("AR1 must not run a per-vertex generic optimizer")
    monkeypatch.setattr(extended, "fit_components", forbidden)
    result=extended.fit_extended_reml(runs, library, {"variant":"ar1"})
    assert result.diagnostics["component_names"] == ["independent_task_junk"]
