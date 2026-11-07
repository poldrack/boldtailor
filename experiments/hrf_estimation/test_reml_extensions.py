"""Extended REML must retain an exactly comparable restricted likelihood."""

from dataclasses import replace
import importlib

import numpy as np
import pytest
from scipy.linalg import block_diag, null_space
from scipy.optimize._numdiff import approx_derivative

from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.test_library_cv import planted_runs


def api():
    return importlib.import_module("experiments.hrf_estimation.reml_extensions")


def test_ar1_whitening_matches_stationary_covariance():
    rho = .6
    n = 12
    w = api().whiten(np.eye(n), rho)
    covariance = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / (1-rho**2)
    np.testing.assert_allclose(w @ covariance @ w.T, np.eye(n), atol=1e-12)


@pytest.mark.parametrize("separate,shared", [(False, False), (True, False), (False, True), (True, True)])
def test_low_rank_objective_and_gradient_equal_dense(planted_runs, separate, shared):
    runs, library = planted_runs
    rng = np.random.default_rng(11)
    runs = tuple(replace(r, signals=r.signals + rng.normal(size=r.signals.shape)) for r in runs[:2])
    kernel = library.candidates[1]
    common = api().prepare_common(runs, rho=.4)
    terms = api().component_terms(runs, kernel, common, separate=separate, shared=shared)
    matrices, ys = [], []
    for run in runs:
        parts = build_design(run, kernel)
        c = api().whiten(parts.nuisance, .4)
        basis = null_space(c.T)
        matrices.append(basis.T @ api().whiten(np.column_stack([parts.task, parts.junk]), .4))
        ys.append(basis.T @ api().whiten(run.signals, .4))
    y = np.vstack(ys)
    independent = block_diag(*matrices)
    blocks = ([independent[:, np.tile([True,True,False,False,False],len(runs))],
               independent[:, np.tile([False,False,True,False,False],len(runs))],
               independent[:, np.tile([False,False,False,True,True],len(runs))]]
              if separate else [independent])
    if shared:
        joined = np.vstack(matrices)[:, :3]
        blocks.extend([joined[:, :2], joined[:, 2:3]] if separate else [joined])
    ratios = np.arange(1, len(blocks)+1) * .2
    covariance = np.eye(len(y)) + sum(r * b @ b.T for r,b in zip(ratios, blocks))
    q = np.sum(y[:, 0] * np.linalg.solve(covariance, y[:, 0]))
    expected = len(y)*np.log(q/len(y)) + np.linalg.slogdet(covariance)[1]
    actual, gradient, sigma = api().component_objective(ratios, terms, 0)
    assert actual == pytest.approx(expected, abs=1e-8)
    assert sigma == pytest.approx(q/len(y), abs=1e-10)
    numerical = approx_derivative(lambda x: api().component_objective(x, terms, 0)[0], ratios)
    np.testing.assert_allclose(gradient, numerical.ravel(), atol=1e-5, rtol=1e-5)
    assert len(terms["names"]) == len(blocks)


def test_noise_estimation_uses_training_only_and_is_candidate_independent(planted_runs):
    runs, _ = planted_runs
    first = api().estimate_noise(runs[:3], "amplitude")
    runs[3].signals[:] = 1e15
    second = api().estimate_noise(runs[:3], "amplitude")
    assert first == second
    assert first["training_run_ids"] == [r.run_id for r in runs[:3]]
    assert -.8 <= first["rho"] <= .8


def test_component_optimizer_reduces_to_original_scalar_reml(planted_runs):
    from experiments.hrf_estimation.reml import prepare_terms, estimate_ratio
    runs, library = planted_runs
    rng = np.random.default_rng(31)
    runs = tuple(replace(r, signals=r.signals + rng.normal(0, .4, r.signals.shape)) for r in runs)
    kernel = library.candidates[1]
    reference = estimate_ratio(prepare_terms(runs, kernel))
    terms = api().component_terms(runs, kernel, api().prepare_common(runs))
    for f in range(2):
        fit = api().fit_components(terms, f, {})
        assert fit["criterion"] == pytest.approx(reference["criterion"][f], abs=1e-6)
        assert fit["ratios"][0] == pytest.approx(reference["ratio"][f], rel=2e-4)


@pytest.mark.parametrize("variant", ["ar1", "components", "shared", "combined"])
def test_variants_recover_shapes_and_keep_training_metadata(planted_runs, variant):
    runs, library = planted_runs
    rng = np.random.default_rng(49)
    noisy = tuple(replace(r, signals=r.signals + rng.normal(0, .1, r.signals.shape)) for r in runs)
    estimate = api().fit_extended_reml(noisy, library, {"variant": variant})
    assert [k.id for k in estimate.kernels] == [0, 1]
    assert estimate.training_run_ids == tuple(r.run_id for r in noisy)
    assert "candidate_gap" in estimate.diagnostics
    assert estimate.diagnostics["variant"] == variant
    assert len(estimate.diagnostics["component_names"]) == {"ar1":1,"components":3,"shared":2,"combined":5}[variant]
    constant = tuple(replace(r, signals=np.ones_like(r.signals)) for r in runs)
    invalid = api().fit_extended_reml(constant, library, {"variant": variant})
    assert invalid.kernels == (None, None)
    assert invalid.status == ("no_signal", "no_signal")


def test_variance_boundary_is_exactly_zero(planted_runs):
    runs, library = planted_runs
    rng = np.random.default_rng(3)
    changed = []
    for r in runs:
        parts = build_design(r, library.candidates[0])
        x = np.column_stack([parts.nuisance, parts.task, parts.junk])
        y = rng.normal(size=r.signals.shape)
        changed.append(replace(r, signals=y-x@np.linalg.lstsq(x,y,rcond=None)[0]))
    terms = api().component_terms(tuple(changed), library.candidates[0],
                                   api().prepare_common(tuple(changed)), separate=True, shared=True)
    fit = api().fit_components(terms, 0, {})
    np.testing.assert_array_equal(fit["ratios"], np.zeros(5))
