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
