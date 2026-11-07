"""Check direct REML against a dense, independently assembled covariance."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import null_space

from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.test_library_cv import planted_runs


def test_restricted_objective_equals_dense_covariance(planted_runs):
    runs, library = planted_runs
    api = importlib.import_module("experiments.hrf_estimation.reml")
    rng = np.random.default_rng(91)
    runs = tuple(
        replace(r, signals=r.signals + rng.normal(0, 0.2, r.signals.shape))
        for r in runs[:2]
    )
    candidate = library.candidates[1]
    terms = api.prepare_terms(runs, candidate)
    for ratio in [0.0, 0.01, 1.0, 10.0]:
        q = np.zeros(2)
        logdet = 0.0
        n = 0
        for run in runs:
            parts = build_design(run, candidate)
            basis = null_space(parts.nuisance.T)
            z = basis.T @ np.column_stack([parts.task, parts.junk])
            y = basis.T @ run.signals
            covariance = np.eye(len(y)) + ratio * z @ z.T
            q += np.sum(y * np.linalg.solve(covariance, y), axis=0)
            logdet += np.linalg.slogdet(covariance)[1]
            n += len(y)
        expected = n * np.log(q / n) + logdet
        np.testing.assert_allclose(
            api.criterion_grid(np.array([ratio]), terms)[0],
            expected,
            rtol=1e-9,
            atol=1e-8,
        )


def test_reml_is_invariant_to_fixed_nuisance_scaling(planted_runs):
    runs, library = planted_runs
    api = importlib.import_module("experiments.hrf_estimation.reml")
    first = api.prepare_terms(runs, library.candidates[1])
    second = api.prepare_terms(
        tuple(replace(r, confounds=r.confounds * 17) for r in runs),
        library.candidates[1],
    )
    np.testing.assert_allclose(
        api.criterion_grid(np.array([0.0, 1.0, 20.0]), first),
        api.criterion_grid(np.array([0.0, 1.0, 20.0]), second),
        atol=1e-8,
    )


def test_reml_recovers_planted_hrfs_and_flags_no_signal(planted_runs):
    runs, library = planted_runs
    fit = importlib.import_module("experiments.hrf_estimation.reml").fit_library_reml
    rng = np.random.default_rng(99)
    noisy = tuple(
        replace(r, signals=r.signals + rng.normal(0, 0.1, r.signals.shape))
        for r in runs
    )
    result = fit(noisy, library, {})
    assert [k.id for k in result.kernels] == [0, 1]
    assert np.all(np.asarray(result.diagnostics["variance_ratio"]) > 0)
    empty = fit(
        tuple(replace(r, signals=np.ones_like(r.signals)) for r in runs), library, {}
    )
    assert empty.kernels == (None, None)
    assert empty.status == ("no_signal", "no_signal")


def test_zero_random_effect_boundary_is_checked_explicitly(planted_runs):
    runs, library = planted_runs
    api = importlib.import_module("experiments.hrf_estimation.reml")
    rng = np.random.default_rng(18)
    null = []
    for run in runs:
        p = build_design(run, library.candidates[0])
        x = np.column_stack([p.nuisance, p.task, p.junk])
        y = rng.normal(size=run.signals.shape)
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        null.append(replace(run, signals=residual))
    terms = api.prepare_terms(tuple(null), library.candidates[0])
    result = api.estimate_ratio(terms)
    np.testing.assert_array_equal(result["ratio"], [0, 0])
    assert result["at_zero"].all()
