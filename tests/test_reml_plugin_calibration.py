"""Bounded Monte Carlo checks of scientific interpretation, not significance gates.

Replicate features are independent draws with shared designs. Tolerances are
declared in scientific units: 0.08 variance, 0.25 median ratio, and 0.03 tail
probability. They allow finite-sample and Monte Carlo variation, not arbitrary
numerical changes. These checks cover specified matched-model scenarios only.
"""

import numpy as np
import pytest

from boldtailor.ar1 import estimate_rho
from boldtailor.reml import RemlGroup, fit_reml, select_random_design
from tests.reml_oracles import simulate


@pytest.fixture(params=[(-0.5, 0.0), (0.5, 0.0), (-0.5, 1.0), (0.5, 1.0)])
def plugin_problem(request):
    rho, ratio = request.param
    signals, fixed, random = simulate(
        np.random.default_rng(620),
        scans=(96,) * 4,
        n_fixed=3,
        n_random=3,
        ratio=ratio,
        rho=rho,
        n_features=1500,
    )
    groups = [RemlGroup(y, c) for y, c in zip(signals, fixed, strict=True)]
    reference = [np.column_stack([c, z]) for c, z in zip(fixed, random, strict=True)]
    # The scenarios include rho = -0.5, outside the default fMRI range.
    estimated = estimate_rho(signals, reference, rho_range=(-0.95, 0.95)).pooled
    return groups, random, rho, ratio, estimated


def test_estimated_rho_preserves_matched_model_recovery_and_null_tail(plugin_problem):
    groups, random, rho, ratio, estimated = plugin_problem
    assert abs(estimated - rho) < 0.03
    oracle = fit_reml(groups, random, rho=rho)
    plugin = fit_reml(groups, random, rho=estimated)
    assert abs(np.mean(plugin.noise_variance) - 1.0) < 0.08
    if ratio:
        assert abs(np.median(plugin.ratio) - ratio) < 0.25
    else:
        null_design = [None] * len(groups)
        oracle_lrt = fit_reml(groups, null_design, rho=rho).criterion - oracle.criterion
        plugin_lrt = (
            fit_reml(groups, null_design, rho=estimated).criterion - plugin.criterion
        )
        threshold = np.quantile(oracle_lrt, 0.95)
        assert abs(np.mean(plugin_lrt > threshold) - 0.05) < 0.03
        assert abs(np.mean(plugin.ratio == 0) - np.mean(oracle.ratio == 0)) < 0.04


def test_null_library_winners_are_not_a_significance_test():
    rng = np.random.default_rng(621)
    n, features = 48, 2000
    groups = [
        RemlGroup(rng.normal(size=(n, features)), np.ones((n, 1))) for _ in range(3)
    ]
    candidates = [[rng.normal(size=(n, 3)) for _ in groups] for _ in range(8)]
    one = select_random_design(groups, candidates[:1])
    many = select_random_design(groups, candidates)
    # Nested libraries cannot lose existing support. Even one candidate is
    # selected far more than 5% of the time under a true null: winning != p<.05.
    assert np.all((one.winner < 0) | (many.winner >= 0))
    assert np.mean(one.winner >= 0) > 0.15
    assert np.mean(many.winner >= 0) > np.mean(one.winner >= 0) + 0.1


def test_heterogeneous_noise_is_a_pooled_variance_not_each_runs_variance():
    rng = np.random.default_rng(622)
    scans, variances = (30, 60, 90), (0.25, 1.0, 4.0)
    groups = [
        RemlGroup(rng.normal(scale=np.sqrt(v), size=(n, 3000)), np.ones((n, 1)))
        for n, v in zip(scans, variances, strict=True)
    ]
    fit = fit_reml(groups, [None] * 3)
    expected = np.average(variances, weights=np.array(scans) - 1)
    assert abs(fit.noise_variance.mean() - expected) < 0.02
    assert np.all(np.abs(fit.noise_variance.mean() - np.array(variances)) > 0.2)


def _correlation(n, rho):
    return rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))


def test_omitted_smooth_task_signal_contaminates_noise_autocorrelation():
    """Reference completeness is a scientific assumption, not a neutral knob."""
    rng = np.random.default_rng(623)
    signals, complete, incomplete = [], [], []
    for _ in range(4):
        n = 120
        fixed = np.ones((n, 1))
        task = np.linalg.cholesky(_correlation(n, 0.9)) @ rng.normal(size=(n, 3))
        noise = np.linalg.cholesky(_correlation(n, 0.4)) @ rng.normal(size=(n, 1000))
        signals.append(task @ rng.normal(scale=3, size=(3, 1000)) + noise)
        complete.append(np.column_stack([fixed, task]))
        incomplete.append(fixed)
    identified = estimate_rho(signals, complete).pooled
    contaminated = estimate_rho(signals, incomplete).pooled
    assert abs(identified - 0.4) < 0.04
    assert contaminated > identified + 0.2


def test_group_heterogeneous_rho_is_not_recovered_by_one_pooled_scalar():
    rng = np.random.default_rng(624)
    truth = np.array([-0.5, 0.7])
    n = 120
    signals = [
        np.linalg.cholesky(_correlation(n, r)) @ rng.normal(size=(n, 2000))
        for r in truth
    ]
    estimates = estimate_rho(signals, [np.ones((n, 1))] * 2)
    assert np.all(np.abs(np.nanmedian(estimates.values, axis=1) - truth) < 0.04)
    assert np.all(np.abs(estimates.pooled - truth) > 0.2)


def test_correlated_random_effects_require_their_covariance_factor():
    rng = np.random.default_rng(625)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    factor = rotation @ np.diag(np.sqrt([0.05, 1.0, 10.0]))
    groups, matched, isotropic = [], [], []
    for _ in range(24):
        z = rng.normal(size=(40, 3))
        correlated = z @ factor
        signal = correlated @ rng.normal(size=(3, 1000)) + rng.normal(size=(40, 1000))
        groups.append(RemlGroup(signal, np.ones((40, 1))))
        matched.append(correlated)
        isotropic.append(z)
    correct = fit_reml(groups, matched)
    misspecified = fit_reml(groups, isotropic)
    assert abs(np.median(correct.ratio) - 1.0) < 0.15
    assert abs(correct.noise_variance.mean() - 1.0) < 0.05
    # A scalar fitted to the wrong coefficient covariance is a different
    # estimand, not an estimate of the generating scale (one).
    assert np.median(misspecified.ratio) > 2.0
