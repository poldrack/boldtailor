"""Seeded Monte Carlo checks of the statistical behaviour of scalar REML.

Monte Carlo replicates are features: one simulation and one fit per scenario.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from boldtailor._reml_criterion import profiled_variance
from boldtailor._reml_terms import project_fixed, reml_terms
from boldtailor.ar1 import estimate_rho
from boldtailor.reml import RemlGroup, fit_reml, select_random_effects_design
from tests.reml_oracles import (
    ar1_correlation,
    dense_criterion,
    dense_fit,
    null_lrt_reference,
    simulate,
)

BOOTSTRAP_RESAMPLES = 200
ZERO_STATISTIC = 1e-9


@pytest.fixture(
    params=[
        pytest.param(2_000, id="default"),
        pytest.param(50_000, id="slow", marks=pytest.mark.slow),
    ]
)
def replicates(request):
    return request.param


# ---------------------------------------------------------------- helpers


def mean_se(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def median_se(values, seed=0):
    """Bootstrap standard error of the median."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    medians = [
        np.median(values[rng.integers(0, values.size, values.size)])
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return float(np.std(medians, ddof=1))


def binomial_se(p, n):
    return float(np.sqrt(p * (1.0 - p) / n))


def as_groups(signals, fixed):
    return [RemlGroup(y, c) for y, c in zip(signals, fixed, strict=True)]


def combined_se(*errors):
    return float(np.sqrt(np.sum(np.square(errors))))


# ---------------------------------------------------------------- unbiasedness


def unbiased_statistics(replicates):
    rng = np.random.default_rng(801)
    signals, fixed, random = simulate(
        rng,
        scans=(80,) * 3,
        n_fixed=25,
        n_random=4,
        ratio=1.0,
        sigma2=2.0,
        n_features=replicates,
    )
    projections = [project_fixed(y, c) for y, c in zip(signals, fixed, strict=True)]
    terms = [reml_terms(p, z) for p, z in zip(projections, random, strict=True)]
    reml = profiled_variance(np.ones(replicates), terms)
    ml = reml * sum(p.dof for p in projections) / sum(len(y) for y in signals)
    return reml, ml


def test_noise_variance_unbiased_at_true_ratio(replicates):
    """Verify noise variance unbiased at true ratio."""
    reml, ml = unbiased_statistics(replicates)
    assert abs(reml.mean() - 2.0) < 3 * mean_se(reml)
    assert abs(ml.mean() - 2.0) > 10 * mean_se(ml)


# ---------------------------------------------------------------- consistency

GROUP_COUNTS = (2, 8, 32)


def consistency_statistics(replicates, n_groups):
    rng = np.random.default_rng(802 + n_groups)
    signals, fixed, random = simulate(
        rng, scans=(60,) * n_groups, ratio=1.0, sigma2=1.0, n_features=replicates
    )
    fit = fit_reml(as_groups(signals, fixed), random)
    return {
        "ratio": (abs(np.median(fit.ratio) - 1.0), median_se(fit.ratio)),
        "noise": (abs(fit.noise_variance.mean() - 1.0), mean_se(fit.noise_variance)),
    }


def assert_non_increasing(biases):
    for (previous, previous_se), (current, current_se) in zip(
        biases, biases[1:], strict=False
    ):
        assert current <= previous + 2 * combined_se(previous_se, current_se)


def test_estimates_are_consistent(replicates):
    """Verify estimates are consistent."""
    stats = [consistency_statistics(replicates, g) for g in GROUP_COUNTS]
    assert_non_increasing([s["ratio"] for s in stats])
    assert_non_increasing([s["noise"] for s in stats])
    assert stats[-1]["ratio"][0] < 0.05
    assert stats[-1]["noise"][0] < 0.02


# ---------------------------------------------------------------- null calibration

NULL_SEED = 803
NULL_REFERENCE_DRAWS = 200_000


def null_problem(replicates):
    rng = np.random.default_rng(NULL_SEED)
    return simulate(
        rng, scans=(60, 60), n_fixed=6, n_random=3, ratio=0.0, n_features=replicates
    )


def restricted_lrt(signals, fixed, random):
    groups = as_groups(signals, fixed)
    fit = fit_reml(groups, random)
    at_zero = fit_reml(groups, [None] * len(groups))
    statistic = at_zero.criterion - fit.criterion
    return np.where(np.array(fit.status) == "variance_zero", 0.0, statistic)


def null_statistics(replicates):
    signals, fixed, random = null_problem(replicates)
    statistic = restricted_lrt(signals, fixed, random)
    reference = null_lrt_reference(fixed, random, NULL_REFERENCE_DRAWS, NULL_SEED)
    return statistic, reference


def two_sample_se(p, replicates, draws):
    """SE of an observed rate against a reference rate from its own draws."""
    return combined_se(binomial_se(p, replicates), binomial_se(p, draws))


def test_null_lrt_matches_finite_sample_reference(replicates):
    """Verify null lrt matches finite sample reference."""
    statistic, reference = null_statistics(replicates)
    draws = reference.size
    p_zero = np.mean(reference <= ZERO_STATISTIC)
    observed = np.mean(statistic <= ZERO_STATISTIC)
    assert abs(observed - p_zero) < 4 * two_sample_se(p_zero, replicates, draws)
    for level in (0.90, 0.95):
        exceedance = np.mean(statistic > np.quantile(reference, level))
        tail = 1.0 - level
        assert abs(exceedance - tail) < 4 * two_sample_se(tail, replicates, draws)
    assert abs(p_zero - 0.5) > 0.02


def dense_restricted_lrt(signals, fixed, random):
    ratio, _ = dense_fit(signals, fixed, random)
    statistic = np.empty(ratio.size)
    for f, value in enumerate(ratio):
        feature = [y[:, [f]] for y in signals]
        at_zero = dense_criterion(feature, fixed, random, 0.0)[0]
        statistic[f] = at_zero - dense_criterion(feature, fixed, random, value)[0]
    return statistic


@pytest.mark.slow
def test_null_lrt_matches_dense_fits():
    """Verify null lrt matches dense fits."""
    signals, fixed, random = null_problem(500)
    assert_allclose(
        restricted_lrt(signals, fixed, random),
        dense_restricted_lrt(signals, fixed, random),
        rtol=0.0,
        atol=1e-6,
    )


# ---------------------------------------------------------------- rho correction

RHO_SCANS = 200
RHO_GROUPS = 6


def drift_basis(n, count):
    """Constant plus ``count`` discrete cosine drift regressors."""
    time = (np.arange(n) + 0.5) / n
    cosines = [np.cos(np.pi * k * time) for k in range(1, count + 1)]
    return np.column_stack([np.ones(n), *cosines])


def motion_regressors(rng, n, count):
    """Random-walk motion traces and their backward differences."""
    walks = np.cumsum(rng.normal(scale=0.1, size=(n, count)), axis=0)
    return np.column_stack([walks, np.diff(walks, axis=0, prepend=0.0)])


def task_regressors(rng, n, count):
    """Random block onsets convolved with a gamma-shaped response."""
    lags = np.arange(16.0)
    kernel = lags**5 * np.exp(-lags) / 120.0
    blocks = (rng.uniform(size=(n, count)) < 0.08).astype(float)
    return np.column_stack([np.convolve(b, kernel)[:n] for b in blocks.T])


def reference_design(rng, n=RHO_SCANS):
    """A 30-column nuisance-plus-task reference design."""
    design = np.column_stack(
        [drift_basis(n, 11), motion_regressors(rng, n, 6), task_regressors(rng, n, 6)]
    )
    assert design.shape == (n, 30)
    return design


def ar1_signals(rng, design, rho, n_features):
    noise = np.linalg.cholesky(ar1_correlation(len(design), rho))
    beta = rng.normal(size=(design.shape[1], n_features))
    return design @ beta + noise @ rng.normal(size=(len(design), n_features))


def rho_statistics(replicates, rho):
    rng = np.random.default_rng(804)
    designs = [reference_design(rng) for _ in range(RHO_GROUPS)]
    signals = [ar1_signals(rng, d, rho, replicates) for d in designs]
    return estimate_rho(signals, designs)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6])
def test_rho_correction_removes_projection_bias(replicates, rho):
    """Verify rho correction removes projection bias."""
    estimate = rho_statistics(replicates, rho)
    assert abs(estimate.pooled - rho) < 0.02
    if rho == 0.6:
        assert abs(np.median(estimate.raw) - rho) > 0.06


# ---------------------------------------------------------------- AR(1) fit


def ar1_fit_statistics(replicates):
    rng = np.random.default_rng(805)
    signals, fixed, random = simulate(
        rng,
        scans=(100,) * 16,
        ratio=1.0,
        sigma2=1.0,
        rho=0.5,
        n_features=replicates,
    )
    groups = as_groups(signals, fixed)
    return fit_reml(groups, random, rho=0.5), fit_reml(groups, random, rho=0.0)


def recovery_errors(fit):
    return abs(np.median(fit.ratio) - 1.0), abs(fit.noise_variance.mean() - 1.0)


def test_ar1_fit_recovers_parameters(replicates):
    """Verify AR(1) fit recovers parameters."""
    ar1, iid = ar1_fit_statistics(replicates)
    ratio_error, noise_error = recovery_errors(ar1)
    assert ratio_error < 0.05
    assert noise_error < 0.02
    # The iid fit understates the marginal noise variance: the fixed design
    # (the intercept above all) absorbs more than its share of AR(1) variance.
    # Both fits use the same data, so the paired SE of the per-feature
    # differences is the correct Monte Carlo error. No ratio comparison is
    # asserted: the iid fit's median ratio is not worse (50k replicates:
    # iid 1.006 vs AR(1) 0.986).
    shortfall = ar1.noise_variance - iid.noise_variance
    assert shortfall.mean() > 4 * mean_se(shortfall)


# ---------------------------------------------------------------- selection

SELECTION_RATIOS = (0.02, 0.2, 2.0)
N_CANDIDATES = 5
SHARED_FRACTION = 0.8


def correlated_candidates(rng, n, n_random):
    """Candidate designs sharing a component: column correlation ~0.8."""
    shared = rng.normal(size=(n, n_random))
    return [
        np.sqrt(SHARED_FRACTION) * shared
        + np.sqrt(1.0 - SHARED_FRACTION) * rng.normal(size=(n, n_random))
        for _ in range(N_CANDIDATES)
    ]


def planted_group(rng, n, ratio, n_features, n_fixed=4, n_random=3):
    fixed = np.column_stack([np.ones(n), rng.normal(size=(n, n_fixed - 1))])
    candidates = correlated_candidates(rng, n, n_random)
    u = rng.normal(scale=np.sqrt(ratio), size=(n_random, n_features))
    beta = rng.normal(size=(n_fixed, n_features))
    noise = rng.normal(size=(n, n_features))
    return fixed @ beta + candidates[0] @ u + noise, fixed, candidates


def selection_rate(replicates, ratio):
    rng = np.random.default_rng(806)
    parts = [planted_group(rng, 60, ratio, replicates) for _ in range(4)]
    groups = [RemlGroup(y, c) for y, c, _ in parts]
    candidates = [[p[2][k] for p in parts] for k in range(N_CANDIDATES)]
    selection = select_random_effects_design(groups, candidates)
    rate = float(np.mean(selection.winner == 0))
    return rate, binomial_se(rate, replicates)


def test_selection_rate_rises_with_snr(replicates):
    """Verify selection rate rises with snr."""
    rates = [selection_rate(replicates, r) for r in SELECTION_RATIOS]
    for (previous, previous_se), (current, current_se) in zip(
        rates, rates[1:], strict=False
    ):
        assert current - previous > 2 * combined_se(previous_se, current_se)
    assert rates[-1][0] > 0.95
