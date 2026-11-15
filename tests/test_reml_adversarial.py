"""Adversarial regressions from the 2026-11-15 REML review, via the public API."""

import numpy as np
import pytest

from boldtailor.reml import RemlGroup, fit_reml, select_random_design

N_SCANS = 12
SPECTRA = ("wide", "moderate", "clustered", "repeated")


def _no_fixed(signals):
    return [RemlGroup(signals, np.empty((len(signals), 0)))]


def _dense_criterion(y, z, ratio):
    """Profiled criterion from the dense covariance I + ratio ZZ'."""
    v = np.eye(len(y)) + ratio * (z @ z.T)
    q = (y.T @ np.linalg.solve(v, y)).item()
    return len(y) * np.log(q / len(y)) + np.linalg.slogdet(v)[1]


# ------------------------------------------------- R2: missed positive optimum


@pytest.fixture
def narrow_minimum():
    """Review counterexample: a minimum narrower than the search grid spacing."""
    y = np.array(
        [
            5.659277805656175,
            1.3575913902516368,
            12.166250878076637,
            440.03410697431354,
            0.024484911495792103,
            0.2989731950657336,
            754.0944608984961,
            0.023686778553470386,
            23.645757013142806,
            0.005169198169685577,
        ]
    )[:, None]
    z = np.zeros((10, 8))
    z[:8] = np.diag(np.sqrt(np.logspace(-10, 10, 8)))
    return y, z


def test_optimum_is_no_worse_than_known_positive_ratio(narrow_minimum):
    y, z = narrow_minimum
    fit = fit_reml(_no_fixed(y), [z])
    assert fit.criterion[0] <= _dense_criterion(y, z, 7.126268271680137e-7) + 1e-9
    assert fit.status == ("ok",)
    assert fit.ratio[0] == pytest.approx(7.1263e-7, rel=1e-3)


def test_narrow_minimum_is_selectable(narrow_minimum):
    y, z = narrow_minimum
    assert select_random_design(_no_fixed(y), [[z]]).winner[0] == 0


def _log10_spectrum(rng, kind, k):
    if kind == "wide":
        return rng.uniform(-10, 10, size=k)
    if kind == "moderate":
        return rng.uniform(-2, 2, size=k)
    if kind == "clustered":
        return rng.choice([-6.0, 5.0], size=k) + rng.normal(scale=0.05, size=k)
    return np.repeat(rng.uniform(-8, 8, size=2), k)[:k]


def _diagonal_problem(seed, kind):
    """Heavy-tailed signal against an orthogonal design with a chosen spectrum."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, N_SCANS - 1))
    eigenvalues = 10.0 ** _log10_spectrum(rng, kind, k)
    y = rng.normal(size=N_SCANS) * 10.0 ** rng.uniform(-2, 3, size=N_SCANS)
    z = np.zeros((N_SCANS, k))
    z[:k] = np.diag(np.sqrt(eigenvalues))
    return y[:, None], z, eigenvalues


def _scan_minimum(y, eigenvalues, points=40001):
    """Lowest criterion over zero and a fine log scan, written out directly."""
    low = -np.log(eigenvalues.max()) - np.log(1e8)
    high = -np.log(eigenvalues.min()) + np.log(1e8)
    ratios = np.concatenate([[0.0], np.exp(np.linspace(low, high, points))])
    scaled = ratios[:, None] * eigenvalues
    y2 = y[:, 0] ** 2
    k = eigenvalues.size
    q = y2[k:].sum() + (y2[:k] / (1.0 + scaled)).sum(axis=1)
    return (len(y) * np.log(q / len(y)) + np.log1p(scaled).sum(axis=1)).min()


@pytest.mark.parametrize("kind", SPECTRA)
@pytest.mark.parametrize("seed", range(60))
def test_optimum_is_no_worse_than_fine_scan(seed, kind):
    y, z, eigenvalues = _diagonal_problem(seed, kind)
    fit = fit_reml(_no_fixed(y), [z])
    assert fit.criterion[0] <= _scan_minimum(y, eigenvalues) + 1e-8
