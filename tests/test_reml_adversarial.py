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


# ------------------------------------------- R1: unidentifiable variance split


def _assert_unidentifiable(fit):
    assert set(fit.status) == {"unidentifiable"}
    for name in ("ratio", "noise_variance", "effect_variance"):
        assert np.isnan(getattr(fit, name)).all()
    assert np.isfinite(fit.criterion).all()
    assert not fit.at_upper.any()


def _complement_basis(fixed):
    """Orthonormal basis of the orthogonal complement of the fixed design."""
    u, _, _ = np.linalg.svd(fixed, full_matrices=True)
    return u[:, fixed.shape[1] :]


def _ar1_root(n, rho):
    """Z with ZZ' equal to the AR(1) correlation matrix."""
    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.linalg.cholesky(float(rho) ** lags)


@pytest.mark.parametrize("seed", range(8))
def test_identity_random_covariance_is_unidentifiable(seed):
    y = np.random.default_rng(seed).normal(size=(10, 3))
    _assert_unidentifiable(fit_reml(_no_fixed(y), [np.eye(10)]))


@pytest.mark.parametrize("seed", range(4))
def test_confounding_after_fixed_projection_is_unidentifiable(seed):
    rng = np.random.default_rng(seed)
    fixed = np.column_stack([np.ones(14), rng.normal(size=(14, 2))])
    y = rng.normal(size=(14, 2))
    z = 1.7 * _complement_basis(fixed)
    _assert_unidentifiable(fit_reml([RemlGroup(y, fixed)], [z]))


@pytest.mark.parametrize("rho", [0.3, 0.6])
def test_random_covariance_proportional_to_ar1_noise_is_unidentifiable(rho):
    y = np.random.default_rng(1).normal(size=(12, 2))
    z = 0.8 * _ar1_root(12, rho)
    _assert_unidentifiable(fit_reml(_no_fixed(y), [z], rho=rho))


def _scaled_identity_groups(seed, scales, n=40, n_features=3, tau2=4.0):
    """Groups with Z_g = c_g I: variance (1 + tau2 c_g^2) per group."""
    rng = np.random.default_rng(seed)
    groups = [
        RemlGroup(
            rng.normal(scale=np.sqrt(1 + tau2 * c**2), size=(n, n_features)),
            np.empty((n, 0)),
        )
        for c in scales
    ]
    return groups, [c * np.eye(n) for c in scales]


def test_common_scale_across_groups_is_unidentifiable():
    groups, random = _scaled_identity_groups(0, (2.0, 2.0))
    _assert_unidentifiable(fit_reml(groups, random))


def test_differing_scales_across_groups_identify_the_ratio():
    groups, random = _scaled_identity_groups(0, (0.2, 3.0), n=400)
    fit = fit_reml(groups, random)
    assert fit.status == ("ok",) * 3
    np.testing.assert_allclose(fit.ratio, 4.0, rtol=0.5)
    np.testing.assert_allclose(fit.noise_variance, 1.0, rtol=0.3)


def test_unsaturated_isotropic_design_stays_identifiable():
    rng = np.random.default_rng(3)
    z = np.linalg.qr(rng.normal(size=(60, 5)))[0]
    y = z @ rng.normal(scale=3.0, size=(5, 2)) + rng.normal(size=(60, 2))
    assert fit_reml(_no_fixed(y), [z]).status == ("ok", "ok")


def test_near_confounded_design_is_not_flagged_unidentifiable():
    rng = np.random.default_rng(0)
    z = np.eye(10) + 1e-2 * rng.normal(size=(10, 10))
    fit = fit_reml(_no_fixed(rng.normal(size=(10, 3))), [z])
    assert "unidentifiable" not in fit.status


@pytest.fixture
def planted_with_saturated_decoy():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(30, 3))
    y = z @ rng.normal(scale=3.0, size=(3, 2)) + rng.normal(size=(30, 2))
    return _no_fixed(y), [z], [np.eye(30)]


@pytest.mark.parametrize("planted_first", [True, False])
def test_unidentifiable_candidate_never_wins(
    planted_with_saturated_decoy, planted_first
):
    groups, planted, decoy = planted_with_saturated_decoy
    candidates = [planted, decoy] if planted_first else [decoy, planted]
    selection = select_random_design(groups, candidates)
    expected = 0 if planted_first else 1
    np.testing.assert_array_equal(selection.winner, expected)
    assert selection.fit.status == ("ok", "ok")
    assert np.isnan(selection.gap).all()


def test_only_unidentifiable_candidates_leave_the_null_fit(
    planted_with_saturated_decoy,
):
    groups, _, decoy = planted_with_saturated_decoy
    selection = select_random_design(groups, [decoy])
    null = fit_reml(groups, [None])
    np.testing.assert_array_equal(selection.winner, -1)
    assert selection.fit.status == null.status
    np.testing.assert_array_equal(selection.fit.criterion, null.criterion)
    np.testing.assert_array_equal(selection.fit.ratio, 0.0)
