"""Adversarial regressions from the 2026-11-15 REML review, via the public API."""

import numpy as np
import pytest

from boldtailor.reml import RemlGroup, fit_reml, select_random_effects_design

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
    assert select_random_effects_design(_no_fixed(y), [[z]]).winner[0] == 0


def _log10_spectrum(rng, kind, k):
    if kind == "wide":
        return rng.uniform(-10, 10, size=k)
    if kind == "moderate":
        return rng.uniform(-2, 2, size=k)
    if kind == "clustered":
        return rng.choice([-6.0, 5.0], size=k) + rng.normal(scale=0.05, size=k)
    return np.resize(rng.uniform(-8, 8, size=2), k)


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
    selection = select_random_effects_design(groups, candidates)
    expected = 0 if planted_first else 1
    np.testing.assert_array_equal(selection.winner, expected)
    assert selection.fit.status == ("ok", "ok")
    assert np.isnan(selection.gap).all()


def test_only_unidentifiable_candidates_leave_the_null_fit(
    planted_with_saturated_decoy,
):
    groups, _, decoy = planted_with_saturated_decoy
    selection = select_random_effects_design(groups, [decoy])
    null = fit_reml(groups, [None])
    np.testing.assert_array_equal(selection.winner, -1)
    assert selection.fit.status == null.status
    np.testing.assert_array_equal(selection.fit.criterion, null.criterion)
    np.testing.assert_array_equal(selection.fit.ratio, 0.0)


# ------------------------------------------- R3: fixed-span shifts and no_signal

SHIFT_SCALES = (1e4, 1e6, 1e8, 1e10)


def _shift_tolerance(n_scans, scale):
    """A hundred times the projection error of a shift of this scale, relative
    to unit-scale residuals: n eps scale."""
    return 100 * n_scans * np.finfo(float).eps * scale


def _planted_problem(seed, n_fixed, rho, n=40):
    rng = np.random.default_rng(seed)
    fixed = np.column_stack([np.ones(n), rng.normal(size=(n, n_fixed - 1))])
    planted = rng.normal(size=(n, 3))
    decoy = rng.normal(size=(n, 3))
    noise = rng.normal(size=(n, 4))
    noise[1:] += rho * noise[:-1]
    signals = planted @ rng.normal(scale=3.0, size=(3, 4)) + noise
    return signals, fixed, planted, decoy


def _shifted(signals, fixed, scale, seed=0):
    coefficients = np.random.default_rng(seed).normal(size=(fixed.shape[1], 1))
    return signals + scale * (fixed @ coefficients)


@pytest.mark.parametrize("scale", SHIFT_SCALES)
@pytest.mark.parametrize("rho", [0.0, 0.4])
@pytest.mark.parametrize("n_fixed", [1, 3])
def test_fit_is_invariant_to_resolvable_fixed_span_shifts(n_fixed, rho, scale):
    signals, fixed, planted, _ = _planted_problem(19, n_fixed, rho)
    base = fit_reml([RemlGroup(signals, fixed)], [planted], rho=rho)
    moved = _shifted(signals, fixed, scale)
    shifted = fit_reml([RemlGroup(moved, fixed)], [planted], rho=rho)
    tolerance = _shift_tolerance(len(signals), scale)
    assert base.status == ("ok",) * 4
    assert shifted.status == base.status
    np.testing.assert_allclose(shifted.ratio, base.ratio, rtol=tolerance)
    np.testing.assert_allclose(
        shifted.noise_variance, base.noise_variance, rtol=tolerance
    )


@pytest.mark.parametrize("scale", SHIFT_SCALES)
def test_selection_is_invariant_to_resolvable_fixed_span_shifts(scale):
    signals, fixed, planted, decoy = _planted_problem(7, 2, 0.0)
    moved = _shifted(signals, fixed, scale)
    selection = select_random_effects_design(
        [RemlGroup(moved, fixed)], [[decoy], [planted]]
    )
    np.testing.assert_array_equal(selection.winner, 1)
    assert selection.fit.status == ("ok",) * 4


@pytest.mark.parametrize("scale", [1.0, 1e6])
@pytest.mark.parametrize("rho", [0.0, 0.6])
@pytest.mark.parametrize("n_fixed", [1, 3, 8])
def test_signals_inside_the_fixed_span_are_no_signal(n_fixed, rho, scale):
    _, fixed, planted, _ = _planted_problem(3, n_fixed, rho, n=200)
    coefficients = np.random.default_rng(4).normal(size=(n_fixed, 6))
    signals = scale * (fixed @ coefficients)
    fit = fit_reml([RemlGroup(signals, fixed)], [planted], rho=rho)
    assert fit.status == ("no_signal",) * 6


def test_shift_that_destroys_the_residual_is_no_signal():
    signals, fixed, planted, _ = _planted_problem(19, 1, 0.0)
    fit = fit_reml([RemlGroup(signals + 1e18, fixed)], [planted])
    assert fit.status == ("no_signal",) * 4


def _weak_column_problem(seed=0):
    """Signals in the span of an ill-conditioned saturated design, dominated
    by its weakest column, where the computed span is least accurate."""
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(10, 9)) * np.logspace(0, 9, 9)
    weights = rng.normal(size=(9, 40)) * np.logspace(0, -12, 9)[:, None]
    return design @ weights, design


def test_ill_conditioned_saturated_fixed_span_signal_is_no_signal():
    signals, fixed = _weak_column_problem()
    random = np.random.default_rng(1).normal(size=(10, 2))
    fit = fit_reml([RemlGroup(signals, fixed)], [random])
    assert fit.status == ("no_signal",) * 40


# ------------------------------ follow-up F1: column units and unresolved fits


@pytest.fixture
def scaled_fixed_problem():
    """Follow-up review probe: orthogonal fixed columns in unequal units."""
    eye = np.eye(30)
    fixed = eye[:, :2] * np.array([1e-12, 1.0])
    signal = (2 * eye[:, 2] + eye[:, 3])[:, None]
    return fixed, eye[:, [2]], signal, signal + 100 * eye[:, [0]]


def test_exactly_removable_shift_preserves_reml(scaled_fixed_problem):
    fixed, random, signal, shifted = scaled_fixed_problem
    base = fit_reml([RemlGroup(signal, fixed)], [random])
    moved = fit_reml([RemlGroup(shifted, fixed)], [random])
    assert base.status == ("ok",)
    assert moved.status == base.status
    np.testing.assert_allclose(moved.ratio, base.ratio, rtol=1e-12)


def _drift_problem(seed=2, n=300):
    """Planted effect on a fixed design with unnormalised drift and small units."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    fixed = np.column_stack([np.ones(n), t, t**2, t**3, 0.05 * rng.normal(size=(n, 6))])
    planted = rng.normal(size=(n, 3))
    signals = planted @ rng.normal(scale=3.0, size=(3, 4)) + rng.normal(size=(n, 4))
    return signals, fixed, planted


@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8])
def test_fit_is_invariant_to_shifts_in_a_fixed_span_of_unequal_units(scale):
    signals, fixed, planted = _drift_problem()
    unit_shift = _unit_columns(fixed) @ np.random.default_rng(0).normal(size=(10, 1))
    moved = signals + scale * unit_shift
    base = fit_reml([RemlGroup(signals, fixed)], [planted])
    shifted = fit_reml([RemlGroup(moved, fixed)], [planted])
    tolerance = max(_shift_tolerance(len(signals), scale), 1e-9)
    assert base.status == ("ok",) * 4
    assert shifted.status == base.status
    np.testing.assert_allclose(shifted.ratio, base.ratio, rtol=tolerance)


def _unit_columns(design):
    return design / np.linalg.norm(design, axis=0)


@pytest.mark.parametrize("scales", [(1e-20, 1.0, 1e15), (1e6, 1e-9, 1.0)])
def test_fit_does_not_depend_on_fixed_column_units(scales):
    signals, fixed, planted, _ = _planted_problem(19, 3, 0.0)
    base = fit_reml([RemlGroup(signals, fixed)], [planted])
    rescaled = fit_reml([RemlGroup(signals, fixed * np.array(scales))], [planted])
    assert rescaled.status == base.status
    assert rescaled.restricted_dof == base.restricted_dof
    np.testing.assert_allclose(rescaled.ratio, base.ratio, rtol=1e-8)
    np.testing.assert_allclose(rescaled.criterion, base.criterion, rtol=1e-10)


def test_tiny_unit_fixed_column_still_costs_a_degree_of_freedom():
    signals, fixed, planted, _ = _planted_problem(19, 3, 0.0)
    tiny = fixed * np.array([1.0, 1e-30, 1.0])
    assert fit_reml([RemlGroup(signals, tiny)], [planted]).restricted_dof == 37


def _collinear_fixed(rng, n=30, gap=1e-9):
    fixed = rng.normal(size=(n, 4))
    fixed[:, 3] = fixed[:, 0] + gap * rng.normal(size=n)
    return fixed, (fixed[:, [3]] - fixed[:, [0]]) / gap


def test_residual_below_the_span_uncertainty_is_unresolved():
    """A genuine residual of 1e-8 beside an order-one component along the
    weak direction of a near-collinear fixed design cannot be certified."""
    rng = np.random.default_rng(0)
    fixed, weak = _collinear_fixed(rng)
    random = rng.normal(size=(30, 2))
    signals = weak @ rng.normal(size=(1, 5)) + 1e-8 * rng.normal(size=(30, 5))
    fit = fit_reml([RemlGroup(signals, fixed)], [random])
    assert fit.status == ("unresolved",) * 5
    for name in ("ratio", "noise_variance", "effect_variance", "criterion"):
        assert np.isnan(getattr(fit, name)).all()
    selection = select_random_effects_design([RemlGroup(signals, fixed)], [[random]])
    np.testing.assert_array_equal(selection.winner, -1)
    assert selection.fit.status == ("unresolved",) * 5
    assert np.isnan(selection.gap).all()


def test_collinear_fixed_design_alone_does_not_make_a_fit_unresolved():
    """The same residual without the weak-direction component is resolved."""
    rng = np.random.default_rng(0)
    fixed, _ = _collinear_fixed(rng)
    random = rng.normal(size=(30, 2))
    signals = fixed[:, :3] @ rng.normal(size=(3, 5)) + 1e-8 * rng.normal(size=(30, 5))
    fit = fit_reml([RemlGroup(signals, fixed)], [random])
    assert set(fit.status) <= {"ok", "variance_zero"}


def test_unresolved_features_do_not_affect_resolved_ones():
    rng = np.random.default_rng(0)
    fixed, weak = _collinear_fixed(rng)
    random = rng.normal(size=(30, 2))
    live = random @ rng.normal(size=(2, 2)) + rng.normal(size=(30, 2))
    uncertain = weak @ rng.normal(size=(1, 1)) + 1e-8 * rng.normal(size=(30, 1))
    mixed = np.column_stack([live[:, [0]], uncertain, live[:, [1]]])
    fit = fit_reml([RemlGroup(mixed, fixed)], [random])
    reference = fit_reml([RemlGroup(live, fixed)], [random])
    assert fit.status[1] == "unresolved"
    np.testing.assert_array_equal(fit.ratio[[0, 2]], reference.ratio)
    np.testing.assert_array_equal(fit.criterion[[0, 2]], reference.criterion)
