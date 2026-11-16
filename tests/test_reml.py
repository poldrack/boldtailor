import dataclasses
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from boldtailor._reml_search import minimize_ratio
from boldtailor._reml_terms import project_fixed, reml_terms
import boldtailor.reml as reml
from boldtailor.reml import RemlFit, RemlGroup, fit_reml, select_random_effects_design
from tests.reml_oracles import dense_fit, simulate

N_FEATURES = 4


@pytest.fixture
def data():
    return simulate(
        np.random.default_rng(11), scans=(50, 65), ratio=1.0, n_features=N_FEATURES
    )


def _groups(signals, fixed):
    return [RemlGroup(y, c) for y, c in zip(signals, fixed, strict=True)]


def _fit(signals, fixed, random, rho=0.0):
    return fit_reml(_groups(signals, fixed), random, rho=rho)


def _columns(signals, index):
    return [y[:, index] for y in signals]


def test_fit_matches_layers(data):
    """Compare the public REML fit with explicit projection, criterion and optimization
    steps.
    """
    signals, fixed, random = data
    terms = [
        reml_terms(project_fixed(y, c), z)
        for y, c, z in zip(signals, fixed, random, strict=True)
    ]
    optimum = minimize_ratio(terms)
    fit = _fit(*data)
    assert isinstance(fit, RemlFit)
    assert_array_equal(fit.ratio, optimum.ratio)
    assert_array_equal(fit.criterion, optimum.criterion)
    assert_array_equal(fit.at_upper, optimum.at_upper)
    assert_allclose(fit.effect_variance, fit.ratio * fit.noise_variance, rtol=1e-15)
    assert fit.restricted_dof == sum(t.dof for t in terms)
    assert fit.status == ("ok",) * N_FEATURES
    assert_array_equal(fit.rho, np.zeros(N_FEATURES))


def test_noise_free_signal_is_ok_at_upper_edge():
    """Verify noise free signal is ok at upper edge."""
    rng = np.random.default_rng(4)
    _, fixed, random = simulate(rng, scans=(60,), n_features=2)
    signals = [fixed[0] @ rng.normal(size=(4, 2)) + random[0] @ rng.normal(size=(3, 2))]
    fit = _fit(signals, fixed, random)
    assert fit.at_upper.all()
    assert fit.status == ("ok", "ok")


def test_results_are_read_only(data):
    """Verify results are read only."""
    fit = _fit(*data)
    for name in ("ratio", "noise_variance", "effect_variance", "criterion"):
        with pytest.raises(ValueError):
            getattr(fit, name)[0] = 1.0
    with pytest.raises(ValueError):
        fit.at_upper[0] = True
    with pytest.raises(ValueError):
        fit.rho[0] = 0.1
    assert fit.at_upper.dtype == bool


def test_group_owns_read_only_float64_arrays(data):
    """Verify group owns read only float64 arrays."""
    signals, fixed, _ = data
    group = RemlGroup(signals[0], fixed[0])
    signals[0][0, 0] = 1e9
    assert group.signals[0, 0] != 1e9
    with pytest.raises(ValueError):
        group.signals[0, 0] = 0.0


def test_float32_input_is_converted(data):
    """Verify float32 input is converted."""
    signals, fixed, random = data
    as32 = [[a.astype(np.float32) for a in part] for part in (signals, fixed, random)]
    group = RemlGroup(as32[0][0], as32[1][0])
    assert group.signals.dtype == np.float64
    assert group.fixed_design.dtype == np.float64
    expected = _fit(*[[a.astype(np.float64) for a in part] for part in as32])
    fit = _fit(*as32)
    assert fit.ratio.dtype == np.float64
    assert_array_equal(fit.ratio, expected.ratio)


def test_per_feature_rho_equals_separate_fits():
    """Verify per feature rho equals separate fits."""
    signals, fixed, random = simulate(
        np.random.default_rng(5), scans=(50, 65), rho=0.3, n_features=N_FEATURES
    )
    rho = np.array([0.0, 0.3, 0.3, 0.6])
    fit = _fit(signals, fixed, random, rho=rho)
    for value, index in ((0.0, [0]), (0.3, [1, 2]), (0.6, [3])):
        single = _fit(_columns(signals, index), fixed, random, rho=value)
        assert_array_equal(fit.ratio[index], single.ratio)
        assert_array_equal(fit.noise_variance[index], single.noise_variance)
        assert_array_equal(fit.criterion[index], single.criterion)
        assert fit.restricted_dof == single.restricted_dof
    assert_array_equal(fit.rho, rho)


def _with_dead_columns(signals):
    out = []
    for y in signals:
        dead = np.column_stack(
            [y[:, 0], np.zeros(len(y)), y[:, 1], np.full(len(y), 3.5)]
        )
        out.append(dead)
    return out


def test_dead_features_do_not_affect_others(data):
    """Verify dead features do not affect others."""
    signals, fixed, random = data
    live = _columns(signals, [0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = _fit(_with_dead_columns(live), fixed, random)
        reference = _fit(live, fixed, random)
    assert (
        fit.status[1] == "no_residual_signal" and fit.status[3] == "no_residual_signal"
    )
    for name in ("noise_variance", "effect_variance", "criterion"):
        assert np.isnan(getattr(fit, name)[[1, 3]]).all()
        assert_array_equal(getattr(fit, name)[[0, 2]], getattr(reference, name))
    assert np.isnan(fit.ratio[[1, 3]]).all()
    assert_array_equal(fit.ratio[[0, 2]], reference.ratio)
    assert not fit.at_upper[[1, 3]].any()
    assert (fit.status[0], fit.status[2]) == reference.status


def test_all_features_dead_is_no_residual_signal(data):
    """Verify all features dead is no residual signal."""
    _, fixed, random = data
    signals = [np.zeros((len(c), 2)) for c in fixed]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = _fit(signals, fixed, random)
    assert fit.status == ("no_residual_signal", "no_residual_signal")
    assert np.isnan(fit.noise_variance).all()


def _ragged_data():
    rng = np.random.default_rng(21)
    scans, n_fixed = (40, 73, 55), (3, 5, 4)
    fixed = [
        np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
        for n, p in zip(scans, n_fixed)
    ]
    random = [rng.normal(size=(n, 3)) for n in scans]
    random[1][:, 2] = 0.0
    signals = [
        c @ rng.normal(size=(c.shape[1], 2))
        + z @ rng.normal(size=(3, 2))
        + rng.normal(size=(len(c), 2))
        for c, z in zip(fixed, random)
    ]
    return signals, fixed, random


def test_ragged_groups_match_dense_oracle():
    """Verify ragged groups match dense oracle."""
    signals, fixed, random = _ragged_data()
    expected, _ = dense_fit(signals, fixed, random)
    fit = _fit(signals, fixed, random)
    assert_allclose(fit.ratio, expected, rtol=1e-5)


def _corrupt(data, part, value):
    copies = [[a.copy() for a in p] for p in data]
    copies[part][1][2, 0] = value
    return copies


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize(
    "part,name", [(0, "signals"), (1, "fixed_design"), (2, "random_design")]
)
def test_non_finite_input_raises(data, part, name, value):
    """Verify non finite input raises."""
    with pytest.raises(ValueError, match=rf"{name}.*group 1|group 1.*{name}"):
        _fit(*_corrupt(data, part, value))


def test_no_groups_raises():
    """Verify no groups raises."""
    with pytest.raises(ValueError):
        fit_reml([], [])


def test_zero_features_raises(data):
    """Verify zero features raises."""
    signals, fixed, random = data
    empty = [y[:, :0] for y in signals]
    with pytest.raises(ValueError, match="signals"):
        _fit(empty, fixed, random)


def test_invalid_shapes_raise(data):
    """Verify invalid shapes raise."""
    signals, fixed, random = data
    groups = _groups(signals, fixed)
    with pytest.raises(ValueError, match="random_design"):
        fit_reml(groups, random[:1])
    with pytest.raises(ValueError, match="scan"):
        fit_reml([RemlGroup(signals[0], fixed[0][:-1])], random[:1])
    with pytest.raises(ValueError, match="scan"):
        fit_reml(groups[:1], [random[0][:-1]])
    with pytest.raises(ValueError, match="feature"):
        fit_reml([groups[0], RemlGroup(signals[1][:, :2], fixed[1])], random)


@pytest.mark.parametrize(
    "rho",
    [
        1.0,
        -1.0,
        1.5,
        np.nan,
        np.zeros(N_FEATURES - 1),
        np.zeros((N_FEATURES, 1)),
        np.array([0.0, 0.2, 1.0, 0.1]),
    ],
)
def test_invalid_rho_raises(data, rho):
    """Verify invalid rho raises."""
    with pytest.raises(ValueError, match="rho"):
        _fit(*data, rho=rho)


def test_none_random_design_contributes_no_directions(data):
    """Verify none random design contributes no directions."""
    signals, fixed, random = data
    fit = _fit(signals, fixed, [random[0], None])
    empty = _fit(signals, fixed, [random[0], np.empty((len(fixed[1]), 0))])
    assert_array_equal(fit.ratio, empty.ratio)


def test_variance_zero_status():
    """Report a fitted zero random-effects variance separately from absent residual
    signal.
    """
    signals, fixed, random = simulate(np.random.default_rng(8), scans=(60,))
    both = np.column_stack([fixed[0], random[0]])
    residual = signals[0] - both @ np.linalg.lstsq(both, signals[0], rcond=None)[0]
    fit = _fit([residual + fixed[0] @ np.ones((4, 1))], fixed, random)
    assert fit.status == ("variance_zero",)
    assert_array_equal(fit.ratio, 0.0)
    assert_array_equal(fit.effect_variance, 0.0)
    assert np.isfinite(fit.noise_variance).all()


def test_no_restricted_dof_raises():
    """Verify no restricted degrees-of-freedom raises."""
    rng = np.random.default_rng(0)
    signals, fixed = [rng.normal(size=(4, 2))], [rng.normal(size=(4, 4))]
    with pytest.raises(ValueError, match="degrees of freedom"):
        _fit(signals, fixed, [rng.normal(size=(4, 1))])


@pytest.fixture
def dof_differs_at_half(monkeypatch):
    """project_fixed reporting one fewer dof at rho 0.5, as a rank flip would."""
    original = reml.project_fixed

    def shifted(*args, **kwargs):
        projection = original(*args, **kwargs)
        if projection.rho == 0.5:
            return dataclasses.replace(projection, dof=projection.dof - 1)
        return projection

    monkeypatch.setattr(reml, "project_fixed", shifted)


def test_restricted_dof_must_agree_across_rho(data, dof_differs_at_half):
    """Verify restricted degrees-of-freedom must agree across rho."""
    signals, fixed, random = data
    rho = np.array([0.0, 0.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="restricted degrees of freedom"):
        _fit(signals, fixed, random, rho=rho)
    with pytest.raises(ValueError, match="restricted degrees of freedom"):
        select_random_effects_design(_groups(signals, fixed), [random], rho=rho)


def test_unrefined_flag_covers_the_upper_edge_and_is_read_only(data):
    """Verify unrefined flag covers the upper edge and is read only."""
    rng = np.random.default_rng(4)
    _, fixed, random = simulate(rng, scans=(60,), n_features=2)
    signals = [fixed[0] @ rng.normal(size=(4, 2)) + random[0] @ rng.normal(size=(3, 2))]
    edge = _fit(signals, fixed, random)
    assert edge.at_upper.all() and edge.unrefined.all()
    ordinary = _fit(*data)
    assert ordinary.unrefined.dtype == bool and not ordinary.unrefined.any()
    with pytest.raises(ValueError):
        ordinary.unrefined[0] = True


def test_features_without_a_fit_are_not_unrefined(data):
    """Verify features without a fit are not unrefined."""
    signals, fixed, random = data
    fit = _fit(_with_dead_columns(_columns(signals, [0, 1])), fixed, random)
    assert fit.status[1] == "no_residual_signal"
    assert not fit.unrefined.any()


def test_fit_values_pools_terms_once(data, monkeypatch):
    """Verify fit values pools terms once."""
    import boldtailor._reml_criterion as crit

    calls = 0
    original = crit.pool_terms

    def counted(terms):
        nonlocal calls
        calls += 1
        return original(terms)

    monkeypatch.setattr(crit, "pool_terms", counted)
    monkeypatch.setattr(reml, "pool_terms", counted)
    import boldtailor._reml_search as search

    monkeypatch.setattr(search, "pool_terms", counted)
    _fit(*data)
    assert calls == 1


def test_live_subset_computes_fixed_basis_once_when_mixing_dead_features(
    data, monkeypatch
):
    """Verify live subset computes fixed basis once when mixing dead features."""
    import boldtailor._reml_terms as terms_mod

    calls = 0
    original = terms_mod.conditioned_basis

    def counted(matrix):
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(terms_mod, "conditioned_basis", counted)
    signals, fixed, random = data
    _fit(_with_dead_columns(_columns(signals, [0, 1])), fixed, random)
    assert calls == len(signals)


@pytest.mark.parametrize("rho", [0.0, 0.6])
@pytest.mark.parametrize("signal_level", [0.0, 3.5])
def test_no_residual_signal_has_no_estimated_variance_ratio(rho, signal_level):
    """Verify no residual signal has no estimated variance ratio."""
    fixed = np.ones((30, 1))
    signals = np.full((30, 2), signal_level)
    random = np.random.default_rng(16).normal(size=(30, 3))
    groups = [RemlGroup(signals, fixed)]
    fit = fit_reml(groups, [random], rho=rho)
    selection = select_random_effects_design(groups, [[random], [None]], rho=rho)
    for result in (fit, selection.fit):
        assert result.status == ("no_residual_signal",) * 2
        for field in ("ratio", "noise_variance", "effect_variance", "criterion"):
            assert np.isnan(getattr(result, field)).all()
    assert_array_equal(selection.winner, [-1, -1])
    assert np.isnan(selection.gap).all()
