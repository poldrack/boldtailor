"""Property-based invariances and equivariances of the public REML API."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from boldtailor.reml import RemlGroup, fit_reml, select_random_effects_design
from tests.reml_oracles import simulate

RTOL = 1e-6
PROPERTY_SETTINGS = settings(derandomize=True, deadline=None, max_examples=50)
SCALES = st.floats(min_value=-6.0, max_value=6.0).map(lambda e: 10.0**e)


@st.composite
def parameters(draw):
    """Simulation settings; hypothesis prints these for a failing example."""
    n_groups = draw(st.integers(min_value=1, max_value=4))
    return {
        "seed": draw(st.integers(min_value=0, max_value=2**32 - 1)),
        "scans": tuple(draw(st.integers(12, 40)) for _ in range(n_groups)),
        "n_fixed": draw(st.integers(1, 4)),
        "n_random": draw(st.integers(1, 4)),
        "ratio": draw(st.sampled_from([0.0, 0.1, 1.0, 10.0])),
        "n_features": draw(st.integers(1, 3)),
    }


def _simulated(params):
    """Simulated (signals, fixed, random) plus a generator for transformations."""
    settings_ = {k: v for k, v in params.items() if k != "seed"}
    rng = np.random.default_rng(params["seed"])
    return (*simulate(rng, **settings_), rng)


def _groups(signals, fixed):
    return [RemlGroup(y, c) for y, c in zip(signals, fixed, strict=True)]


def _fit(signals, fixed, random, **kwargs):
    return fit_reml(_groups(signals, fixed), random, **kwargs)


def _orthogonal(rng, size):
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    return q * np.sign(np.diag(r))


def _well_conditioned(rng, size):
    scales = rng.uniform(0.5, 2.0, size=size)
    return _orthogonal(rng, size) @ np.diag(scales) @ _orthogonal(rng, size)


def _assert_same_fit(actual, expected):
    assert actual.status == expected.status
    assert_array_equal(actual.at_upper, expected.at_upper)
    assert_array_equal(actual.unrefined, expected.unrefined)
    assert_allclose(actual.ratio, expected.ratio, rtol=RTOL)
    assert_allclose(actual.noise_variance, expected.noise_variance, rtol=RTOL)
    assert_allclose(actual.effect_variance, expected.effect_variance, rtol=RTOL)
    assert_allclose(actual.criterion, expected.criterion, rtol=0, atol=1e-6)


def _permuted_fit(fit, order):
    return {
        "status": tuple(fit.status[i] for i in order),
        "at_upper": fit.at_upper[order],
        "ratio": fit.ratio[order],
        "noise_variance": fit.noise_variance[order],
        "effect_variance": fit.effect_variance[order],
        "criterion": fit.criterion[order],
    }


@PROPERTY_SETTINGS
@given(parameters())
def test_fixed_reparameterisation_is_invariant(params):
    """Verify fixed reparameterisation is invariant."""
    signals, fixed, random, rng = _simulated(params)
    moved = [c @ _well_conditioned(rng, c.shape[1]) for c in fixed]
    _assert_same_fit(_fit(signals, moved, random), _fit(signals, fixed, random))


@PROPERTY_SETTINGS
@given(parameters())
def test_random_rotation_is_invariant(params):
    """Verify random rotation is invariant."""
    signals, fixed, random, rng = _simulated(params)
    rotated = [z @ _orthogonal(rng, z.shape[1]) for z in random]
    _assert_same_fit(_fit(signals, fixed, rotated), _fit(signals, fixed, random))


@PROPERTY_SETTINGS
@given(parameters(), st.randoms(use_true_random=False))
def test_group_permutation_is_invariant(params, shuffler):
    """Verify group permutation is invariant."""
    signals, fixed, random, _ = _simulated(params)
    order = list(range(len(signals)))
    shuffler.shuffle(order)
    permuted = [[part[i] for i in order] for part in (signals, fixed, random)]
    _assert_same_fit(_fit(*permuted), _fit(signals, fixed, random))


@PROPERTY_SETTINGS
@given(parameters(), st.randoms(use_true_random=False))
def test_feature_permutation_is_equivariant(params, shuffler):
    """Verify feature permutation is equivariant."""
    signals, fixed, random, _ = _simulated(params)
    order = list(range(signals[0].shape[1]))
    shuffler.shuffle(order)
    actual = _fit([y[:, order] for y in signals], fixed, random)
    expected = _permuted_fit(_fit(signals, fixed, random), order)
    assert actual.status == expected["status"]
    assert_array_equal(actual.at_upper, expected["at_upper"])
    for name in ("ratio", "noise_variance", "effect_variance"):
        assert_allclose(getattr(actual, name), expected[name], rtol=RTOL)
    assert_allclose(actual.criterion, expected["criterion"], rtol=0, atol=1e-6)


@PROPERTY_SETTINGS
@given(parameters(), SCALES)
def test_scaling_signals_is_equivariant(params, k):
    """Verify scaling signals is equivariant."""
    signals, fixed, random, _ = _simulated(params)
    scaled = _fit([k * y for y in signals], fixed, random)
    base = _fit(signals, fixed, random)
    assert scaled.status == base.status
    assert_array_equal(scaled.at_upper, base.at_upper)
    assert_allclose(scaled.ratio, base.ratio, rtol=RTOL)
    assert_allclose(scaled.noise_variance, k**2 * base.noise_variance, rtol=RTOL)
    assert_allclose(scaled.effect_variance, k**2 * base.effect_variance, rtol=RTOL)


@PROPERTY_SETTINGS
@given(parameters(), SCALES)
def test_scaling_random_design_is_equivariant(params, k):
    """Verify scaling random design is equivariant."""
    signals, fixed, random, _ = _simulated(params)
    scaled = _fit(signals, fixed, [k * z for z in random])
    base = _fit(signals, fixed, random)
    assert scaled.status == base.status
    assert_array_equal(scaled.at_upper, base.at_upper)
    assert_allclose(scaled.ratio, base.ratio / k**2, rtol=RTOL)
    assert_allclose(scaled.effect_variance, base.effect_variance / k**2, rtol=RTOL)
    assert_allclose(scaled.noise_variance, base.noise_variance, rtol=RTOL)
    assert_allclose(scaled.criterion, base.criterion, rtol=0, atol=1e-6)


@PROPERTY_SETTINGS
@given(parameters())
def test_adding_fixed_span_to_signals_changes_nothing(params):
    """Verify adding fixed span to signals changes nothing."""
    signals, fixed, random, rng = _simulated(params)
    shifted = [
        y + c @ rng.normal(scale=10.0, size=(c.shape[1], y.shape[1]))
        for y, c in zip(signals, fixed, strict=True)
    ]
    _assert_same_fit(_fit(shifted, fixed, random), _fit(signals, fixed, random))


@PROPERTY_SETTINGS
@given(parameters())
def test_appending_random_column_inside_fixed_span_changes_nothing(params):
    """Verify appending random column inside fixed span changes nothing."""
    signals, fixed, random, rng = _simulated(params)
    widened = [
        np.column_stack([z, c @ rng.normal(size=c.shape[1])])
        for z, c in zip(random, fixed, strict=True)
    ]
    _assert_same_fit(_fit(signals, fixed, widened), _fit(signals, fixed, random))


def _assert_identical_fit(actual, expected):
    assert actual.status == expected.status
    assert actual.restricted_dof == expected.restricted_dof
    for name in ("ratio", "noise_variance", "effect_variance", "criterion"):
        assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert_array_equal(actual.at_upper, expected.at_upper)
    assert_array_equal(actual.rho, expected.rho)


@PROPERTY_SETTINGS
@given(parameters())
def test_zero_rho_matches_default(params):
    """Verify zero rho matches default."""
    signals, fixed, random, _ = _simulated(params)
    default = _fit(signals, fixed, random)
    n_features = signals[0].shape[1]
    _assert_identical_fit(_fit(signals, fixed, random, rho=0.0), default)
    _assert_identical_fit(
        _fit(signals, fixed, random, rho=np.zeros(n_features)), default
    )


def _candidates(random, rng):
    return [
        list(random),
        [z[:, :1] for z in random],
        [rng.normal(size=z.shape) for z in random],
        [None] * len(random),
    ]


@PROPERTY_SETTINGS
@given(parameters(), st.permutations(range(4)))
def test_selection_is_invariant_to_candidate_order(params, order):
    """Verify selection is invariant to candidate order."""
    signals, fixed, random, rng = _simulated(params)
    groups = _groups(signals, fixed)
    candidates = _candidates(random, rng)
    base = select_random_effects_design(groups, candidates)
    permuted = select_random_effects_design(groups, [candidates[i] for i in order])
    mapped = np.where(permuted.winner >= 0, np.asarray(order)[permuted.winner], -1)
    decisive = base.gap > 1e-8
    assert_array_equal(mapped[decisive], base.winner[decisive])
    assert_array_equal(permuted.winner == -1, base.winner == -1)
    assert_array_equal(permuted.any_at_upper, base.any_at_upper)
    assert_array_equal(permuted.any_unrefined, base.any_unrefined)
