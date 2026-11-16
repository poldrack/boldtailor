"""Property-based tests of the shared numerical policy in ``boldtailor._numerics``.

Designs are drawn with column units spread over sixty decades and, by kind,
with nearly collinear, exactly redundant or zero columns, because those are
the inputs on which the rank and floor rules have failed before.
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from boldtailor._numerics import (
    PROJECTION_ERROR_FACTOR,
    conditioned_basis,
    orthonormal_basis,
    project_out,
    rank_tolerance,
    residualise,
)

EPS = np.finfo(float).eps
PROPERTY_SETTINGS = settings(derandomize=True, deadline=None, max_examples=60)
KINDS = ("plain", "collinear", "redundant", "zero_column")
DECADES = st.floats(min_value=-30.0, max_value=30.0)
N_FEATURES = 4


@st.composite
def designs(draw, kinds=KINDS):
    """Settings of one design; hypothesis prints these for a failing example."""
    n = draw(st.integers(min_value=12, max_value=60))
    p = draw(st.integers(min_value=1, max_value=8))
    return {
        "seed": draw(st.integers(min_value=0, max_value=2**32 - 1)),
        "n": n,
        "p": p,
        "kind": draw(st.sampled_from(kinds)),
        "decades": tuple(draw(DECADES) for _ in range(p)),
        "gap": 10.0 ** -draw(st.integers(min_value=3, max_value=9)),
    }


def _base_columns(rng, params):
    """Unit-scale columns with the structure the design's kind asks for."""
    base = rng.normal(size=(params["n"], params["p"]))
    if params["kind"] == "zero_column":
        base[:, -1] = 0.0
    if params["p"] > 1 and params["kind"] == "collinear":
        base[:, -1] = base[:, 0] + params["gap"] * rng.normal(size=params["n"])
    if params["p"] > 1 and params["kind"] == "redundant":
        base[:, -1] = 2.0 * base[:, 0]
    return base


def _build(params):
    """The design in its drawn column units, and a generator for signals."""
    rng = np.random.default_rng(params["seed"])
    return _base_columns(rng, params) * 10.0 ** np.array(params["decades"]), rng


def _expected_rank(params):
    lost = params["kind"] == "zero_column"
    lost |= params["kind"] == "redundant" and params["p"] > 1
    return params["p"] - int(lost)


def _projector(basis):
    return basis @ basis.T


def _span_tolerance(n, amplification):
    """A hundred times the rounding of a span whose weakest direction is
    amplified this much: 100 n eps max(s_max / s_i)."""
    return 100 * n * EPS * max(amplification.max(initial=1.0), 1.0)


def _nonzero_unit_columns(design):
    """The design's nonzero columns at unit norm, computed without overflow."""
    peak = np.abs(design).max(axis=0)
    scaled = design[:, peak > 0] / peak[peak > 0]
    return scaled / np.linalg.norm(scaled, axis=0)


# ------------------------------------------------------------- rank_tolerance


@PROPERTY_SETTINGS
@given(
    st.integers(1, 500),
    st.integers(1, 500),
    st.floats(min_value=1e-200, max_value=1e200),
    st.integers(-40, 40),
)
def test_rank_tolerance_scales_with_the_norm_and_ignores_orientation(n, p, norm, k):
    """Verify rank tolerance scales with the norm and ignores orientation."""
    tolerance = rank_tolerance((n, p), norm)
    assert tolerance == rank_tolerance((p, n), norm) > 0
    assert rank_tolerance((n, p), norm * 2.0**k) == tolerance * 2.0**k
    assert rank_tolerance((n + 1, p + 1), norm) > tolerance


# ---------------------------------------------------------- conditioned_basis


@PROPERTY_SETTINGS
@given(designs())
def test_basis_is_orthonormal_with_the_rank_of_the_span(params):
    """Verify basis is orthonormal with the rank of the span."""
    design, _ = _build(params)
    basis, amplification = conditioned_basis(design)
    rank = _expected_rank(params)
    assert basis.shape == (params["n"], rank)
    assert amplification.shape == (rank,)
    assert_allclose(basis.T @ basis, np.eye(rank), atol=100 * params["n"] * EPS)
    assert_array_equal(orthonormal_basis(design), basis)


@PROPERTY_SETTINGS
@given(designs())
def test_amplification_starts_at_one_and_does_not_decrease(params):
    """Verify amplification starts at one and does not decrease."""
    amplification = conditioned_basis(_build(params)[0])[1]
    assume(amplification.size > 0)
    assert amplification[0] == 1.0
    assert np.all(np.diff(amplification) >= 0)
    assert np.all(np.isfinite(amplification))


@PROPERTY_SETTINGS
@given(designs())
def test_basis_contains_every_column_of_the_design(params):
    """Verify basis contains every column of the design."""
    design, _ = _build(params)
    unit = _nonzero_unit_columns(design)
    residual = residualise(unit, orthonormal_basis(design))
    assert np.all(np.linalg.norm(residual, axis=0) <= 1e-10)


@PROPERTY_SETTINGS
@given(designs(), st.lists(DECADES, min_size=8, max_size=8))
def test_basis_rank_and_amplification_ignore_column_units(params, decades):
    """Verify basis rank and amplification ignore column units."""
    design, _ = _build(params)
    rescaled = design * 10.0 ** np.array(decades[: params["p"]])
    basis, amplification = conditioned_basis(design)
    other, other_amplification = conditioned_basis(rescaled)
    tolerance = _span_tolerance(params["n"], amplification)
    assert other.shape == basis.shape
    assert_allclose(_projector(other), _projector(basis), atol=tolerance)
    assert_allclose(other_amplification, amplification, rtol=tolerance)


@PROPERTY_SETTINGS
@given(designs(), st.randoms(use_true_random=False))
def test_basis_span_and_amplification_ignore_column_order(params, shuffler):
    """Verify basis span and amplification ignore column order."""
    design, _ = _build(params)
    order = list(range(params["p"]))
    shuffler.shuffle(order)
    basis, amplification = conditioned_basis(design)
    other, other_amplification = conditioned_basis(design[:, order])
    tolerance = _span_tolerance(params["n"], amplification)
    assert_allclose(_projector(other), _projector(basis), atol=tolerance)
    assert_allclose(other_amplification, amplification, rtol=tolerance)


@PROPERTY_SETTINGS
@given(designs(), st.integers(0, 8), st.integers(1, 3))
def test_zero_columns_change_nothing(params, position, count):
    """Preserve the projected subspace when zero design columns are appended."""
    design, _ = _build(params)
    at = min(position, params["p"])
    padded = np.insert(design, [at] * count, 0.0, axis=1)
    basis, amplification = conditioned_basis(design)
    other, other_amplification = conditioned_basis(padded)
    assert_array_equal(other, basis)
    assert_array_equal(other_amplification, amplification)


@PROPERTY_SETTINGS
@given(designs(kinds=("plain", "collinear")), DECADES)
def test_a_column_inside_the_span_leaves_the_span_unchanged(params, decade):
    """Verify a column inside the span leaves the span unchanged."""
    design, rng = _build(params)
    unit = _nonzero_unit_columns(design)
    inside = unit @ rng.normal(size=unit.shape[1]) * 10.0**decade
    basis, amplification = conditioned_basis(design)
    other = orthonormal_basis(np.column_stack([design, inside]))
    assert other.shape == basis.shape
    tolerance = _span_tolerance(params["n"], amplification)
    assert_allclose(_projector(other), _projector(basis), atol=tolerance)


@PROPERTY_SETTINGS
@given(st.integers(1, 60), st.integers(0, 5))
def test_a_design_without_nonzero_columns_has_an_empty_basis(n, p):
    """Verify a design without nonzero columns has an empty basis."""
    basis, amplification = conditioned_basis(np.zeros((n, p)))
    assert basis.shape == (n, 0) and amplification.shape == (0,)


# ---------------------------------------------------------------- residualise


def _signals(rng, n, decade=0.0):
    return rng.normal(size=(n, N_FEATURES)) * 10.0**decade


@PROPERTY_SETTINGS
@given(designs(), DECADES)
def test_residuals_are_orthogonal_to_the_basis_and_not_longer(params, decade):
    """Verify residuals are orthogonal to the basis and not longer."""
    design, rng = _build(params)
    basis = orthonormal_basis(design)
    signals = _signals(rng, params["n"], decade)
    residual = residualise(signals, basis)
    norms = np.linalg.norm(signals, axis=0)
    rounding = 100 * params["n"] * EPS * norms
    assert np.all(np.abs(basis.T @ residual) <= rounding)
    assert np.all(np.linalg.norm(residual, axis=0) <= norms + rounding)


@PROPERTY_SETTINGS
@given(designs(), DECADES)
def test_residualising_twice_changes_nothing_beyond_rounding(params, decade):
    """Verify residualising twice changes nothing beyond rounding."""
    design, rng = _build(params)
    basis = orthonormal_basis(design)
    signals = _signals(rng, params["n"], decade)
    once = residualise(signals, basis)
    rounding = 100 * params["n"] * EPS * np.linalg.norm(signals, axis=0)
    assert np.all(np.abs(residualise(once, basis) - once) <= rounding)


@PROPERTY_SETTINGS
@given(designs(), st.floats(-1e3, 1e3), st.floats(-1e3, 1e3))
def test_residualise_is_linear(params, a, b):
    """Preserve linear combinations when residualizing signals against a fixed basis."""
    design, rng = _build(params)
    basis = orthonormal_basis(design)
    first, second = _signals(rng, params["n"]), _signals(rng, params["n"])
    combined = residualise(a * first + b * second, basis)
    separate = a * residualise(first, basis) + b * residualise(second, basis)
    scale = (abs(a) + abs(b) + 1.0) * np.sqrt(params["n"])
    assert_allclose(combined, separate, atol=1000 * params["n"] * EPS * scale)


@PROPERTY_SETTINGS
@given(designs())
def test_residual_of_the_basis_itself_is_rounding(params):
    """Verify residual of the basis itself is rounding."""
    basis = orthonormal_basis(_build(params)[0])
    assert_allclose(residualise(basis, basis), 0.0, atol=100 * params["n"] * EPS)


@PROPERTY_SETTINGS
@given(st.integers(1, 60), st.integers(0, 2**32 - 1))
def test_an_empty_basis_leaves_signals_unchanged(n, seed):
    """Verify an empty basis leaves signals unchanged."""
    signals = _signals(np.random.default_rng(seed), n)
    assert_array_equal(residualise(signals, np.empty((n, 0))), signals)


# ---------------------------------------------------------------- project_out


def _projected(design, signals):
    basis, amplification = conditioned_basis(design)
    return project_out(signals, basis, amplification), basis, amplification


@PROPERTY_SETTINGS
@given(designs(), DECADES)
def test_project_out_agrees_with_residualise_and_orders_its_floors(params, decade):
    """Verify project out agrees with residualise and orders its floors."""
    design, rng = _build(params)
    signals = _signals(rng, params["n"], decade)
    (residual, rounding, uncertainty), basis, _ = _projected(design, signals)
    assert_array_equal(residual, residualise(signals, basis))
    assert rounding.shape == uncertainty.shape == (N_FEATURES,)
    assert np.all(rounding > 0)
    assert np.all(uncertainty >= rounding)


@PROPERTY_SETTINGS
@given(designs(), DECADES)
def test_rounding_floor_is_a_fixed_fraction_of_the_signal_energy(params, decade):
    """Verify rounding floor is a fixed fraction of the signal energy."""
    design, rng = _build(params)
    signals = _signals(rng, params["n"], decade)
    (_, rounding, _), _, _ = _projected(design, signals)
    relative = PROJECTION_ERROR_FACTOR * params["n"] * EPS
    assert_allclose(rounding, relative**2 * np.sum(signals**2, axis=0), rtol=1e-12)


@PROPERTY_SETTINGS
@given(designs(), st.integers(-60, 60))
def test_project_out_is_equivariant_to_signal_scale(params, k):
    """Verify project out is equivariant to signal scale."""
    design, rng = _build(params)
    signals = _signals(rng, params["n"])
    (residual, rounding, uncertainty), _, _ = _projected(design, signals)
    (scaled, scaled_rounding, scaled_uncertainty), _, _ = _projected(
        design, signals * 2.0**k
    )
    assert_array_equal(scaled, residual * 2.0**k)
    assert_allclose(scaled_rounding, rounding * 4.0**k, rtol=1e-12)
    assert_allclose(scaled_uncertainty, uncertainty * 4.0**k, rtol=1e-12)


@PROPERTY_SETTINGS
@given(designs())
def test_features_are_projected_independently(params):
    """Match joint feature projection to separate projections of each feature."""
    design, rng = _build(params)
    signals = _signals(rng, params["n"])
    (residual, rounding, uncertainty), _, _ = _projected(design, signals)
    (one, one_rounding, one_uncertainty), _, _ = _projected(design, signals[:, [2]])
    scale = 100 * params["n"] * EPS * np.linalg.norm(signals[:, 2])
    assert_allclose(one[:, 0], residual[:, 2], atol=scale)
    assert_allclose(one_rounding, rounding[[2]], rtol=1e-12)
    assert_allclose(one_uncertainty, uncertainty[[2]], rtol=1e-9)


@PROPERTY_SETTINGS
@given(designs(), st.floats(min_value=1.0, max_value=1e6))
def test_uncertainty_floor_grows_with_amplification(params, factor):
    """Verify uncertainty floor grows with amplification."""
    design, rng = _build(params)
    signals = _signals(rng, params["n"])
    basis, amplification = conditioned_basis(design)
    _, rounding, uncertainty = project_out(signals, basis, amplification)
    _, same_rounding, larger = project_out(signals, basis, amplification * factor)
    assert_array_equal(same_rounding, rounding)
    assert np.all(larger >= uncertainty)


@PROPERTY_SETTINGS
@given(designs())
def test_floors_coincide_without_amplification(params):
    """Match rounding and uncertainty floors when the basis has no conditioning
    amplification.
    """
    design, rng = _build(params)
    basis = orthonormal_basis(design)
    signals = _signals(rng, params["n"])
    _, rounding, uncertainty = project_out(signals, basis, np.ones(basis.shape[1]))
    assert_allclose(uncertainty, rounding, rtol=1e-12)


@PROPERTY_SETTINGS
@given(designs(), st.lists(DECADES, min_size=8, max_size=8))
def test_projection_and_floors_ignore_column_units(params, decades):
    """Follow-up review F1 as a property: units alone change nothing."""
    design, rng = _build(params)
    signals = _signals(rng, params["n"])
    rescaled = design * 10.0 ** np.array(decades[: params["p"]])
    (residual, rounding, uncertainty), _, amplification = _projected(design, signals)
    (other, other_rounding, other_uncertainty), _, _ = _projected(rescaled, signals)
    tolerance = _span_tolerance(params["n"], amplification)
    assert_allclose(other, residual, atol=tolerance * np.sqrt(params["n"]) * 10)
    assert_array_equal(other_rounding, rounding)
    assert_allclose(other_uncertainty, uncertainty, rtol=max(tolerance, 1e-9))


# ------------------------------------------- floors against span and residual


def _span_signals(rng, design, heavy):
    """Signals in the span of the design, loading its columns unevenly."""
    unit = _nonzero_unit_columns(design)
    amplitudes = 10.0 ** rng.uniform(-heavy, heavy, size=(unit.shape[1], 1))
    return unit @ (rng.normal(size=(unit.shape[1], N_FEATURES)) * amplitudes)


@PROPERTY_SETTINGS
@given(designs(), st.floats(min_value=0.0, max_value=6.0))
def test_span_signals_stay_below_the_uncertainty_floor(params, heavy):
    """Verify span signals stay below the uncertainty floor."""
    design, rng = _build(params)
    assume(_expected_rank(params) > 0)
    signals = _span_signals(rng, design, heavy)
    (residual, _, uncertainty), _, _ = _projected(design, signals)
    assert np.all(np.sum(residual**2, axis=0) < uncertainty)


@PROPERTY_SETTINGS
@given(designs(kinds=("plain", "redundant", "zero_column")), st.floats(0.0, 6.0))
def test_span_signals_of_well_conditioned_designs_are_below_rounding(params, heavy):
    """With unit columns that are not nearly collinear, whatever their units,
    the plain rounding floor is enough: such signals have no resolved residual energy.
    """
    design, rng = _build(params)
    signals = _span_signals(rng, design, heavy)
    (residual, rounding, _), _, amplification = _projected(design, signals)
    assume(amplification.size > 0 and amplification.max() <= 4.0)
    assert np.all(np.sum(residual**2, axis=0) < rounding)


def _unit_residuals(rng, basis):
    """Signals orthogonal to the basis, each of unit norm."""
    residual = residualise(rng.normal(size=(len(basis), N_FEATURES)), basis)
    return residual / np.linalg.norm(residual, axis=0)


@PROPERTY_SETTINGS
@given(designs(), st.floats(min_value=0.0, max_value=6.0))
def test_unit_residual_survives_a_span_shift_in_any_column_units(params, decade):
    """A unit residual beside a span component up to 1e6 times larger stays
    resolved and keeps its energy, as long as the columns are not so nearly
    collinear that the span itself is uncertain at that level."""
    design, rng = _build(params)
    basis, amplification = conditioned_basis(design)
    assume(params["n"] - basis.shape[1] >= 3)
    assume(amplification.max(initial=1.0) <= 1e3)
    shift = 10.0**decade * _span_signals(rng, design, 0.0)
    signals = _unit_residuals(rng, basis) + shift
    residual, _, uncertainty = project_out(signals, basis, amplification)
    energy = np.sum(residual**2, axis=0)
    tolerance = _span_tolerance(params["n"], amplification) * 10.0**decade * 10
    assert np.all(energy > uncertainty)
    assert_allclose(energy, 1.0, atol=tolerance + 1e-12)


@PROPERTY_SETTINGS
@given(designs(), DECADES)
def test_a_basis_vector_raises_the_floor_by_its_own_amplification(params, decade):
    """A signal along one basis vector has uncertainty floor amplification**2
    times its rounding floor: the weight is applied direction by direction."""
    design, _ = _build(params)
    basis, amplification = conditioned_basis(design)
    assume(basis.shape[1] > 0)
    _, rounding, uncertainty = project_out(basis * 10.0**decade, basis, amplification)
    assert_allclose(uncertainty, amplification**2 * rounding, rtol=1e-6)


@PROPERTY_SETTINGS
@given(designs(kinds=("collinear",)), st.floats(min_value=-3.0, max_value=3.0))
def test_span_signals_along_a_weak_direction_stay_below_uncertainty(params, decade):
    """Loading the direction that nearly collinear columns barely determine
    is the worst case for the computed span; the amplified floor covers it."""
    assume(params["p"] > 1)
    design, rng = _build(params)
    unit = _nonzero_unit_columns(design)
    weak = (unit[:, [-1]] - unit[:, [0]]) / params["gap"]
    loading = 10.0**decade * rng.normal(size=(1, N_FEATURES))
    signals = weak @ loading + _span_signals(rng, design, 0.0)
    (residual, _, uncertainty), _, _ = _projected(design, signals)
    assert np.all(np.sum(residual**2, axis=0) < uncertainty)


@PROPERTY_SETTINGS
@given(designs(kinds=("plain", "redundant", "zero_column")))
def test_residual_eleven_decades_below_the_span_component_is_resolved(params):
    """The resolvable range, stated outright: for columns that are not nearly
    collinear, a residual 1e-11 times the span component is still resolved."""
    design, rng = _build(params)
    basis, amplification = conditioned_basis(design)
    assume(params["n"] - basis.shape[1] >= 3)
    assume(amplification.max(initial=1.0) <= 4.0)
    signals = _unit_residuals(rng, basis) + 1e11 * _span_signals(rng, design, 0.0)
    residual, _, uncertainty = project_out(signals, basis, amplification)
    assert np.all(np.sum(residual**2, axis=0) > uncertainty)
