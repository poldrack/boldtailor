import numpy as np
import pytest
from numpy.testing import assert_allclose

from boldtailor._numerics import (
    PROJECTION_ERROR_FACTOR,
    conditioned_basis,
    orthonormal_basis,
    project_out,
    rank_tolerance,
    residualise,
)

EPS = np.finfo(float).eps


def _design(rng, n, p, condition=1.0):
    design = rng.normal(size=(n, p)) * np.logspace(0, np.log10(condition), p)
    design[:, 0] = 1.0
    return design


def test_basis_is_orthonormal_and_spans_the_design():
    """Verify basis is orthonormal and spans the design."""
    design = _design(np.random.default_rng(0), 40, 5)
    basis = orthonormal_basis(design)
    assert_allclose(basis.T @ basis, np.eye(5), atol=1e-13)
    assert_allclose(basis @ (basis.T @ design), design, atol=1e-12)


def test_basis_drops_redundant_columns():
    """Return an orthonormal basis with duplicate or dependent design columns removed."""
    design = _design(np.random.default_rng(1), 30, 3)
    redundant = np.column_stack([design, design[:, 1] - 2 * design[:, 2]])
    assert orthonormal_basis(redundant).shape == (30, 3)


def test_basis_of_no_columns_is_empty():
    """Verify basis of no columns is empty."""
    assert orthonormal_basis(np.empty((7, 0))).shape == (7, 0)


def test_basis_accepts_array_likes():
    """Accept list-like design inputs when constructing an orthonormal basis."""
    assert orthonormal_basis([[1, 0], [1, 1], [1, 2]]).shape == (3, 2)


def test_basis_is_row_major_like_its_owned_copies():
    """A stored copy of the basis must reach BLAS in the same memory layout.

    Projections keep a C-ordered copy; a column-major original makes the same
    product round differently on some BLAS builds, at the last bit.
    """
    basis, _ = conditioned_basis(_design(np.random.default_rng(2), 30, 4))
    assert basis.flags.c_contiguous
    assert basis.flags.owndata


def test_rank_tolerance_scales_with_size_and_norm():
    """Verify rank tolerance scales with size and norm."""
    assert rank_tolerance((50, 4), 3.0) == pytest.approx(50 * EPS * 3.0)


def test_residuals_are_orthogonal_to_the_basis():
    """Verify residuals are orthogonal to the basis."""
    rng = np.random.default_rng(2)
    basis = orthonormal_basis(_design(rng, 40, 4))
    residual = residualise(rng.normal(size=(40, 3)), basis)
    assert_allclose(basis.T @ residual, 0.0, atol=1e-13)


def _unit_columns(design):
    return design / np.linalg.norm(design, axis=0)


def test_amplification_is_the_inverse_relative_singular_value_of_unit_columns():
    """Verify amplification is the inverse relative singular value of unit columns."""
    design = _design(np.random.default_rng(4), 30, 3, condition=1e4)
    basis, amplification = conditioned_basis(design)
    singular = np.linalg.svd(_unit_columns(design), compute_uv=False)
    assert_allclose(basis, orthonormal_basis(design), atol=0.0)
    assert_allclose(amplification, singular[0] / singular, rtol=1e-12)


@pytest.mark.parametrize("scales", [(1e-20, 1.0, 1e15), (1e-200, 1e100, 1.0)])
def test_basis_and_amplification_do_not_depend_on_column_units(scales):
    """Verify basis and amplification do not depend on column units."""
    design = _design(np.random.default_rng(6), 30, 3)
    basis, amplification = conditioned_basis(design)
    rescaled, rescaled_amplification = conditioned_basis(design * np.array(scales))
    assert_allclose(rescaled @ rescaled.T, basis @ basis.T, atol=1e-13)
    assert_allclose(rescaled_amplification, amplification, rtol=1e-10)


def test_only_exactly_zero_columns_are_dropped_for_their_norm():
    """Verify only exactly zero columns are dropped for their norm."""
    design = _design(np.random.default_rng(7), 30, 3)
    tiny = np.column_stack([design, 1e-250 * design[:, 1] ** 2, np.zeros(30)])
    assert orthonormal_basis(tiny).shape == (30, 4)
    assert orthonormal_basis(np.zeros((5, 2))).shape == (5, 0)


def test_orthogonal_columns_in_unequal_units_have_no_amplification():
    """Verify orthogonal columns in unequal units have no amplification."""
    design = np.eye(30)[:, :2] * np.array([1e-12, 1.0])
    assert_allclose(conditioned_basis(design)[1], 1.0, rtol=1e-12)


def test_project_out_returns_the_residual():
    """Verify project out returns the residual."""
    rng = np.random.default_rng(5)
    basis, amplification = conditioned_basis(_design(rng, 40, 4))
    signals = rng.normal(size=(40, 3))
    residual = project_out(signals, basis, amplification)[0]
    assert_allclose(residual, residualise(signals, basis), atol=0.0)


def test_floors_are_squared_norm_thresholds():
    """The rounding floor counts ||y||^2; the uncertainty floor counts energy
    along a basis vector amplification**2 times and the rest once."""
    basis = np.eye(25)[:, :2]
    amplification = np.array([1.0, 1e3])
    signals = np.zeros((25, 3))
    signals[0, 0], signals[1, 1], signals[7, 2] = 3.0, 3.0, 3.0
    _, rounding, uncertainty = project_out(signals, basis, amplification)
    unit = (PROJECTION_ERROR_FACTOR * 25 * EPS * 3.0) ** 2
    assert_allclose(rounding, unit * np.ones(3), rtol=1e-12)
    assert_allclose(uncertainty, unit * np.array([1.0, 1e6, 1.0]), rtol=1e-12)


def _span_signals(rng, design, weak_last=False):
    """Signals exactly in the design's span, optionally loading its last column."""
    p = design.shape[1]
    amplitudes = 10.0 ** rng.uniform(-3, 6, size=(p, 1))
    if weak_last:
        amplitudes = np.logspace(-12, 0, p)[:, None]
    return design @ (rng.normal(size=(p, 40)) * amplitudes)


def _energies(signals, design):
    basis, amplification = conditioned_basis(design)
    assert basis.shape[1] == design.shape[1]
    residual, rounding, uncertainty = project_out(signals, basis, amplification)
    return np.sum(residual**2, axis=0), rounding, uncertainty


@pytest.mark.parametrize("weak_last", [False, True])
@pytest.mark.parametrize("condition", [1.0, 1e6, 1e12])
@pytest.mark.parametrize("n,p", [(10, 1), (10, 9), (30, 29), (100, 9), (1000, 400)])
def test_rounding_floor_covers_span_signals_of_unit_scaled_designs(
    n, p, condition, weak_last
):
    """Calibration of the factor: when ill-conditioning comes only from column
    units, the residue of exact-span signals stays a factor four in norm below
    the plain rounding floor, whichever columns carry the signal."""
    rng = np.random.default_rng(n + p)
    for _ in range(5):
        design = rng.normal(size=(n, p)) * np.logspace(0, np.log10(condition), p)
        energy, rounding, _ = _energies(_span_signals(rng, design, weak_last), design)
        assert np.all(16 * energy < rounding)


def _collinear_problem(rng, n, p, gap):
    """A design whose last column nearly repeats its first, and span signals
    dominated by the difference of the two."""
    design = rng.normal(size=(n, p))
    design[:, -1] = design[:, 0] + gap * rng.normal(size=n)
    weak = design[:, [-1]] - design[:, [0]]
    signals = weak @ rng.normal(size=(1, 40)) / gap + design @ rng.normal(size=(p, 40))
    return signals, design


@pytest.mark.parametrize("gap", [1e-3, 1e-6, 1e-9])
@pytest.mark.parametrize("n,p", [(10, 9), (30, 5), (300, 30)])
def test_uncertainty_floor_covers_span_signals_of_collinear_designs(n, p, gap):
    """Verify uncertainty floor covers span signals of collinear designs."""
    rng = np.random.default_rng(n * p)
    for _ in range(5):
        energy, _, uncertainty = _energies(*_collinear_problem(rng, n, p, gap))
        assert np.all(16 * energy < uncertainty)


def test_rounding_floor_alone_would_miss_collinear_residue():
    """Why the uncertainty floor exists: near-collinear columns leave residue
    far above the plain rounding floor."""
    signals, design = _collinear_problem(np.random.default_rng(90), 10, 9, 1e-9)
    energy, rounding, uncertainty = _energies(signals, design)
    assert np.any(energy > 100 * rounding)
    assert np.all(16 * energy < uncertainty)


def _drift_design(n=300):
    """Intercept, unnormalised cubic drift and small-unit motion columns."""
    t = np.arange(n, dtype=float)
    motion = 0.05 * np.random.default_rng(0).normal(size=(n, 6))
    return np.column_stack([np.ones(n), t, t**2, t**3, motion])


def test_unit_noise_on_a_drift_design_is_far_above_both_floors():
    """Verify unit noise on a drift design is far above both floors."""
    design = _drift_design()
    rng = np.random.default_rng(1)
    scale = np.array([1e4, 1.0, 1e-2, 1e-4] + [5.0] * 6)[:, None]
    signals = rng.normal(size=(300, 5)) + design @ (rng.normal(size=(10, 5)) * scale)
    energy, rounding, uncertainty = _energies(signals, design)
    assert np.all(energy > 1e12 * uncertainty)
    assert np.all(uncertainty < 1e6 * rounding)


@pytest.mark.parametrize("offset", [1e4, 1e8, 1e10])
def test_small_residual_on_a_large_nuisance_is_above_the_floors(offset):
    """Verify small residual on a large nuisance is above the floors."""
    rng = np.random.default_rng(3)
    design = _design(rng, 30, 2)
    signals = rng.normal(size=(30, 4)) + offset
    energy, _, uncertainty = _energies(signals, design)
    assert np.all(energy > uncertainty)


def test_exactly_removable_shift_in_unequal_units_is_above_the_floors():
    """Follow-up review F1: column units alone must not raise the floor."""
    eye = np.eye(30)
    design = eye[:, :2] * np.array([1e-12, 1.0])
    signals = (2 * eye[:, 2] + eye[:, 3] + 100 * eye[:, 0])[:, None]
    energy, _, uncertainty = _energies(signals, design)
    assert_allclose(energy, 5.0, rtol=1e-12)
    assert np.all(uncertainty < 1e-20)


def test_project_out_does_not_overflow_uncertainty_floor_for_large_signals():
    """Large signals with finite energy and high amplification must not overflow."""
    basis = np.eye(25)[:, :2]
    amplification = np.array([1.0, 1e6])
    signals = np.zeros((25, 2))
    signals[1, 0] = 1e150
    signals[7, 1] = 1e150
    _, rounding, uncertainty = project_out(signals, basis, amplification)
    assert np.all(np.isfinite(rounding))
    assert np.all(np.isfinite(uncertainty))
