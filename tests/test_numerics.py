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
    design = _design(np.random.default_rng(0), 40, 5)
    basis = orthonormal_basis(design)
    assert_allclose(basis.T @ basis, np.eye(5), atol=1e-13)
    assert_allclose(basis @ (basis.T @ design), design, atol=1e-12)


def test_basis_drops_redundant_columns():
    design = _design(np.random.default_rng(1), 30, 3)
    redundant = np.column_stack([design, design[:, 1] - 2 * design[:, 2]])
    assert orthonormal_basis(redundant).shape == (30, 3)


def test_basis_of_no_columns_is_empty():
    assert orthonormal_basis(np.empty((7, 0))).shape == (7, 0)


def test_basis_accepts_array_likes():
    assert orthonormal_basis([[1, 0], [1, 1], [1, 2]]).shape == (3, 2)


def test_rank_tolerance_scales_with_size_and_norm():
    assert rank_tolerance((50, 4), 3.0) == pytest.approx(50 * EPS * 3.0)


def test_residuals_are_orthogonal_to_the_basis():
    rng = np.random.default_rng(2)
    basis = orthonormal_basis(_design(rng, 40, 4))
    residual = residualise(rng.normal(size=(40, 3)), basis)
    assert_allclose(basis.T @ residual, 0.0, atol=1e-13)


def test_amplification_is_the_inverse_relative_singular_value():
    design = _design(np.random.default_rng(4), 30, 3, condition=1e4)
    basis, amplification = conditioned_basis(design)
    singular = np.linalg.svd(design, compute_uv=False)
    assert_allclose(basis, orthonormal_basis(design), atol=0.0)
    assert_allclose(amplification, singular[0] / singular, rtol=1e-12)


def test_project_out_returns_the_residual():
    rng = np.random.default_rng(5)
    basis, amplification = conditioned_basis(_design(rng, 40, 4))
    signals = rng.normal(size=(40, 3))
    residual, _ = project_out(signals, basis, amplification)
    assert_allclose(residual, residualise(signals, basis), atol=0.0)


def test_floor_is_a_squared_norm_threshold_weighted_by_conditioning():
    """Energy along a basis vector counts amplification**2 times; the rest once."""
    basis = np.eye(25)[:, :2]
    amplification = np.array([1.0, 1e3])
    signals = np.zeros((25, 3))
    signals[0, 0], signals[1, 1], signals[7, 2] = 3.0, 3.0, 3.0
    _, floor = project_out(signals, basis, amplification)
    unit = (PROJECTION_ERROR_FACTOR * 25 * EPS * 3.0) ** 2
    assert_allclose(floor, unit * np.array([1.0, 1e6, 1.0]), rtol=1e-12)


def _residue(rng, n, p, condition):
    """Residual energy and floor of signals lying exactly in the design's span."""
    design = _design(rng, n, p, condition)
    amplitudes = 10.0 ** rng.uniform(-3, 6, size=(p, 1))
    signals = design @ (rng.normal(size=(p, 40)) * amplitudes)
    basis, amplification = conditioned_basis(design)
    assert basis.shape[1] == p
    residual, floor = project_out(signals, basis, amplification)
    return np.sum(residual**2, axis=0), floor


@pytest.mark.parametrize("condition", [1.0, 1e6, 1e9])
@pytest.mark.parametrize("n,p", [(10, 1), (10, 9), (30, 29), (100, 9), (1000, 400)])
def test_floor_exceeds_projection_residue_with_a_fourfold_margin(n, p, condition):
    """Calibration of the factor: the residue of exact-span signals stays a
    factor four in norm below the floor across sizes, ranks, scales and
    conditioning, including nearly saturated ill-conditioned designs."""
    rng = np.random.default_rng(n + p)
    for _ in range(5):
        energy, floor = _residue(rng, n, p, condition)
        assert np.all(16 * energy < floor)


def test_floor_without_conditioning_would_miss_ill_conditioned_residue():
    """Why the floor is weighted: against ||y|| alone the residue of an
    ill-conditioned saturated design is far above any modest multiple."""
    rng = np.random.default_rng(19)
    design = _design(rng, 10, 9, 1e9)
    signals = design @ (rng.normal(size=(9, 40)) * 10.0 ** rng.uniform(-3, 6, (9, 1)))
    basis, amplification = conditioned_basis(design)
    residual, floor = project_out(signals, basis, amplification)
    plain = (PROJECTION_ERROR_FACTOR * 10 * EPS) ** 2 * np.sum(signals**2, axis=0)
    assert np.any(np.sum(residual**2, axis=0) > plain)
    assert np.all(np.sum(residual**2, axis=0) < floor)


@pytest.mark.parametrize("offset", [1e4, 1e8, 1e10])
def test_small_residual_on_a_large_nuisance_is_above_the_floor(offset):
    rng = np.random.default_rng(3)
    basis, amplification = conditioned_basis(_design(rng, 30, 2))
    signals = rng.normal(size=(30, 4)) + offset
    residual, floor = project_out(signals, basis, amplification)
    assert np.all(np.sum(residual**2, axis=0) > floor)
