"""Independent regressions from the scientific re-review."""

import numpy as np
import pytest

from boldtailor._reml_terms import project_fixed, reml_terms
from boldtailor.ar1 import estimate_rho, prepare_reference
from boldtailor.reml import RemlGroup, fit_reml


@pytest.fixture
def saturated_diagonal():
    singular = np.logspace(-4, 4, 10)
    return singular[:, None], np.empty((10, 0)), np.diag(singular)


def _diagonal_criterion(signal, random, ratio):
    """Independent exact-diagonal covariance calculation, without subtraction."""
    diagonal = 1.0 + ratio * np.diag(random) ** 2
    quadratic = np.sum(signal[:, 0] ** 2 / diagonal)
    return len(signal) * np.log(quadratic / len(signal)) + np.log(diagonal).sum()


def test_full_restricted_random_rank_has_no_orthogonal_remainder(saturated_diagonal):
    """Verify full restricted random rank has no orthogonal remainder."""
    signal, fixed, random = saturated_diagonal
    terms = reml_terms(project_fixed(signal, fixed), random)
    assert len(terms.singular2) == terms.dof
    np.testing.assert_array_equal(terms.residual_ss, 0.0)


def test_fit_matches_its_covariance_and_a_better_feasible_point(saturated_diagonal):
    """Verify fit matches its covariance and a better feasible point."""
    signal, fixed, random = saturated_diagonal
    fit = fit_reml([RemlGroup(signal, fixed)], [random])
    direct = _diagonal_criterion(signal, random, fit.ratio[0])
    np.testing.assert_allclose(fit.criterion[0], direct, rtol=0, atol=1e-8)
    alternative = _diagonal_criterion(signal, random, 1e16)
    assert fit.criterion[0] <= alternative + 1e-8


def test_iid_row_permutation_preserves_profile_likelihood(saturated_diagonal):
    """Verify independent-noise row permutation preserves profile likelihood."""
    signal, fixed, random = saturated_diagonal
    original = fit_reml([RemlGroup(signal, fixed)], [random])
    reversed_fit = fit_reml([RemlGroup(signal[::-1], fixed)], [random[::-1]])
    np.testing.assert_allclose(
        original.criterion, reversed_fit.criterion, rtol=0, atol=1e-8
    )


def test_small_positive_orthogonal_noise_is_preserved(saturated_diagonal):
    """Verify small positive orthogonal noise is preserved."""
    signal, _, random = saturated_diagonal
    signal = np.vstack([signal, [[1e-5]]])
    random = np.vstack([random, np.zeros((1, 10))])
    fixed = np.empty((11, 0))
    fit = fit_reml([RemlGroup(signal, fixed)], [random])
    ratio = 1e10
    diagonal = 1 + ratio * np.sum(random**2, axis=1)
    quadratic = np.sum(signal[:, 0] ** 2 / diagonal)
    feasible = 11 * np.log(quadratic / 11) + np.log(diagonal).sum()
    assert fit.criterion[0] <= feasible + 1e-8


def test_flat_ar1_moment_curve_does_not_produce_a_rho_estimate():
    """Verify flat AR(1) moment curve does not produce a rho estimate."""
    eye = np.eye(10)
    reference = prepare_reference(eye[:, 1:])
    np.testing.assert_array_equal(reference.expected, 0.0)
    estimate = estimate_rho([eye[:, [0]]], [reference])
    np.testing.assert_array_equal(estimate.raw, 0.0)
    assert np.isnan(estimate.values).all()
    assert estimate.pooled == 0.0
