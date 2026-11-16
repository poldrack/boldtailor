"""Validated and owned REML HRF settings."""

import importlib

import numpy as np
import pytest

from boldtailor.hrf_library import HrfLibrary


@pytest.fixture
def options_type():
    """Resolve the public settings class at the behavior boundary."""
    return importlib.import_module("boldtailor.hrf_options").RemlHrfOptions


def test_options_own_rho_and_subset_features(options_type):
    """Changing a caller's rho cannot change selection or reordered blocks."""
    rho = np.array([0.1, 0.3, -0.2])
    options = options_type(rho=rho)
    rho[:] = 0.9
    np.testing.assert_array_equal(options.rho, [0.1, 0.3, -0.2])
    assert not options.rho.flags.writeable
    np.testing.assert_array_equal(options.subset([2, 0]).rho, [-0.2, 0.1])
    assert options_type(rho=0.2).subset([2, 0]).rho == 0.2


@pytest.mark.parametrize("rho", [True, [False], np.nan, np.inf, 1.0, -1.0, [[0.2]], []])
def test_invalid_rho_is_rejected(options_type, rho):
    """Invalid correlations fail before constructing candidate models."""
    with pytest.raises(ValueError, match="rho"):
        options_type(rho=rho)


@pytest.mark.parametrize("reference", [True, -1, 0.5, "0"])
def test_invalid_reference_is_rejected(options_type, reference):
    """Reference identifiers are nonnegative integer library positions."""
    with pytest.raises(ValueError, match="reference"):
        options_type(reference_hrf_id=reference)


def test_options_validate_against_data_and_library(options_type):
    """Feature mismatch and an absent reference cannot silently select a model."""
    library = HrfLibrary.from_parameters([])
    with pytest.raises(ValueError, match="feature"):
        options_type(rho=[0.1, 0.2]).validate(3, library)
    with pytest.raises(ValueError, match="reference"):
        options_type(reference_hrf_id=1).validate(3, library)
    options_type().validate(3, library)
