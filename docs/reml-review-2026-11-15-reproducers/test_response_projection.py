"""Follow-up review evidence; expected failures on commit 87affe6.

These probes are outside the default test suite. No implementation is changed.
"""

import numpy as np
import pytest

from boldtailor.ar1 import estimate_rho, prepare_reference
from boldtailor.reml import RemlGroup, fit_reml


@pytest.fixture
def scaled_fixed_problem():
    eye = np.eye(30)
    fixed = eye[:, :2].copy()
    fixed[:, 0] *= 1e-12
    signal = (2 * eye[:, 2] + eye[:, 3])[:, None]
    shifted = signal + 100 * eye[:, [0]]
    return fixed, eye[:, [2]], signal, shifted


def test_exactly_removable_shift_preserves_reml(scaled_fixed_problem):
    fixed, random, signal, shifted = scaled_fixed_problem
    base = fit_reml([RemlGroup(signal, fixed)], [random])
    moved = fit_reml([RemlGroup(shifted, fixed)], [random])
    assert base.status == ("ok",)
    assert moved.status == base.status
    np.testing.assert_allclose(moved.ratio, base.ratio, rtol=1e-12)


def test_exactly_removable_shift_preserves_ar1(scaled_fixed_problem):
    fixed, _, signal, shifted = scaled_fixed_problem
    reference = [prepare_reference(fixed)]
    base = estimate_rho([signal], reference)
    moved = estimate_rho([shifted], reference)
    np.testing.assert_allclose(base.raw, [[0.4]])
    np.testing.assert_allclose(moved.raw, base.raw, rtol=1e-12)
