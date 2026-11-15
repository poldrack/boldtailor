"""Adversarial review probes; expected to fail on the reviewed code."""
import numpy as np
import pytest
from boldtailor.reml import RemlGroup, fit_reml, select_random_design

@pytest.fixture
def identity_problem():
    y = np.random.default_rng(0).normal(size=(10, 1))
    return [RemlGroup(y, np.empty((10, 0)))], np.eye(10)

def test_unidentifiable_variances_are_not_reported_as_supported(identity_problem):
    groups, z = identity_problem
    result = select_random_design(groups, [[z]])
    # Covariance is (sigma2 + tau2) I: data cannot identify the split.
    assert result.fit.status[0] != "ok"

@pytest.fixture
def narrow_minimum():
    y = np.array([5.659277805656175, 1.3575913902516368,
        12.166250878076637, 440.03410697431354, 0.024484911495792103,
        0.2989731950657336, 754.0944608984961, 0.023686778553470386,
        23.645757013142806, 0.005169198169685577])[:, None]
    z = np.zeros((10, 8))
    z[:8] = np.diag(np.sqrt(np.logspace(-10, 10, 8)))
    return y, z

def test_optimum_is_no_worse_than_known_positive_ratio(narrow_minimum):
    y, z = narrow_minimum
    fit = fit_reml([RemlGroup(y, np.empty((10, 0)))], [z])
    v = np.eye(10) + 7.126268271680137e-7 * (z @ z.T)
    q = (y.T @ np.linalg.solve(v, y)).item()
    objective = 10 * np.log(q / 10) + np.linalg.slogdet(v)[1]
    assert fit.criterion[0] <= objective + 1e-9

@pytest.fixture
def fixed_offset_problem():
    rng = np.random.default_rng(19)
    c = np.ones((30, 1))
    z = rng.normal(size=(30, 2))
    y = z @ np.array([[2.], [3.]]) + rng.normal(size=(30, 1))
    return y, c, z

def test_fixed_effect_offset_preserves_signal_status(fixed_offset_problem):
    y, c, z = fixed_offset_problem
    base = fit_reml([RemlGroup(y, c)], [z])
    shifted = fit_reml([RemlGroup(y + 1e8, c)], [z])
    assert shifted.status == base.status
