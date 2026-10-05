import numpy as np
import pytest

from experiments.nsd_replication.stats import criterion_met, mean_ci, relative_difference, tost_paired


def test_tost_equivalent_when_tight():
    diff = np.array([0.01, -0.01, 0.0, 0.005, -0.005, 0.002, -0.002, 0.0])
    result = tost_paired(diff, margin=0.05)
    assert result["equivalent"] and result["p"] < 0.05


def test_tost_not_equivalent_when_shifted():
    diff = np.full(8, 0.1) + np.linspace(-0.01, 0.01, 8)
    assert not tost_paired(diff, margin=0.05)["equivalent"]


def test_tost_rejects_bad_margin():
    with pytest.raises(ValueError, match="margin"):
        tost_paired(np.zeros(4), margin=0)


def test_relative_difference():
    np.testing.assert_allclose(relative_difference(np.array([1.1, 0.9]), np.array([1.0, 1.0])), [0.1, -0.1])


def test_criterion_three_of_four():
    values = {"sub-01": (0.3, 0.2), "sub-02": (0.3, 0.2), "sub-03": (0.3, 0.2), "sub-04": (0.1, 0.2)}
    result = criterion_met(values, minimum=3)
    assert result["replicated"] and result["per_subject"]["sub-04"] is False


def test_mean_ci_contains_mean():
    mean, low, high = mean_ci(np.array([1.0, 2.0, 3.0]))
    assert low < mean == 2.0 < high
