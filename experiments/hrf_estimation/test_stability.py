import importlib

import numpy as np
import pytest

from boldtailor.hrf_library import HrfCandidate
from experiments.hrf_estimation.estimators import HrfEstimate


def fitted(onset=0):
    kernel = HrfCandidate(0, "double_gamma", (4, 12, 0.8, 1.5, 6, onset, 36))
    return HrfEstimate(
        np.array([4, 7]), (kernel, None), ("ok", "no_signal"), ("run",), {}
    )


def test_identical_and_shifted_hrf_stability():
    compare = importlib.import_module(
        "experiments.hrf_estimation.stability"
    ).compare_hrfs
    same = compare(fitted(), fitted())
    assert same.loc[4, "curve_correlation"] == pytest.approx(1)
    assert same.loc[4, "curve_rmse"] == pytest.approx(0)
    assert np.isnan(same.loc[7, "curve_correlation"])
    shifted = compare(fitted(), fitted(1))
    assert shifted.loc[4, "peak_difference"] == pytest.approx(1, abs=0.11)
    assert shifted.loc[4, "fwhm_difference"] == pytest.approx(0, abs=0.01)
    assert shifted.loc[4, "curve_correlation"] < 1
    with pytest.raises(ValueError, match="feature"):
        compare(
            fitted(),
            HrfEstimate(np.array([7, 4]), fitted().kernels, ("ok", "ok"), ("run",), {}),
        )
