import importlib

import pytest

from experiments.hrf_estimation.test_continuous import offgrid
from experiments.hrf_estimation.test_library_cv import planted_runs


def test_inner_validation_prefers_unpenalized_offgrid_recovery(offgrid):
    runs, library, _ = offgrid
    choose = importlib.import_module(
        "experiments.hrf_estimation.tuning"
    ).choose_regularization
    settings = {
        "library": library,
        "lambda_grid": [0.0, 1.0],
        "max_starts": 1,
        "maxiter": 100,
        "maxfun": 1200,
    }
    assert choose(runs, settings) == 0.0
    # This endpoint has no access to any outer-test runs; one training run cannot tune.
    with pytest.raises(ValueError, match="two"):
        choose(runs[:1], settings)


def test_regularization_grid_must_be_nonnegative(offgrid):
    runs, library, _ = offgrid
    choose = importlib.import_module(
        "experiments.hrf_estimation.tuning"
    ).choose_regularization
    with pytest.raises(ValueError, match="lambda"):
        choose(runs, {"library": library, "lambda_grid": [-1, 0]})
