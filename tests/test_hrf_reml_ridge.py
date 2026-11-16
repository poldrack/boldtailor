"""Inner ridge folds select likelihood HRFs using training runs only."""

import numpy as np
import pytest

from boldtailor.data import from_arrays
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.hrf_selection import select_hrfs, subset_runs
from boldtailor.fractional_ridge import score_fraction_candidates
from boldtailor.ridge_selection import score_ridge_candidates
from boldtailor.model import TaskModel, Modulator
from tests.hrf_reml_fixtures import reml_hrf_problem, planted_reml_hrfs


@pytest.mark.parametrize("fractional", [False, True])
def test_reml_ridge_folds_use_only_training_runs(planted_reml_hrfs, fractional):
    """Each inner fold retains its method, reference mean, and supplied rho."""
    data, library = planted_reml_hrfs
    data = subset_runs(data, [0, 1])
    predictors = [e[["response_time"]] for e in data.events]
    task = TaskModel((Modulator("response_time", missing="indicator"),))
    options = RemlHrfOptions(rho=0.15)
    scorer = score_fraction_candidates if fractional else score_ridge_candidates
    grid = dict(fractions=[0.5, 1.0]) if fractional else dict(alphas=[0.0, 1.0])
    result = scorer(
        data,
        predictors,
        library=library,
        hrf_method="reml",
        reml_options=options,
        task_model=task,
        **grid,
    )
    activity = result.provenance.to_dict()["activities"][-1]
    for held in range(2):
        train = [1 - held]
        expected = select_hrfs(
            subset_runs(data, train),
            library=library,
            method="reml",
            reml_options=options,
            task_model=task,
        )
        np.testing.assert_array_equal(
            result.fold_hrf_indices[held], expected.hrf_indices
        )
        record = activity["folds"][held]["hrf_selection"]
        assert record["method"] == "reml"
        assert record["rho"] == 0.15
        assert record["task_model"] == task.to_dict()
    assert activity["validation_target"] == (
        "fixed_ols_betas" if fractional else "candidate_regularized_betas"
    )
    changed = from_arrays(
        [data.signals[0] * 7, data.signals[1]],
        data.events,
        frame_times=data.frame_times,
        confounds=data.confounds,
    )
    altered = scorer(
        changed,
        predictors,
        library=library,
        hrf_method="reml",
        reml_options=options,
        task_model=task,
        **grid,
    )
    np.testing.assert_array_equal(
        result.fold_hrf_indices[0], altered.fold_hrf_indices[0]
    )
    assert (
        activity["folds"][0]["hrf_selection"]
        == altered.provenance.to_dict()["activities"][-1]["folds"][0]["hrf_selection"]
    )


def test_reml_ridge_requires_a_library(planted_reml_hrfs):
    """An explicit REML request cannot be ignored by the canonical-only path."""
    data, _ = planted_reml_hrfs
    with pytest.raises(ValueError, match="library"):
        score_fraction_candidates(
            data,
            [e[["response_time"]] for e in data.events],
            fractions=[1.0],
            hrf_method="reml",
        )
