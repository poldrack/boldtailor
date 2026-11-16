"""Reference means stay fixed while candidate trial covariances change."""

import importlib

import numpy as np
import pandas as pd
import pytest

from boldtailor.data import from_arrays
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor._hrf_design import trial_regressors
from boldtailor.model import TaskModel
from boldtailor.reml import fit_reml
from tests.hrf_reml_fixtures import reml_hrf_problem


def test_reference_design_is_fixed_and_contains_all_terms(reml_hrf_problem):
    """Candidate changes cannot move mean, nuisance, or missing-indicator columns."""
    adapter = importlib.import_module("boldtailor._hrf_reml")
    data, library, model = reml_hrf_problem
    groups = adapter.prepare_reml_groups(data, library, model, RemlHrfOptions(), ["a", "b", "c"])
    original = [g.fixed_design.copy() for g in groups]
    for r, group in enumerate(groups):
        events = data.events[r]
        trials = trial_regressors(events, data.frame_times[r], library.candidates[0])
        rt = events.response_time.fillna(0).to_numpy()
        weights = np.column_stack([np.ones(8), rt, (events.trial_type == "house"), events.response_time.isna()])
        expected = np.column_stack([data.confounds[r], np.ones(len(trials)), trials @ weights])
        np.testing.assert_allclose(group.fixed_design, expected, atol=1e-12)
        assert group.fixed_design.shape[1] == 6
        assert not group.fixed_design.flags.writeable
    designs = list(adapter.iter_trial_designs(data, library, [1, 0], ["a", "b", "c"]))
    for index, cid in enumerate([1, 0]):
        for r, matrix in enumerate(designs[index]):
            np.testing.assert_allclose(matrix, trial_regressors(data.events[r], data.frame_times[r], library.candidates[cid]), atol=1e-12)
            np.testing.assert_array_equal(groups[r].fixed_design, original[r])
            assert matrix.shape == (data.signals[r].shape[0], 8)
    assert not np.allclose(designs[0][0], designs[1][0])


def test_wide_random_design_reaches_reml(reml_hrf_problem):
    """More trial columns than scans is legal for a random covariance model."""
    adapter = importlib.import_module("boldtailor._hrf_reml")
    _, library, _ = reml_hrf_problem
    events = pd.DataFrame({"onset": np.linspace(0, 10, 30), "duration": np.zeros(30)})
    data = from_arrays(np.random.default_rng(3).normal(size=(20, 2)), events, tr=1.)
    groups = adapter.prepare_reml_groups(data, library, TaskModel(), RemlHrfOptions(), ["wide"])
    random = next(adapter.iter_trial_designs(data, library, [0], ["wide"]))
    assert random[0].shape == (20, 30)
    fitted = fit_reml(groups, random)
    assert fitted.restricted_dof > 0


@pytest.mark.parametrize("bad", ["timing", "category"])
def test_bad_events_fail_with_run_context(reml_hrf_problem, bad):
    """Malformed events are input errors rather than silently dropped candidates."""
    adapter = importlib.import_module("boldtailor._hrf_reml")
    data, library, model = reml_hrf_problem
    events = data.events[0]
    if bad == "timing":
        events.loc[0, "onset"] = 1000.
    else:
        events["trial_type"] = "face"
    changed = from_arrays(data.signals[0], events, tr=1., confounds=data.confounds[0])
    with pytest.raises(ValueError, match="bad-run"):
        adapter.prepare_reml_groups(changed, library, model, RemlHrfOptions(), ["bad-run"])
