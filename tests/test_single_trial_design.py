import importlib

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor


def compiler():
    try:
        return importlib.import_module(
            "boldtailor._single_trial_design"
        ).compile_trial_run
    except ModuleNotFoundError:
        pytest.fail("Single-trial compiler is not implemented")


@pytest.fixture
def inputs():
    events = pd.DataFrame(
        {
            "onset": [12.0, 8.0],
            "duration": [3.0, 1.0],
            "73k_id": [99, 99],
            "response_time": [0.5, 2.0],
        }
    )
    times = 0.775 + 1.6 * np.arange(40)
    confounds = pd.DataFrame(
        {"motion": np.linspace(-1, 1, 40)}, index=np.arange(40) + 8
    )
    return events, times, confounds


def test_subtr_timing_repeat_identity_and_row_order(inputs):
    events, times, confounds = inputs
    x, n, trials = compiler()(events, times, confounds, "run-01")
    assert trials.event_index.tolist() == [0, 1]
    assert trials.trial_id.tolist() == ["run-01_trial-0001", "run-01_trial-0002"]
    assert list(x) == trials.trial_id.tolist()
    assert list(n) == ["motion", "constant"]
    np.testing.assert_array_equal(n.motion, confounds.motion)
    for i, (onset, duration) in enumerate([(12.0, 3.0), (8.0, 1.0)]):
        expected, _ = compute_regressor(
            np.array([[onset], [duration], [1.0]]), "spm", times
        )
        np.testing.assert_allclose(x.iloc[:, i], expected[:, 0])
    pd.testing.assert_frame_equal(trials[events.columns], events)


def test_rt_and_image_identity_have_no_design_effect(inputs):
    events, times, confounds = inputs
    x, _, _ = compiler()(events, times, confounds, "run-01")
    changed = events.assign(response_time=[np.nan, -1], **{"73k_id": [12, 13]})
    x2, _, _ = compiler()(changed, times, confounds, "run-01")
    pd.testing.assert_frame_equal(x, x2)


def test_zero_duration_has_nilearn_impulse_convention(inputs):
    events, times, confounds = inputs
    events.loc[0, "duration"] = 0.0
    x, _, _ = compiler()(events, times, confounds, "run-01")
    expected, _ = compute_regressor(np.array([[12.0], [0.0], [1.0]]), "spm", times)
    np.testing.assert_allclose(x.iloc[:, 0], expected[:, 0])
    assert np.any(x.iloc[:, 0] != 0)


@pytest.mark.parametrize(
    "column,value", [("onset", np.nan), ("duration", -1), ("duration", np.inf)]
)
def test_invalid_timing_is_rejected(inputs, column, value):
    events, times, confounds = inputs
    events.loc[0, column] = value
    with pytest.raises(ValueError, match="timing|duration|finite"):
        compiler()(events, times, confounds, "run-01")


@pytest.mark.parametrize("onset", [-1000.0, 1000.0])
def test_events_without_supported_response_are_rejected(inputs, onset):
    events, times, confounds = inputs
    events.loc[0, "onset"] = onset
    with pytest.raises(ValueError, match="support|onset|sample"):
        compiler()(events, times, confounds, "run-01")


@pytest.mark.parametrize(
    "name", ["trial_id", "trial_index", "run_index", "run_label", "event_index"]
)
def test_reserved_event_names_are_rejected(inputs, name):
    events, times, confounds = inputs
    events[name] = 1
    with pytest.raises(ValueError, match="reserved"):
        compiler()(events, times, confounds, "run-01")


def test_invalid_nuisance_is_rejected(inputs):
    events, times, confounds = inputs
    for bad in (
        confounds.assign(constant=1),
        confounds.iloc[:-1],
        pd.concat([confounds, confounds], axis=1),
        confounds * np.nan,
    ):
        with pytest.raises(ValueError):
            compiler()(events, times, bad, "run-01")


def test_input_tables_are_not_mutated(inputs):
    events, times, confounds = inputs
    original_events, original_confounds = events.copy(), confounds.copy()
    _, _, trials = compiler()(events, times, confounds, "run-01")
    trials.loc[0, "onset"] = 999
    pd.testing.assert_frame_equal(events, original_events)
    pd.testing.assert_frame_equal(confounds, original_confounds)
