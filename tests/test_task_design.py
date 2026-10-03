"""Raw trials expand to Nilearn conditions; Nilearn builds the task columns."""

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import Modulator, TaskModel
from tests.oracles import scaled_condition, peak_kernel

NSD = TaskModel(
    (
        Modulator("response_time", missing="indicator"),
        Modulator("trial_type"),
    )
)


@pytest.fixture
def events():
    return pd.DataFrame(
        dict(
            onset=[8.0, 22.0, 38.0, 60.0, 90.0, 112.0],
            duration=[3.0, 1.0, 2.0, 3.0, 1.5, 2.5],
            trial_type=[0, 1, 0, 1, 1, 0],
            response_time=[1.0, np.nan, 3.0, 2.0, 4.0, 5.0],
            image=[10, 11, 12, 13, 14, 15],
        )
    )


def test_expansion_orders_regressors_and_computes_amplitudes(events):
    from boldtailor._task_design import expand_events

    original = events.copy(deep=True)
    result = expand_events(events, NSD)
    expected = {
        "task": [1, 1, 1, 1, 1, 1],
        "response_time": [1, 0, 3, 2, 4, 5],
        "trial_type": [0, 1, 0, 1, 1, 0],
        "missing_response_time": [0, 1, 0, 0, 0, 0],
    }
    assert list(dict.fromkeys(result.trial_type)) == list(expected)
    assert list(result.columns) == ["onset", "duration", "trial_type", "modulation"]
    assert len(result) == 4 * len(events)
    for name, amplitudes in expected.items():
        rows = result.loc[result.trial_type == name]
        np.testing.assert_allclose(rows.modulation, amplitudes)
        np.testing.assert_array_equal(
            rows[["onset", "duration"]], events[["onset", "duration"]]
        )
    pd.testing.assert_frame_equal(events, original)


def test_indicator_absent_when_nothing_is_missing(events):
    from boldtailor._task_design import expand_events

    complete = events.assign(response_time=[1.0, 2.0, 3.0, 2.0, 4.0, 5.0])
    result = expand_events(complete, NSD)
    assert set(result.trial_type) == {"task", "response_time", "trial_type"}
    rt = result.loc[result.trial_type == "response_time", "modulation"]
    np.testing.assert_allclose(rt, complete.response_time)


def test_task_only_model_expands_to_unit_task_rows(events):
    from boldtailor._task_design import expand_events

    result = expand_events(events[["onset", "duration"]], TaskModel())
    assert list(result.trial_type.unique()) == ["task"]
    np.testing.assert_array_equal(result.modulation, 1.0)


def test_modulator_has_no_center_option():
    with pytest.raises(TypeError):
        Modulator("response_time", center=False)
    assert Modulator("response_time").to_dict() == {
        "column": "response_time",
        "missing": "error",
    }


def test_error_policy_rejects_missing_values(events):
    from boldtailor._task_design import expand_events

    with pytest.raises(ValueError, match="run 2.*response_time.*finite"):
        expand_events(events, TaskModel((Modulator("response_time"),)), run=2)


def test_text_in_modulator_column_is_an_error_not_missing(events):
    from boldtailor._task_design import expand_events

    events["response_time"] = events.response_time.astype(object)
    events.loc[1, "response_time"] = "invalid"
    with pytest.raises(ValueError, match="response_time.*numeric"):
        expand_events(events, NSD)


def test_all_missing_indicator_modulator_is_rejected(events):
    from boldtailor._task_design import expand_events

    with pytest.raises(ValueError, match="response_time.*observed"):
        expand_events(events.assign(response_time=np.nan), NSD)


def test_regressor_without_nonzero_amplitude_is_rejected(events):
    from boldtailor._task_design import expand_events

    with pytest.raises(ValueError, match="run 1.*trial_type.*nonzero"):
        expand_events(events.assign(trial_type=0), NSD, run=1)
    with pytest.raises(ValueError, match="response_time.*nonzero"):
        expand_events(events.assign(response_time=0.0), NSD)


def test_missing_modulator_column_or_timing_is_rejected(events):
    from boldtailor._task_design import expand_events

    with pytest.raises(ValueError, match="trial_type"):
        expand_events(events.drop(columns="trial_type"), NSD)
    with pytest.raises(ValueError, match="onset"):
        expand_events(events.drop(columns="onset"), TaskModel())


@pytest.mark.parametrize("cid", [0, 1])
def test_task_columns_match_compute_regressor_per_condition(events, cid, capsys):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    library = HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])
    times = 0.775 + 1.6 * np.arange(90)
    expanded = expand_events(events, NSD)
    hrf = hrf_model(library.candidates[cid])
    result = task_columns(expanded, times, hrf, min_onset=-24.0, oversampling=50)
    assert list(result.columns) == [
        "task",
        "response_time",
        "trial_type",
        "missing_response_time",
    ]
    np.testing.assert_array_equal(result.index, times)
    for name in result.columns:
        rows = expanded.loc[expanded.trial_type == name]
        expected, _ = compute_regressor(
            scaled_condition(rows.onset, rows.duration, rows.modulation, hrf, times),
            hrf,
            times,
            oversampling=50,
            min_onset=-24.0,
        )
        np.testing.assert_allclose(result[name], expected[:, 0], atol=1e-13)
    assert capsys.readouterr().out == ""


def test_task_columns_accept_string_hrfs_and_honor_settings(events):
    from boldtailor._task_design import expand_events, task_columns

    times = 0.775 + 1.6 * np.arange(90)
    expanded = expand_events(events, TaskModel())
    spm = task_columns(expanded, times, "spm", min_onset=-10.0, oversampling=20)
    glover = task_columns(expanded, times, "glover", min_onset=-10.0, oversampling=20)
    for name, actual in (("spm", spm), ("glover", glover)):
        expected, _ = compute_regressor(
            scaled_condition(
                expanded.onset,
                expanded.duration,
                expanded.modulation,
                peak_kernel(name),
                times,
                oversampling=20,
                min_onset=-10.0,
            ),
            peak_kernel(name),
            times,
            oversampling=20,
            min_onset=-10.0,
        )
        np.testing.assert_allclose(actual["task"], expected[:, 0], atol=1e-12)
    assert not np.allclose(spm["task"], glover["task"])


def test_string_hrf_keeps_kernel_suffix_in_regressor_names(events):
    from boldtailor._task_design import expand_events, task_columns

    model = TaskModel((Modulator("foo_kernel"),))
    frame = events.assign(foo_kernel=[1, 0, 1, 0, 1, 0])
    times = 0.775 + 1.6 * np.arange(90)
    result = task_columns(expand_events(frame, model), times, "spm")
    assert list(result.columns) == ["task", "foo_kernel"]


@pytest.mark.parametrize("hrf", ["spm", "glover", 1])
def test_task_column_peaks_do_not_depend_on_oversampling(hrf):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    if hrf == 1:
        library = HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])
        hrf = hrf_model(library.candidates[1])
    times = np.arange(0, 60, 0.1)
    expanded = expand_events(
        pd.DataFrame(dict(onset=[5.0], duration=[3.0])), TaskModel()
    )
    peaks = [
        task_columns(expanded, times, hrf, oversampling=o)["task"].max()
        for o in (20, 50)
    ]
    # Unit oversampled peaks; slack is 0.1 s frame sampling of the peak only.
    assert peaks[0] == pytest.approx(peaks[1], abs=1e-3)
    assert peaks[1] == pytest.approx(1.0, abs=5e-4)
