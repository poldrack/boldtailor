import numpy as np
import pandas as pd
import pytest

from boldtailor.data import from_arrays


@pytest.fixture
def events():
    return pd.DataFrame(
        {
            "onset": [0.0, 4.0],
            "duration": [1.0, 1.0],
            "trial_type": ["face", "house"],
        }
    )


def test_from_arrays_normalizes_one_run_from_tr(events):
    signals = np.arange(20.0).reshape(10, 2)

    data = from_arrays(signals, events, tr=2.0)

    assert data.n_runs == 1
    assert data.n_features == 2
    assert data.timing_source == "tr"
    np.testing.assert_array_equal(data.signals[0], signals)
    np.testing.assert_array_equal(data.frame_times[0], np.arange(10) * 2.0)
    assert list(data.confounds[0].columns) == []


def test_from_arrays_accepts_run_wise_frame_times(events):
    signals = [np.ones((10, 2)), np.ones((12, 2))]
    frame_times = [np.arange(10) * 1.5 + 0.75, np.arange(12) * 2.0 + 1.0]

    data = from_arrays(
        signals,
        [events, events.copy()],
        frame_times=frame_times,
    )

    assert data.timing_source == "frame_times"
    np.testing.assert_array_equal(data.frame_times[0], frame_times[0])
    np.testing.assert_array_equal(data.frame_times[1], frame_times[1])


@pytest.mark.parametrize(
    ("tr", "frame_times"),
    [(None, None), (2.0, np.arange(10) * 2.0)],
)
def test_from_arrays_requires_exactly_one_timing_source(events, tr, frame_times):
    with pytest.raises(ValueError, match="exactly one of tr or frame_times"):
        from_arrays(
            np.ones((10, 2)),
            events,
            tr=tr,
            frame_times=frame_times,
        )


@pytest.mark.parametrize(
    ("frame_times", "message"),
    [
        (np.array([0.0, 1.0, np.nan]), "finite"),
        (np.array([0.0, 2.0, 1.0]), "strictly increasing"),
        (np.arange(9, dtype=float), "10 entries"),
    ],
)
def test_from_arrays_rejects_invalid_frame_times(events, frame_times, message):
    with pytest.raises(ValueError, match=message):
        from_arrays(
            np.ones((10, 2)),
            events,
            frame_times=frame_times,
        )


def test_from_arrays_owns_strictly_immutable_copies(events):
    signals = np.arange(20.0).reshape(10, 2)
    frame_times = np.arange(10, dtype=float) * 2.0
    confounds = pd.DataFrame({"motion": np.linspace(0.0, 1.0, 10)})

    data = from_arrays(
        signals,
        events,
        frame_times=frame_times,
        confounds=confounds,
    )
    signals[0, 0] = -99.0
    frame_times[0] = -99.0
    events.loc[0, "trial_type"] = "changed"
    confounds.loc[0, "motion"] = -99.0

    assert data.signals[0][0, 0] == 0.0
    assert data.frame_times[0][0] == 0.0
    assert data.events[0].loc[0, "trial_type"] == "face"
    assert data.confounds[0].loc[0, "motion"] == 0.0
    for values in (data.signals[0], data.frame_times[0]):
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)


def test_from_arrays_owns_nested_tabular_payloads(events):
    events["metadata"] = [["event"], ["other-event"]]
    confounds = pd.DataFrame({"unselected": [["confound"] for _ in range(10)]})

    data = from_arrays(np.ones((10, 2)), events, tr=2.0, confounds=confounds)

    events.at[0, "metadata"].append("source-mutated")
    confounds.at[0, "unselected"].append("source-mutated")

    assert data.events[0].at[0, "metadata"] == ["event"]
    assert data.confounds[0].at[0, "unselected"] == ["confound"]

    data.events[0].at[0, "metadata"].append("accessor-mutated")
    data.confounds[0].at[0, "unselected"].append("accessor-mutated")

    assert data.events[0].at[0, "metadata"] == ["event"]
    assert data.confounds[0].at[0, "unselected"] == ["confound"]


def test_from_arrays_accepts_negative_onsets_without_trial_type():
    events = pd.DataFrame({"onset": [-2.0, 4.0], "duration": [1.0, 0.0]})

    data = from_arrays(np.ones((10, 2)), events, tr=2.0)

    pd.testing.assert_frame_equal(data.events[0], events)


@pytest.mark.parametrize(
    ("signals", "tr", "message"),
    [
        (np.ones((2, 3, 4)), 2.0, "time x features"),
        (np.array([[1.0, np.nan]]), 2.0, "finite"),
        (np.ones((2, 3)), 0.0, "TR must be positive"),
    ],
)
def test_from_arrays_rejects_invalid_signals(events, signals, tr, message):
    with pytest.raises(ValueError, match=message):
        from_arrays(signals, events.iloc[:1], tr=tr)


@pytest.mark.parametrize(
    "tr",
    [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.array(2.0),
        np.array([2.0, 2.0]),
        "2.0",
        1 + 0j,
        object(),
    ],
)
def test_from_arrays_rejects_nonreal_tr_values(events, tr):
    with pytest.raises(ValueError, match="TR must be positive"):
        from_arrays(np.ones((10, 2)), events, tr=tr)


@pytest.mark.parametrize("tr", [2, 2.5, np.int64(3), np.float64(1.5)])
def test_from_arrays_accepts_real_scalar_tr_values(events, tr):
    data = from_arrays(np.ones((10, 2)), events, tr=tr)

    assert data.frame_times[0][1] == float(tr)


def test_from_arrays_rejects_incompatible_runs(events):
    signals = [np.ones((10, 2)), np.ones((10, 3))]

    with pytest.raises(ValueError, match="same number of features"):
        from_arrays(signals, [events, events.copy()], tr=2.0)


def test_from_arrays_rejects_wrong_confound_length(events):
    with pytest.raises(ValueError, match="confounds.*10 rows"):
        from_arrays(
            np.ones((10, 2)),
            events,
            tr=2.0,
            confounds=pd.DataFrame({"motion": np.ones(9)}),
        )


def test_from_arrays_rejects_invalid_event_timing(events):
    events.loc[0, "duration"] = -1.0

    with pytest.raises(ValueError, match="durations must be non-negative"):
        from_arrays(np.ones((10, 2)), events, tr=2.0)
