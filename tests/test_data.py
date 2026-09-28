import numpy as np
import pandas as pd
import pytest
from types import MappingProxyType

from boldtailor.data import from_arrays
from boldtailor.provenance import RunSources, SourceRef


@pytest.fixture
def events():
    return pd.DataFrame(
        {
            "onset": [0.0, 4.0],
            "duration": [1.0, 1.0],
            "trial_type": ["face", "house"],
        }
    )


def _complete_sources(run_index: int) -> RunSources:
    stem = f"sub-01_task-localizer_run-{run_index:02d}"
    return RunSources(
        signal=SourceRef(
            role="signal",
            uri=f"sub-01/func/{stem}_bold.tsv",
            media_type="text/tab-separated-values",
            byte_size=1024 + run_index,
            modified_at="2026-08-08T12:00:00Z",
        ),
        events=SourceRef(
            role="events",
            uri=f"sub-01/func/{stem}_events.tsv",
            media_type="text/tab-separated-values",
            byte_size=256 + run_index,
            modified_at="2026-08-08T12:01:00Z",
        ),
        confounds=SourceRef(
            role="confounds",
            uri=f"sub-01/func/{stem}_confounds.tsv",
            media_type="text/tab-separated-values",
            byte_size=512 + run_index,
            modified_at="2026-08-08T12:02:00Z",
        ),
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
    assert data.provenance.execution_id
    assert data.provenance.metadata_fingerprint is None
    assert data.provenance.sources[0].signal.role == "signal"
    assert data.provenance.sources[0].signal.uri is None
    assert tuple(event["event"] for event in data.provenance.events) == (
        "normalization_started",
        "normalization_completed",
    )
    assert any(
        warning["code"] == "provenance_quality" for warning in data.provenance.warnings
    )


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


def test_from_arrays_owns_readonly_copies(events):
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
        assert type(values) is np.ndarray
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0


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


def test_from_arrays_accepts_run_wise_sources_and_stable_fingerprint(events):
    signals = [np.ones((10, 2)), np.ones((10, 2)) * 2.0]
    run_events = [events, events.copy()]
    run_confounds = [
        pd.DataFrame({"motion": np.linspace(0.0, 1.0, 10)}),
        pd.DataFrame({"motion": np.linspace(1.0, 2.0, 10)}),
    ]
    metadata = {"labels": ["face", "house"], "details": {"task": "localizer"}}
    sources = [_complete_sources(1), _complete_sources(2)]

    first = from_arrays(
        signals,
        run_events,
        tr=2.0,
        confounds=run_confounds,
        sources=sources,
        provenance_metadata=metadata,
    )
    second = from_arrays(
        signals,
        [events.copy(), events.copy()],
        tr=2.0,
        confounds=run_confounds,
        sources=sources,
        provenance_metadata=metadata,
    )
    metadata["labels"].append("source-mutated")
    metadata["details"]["task"] = "mutated"

    assert first.provenance.execution_id != second.provenance.execution_id
    assert (
        first.provenance.metadata_fingerprint == second.provenance.metadata_fingerprint
    )
    assert first.provenance.metadata_fingerprint is not None
    assert len(first.provenance.sources) == 2
    assert (
        first.provenance.sources[0].signal.uri
        == "sub-01/func/sub-01_task-localizer_run-01_bold.tsv"
    )
    activity = first.provenance.activities[0]
    assert activity["name"] == "normalize"
    assert activity["stage"] == "data"
    assert activity["metadata"]["labels"] == ("face", "house")
    assert activity["metadata"]["details"]["task"] == "localizer"
    assert isinstance(activity["metadata"], MappingProxyType)
    assert isinstance(activity["metadata"]["details"], MappingProxyType)
    with pytest.raises(TypeError):
        activity["metadata"]["details"]["task"] = "changed"


def test_from_arrays_rejects_source_count_mismatch(events):
    signals = [np.ones((10, 2)), np.ones((10, 2))]

    with pytest.raises(
        ValueError, match="sources must contain one value per signal run"
    ):
        from_arrays(
            signals,
            [events, events.copy()],
            tr=2.0,
            sources=[_complete_sources(1)],
        )


def test_from_arrays_rejects_path_like_provenance_metadata(events):
    with pytest.raises(ValueError, match="path-like"):
        from_arrays(
            np.ones((10, 2)),
            events,
            tr=2.0,
            provenance_metadata={"cwd": "./secret"},
        )
