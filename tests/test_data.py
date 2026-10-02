from types import MappingProxyType

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


def test_from_arrays_accepts_negative_onsets_without_trial_type():
    events = pd.DataFrame({"onset": [-2.0, 4.0], "duration": [1.0, 0.0]})

    data = from_arrays(np.ones((10, 2)), events, tr=2.0)

    pd.testing.assert_frame_equal(data.events[0], events)


_TR_ERROR = "TR must be positive"
_BAD_DURATION = pd.DataFrame({"onset": [0.0, 4.0], "duration": [-1.0, 1.0]})


def _two_runs(features=(2, 2)):
    return [np.ones((10, n)) for n in features]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tr": None}, "exactly one of tr or frame_times"),
        ({"frame_times": np.arange(10) * 2.0}, "exactly one of tr or frame_times"),
        (
            {"tr": None, "frame_times": np.array([0.0, 1.0, np.nan])},
            "finite",
        ),
        (
            {"tr": None, "frame_times": np.array([0.0, 2.0, 1.0])},
            "strictly increasing",
        ),
        ({"tr": None, "frame_times": np.arange(9, dtype=float)}, "10 entries"),
        ({"signals": np.ones((2, 3, 4)), "events": "one"}, "time x features"),
        ({"signals": np.array([[1.0, np.nan]]), "events": "one"}, "finite"),
        ({"signals": np.ones((2, 3)), "events": "one", "tr": 0.0}, _TR_ERROR),
        ({"tr": True}, _TR_ERROR),
        ({"tr": np.bool_(False)}, _TR_ERROR),
        ({"tr": np.array(2.0)}, _TR_ERROR),
        ({"tr": np.array([2.0, 2.0])}, _TR_ERROR),
        ({"tr": "2.0"}, _TR_ERROR),
        ({"tr": 1 + 0j}, _TR_ERROR),
        ({"tr": object()}, _TR_ERROR),
        (
            {"signals": _two_runs((2, 3)), "events": "two"},
            "same number of features",
        ),
        (
            {"confounds": pd.DataFrame({"motion": np.ones(9)})},
            "confounds.*10 rows",
        ),
        ({"events": _BAD_DURATION}, "durations must be non-negative"),
        (
            {"signals": _two_runs(), "events": "two", "sources": 1},
            "sources must contain one value per signal run",
        ),
        ({"provenance_metadata": {"cwd": "./secret"}}, "path-like"),
    ],
)
def test_from_arrays_rejects_invalid_arguments(
    events, complete_sources, overrides, message
):
    run_events = {"one": events.iloc[:1], "two": [events, events.copy()]}
    kwargs = {"signals": np.ones((10, 2)), "events": events, "tr": 2.0}
    kwargs.update(overrides)
    if isinstance(kwargs["events"], str):
        kwargs["events"] = run_events[kwargs["events"]]
    if "sources" in kwargs:
        kwargs["sources"] = complete_sources(kwargs["sources"])

    with pytest.raises(ValueError, match=message):
        from_arrays(**kwargs)


@pytest.mark.parametrize("tr", [2, 2.5, np.int64(3), np.float64(1.5)])
def test_from_arrays_accepts_real_scalar_tr_values(events, tr):
    data = from_arrays(np.ones((10, 2)), events, tr=tr)

    assert data.frame_times[0][1] == float(tr)


def test_from_arrays_accepts_run_wise_sources_and_stable_fingerprint(
    events, complete_sources
):
    signals = [np.ones((10, 2)), np.ones((10, 2)) * 2.0]
    run_events = [events, events.copy()]
    run_confounds = [
        pd.DataFrame({"motion": np.linspace(0.0, 1.0, 10)}),
        pd.DataFrame({"motion": np.linspace(1.0, 2.0, 10)}),
    ]
    metadata = {"labels": ["face", "house"], "details": {"task": "localizer"}}
    sources = complete_sources(2)

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


@pytest.mark.parametrize("complete", [False, True])
def test_normalization_builds_one_record(
    events, monkeypatch, caplog, complete, complete_sources
):
    import hashlib
    import json
    import logging

    import boldtailor.data as module

    sources = complete_sources(1) if complete else None
    metadata = {"purpose": "normalization"}
    original = module.ProvenanceRecord
    calls = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(module, "ProvenanceRecord", record)
    caplog.set_level(logging.INFO, logger="boldtailor")
    result = from_arrays(
        np.arange(20.0).reshape(10, 2),
        events,
        tr=2.0,
        sources=sources,
        provenance_metadata=metadata,
    )
    assert len(calls) == 1
    records = [
        json.loads(r.getMessage()) for r in caplog.records if r.name == "boldtailor"
    ]
    assert dict(result.provenance.events[-1]) == records[-1]
    if complete:
        payload = {"sources": [s.to_dict() for s in sources]}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert result.provenance.metadata_fingerprint == expected
        assert not result.provenance.warnings
    else:
        assert result.provenance.metadata_fingerprint is None
        assert [w["code"] for w in result.provenance.warnings] == ["provenance_quality"]


@pytest.mark.parametrize("outcome", ["success", "metadata", "constructor"])
def test_normalization_catches_final_failures_and_isolates_context(
    events, caplog, monkeypatch, outcome
):
    import json
    import logging

    import boldtailor.data as module
    from boldtailor.logging import bind_context, emit_event

    caplog.set_level(logging.INFO, logger="boldtailor")
    sources = None
    metadata = {"bad": object()} if outcome == "metadata" else {}
    failure = RuntimeError("private constructor failure")

    def reject(*args, **kwargs):
        raise failure

    if outcome == "constructor":
        monkeypatch.setattr(module.AnalysisData, "__init__", reject)
    outer = dict(
        execution_id="outer",
        data_id="outer-data",
        analysis_id="outer-analysis",
        run_index=2,
    )
    with bind_context(**outer):
        if outcome == "success":
            from_arrays(
                np.arange(20.0).reshape(10, 2),
                events,
                tr=2.0,
                sources=sources,
                provenance_metadata=metadata,
            )
        else:
            with pytest.raises((ValueError, RuntimeError)) as caught:
                from_arrays(
                    np.arange(20.0).reshape(10, 2),
                    events,
                    tr=2.0,
                    sources=sources,
                    provenance_metadata=metadata,
                )
            if outcome == "constructor":
                assert caught.value is failure
        restored = emit_event("after_normalization", stage="test")
    records = [
        json.loads(r.getMessage()) for r in caplog.records if r.name == "boldtailor"
    ][:-1]
    ending = "completed" if outcome == "success" else "failed"
    assert [r["event"] for r in records] == [
        "normalization_started",
        f"normalization_{ending}",
    ]
    for record in records:
        assert not {"data_id", "analysis_id", "run_index"} & record.keys()
        assert record["execution_id"] != "outer"
    if outcome != "success":
        assert records[-1]["error_code"] == (
            "invalid_input" if outcome == "metadata" else "operation_failed"
        )
    assert "private" not in caplog.text
    assert all(restored[k] == v for k, v in outer.items())
