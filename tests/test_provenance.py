from pathlib import Path
import re
import socket

from types import MappingProxyType

import pytest

from boldtailor.provenance import ProvenanceRecord, RunSources, SourceRef


def _complete_source(role: str, uri: str) -> SourceRef:
    return SourceRef(
        role=role,
        uri=uri,
        media_type="text/tab-separated-values",
        byte_size=128,
        modified_at="2026-08-08T12:00:00Z",
        annotations={"columns": ("onset", "duration"), "meta": {"kind": role}},
    )


def _record_payload(**overrides):
    payload = {
        "execution_id": "123e4567-e89b-12d3-a456-426614174000",
        "sources": [
            {
                "signal": _complete_source(
                    "signal", "sub-01/func/sub-01_task-rest_bold.tsv"
                ).to_dict(),
                "events": _complete_source(
                    "events", "sub-01/func/sub-01_task-rest_events.tsv"
                ).to_dict(),
            }
        ],
        "activities": [{"name": "normalize", "stage": "data"}],
        "events": [{"event": "started", "sequence": 1}],
        "warnings": [{"code": "quality", "message": "all good"}],
    }
    payload.update(overrides)
    return payload


def test_source_ref_owns_immutable_annotations():
    annotations = {
        "labels": ["face", "house"],
        "details": {"column": "trial_type"},
    }

    source = SourceRef(
        role="events",
        uri="sub-01/func/sub-01_task-rest_events.tsv",
        annotations=annotations,
    )
    annotations["labels"].append("source-mutated")
    annotations["details"]["column"] = "mutated"

    assert source.annotations["labels"] == ("face", "house")
    assert source.annotations["details"]["column"] == "trial_type"
    assert isinstance(source.annotations, MappingProxyType)
    assert isinstance(source.annotations["details"], MappingProxyType)
    with pytest.raises(TypeError):
        source.annotations["details"]["column"] = "changed"
    with pytest.raises(TypeError):
        source.annotations["new"] = "value"


@pytest.mark.parametrize(
    "uri",
    [
        "sub-01/func/sub-01_task-rest_bold.tsv",
        "bids:raw:sub-01/func/sub-01_task-rest_events.tsv",
    ],
)
def test_source_ref_accepts_dataset_relative_and_bids_uris(uri):
    source = SourceRef(
        role="signal",
        uri=uri,
        media_type="text/tab-separated-values",
        byte_size=10,
        modified_at="2026-08-08T12:00:00Z",
    )

    assert source.role == "signal"
    assert source.uri == uri
    assert source.byte_size == 10


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"role": ""}, "role"),
        ({"uri": "/tmp/secret.tsv"}, "relative"),
        ({"uri": "../secret.tsv"}, "traversal"),
        ({"uri": "http://example.com/file.tsv"}, "BIDS"),
        ({"byte_size": -1}, "non-negative"),
        ({"byte_size": 1.5}, "integer"),
        ({"modified_at": "2026-08-08T12:00:00"}, "UTC"),
        ({"annotations": {"value": float("nan")}}, "JSON-safe"),
        ({"annotations": {"value": float("inf")}}, "JSON-safe"),
        ({"annotations": {"value": object()}}, "JSON-safe"),
        ({"annotations": {"path": Path.home() / "secret.tsv"}}, "path-like"),
    ],
)
def test_source_ref_rejects_invalid_values(kwargs, message):
    payload = {
        "role": "events",
        "uri": "sub-01/func/sub-01_task-rest_events.tsv",
        "media_type": "text/tab-separated-values",
        "byte_size": 10,
        "modified_at": "2026-08-08T12:00:00Z",
        "annotations": {"kind": "events"},
    }
    payload.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SourceRef(**payload)


def test_run_sources_requires_exactly_one_signal_and_events_source():
    sources = RunSources(
        signal=_complete_source("signal", "sub-01/func/sub-01_task-rest_bold.tsv"),
        events=_complete_source("events", "sub-01/func/sub-01_task-rest_events.tsv"),
    )

    assert sources.signal.role == "signal"
    assert sources.events.role == "events"
    assert sources.confounds is None

    with pytest.raises(ValueError, match="SourceRef"):
        RunSources(
            signal="signal",  # type: ignore[arg-type]
            events=_complete_source(
                "events", "sub-01/func/sub-01_task-rest_events.tsv"
            ),
        )


def test_run_sources_rejects_role_mismatches_for_named_slots():
    with pytest.raises(ValueError, match="signal"):
        RunSources(
            signal=_complete_source(
                "events", "sub-01/func/sub-01_task-rest_events.tsv"
            ),
            events=_complete_source(
                "events", "sub-01/func/sub-01_task-rest_events.tsv"
            ),
        )

    with pytest.raises(ValueError, match="events"):
        RunSources(
            signal=_complete_source("signal", "sub-01/func/sub-01_task-rest_bold.tsv"),
            events=_complete_source("signal", "sub-01/func/sub-01_task-rest_bold.tsv"),
        )


def test_provenance_record_owns_nested_inputs_and_preserves_order():
    run_sources = [
        RunSources(
            signal=_complete_source("signal", "sub-01/func/sub-01_task-rest_bold.tsv"),
            events=_complete_source(
                "events", "sub-01/func/sub-01_task-rest_events.tsv"
            ),
            confounds=_complete_source(
                "confounds", "sub-01/func/sub-01_task-rest_confounds.tsv"
            ),
        )
    ]
    activities = [{"name": "normalize", "runs": [0]}]
    events = [
        {"event": "started", "sequence": 1},
        {"event": "completed", "sequence": 2},
    ]
    warnings = [{"code": "quality", "message": "stable"}]

    record = ProvenanceRecord(
        execution_id="123e4567-e89b-12d3-a456-426614174000",
        sources=run_sources,
        activities=activities,
        events=events,
        warnings=warnings,
    )
    activities[0]["runs"].append(1)
    events[0]["event"] = "mutated"
    warnings[0]["message"] = "mutated"

    assert record.schema == "boldtailor.provenance/1"
    assert tuple(event["event"] for event in record.events) == ("started", "completed")
    assert record.activities[0]["runs"] == (0,)
    assert record.warnings[0]["message"] == "stable"
    assert isinstance(record.activities[0], MappingProxyType)
    with pytest.raises(TypeError):
        record.events[0]["event"] = "changed"


def test_provenance_record_round_trips_without_sharing_state():
    record = ProvenanceRecord.from_dict(_record_payload())
    serialized = record.to_dict()
    loaded = ProvenanceRecord.from_dict(serialized)

    serialized["activities"][0]["name"] = "mutated"
    serialized["sources"][0]["signal"]["role"] = "mutated"

    assert loaded.to_dict()["activities"][0]["name"] == "normalize"
    assert loaded.sources[0].signal.role == "signal"
    assert loaded.to_dict() == record.to_dict()


def test_provenance_record_validates_execution_ids_and_schema_versions():
    with pytest.raises(ValueError, match="UUID"):
        ProvenanceRecord.from_dict(_record_payload(execution_id="not-a-uuid"))

    with pytest.raises(ValueError, match="major"):
        ProvenanceRecord.from_dict(_record_payload(schema="boldtailor.provenance/2"))


def test_provenance_record_accepts_additive_version_one_fields():
    loaded = ProvenanceRecord.from_dict(
        _record_payload(future_field={"status": "kept", "count": 2})
    )

    assert loaded.to_dict()["future_field"] == {"status": "kept", "count": 2}


def test_provenance_record_rejects_top_level_digest_field():
    with pytest.raises(ValueError, match="digest"):
        ProvenanceRecord.from_dict(_record_payload(digest="abc123"))


def test_canonical_json_and_metadata_fingerprint_are_stable_across_mapping_order():
    record_a = ProvenanceRecord(
        execution_id="123e4567-e89b-12d3-a456-426614174000",
        sources=(
            RunSources(
                signal=SourceRef(
                    role="signal",
                    uri="sub-01/func/sub-01_task-rest_bold.tsv",
                    media_type="text/tab-separated-values",
                    byte_size=10,
                    modified_at="2026-08-08T12:00:00Z",
                    annotations={"alpha": 1, "beta": {"first": "x", "second": "y"}},
                ),
                events=SourceRef(
                    role="events",
                    uri="sub-01/func/sub-01_task-rest_events.tsv",
                    media_type="text/tab-separated-values",
                    byte_size=11,
                    modified_at="2026-08-08T12:01:00Z",
                    annotations={"beta": {"second": "y", "first": "x"}, "alpha": 1},
                ),
            ),
        ),
        activities=({"name": "normalize", "details": {"b": 2, "a": 1}},),
    )
    record_b = ProvenanceRecord.from_dict(
        {
            "schema": "boldtailor.provenance/1",
            "execution_id": "123e4567-e89b-12d3-a456-426614174000",
            "sources": [
                {
                    "signal": {
                        "role": "signal",
                        "uri": "sub-01/func/sub-01_task-rest_bold.tsv",
                        "media_type": "text/tab-separated-values",
                        "byte_size": 10,
                        "modified_at": "2026-08-08T12:00:00Z",
                        "annotations": {
                            "beta": {"second": "y", "first": "x"},
                            "alpha": 1,
                        },
                    },
                    "events": {
                        "role": "events",
                        "uri": "sub-01/func/sub-01_task-rest_events.tsv",
                        "media_type": "text/tab-separated-values",
                        "byte_size": 11,
                        "modified_at": "2026-08-08T12:01:00Z",
                        "annotations": {
                            "alpha": 1,
                            "beta": {"first": "x", "second": "y"},
                        },
                    },
                }
            ],
            "activities": [{"details": {"a": 1, "b": 2}, "name": "normalize"}],
        }
    )

    assert record_a.canonical_json() == record_b.canonical_json()
    assert record_a.metadata_fingerprint == record_b.metadata_fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", record_a.metadata_fingerprint or "")


def test_incomplete_sources_clear_metadata_fingerprint_and_add_quality_warning():
    record = ProvenanceRecord(
        execution_id="123e4567-e89b-12d3-a456-426614174000",
        sources=(
            RunSources(
                signal=SourceRef(role="signal"),
                events=_complete_source(
                    "events", "sub-01/func/sub-01_task-rest_events.tsv"
                ),
            ),
        ),
    )

    assert record.metadata_fingerprint is None
    assert any("incomplete" in warning["message"] for warning in record.warnings)


def test_serialized_record_excludes_digest_and_sensitive_runtime_strings():
    record = ProvenanceRecord.from_dict(_record_payload())
    serialized = record.canonical_json()

    assert "Digest" not in serialized
    assert "digest" not in serialized
    assert str(Path.home()) not in serialized
    assert "/Users/poldrack/Dropbox/code/boldtailor" not in serialized
    assert Path.home().name not in serialized
    assert socket.gethostname() not in serialized
