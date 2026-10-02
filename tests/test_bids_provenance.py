import builtins
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from boldtailor.bids_provenance import project_bids_provenance
from boldtailor.provenance import ProvenanceRecord

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "bep028" / "expected"


@pytest.fixture
def provenance_record():
    return ProvenanceRecord.from_dict(
        {
            "schema": "boldtailor.provenance/1",
            "execution_id": "12345678-1234-5678-1234-567812345678",
            "sources": [
                {
                    "signal": {
                        "role": "signal",
                        "uri": "sub-01/func/sub-01_task-rest_bold.tsv",
                    },
                    "events": {
                        "role": "events",
                        "uri": "bids:raw:sub-01/func/sub-01_task-rest_events.tsv",
                    },
                }
            ],
            "activities": [
                {
                    "name": "normalize",
                    "stage": "data",
                    "timing_source": "tr",
                    "run_count": 1,
                    "feature_count": 2,
                },
                {
                    "name": "fit",
                    "stage": "fit",
                    "model": {"noise_model": "ols"},
                },
            ],
            "events": [
                {
                    "timestamp": "2026-08-08T12:00:00.000Z",
                    "sequence": 1,
                    "level": "INFO",
                    "event": "normalization_started",
                    "stage": "data",
                    "execution_id": "12345678-1234-5678-1234-567812345678",
                },
                {
                    "timestamp": "2026-08-08T12:00:01.000Z",
                    "sequence": 2,
                    "level": "INFO",
                    "event": "fit_completed",
                    "stage": "fit",
                    "execution_id": "12345678-1234-5678-1234-567812345678",
                    "analysis_id": "analysis-abc",
                },
            ],
            "analysis_fingerprint": "analysis-abc",
        }
    )


@pytest.fixture
def projection_options():
    return {
        "dataset_name": "Boldtailor example derivatives",
        "label": "boldtailor",
        "code_url": "https://github.com/example/boldtailor",
        "container": {
            "URI": "docker://example/boldtailor:0.1.0",
            "Tag": "example/boldtailor:0.1.0",
            "Type": "docker",
        },
        "source_datasets": (
            {"URL": "https://example.org/raw", "Version": "2.0.0"},
            {"DOI": "doi:10.1234/example", "Version": "1.0.0"},
        ),
        "dataset_links": {
            "template": "https://example.org/template",
            "raw": "https://example.org/raw",
        },
        "derivative_sidecars": {
            "sub-01/func/sub-01_task-rest_desc-model_bold.json": (
                "sub-01/func/sub-01_task-rest_bold.tsv",
                "bids:raw:sub-01/func/sub-01_task-rest_events.tsv",
            )
        },
    }


def test_projection_uses_installed_version(
    monkeypatch, provenance_record, projection_options
):
    from importlib import metadata

    original = metadata.version
    monkeypatch.setattr(
        metadata,
        "version",
        lambda name: "9.8.7" if name == "boldtailor" else original(name),
    )
    files = project_bids_provenance(provenance_record, **projection_options)
    description = json.loads(files["dataset_description.json"])
    generated = description["GeneratedBy"]
    assert next(x for x in generated if x["Name"] == "Boldtailor")["Version"] == "9.8.7"
    software = json.loads(files["prov/prov-boldtailor_soft.json"])["Software"]
    assert next(x for x in software if x["Label"] == "Boldtailor")["Version"] == "9.8.7"


def _fixture_projection():
    return {
        path.relative_to(FIXTURE_ROOT).as_posix(): path.read_bytes()
        for path in sorted(FIXTURE_ROOT.rglob("*"))
        if path.is_file()
    }


def test_projection_matches_hand_authored_pinned_draft_fixture(
    provenance_record,
    projection_options,
):
    expected = _fixture_projection()

    projected = project_bids_provenance(provenance_record, **projection_options)

    assert dict(projected) == expected


def test_projection_is_immutable_bytes_and_performs_no_io(
    provenance_record,
    projection_options,
    monkeypatch,
):
    expected_paths = tuple(_fixture_projection())

    def unexpected_io(*args, **kwargs):
        raise AssertionError("projection must not perform filesystem I/O")

    monkeypatch.setattr(builtins, "open", unexpected_io)
    monkeypatch.setattr(Path, "write_bytes", unexpected_io)
    monkeypatch.setattr(Path, "write_text", unexpected_io)

    projected = project_bids_provenance(provenance_record, **projection_options)

    assert isinstance(projected, MappingProxyType)
    assert tuple(projected) == expected_paths
    assert all(isinstance(path, str) for path in projected)
    assert all(isinstance(payload, bytes) for payload in projected.values())
    with pytest.raises(TypeError):
        projected["extra.json"] = b"{}\n"


def test_draft_records_form_semantic_activity_source_graph(
    provenance_record,
    projection_options,
):
    projected = project_bids_provenance(provenance_record, **projection_options)
    activities = json.loads(projected["prov/prov-boldtailor_act.json"])["Activities"]
    entities = json.loads(projected["prov/prov-boldtailor_ent.json"])["Files"]
    software = json.loads(projected["prov/prov-boldtailor_soft.json"])["Software"]
    environments = json.loads(projected["prov/prov-boldtailor_env.json"])[
        "Environments"
    ]
    sidecar = json.loads(projected["sub-01/func/sub-01_task-rest_desc-model_bold.json"])

    activity_ids = {activity["Id"] for activity in activities}
    entity_ids = {entity["Id"] for entity in entities}
    software_ids = {item["Id"] for item in software}
    environment_ids = {item["Id"] for item in environments}
    assert sidecar["GeneratedBy"] == [activities[-1]["Id"]]
    assert set(sidecar["Sources"]) == entity_ids
    assert set(activities[-1]["AssociatedWith"]) == software_ids
    assert set(activities[-1]["Used"]) == entity_ids | environment_ids
    assert set(sidecar["GeneratedBy"]) <= activity_ids


def test_opt_out_omits_only_pinned_draft_projection(
    provenance_record,
    projection_options,
):
    projected = project_bids_provenance(
        provenance_record,
        **projection_options,
        export_bids_prov=False,
    )

    assert tuple(projected) == (
        "dataset_description.json",
        "logs/boldtailor_events.jsonl",
        "logs/boldtailor_provenance.json",
    )


def test_logs_are_deterministic_newline_terminated_and_exclude_sensitive_runtime(
    provenance_record,
    projection_options,
):
    first = project_bids_provenance(provenance_record, **projection_options)
    second = project_bids_provenance(provenance_record, **projection_options)
    canonical = first["logs/boldtailor_provenance.json"]
    events = first["logs/boldtailor_events.jsonl"]

    assert first == second
    assert canonical.endswith(b"\n") and not canonical.endswith(b"\n\n")
    assert events.endswith(b"\n") and not events.endswith(b"\n\n")
    assert len(events.splitlines()) == len(provenance_record.events)
    combined = (canonical + events).decode("utf-8").lower()
    for forbidden in (
        '"digest"',
        '"argv"',
        '"cwd"',
        '"environment_variables"',
        '"hostname"',
        '"username"',
        '"working_directory"',
    ):
        assert forbidden not in combined


@pytest.mark.parametrize(
    "label", ["../escape", "bad/name", "bad\\name", "bad-label", ""]
)
def test_projection_rejects_invalid_bids_provenance_labels(provenance_record, label):
    with pytest.raises(ValueError, match="label"):
        project_bids_provenance(provenance_record, label=label)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.json",
        "/absolute.json",
        "sub-01/../escape.json",
        "sub-01\\escape.json",
        "sub-01/func/not-a-sidecar.tsv",
    ],
)
def test_projection_rejects_unsafe_derivative_sidecar_paths(provenance_record, path):
    with pytest.raises(ValueError, match="sidecar path"):
        project_bids_provenance(
            provenance_record,
            derivative_sidecars={path: ()},
        )


def test_projection_rejects_case_folded_output_collisions(provenance_record):
    sidecars = {
        "sub-01/func/sub-01_desc-model_bold.json": (),
        "sub-01/func/SUB-01_desc-model_bold.json": (),
    }

    with pytest.raises(ValueError, match="collision"):
        project_bids_provenance(
            provenance_record,
            derivative_sidecars=sidecars,
        )


@pytest.mark.parametrize("path", ["dataset_description.json", "provenance.json"])
def test_projection_rejects_sidecars_that_overwrite_reserved_artifacts(
    provenance_record,
    path,
):
    with pytest.raises(ValueError, match="collision"):
        project_bids_provenance(
            provenance_record,
            derivative_sidecars={path: ()},
        )


def test_projection_rejects_relationship_sources_absent_from_record(provenance_record):
    with pytest.raises(ValueError, match="record source"):
        project_bids_provenance(
            provenance_record,
            derivative_sidecars={
                "sub-01/func/sub-01_desc-model_bold.json": (
                    "sub-99/func/sub-99_task-secret_bold.nii.gz",
                )
            },
        )
