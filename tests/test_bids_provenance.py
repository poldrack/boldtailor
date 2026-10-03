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
                    "software": {
                        "python": "3.12.0",
                        "platform": "Test-Platform-1.0",
                        "boldtailor": "0.1.0",
                        "numpy": "2.0.0",
                    },
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


def test_file_entities_carry_digest_only_when_supplied(
    provenance_record, projection_options
):
    payload = provenance_record.to_dict()
    payload["sources"][0]["signal"]["sha256"] = "a" * 64
    record = ProvenanceRecord.from_dict(payload)
    projected = project_bids_provenance(record, **projection_options)
    files = json.loads(projected["prov/prov-boldtailor_ent.json"])["Files"]
    by_location = {item.get("AtLocation"): item for item in files}
    signal = by_location["sub-01/func/sub-01_task-rest_bold.tsv"]
    assert signal["Digest"] == {"sha256": "a" * 64}
    others = [item for item in files if item is not signal]
    assert others and all("Digest" not in item for item in others)


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
        '"argv"',
        '"cwd"',
        '"environment_variables"',
        '"hostname"',
        '"username"',
        '"working_directory"',
    ):
        assert forbidden not in combined


_MODEL_SIDECAR = "sub-01/func/sub-01_desc-model_bold.json"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"label": "../escape"}, "label"),
        ({"label": "bad/name"}, "label"),
        ({"label": "bad\\name"}, "label"),
        ({"label": "bad-label"}, "label"),
        ({"label": ""}, "label"),
        ({"derivative_sidecars": {"../escape.json": ()}}, "sidecar path"),
        ({"derivative_sidecars": {"/absolute.json": ()}}, "sidecar path"),
        ({"derivative_sidecars": {"sub-01/../escape.json": ()}}, "sidecar path"),
        ({"derivative_sidecars": {"sub-01\\escape.json": ()}}, "sidecar path"),
        (
            {"derivative_sidecars": {"sub-01/func/not-a-sidecar.tsv": ()}},
            "sidecar path",
        ),
        ({"derivative_sidecars": {"sub-01/my file.json": ()}}, "relative path"),
        ({"derivative_sidecars": {"dataset_description.json": ()}}, "collision"),
        ({"derivative_sidecars": {"provenance.json": ()}}, "collision"),
        (
            {
                "derivative_sidecars": {
                    _MODEL_SIDECAR: ("sub-99/func/sub-99_task-secret_bold.nii.gz",)
                }
            },
            "record source",
        ),
    ],
)
def test_projection_rejects_invalid_options(provenance_record, options, message):
    with pytest.raises(ValueError, match=message):
        project_bids_provenance(provenance_record, **options)


def _draft(projected, kind):
    return json.loads(projected[f"prov/prov-boldtailor_{kind}.json"])


@pytest.fixture
def record_from_fit(complete_sources):
    import numpy as np
    import pandas as pd

    from boldtailor.data import from_arrays
    from boldtailor.fit import fit
    from boldtailor.model import ModelSpec

    rng = np.random.default_rng(20261002)
    events = pd.DataFrame(
        {"onset": [4.0, 24.0, 44.0, 64.0], "duration": 1.0, "trial_type": "face"}
    )
    data = from_arrays(
        rng.normal(size=(40, 2)), events, tr=2.0, sources=complete_sources(1)
    )
    return fit(data, ModelSpec(contrasts={"face": "face"})).provenance


def test_bids_activity_has_command_and_timestamps(record_from_fit):
    from boldtailor._software import software_environment

    projected = project_bids_provenance(record_from_fit)
    activity = _draft(projected, "act")["Activities"][-1]
    assert activity["Command"] == "boldtailor.fit"
    assert activity["StartedAtTime"] <= activity["EndedAtTime"]
    own = [
        event["timestamp"]
        for event in record_from_fit.events
        if event["execution_id"] == record_from_fit.execution_id
    ]
    assert (activity["StartedAtTime"], activity["EndedAtTime"]) == (min(own), max(own))
    env = _draft(projected, "env")["Environments"][0]
    assert env["Python"] == software_environment()["python"]
    assert env["Platform"] == software_environment()["platform"]
    packages = {item["Label"]: item["Version"] for item in env["Software"]}
    assert packages["nilearn"] == software_environment()["nilearn"]


@pytest.mark.parametrize(
    ("name", "command"),
    [
        ("fit", "boldtailor.fit"),
        ("single_trial", "boldtailor.fit_single_trials"),
        ("hrf_selection", "boldtailor.select_hrf"),
        ("normalize", "boldtailor.from_arrays"),
        ("brand_new_step", "boldtailor.brand_new_step"),
    ],
)
def test_bids_command_names_the_entry_point(provenance_record, name, command):
    payload = provenance_record.to_dict()
    payload["activities"][-1]["name"] = name
    projected = project_bids_provenance(ProvenanceRecord.from_dict(payload))
    assert _draft(projected, "act")["Activities"][-1]["Command"] == command


def test_projection_equals_record_for_sensitive_looking_keys(provenance_record):
    payload = provenance_record.to_dict()
    payload["sources"][0]["signal"]["annotations"] = {"hostname": "h", "cwd": "c"}
    payload["activities"][-1]["cwd"] = "kept"
    record = ProvenanceRecord.from_dict(payload)
    projected = project_bids_provenance(record)
    logged = json.loads(projected["logs/boldtailor_provenance.json"])
    assert logged == record.to_dict()
    assert projected["logs/boldtailor_provenance.json"] == (
        record.canonical_json().encode("utf-8") + b"\n"
    )


def test_bids_projection_uses_the_shared_relative_path_rule():
    import boldtailor.bids_provenance as bids
    import boldtailor.provenance as provenance

    assert bids.validate_relative_path is provenance.validate_relative_path
    assert not hasattr(bids, "_validate_relative_path")


def test_sidecar_paths_differing_only_in_case_are_both_projected(
    provenance_record,
):
    upper = "sub-01/func/SUB-01_desc-model_bold.json"
    projected = project_bids_provenance(
        provenance_record,
        derivative_sidecars={_MODEL_SIDECAR: (), upper: ()},
    )
    assert _MODEL_SIDECAR in projected and upper in projected
