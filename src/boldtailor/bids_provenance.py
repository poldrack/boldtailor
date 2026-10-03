from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from urllib.parse import urlsplit

from boldtailor._software import package_version
from boldtailor.provenance import ProvenanceRecord

STABLE_BIDS_VERSION = "1.11.1"
BEP028_DRAFT_IDENTIFIER = "BEP028"
BEP028_DRAFT_SNAPSHOT = "02172700aac8d1bdd67b45191f43533f426848dc"
BEP028_SUPPORTED_SUBSET = (
    "Activities",
    "Files",
    "Environments",
    "Software",
    "provenance label table",
    "file GeneratedBy and Sources relationships",
)

_BIDS_LABEL = re.compile(r"[A-Za-z0-9]+\Z")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9+_.-]+\Z")
_SOURCE_DATASET_KEYS = frozenset({"URL", "DOI", "Version"})
_CONTAINER_KEYS = frozenset({"Type", "Tag", "URI"})
_SENSITIVE_KEYS = frozenset(
    {
        "argv",
        "cwd",
        "environment_variables",
        "hostname",
        "username",
        "working_directory",
    }
)


def project_bids_provenance(
    record: ProvenanceRecord,
    *,
    dataset_name: str = "Boldtailor derivatives",
    label: str = "boldtailor",
    code_url: str | None = None,
    container: Mapping[str, str] | None = None,
    source_datasets: Sequence[Mapping[str, str]] = (),
    dataset_links: Mapping[str, str] | None = None,
    derivative_sidecars: Mapping[str, Sequence[str]] | None = None,
    export_bids_prov: bool = True,
) -> Mapping[str, bytes]:
    """Project an internal record into deterministic in-memory BIDS artifacts.

    Stable metadata follows BIDS 1.11.1. Draft files implement only the pinned
    BEP028 subset named by ``BEP028_SUPPORTED_SUBSET``; they do not claim
    conformance to a final BIDS provenance standard.
    """
    _require_record(record)
    safe_label = _validate_label(label)
    normalized_sidecars = _normalize_sidecars(record, derivative_sidecars)
    generated_by = _generated_by(code_url, container)
    artifacts = _stable_artifacts(
        record,
        dataset_name=_validate_text(dataset_name, "dataset_name"),
        generated_by=generated_by,
        source_datasets=_normalize_source_datasets(source_datasets),
        dataset_links=_normalize_dataset_links(dataset_links),
    )
    if export_bids_prov:
        _extend_artifacts(
            artifacts,
            _draft_artifacts(
                record,
                label=safe_label,
                code_url=code_url,
                sidecars=normalized_sidecars,
            ),
        )
    return _freeze_artifacts(artifacts)


def _stable_artifacts(
    record: ProvenanceRecord,
    *,
    dataset_name: str,
    generated_by: Mapping[str, object],
    source_datasets: tuple[Mapping[str, str], ...],
    dataset_links: Mapping[str, str],
) -> dict[str, bytes]:
    description: dict[str, object] = {
        "Name": dataset_name,
        "BIDSVersion": STABLE_BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": [generated_by],
    }
    if source_datasets:
        description["SourceDatasets"] = list(source_datasets)
    if dataset_links:
        description["DatasetLinks"] = dict(dataset_links)
    safe_record = _safe_json(record.to_dict())
    return {
        "dataset_description.json": _pretty_json(description),
        "logs/boldtailor_events.jsonl": _json_lines(safe_record["events"]),
        "logs/boldtailor_provenance.json": _compact_json(safe_record),
    }


def _draft_artifacts(
    record: ProvenanceRecord,
    *,
    label: str,
    code_url: str | None,
    sidecars: Mapping[str, tuple[str, ...]],
) -> dict[str, bytes]:
    graph = _draft_graph(record, code_url=code_url)
    prefix = f"prov/prov-{label}"
    artifacts = {
        "provenance.tsv": _provenance_tsv(label),
        "provenance.json": _pretty_json(_provenance_sidecar()),
        f"{prefix}_act.json": _pretty_json({"Activities": graph["activities"]}),
        f"{prefix}_ent.json": _pretty_json(graph["entities"]),
        f"{prefix}_env.json": _pretty_json({"Environments": graph["environments"]}),
        f"{prefix}_soft.json": _pretty_json({"Software": graph["software"]}),
    }
    _extend_artifacts(artifacts, _relationship_sidecars(sidecars, graph))
    return artifacts


def _draft_graph(
    record: ProvenanceRecord,
    *,
    code_url: str | None,
) -> dict[str, object]:
    token = record.execution_id.replace("-", "")
    software_id = f"bids::prov#boldtailor-{token}"
    environment_id = f"bids::prov#environment-{token}"
    entities, source_ids = _entities(record, token)
    activities, activity_ids = _activities(
        record,
        token=token,
        software_id=software_id,
        used=(*source_ids, environment_id),
    )
    software: dict[str, object] = {
        "Id": software_id,
        "Label": "Boldtailor",
        "Version": package_version("boldtailor"),
    }
    if code_url is not None:
        software["AlternativeIdentifier"] = [code_url]
    return {
        "activities": activities,
        "activity_ids": activity_ids,
        "entities": entities,
        "source_ids": source_ids,
        "environments": [
            {
                "Id": environment_id,
                "Label": "Boldtailor execution environment",
            }
        ],
        "software": [software],
    }


def _activities(
    record: ProvenanceRecord,
    *,
    token: str,
    software_id: str,
    used: tuple[str, ...],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    source = tuple(record.activities) or ({"name": "provenance", "stage": "record"},)
    activities = []
    identifiers = []
    for index, activity in enumerate(source):
        safe_activity = _safe_json(activity)
        name = _slug(safe_activity.get("name", f"activity{index}"), "activity")
        identifier = f"bids::prov#{name}-{token}-{index}"
        identifiers.append(identifier)
        activities.append(
            {
                "Id": identifier,
                "Label": name,
                "Command": _activity_command(safe_activity),
                "AssociatedWith": [software_id],
                "Used": list(used),
            }
        )
    return activities, tuple(identifiers)


def _activity_command(activity: Mapping[str, object]) -> str:
    payload = json.dumps(
        activity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"boldtailor Python API activity {payload}"


def _entities(
    record: ProvenanceRecord,
    token: str,
) -> tuple[dict[str, list[dict[str, object]]], tuple[str, ...]]:
    files: dict[str, dict[str, object]] = {}
    anonymous: list[dict[str, object]] = []
    seen_paths: dict[str, str] = {}
    for run_index, run in enumerate(record.sources):
        for source in (run.signal, run.events, run.confounds):
            if source is None:
                continue
            if source.uri is None:
                anonymous.append(_anonymous_entity(source.role, run_index, token))
                continue
            _check_case_collision(source.uri, seen_paths, "source path")
            files.setdefault(source.uri, _file_entity(source.uri, source.sha256))
    document: dict[str, list[dict[str, object]]] = {}
    if files:
        document["Files"] = list(files.values())
    if anonymous:
        document["prov:Entity"] = anonymous
    if not document:
        anonymous.append(_anonymous_entity("input", 0, token))
        document["prov:Entity"] = anonymous
    identifiers = tuple(item["Id"] for group in document.values() for item in group)
    return document, identifiers


def _file_entity(uri: str, sha256: str | None = None) -> dict[str, object]:
    if uri.startswith("bids:"):
        identifier = uri
        location = uri.split(":", 2)[2]
        _validate_relative_path(location, "record source path")
    else:
        _validate_relative_path(uri, "record source path")
        identifier = f"bids::{uri}"
        location = uri
    entity: dict[str, object] = {
        "Id": identifier,
        "Label": PurePosixPath(location).name,
    }
    if not uri.startswith("bids:"):
        entity["AtLocation"] = uri
    if sha256:
        entity["Digest"] = {"sha256": sha256}
    return entity


def _anonymous_entity(role: str, run_index: int, token: str) -> dict[str, object]:
    safe_role = _slug(role, "source")
    return {
        "Id": f"bids::prov#entity-run{run_index}{safe_role}-{token}",
        "Label": f"run{run_index}{safe_role}",
    }


def _relationship_sidecars(
    sidecars: Mapping[str, tuple[str, ...]],
    graph: Mapping[str, object],
) -> dict[str, bytes]:
    activity_ids = graph["activity_ids"]
    source_ids = {_source_identifier(uri) for uri in graph["source_ids"]}
    output = {}
    for path, sources in sidecars.items():
        relationships = {
            "GeneratedBy": [activity_ids[-1]],
            "Sources": sorted(_source_identifier(source) for source in sources),
        }
        if not set(relationships["Sources"]) <= source_ids:
            raise ValueError("derivative relationship must reference a record source")
        output[path] = _pretty_json(relationships)
    return output


def _source_identifier(value: str) -> str:
    return value if value.startswith("bids:") else f"bids::{value}"


def _generated_by(
    code_url: str | None,
    container: Mapping[str, str] | None,
) -> dict[str, object]:
    generated: dict[str, object] = {
        "Name": "Boldtailor",
        "Version": package_version("boldtailor"),
    }
    if code_url is not None:
        generated["CodeURL"] = _validate_uri(code_url, "code_url")
    normalized_container = _normalize_container(container)
    if normalized_container:
        generated["Container"] = normalized_container
    return generated


def _normalize_container(
    container: Mapping[str, str] | None,
) -> dict[str, str]:
    if container is None:
        return {}
    if not isinstance(container, Mapping) or not container:
        raise ValueError("container must be a non-empty mapping")
    unknown = set(container) - _CONTAINER_KEYS
    if unknown:
        raise ValueError(f"unsupported container field: {sorted(unknown)[0]}")
    normalized = {
        key: _validate_text(value, f"container {key}")
        for key, value in container.items()
    }
    if "URI" in normalized:
        normalized["URI"] = _validate_uri(normalized["URI"], "container URI")
    return dict(sorted(normalized.items()))


def _normalize_source_datasets(
    values: Sequence[Mapping[str, str]],
) -> tuple[Mapping[str, str], ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError("source_datasets must be a sequence of mappings")
    normalized = [_normalize_source_dataset(value) for value in values]
    unique = {_canonical_text(value): value for value in normalized}
    return tuple(unique[key] for key in sorted(unique))


def _normalize_source_dataset(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source dataset must be a non-empty mapping")
    unknown = set(value) - _SOURCE_DATASET_KEYS
    if unknown:
        raise ValueError(f"unsupported source dataset field: {sorted(unknown)[0]}")
    normalized = {
        key: _validate_text(item, f"source dataset {key}")
        for key, item in value.items()
    }
    for key in ("URL", "DOI"):
        if key in normalized:
            normalized[key] = _validate_uri(normalized[key], f"source dataset {key}")
    return dict(sorted(normalized.items()))


def _normalize_dataset_links(
    values: Mapping[str, str] | None,
) -> Mapping[str, str]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise ValueError("dataset_links must be a mapping")
    normalized = {}
    for key, value in values.items():
        normalized[_validate_label(key, name="dataset link label")] = (
            _validate_link_location(value)
        )
    return MappingProxyType(dict(sorted(normalized.items())))


def _validate_link_location(value: object) -> str:
    text = _validate_text(value, "dataset link location")
    if urlsplit(text).scheme:
        return _validate_uri(text, "dataset link location")
    return _validate_relative_path(text, "dataset link location")


def _normalize_sidecars(
    record: ProvenanceRecord,
    values: Mapping[str, Sequence[str]] | None,
) -> Mapping[str, tuple[str, ...]]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise ValueError("derivative_sidecars must be a mapping")
    record_sources = _record_source_uris(record)
    normalized = {}
    seen: dict[str, str] = {}
    for path, sources in values.items():
        safe_path = _validate_sidecar_path(path)
        _check_case_collision(safe_path, seen, "output path")
        normalized[safe_path] = _normalize_relationship_sources(
            sources,
            record_sources,
        )
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_relationship_sources(
    values: Sequence[str],
    record_sources: frozenset[str],
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError("derivative sidecar sources must be a sequence")
    normalized = tuple(sorted(set(values)))
    if any(value not in record_sources for value in normalized):
        raise ValueError("derivative relationship must reference a record source")
    return normalized


def _record_source_uris(record: ProvenanceRecord) -> frozenset[str]:
    return frozenset(
        source.uri
        for run in record.sources
        for source in (run.signal, run.events, run.confounds)
        if source is not None and source.uri is not None
    )


def _validate_sidecar_path(value: object) -> str:
    path = _validate_relative_path(value, "derivative sidecar path")
    if not path.endswith(".json"):
        raise ValueError("derivative sidecar path must end in .json")
    return path


def _validate_relative_path(value: object, name: str) -> str:
    text = _validate_text(value, name)
    if text.startswith("/") or "\\" in text:
        raise ValueError(f"{name} must be a dataset-relative POSIX path")
    path = PurePosixPath(text)
    if path.as_posix() != text or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{name} must not contain traversal or empty components")
    if not path.parts or any(
        not _PATH_COMPONENT.fullmatch(part) for part in path.parts
    ):
        raise ValueError(f"{name} contains invalid path characters")
    return text


def _validate_label(value: object, name: str = "label") -> str:
    if not isinstance(value, str) or not _BIDS_LABEL.fullmatch(value):
        raise ValueError(f"{name} must contain only ASCII letters and digits")
    return value


def _slug(value: object, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    slug = "".join(
        character.lower()
        for character in value
        if character.isascii() and character.isalnum()
    )
    return slug or fallback


def _validate_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{name} contains unsafe characters")
    return value


def _validate_uri(value: object, name: str) -> str:
    text = _validate_text(value, name)
    if (
        not urlsplit(text).scheme
        or "\\" in text
        or any(char.isspace() for char in text)
    ):
        raise ValueError(f"{name} must be a URI")
    return text


def _require_record(record: object) -> None:
    if not isinstance(record, ProvenanceRecord):
        raise ValueError("record must be a ProvenanceRecord")


def _check_case_collision(value: str, seen: dict[str, str], name: str) -> None:
    folded = value.casefold()
    previous = seen.get(folded)
    if previous is not None and previous != value:
        raise ValueError(f"case-folded {name} collision: {previous!r} and {value!r}")
    seen[folded] = value


def _extend_artifacts(
    artifacts: dict[str, bytes],
    additions: Mapping[str, bytes],
) -> None:
    seen = {path.casefold(): path for path in artifacts}
    for path, payload in additions.items():
        previous = seen.get(path.casefold())
        if previous is not None:
            raise ValueError(f"artifact path collision: {previous!r} and {path!r}")
        seen[path.casefold()] = path
        artifacts[path] = payload


def _safe_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _safe_json(item)
            for key, item in value.items()
            if key.casefold() not in _SENSITIVE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_json(item) for item in value]
    return value


def _provenance_tsv(label: str) -> bytes:
    draft = f"{BEP028_DRAFT_IDENTIFIER}@{BEP028_DRAFT_SNAPSHOT}"
    text = (
        "provenance_id\tdescription\tbep028_draft\n"
        f"prov-{label}\tBoldtailor provenance projection\t{draft}\n"
    )
    return text.encode("utf-8")


def _provenance_sidecar() -> dict[str, object]:
    return {
        "BEP028Draft": {
            "Identifier": BEP028_DRAFT_IDENTIFIER,
            "Snapshot": (
                "bids-standard/bids-specification@" f"{BEP028_DRAFT_SNAPSHOT}"
            ),
            "Status": "draft",
            "SupportedSubset": list(BEP028_SUPPORTED_SUBSET),
        },
        "provenance_id": {
            "Description": "Identifier of the prov entity described by this row."
        },
        "description": {
            "Description": "Human-readable description of the provenance entity."
        },
        "bep028_draft": {
            "Description": "Pinned BEP028 draft snapshot used for this projection."
        },
    }


def _freeze_artifacts(artifacts: Mapping[str, bytes]) -> Mapping[str, bytes]:
    seen: dict[str, str] = {}
    ordered = {}
    for path in sorted(artifacts):
        safe_path = _validate_relative_path(path, "artifact path")
        _check_case_collision(safe_path, seen, "artifact path")
        payload = artifacts[path]
        if not isinstance(payload, bytes):
            raise ValueError("artifact payloads must be bytes")
        ordered[safe_path] = bytes(payload)
    return MappingProxyType(ordered)


def _pretty_json(value: object) -> bytes:
    text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return text.encode("utf-8")


def _compact_json(value: object) -> bytes:
    return (_canonical_text(value) + "\n").encode("utf-8")


def _json_lines(values: Sequence[object]) -> bytes:
    lines = [_canonical_text(value) for value in values]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _canonical_text(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
