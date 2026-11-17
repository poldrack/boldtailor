from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import hashlib
import json
import math
import re
from pathlib import Path
from types import MappingProxyType
from uuid import UUID
from typing import overload

SCHEMA_ID = "boldtailor.provenance/1"
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9+_.-]+")
_QUALITY_WARNING = {
    "code": "provenance_quality",
    "message": "source metadata incomplete or anonymous; metadata fingerprint unavailable",
}


class SourceRef:
    __slots__ = (
        "_role",
        "_uri",
        "_media_type",
        "_byte_size",
        "_modified_at",
        "_sha256",
        "_annotations",
    )

    def __init__(
        self,
        role: str,
        uri: str | None = None,
        media_type: str | None = None,
        byte_size: int | None = None,
        modified_at: str | None = None,
        sha256: str | None = None,
        annotations: Mapping[str, object] | None = None,
    ) -> None:
        self._role = _validate_role(role)
        self._uri = _validate_uri(uri)
        self._media_type = _validate_optional_text(media_type)
        self._byte_size = _validate_byte_size(byte_size)
        self._modified_at = _validate_modified_at(modified_at)
        self._sha256 = _validate_sha256(sha256)
        self._annotations = _freeze_mapping({} if annotations is None else annotations)

    @property
    def role(self) -> str:
        return self._role

    @property
    def uri(self) -> str | None:
        return self._uri

    @property
    def media_type(self) -> str | None:
        return self._media_type

    @property
    def byte_size(self) -> int | None:
        return self._byte_size

    @property
    def modified_at(self) -> str | None:
        return self._modified_at

    @property
    def sha256(self) -> str | None:
        return self._sha256

    @property
    def annotations(self) -> Mapping[str, object]:
        return self._annotations

    def replace(self, **changes: object) -> SourceRef:
        values: dict[str, object] = {
            "role": self._role,
            "uri": self._uri,
            "media_type": self._media_type,
            "byte_size": self._byte_size,
            "modified_at": self._modified_at,
            "sha256": self._sha256,
            "annotations": self._annotations,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unexpected SourceRef fields: {sorted(unknown)}")
        values.update(changes)
        return SourceRef(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SourceRef):
            return NotImplemented
        return (
            self._role == other._role
            and self._uri == other._uri
            and self._media_type == other._media_type
            and self._byte_size == other._byte_size
            and self._modified_at == other._modified_at
            and self._sha256 == other._sha256
            and self._annotations == other._annotations
        )

    def __repr__(self) -> str:
        return (
            f"SourceRef(role={self._role!r}, uri={self._uri!r}, "
            f"media_type={self._media_type!r}, byte_size={self._byte_size!r}, "
            f"modified_at={self._modified_at!r}, sha256={self._sha256!r}, "
            f"annotations={self._annotations!r})"
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"role": self._role}
        if self._uri is not None:
            data["uri"] = self._uri
        if self._media_type is not None:
            data["media_type"] = self._media_type
        if self._byte_size is not None:
            data["byte_size"] = self._byte_size
        if self._modified_at is not None:
            data["modified_at"] = self._modified_at
        if self._sha256 is not None:
            data["sha256"] = self._sha256
        if self._annotations:
            data["annotations"] = _thaw(self._annotations)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SourceRef:
        return cls(
            role=_validate_role(data["role"]),
            uri=_validate_uri(data.get("uri")),
            media_type=_validate_optional_text(data.get("media_type")),
            byte_size=_validate_byte_size(data.get("byte_size")),
            modified_at=_validate_modified_at(data.get("modified_at")),
            sha256=_validate_sha256(data.get("sha256")),
            annotations=_freeze_mapping(data.get("annotations", {})),
        )


class RunSources:
    __slots__ = ("_signal", "_events", "_confounds")

    def __init__(
        self,
        signal: SourceRef,
        events: SourceRef,
        confounds: SourceRef | None = None,
    ) -> None:
        self._signal = _copy_source_ref(signal, "signal")
        self._events = _copy_source_ref(events, "events")
        self._confounds = (
            None if confounds is None else _copy_source_ref(confounds, "confounds")
        )

    @property
    def signal(self) -> SourceRef:
        return self._signal

    @property
    def events(self) -> SourceRef:
        return self._events

    @property
    def confounds(self) -> SourceRef | None:
        return self._confounds

    def replace(self, **changes: object) -> RunSources:
        values: dict[str, object] = {
            "signal": self._signal,
            "events": self._events,
            "confounds": self._confounds,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unexpected RunSources fields: {sorted(unknown)}")
        values.update(changes)
        return RunSources(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunSources):
            return NotImplemented
        return (
            self._signal == other._signal
            and self._events == other._events
            and self._confounds == other._confounds
        )

    def __repr__(self) -> str:
        return (
            f"RunSources(signal={self._signal!r}, events={self._events!r}, "
            f"confounds={self._confounds!r})"
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "signal": self._signal.to_dict(),
            "events": self._events.to_dict(),
        }
        if self._confounds is not None:
            data["confounds"] = self._confounds.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunSources:
        return cls(
            signal=SourceRef.from_dict(_require_mapping(data.get("signal"), "signal")),
            events=SourceRef.from_dict(_require_mapping(data.get("events"), "events")),
            confounds=_optional_source_ref(data.get("confounds")),
        )


class ProvenanceRecord:
    __slots__ = (
        "_execution_id",
        "_sources",
        "_activities",
        "_events",
        "_warnings",
        "_schema",
        "_extra",
        "_metadata_fingerprint",
    )

    def __init__(
        self,
        execution_id: str,
        sources: Sequence[RunSources] = (),
        activities: Sequence[Mapping[str, object]] = (),
        events: Sequence[Mapping[str, object]] = (),
        warnings: Sequence[Mapping[str, object]] = (),
        schema: str = SCHEMA_ID,
        _extra: Mapping[str, object] | None = None,
    ) -> None:
        self._schema = _validate_schema(schema)
        self._execution_id = _validate_uuid(execution_id)
        self._sources = tuple(_copy_run_sources(source) for source in sources)
        self._activities = _freeze_mapping_sequence(activities)
        self._events = _freeze_mapping_sequence(events)
        fingerprint = _metadata_fingerprint(self._sources)
        self._warnings = _augment_warnings(
            _freeze_mapping_sequence(warnings), fingerprint
        )
        self._extra = _freeze_mapping({} if _extra is None else _extra)
        self._metadata_fingerprint = fingerprint

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @property
    def sources(self) -> tuple[RunSources, ...]:
        return self._sources

    @property
    def activities(self) -> Sequence[Mapping[str, object]]:
        return self._activities

    @property
    def events(self) -> Sequence[Mapping[str, object]]:
        return self._events

    @property
    def warnings(self) -> Sequence[Mapping[str, object]]:
        return self._warnings

    @property
    def schema(self) -> str:
        return self._schema

    def replace(self, **changes: object) -> ProvenanceRecord:
        values: dict[str, object] = {
            "execution_id": self._execution_id,
            "sources": self._sources,
            "activities": self._activities,
            "events": self._events,
            "warnings": self._warnings,
            "schema": self._schema,
            "_extra": self._extra,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unexpected ProvenanceRecord fields: {sorted(unknown)}")
        values.update(changes)
        return ProvenanceRecord(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProvenanceRecord):
            return NotImplemented
        return (
            self._execution_id == other._execution_id
            and self._sources == other._sources
            and self._activities == other._activities
            and self._events == other._events
            and self._warnings == other._warnings
            and self._schema == other._schema
            and self._extra == other._extra
        )

    def __repr__(self) -> str:
        return (
            f"ProvenanceRecord(execution_id={self._execution_id!r}, "
            f"sources={self._sources!r}, activities={self._activities!r}, "
            f"events={self._events!r}, warnings={self._warnings!r}, "
            f"schema={self._schema!r})"
        )

    @property
    def metadata_fingerprint(self) -> str | None:
        return self._metadata_fingerprint

    @property
    def analysis_fingerprint(self) -> str | None:
        value = self._extra.get("analysis_fingerprint")
        return value if isinstance(value, str) else None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema": self._schema,
            "execution_id": self._execution_id,
            "sources": [source.to_dict() for source in self._sources],
            "activities": self.activity_records(),
            "events": [_thaw(event) for event in self._events],
            "warnings": [_thaw(warning) for warning in self._warnings],
            "metadata_fingerprint": self._metadata_fingerprint,
        }
        data.update(_thaw(self._extra))
        return data

    def activity_records(self) -> list[dict[str, object]]:
        """Return owned JSON-compatible activity dictionaries."""
        return [_thaw(activity) for activity in self._activities]

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ProvenanceRecord:
        known = {
            "schema",
            "execution_id",
            "sources",
            "activities",
            "events",
            "warnings",
            "metadata_fingerprint",
        }
        return cls(
            schema=_validate_schema(data.get("schema", SCHEMA_ID)),
            execution_id=_validate_uuid(data["execution_id"]),
            sources=tuple(
                RunSources.from_dict(_require_mapping(item, "sources item"))
                for item in _require_sequence(data.get("sources", ()), "sources")
            ),
            activities=_mapping_items(data.get("activities", ()), "activities"),
            events=_mapping_items(data.get("events", ()), "events"),
            warnings=_mapping_items(data.get("warnings", ()), "warnings"),
            _extra=_extra_fields(data, known),
        )


def analysis_fingerprint(
    metadata_fingerprint: str | None,
    model: Mapping[str, object],
) -> str | None:
    if metadata_fingerprint is None:
        return None
    payload = {
        "data_fingerprint": metadata_fingerprint,
        "model": _thaw(_freeze_mapping(model)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def identity_activity(record: ProvenanceRecord) -> dict[str, object]:
    """The record's last activity without environment-only keys, for embedding in identities."""
    activity = dict(record.activity_records()[-1])
    activity.pop("software", None)
    return activity


def extend_provenance(
    record: ProvenanceRecord,
    *,
    execution_id: str,
    activity: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
    warnings: Sequence[Mapping[str, object]],
    analysis_id: str | None,
) -> ProvenanceRecord:
    return record.replace(
        execution_id=execution_id,
        activities=(*record.activities, activity),
        events=tuple(events),
        warnings=(*record.warnings, *warnings),
        _extra={**record._extra, "analysis_fingerprint": analysis_id},
    )


def _copy_source_ref(value: SourceRef, name: str) -> SourceRef:
    if not isinstance(value, SourceRef):
        raise ValueError(f"{name} must be a SourceRef")
    source = SourceRef.from_dict(value.to_dict())
    if source.role != name:
        raise ValueError(f"{name} source must have role {name!r}")
    return source


def _copy_run_sources(value: RunSources) -> RunSources:
    if not isinstance(value, RunSources):
        raise ValueError("sources must contain RunSources values")
    return RunSources.from_dict(value.to_dict())


def _optional_source_ref(value: object) -> SourceRef | None:
    if value is None:
        return None
    return SourceRef.from_dict(_require_mapping(value, "confounds"))


def _mapping_items(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _require_mapping(item, f"{name} item")
        for item in _require_sequence(value, name)
    )


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _validate_role(role: object) -> str:
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a non-empty string")
    return role


def _validate_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("media_type must be a non-empty string")
    return value


def _validate_byte_size(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("byte_size must be a non-negative integer")
    if value < 0:
        raise ValueError("byte_size must be a non-negative integer")
    return value


def _validate_sha256(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return value


def _validate_modified_at(value: object) -> str | None:
    if value is None:
        return None
    message = "modified_at must be a UTC ISO-8601 string ending in Z or +00:00"
    if not isinstance(value, str) or not value.endswith(("Z", "+00:00")):
        raise ValueError(message)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(message) from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(message)
    return value


def _validate_uri(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("uri must be a non-empty string")
    if value.startswith("bids:"):
        return _validate_bids_uri(value)
    if _has_non_bids_scheme(value):
        raise ValueError("uri must be dataset-relative POSIX text or a valid BIDS URI")
    return validate_relative_path(value, name="uri")


def _validate_bids_uri(value: str) -> str:
    parts = value.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError("uri must be a valid BIDS URI")
    validate_relative_path(parts[2], name="uri")
    return value


def validate_relative_path(text: object, *, name: str = "path") -> str:
    """Return ``text`` if it is a dataset-relative POSIX path of safe components."""
    if not isinstance(text, str) or not text:
        raise ValueError(f"{name} must be a non-empty dataset-relative path")
    if text.startswith("/"):
        raise ValueError(f"{name} must be a dataset-relative path, not absolute")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"{name} must be a relative path without traversal or empty components"
        )
    if not all(_PATH_COMPONENT.fullmatch(part) for part in parts):
        raise ValueError(f"{name} must be a relative path of [A-Za-z0-9+_.-] parts")
    return text


def _has_non_bids_scheme(value: str) -> bool:
    head = value.split("/", 1)[0]
    scheme, separator, _ = head.partition(":")
    if not separator:
        return False
    if not scheme or not scheme[0].isalpha():
        return False
    return all(character.isalnum() or character in "+-." for character in scheme)


def _validate_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("execution_id must be a UUID string")
    try:
        return str(UUID(value))
    except ValueError as error:
        raise ValueError("execution_id must be a UUID string") from error


def _validate_schema(value: object) -> str:
    if value != SCHEMA_ID:
        raise ValueError("unsupported schema major version")
    return SCHEMA_ID


def _freeze_mapping_sequence(
    values: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(_freeze_mapping(value) for value in values)


def _freeze_mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("annotations must be a mapping")
    frozen = {_freeze_key(key): _freeze_json(item) for key, item in value.items()}
    return MappingProxyType(frozen)


def _freeze_key(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("JSON-safe mappings require string keys")
    return value


def _freeze_json(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("values must be JSON-safe")
        return value
    if isinstance(value, Path):
        raise ValueError("path-like objects are not JSON-safe; pass text")
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError("values must be JSON-safe")


@overload
def _thaw(value: Mapping[str, object]) -> dict[str, object]: ...


@overload
def _thaw(value: object) -> object: ...


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _augment_warnings(
    warnings: tuple[Mapping[str, object], ...],
    metadata_fingerprint: str | None,
) -> tuple[Mapping[str, object], ...]:
    if metadata_fingerprint is not None:
        return warnings
    if any(warning.get("code") == _QUALITY_WARNING["code"] for warning in warnings):
        return warnings
    return warnings + (_freeze_mapping(_QUALITY_WARNING),)


def _metadata_fingerprint(sources: Sequence[RunSources]) -> str | None:
    if not _sources_complete(sources):
        return None
    payload = {
        "sources": [source.to_dict() for source in sources],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sources_complete(sources: Sequence[RunSources]) -> bool:
    if not sources:
        return False
    for run in sources:
        for source in (run.signal, run.events, run.confounds):
            if source is None:
                continue
            if not _source_complete(source):
                return False
    return True


def _source_complete(source: SourceRef) -> bool:
    return (
        source.uri is not None
        and source.byte_size is not None
        and source.modified_at is not None
    )


def _extra_fields(data: Mapping[str, object], known: set[str]) -> Mapping[str, object]:
    return {key: value for key, value in data.items() if key not in known}
