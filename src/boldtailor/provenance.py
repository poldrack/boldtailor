from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from uuid import UUID

SCHEMA_ID = "boldtailor.provenance/1"
_FORBIDDEN_TOP_LEVEL_FIELDS = frozenset({"digest"})
_QUALITY_WARNING = {
    "code": "provenance_quality",
    "message": "source metadata incomplete or anonymous; metadata fingerprint unavailable",
}


@dataclass(frozen=True)
class SourceRef:
    role: str
    uri: str | None = None
    media_type: str | None = None
    byte_size: int | None = None
    modified_at: str | None = None
    annotations: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _validate_role(self.role))
        object.__setattr__(self, "uri", _validate_uri(self.uri))
        object.__setattr__(self, "media_type", _validate_optional_text(self.media_type))
        object.__setattr__(self, "byte_size", _validate_byte_size(self.byte_size))
        object.__setattr__(self, "modified_at", _validate_modified_at(self.modified_at))
        object.__setattr__(
            self,
            "annotations",
            _freeze_mapping(self.annotations, path_safe=True),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"role": self.role}
        if self.uri is not None:
            data["uri"] = self.uri
        if self.media_type is not None:
            data["media_type"] = self.media_type
        if self.byte_size is not None:
            data["byte_size"] = self.byte_size
        if self.modified_at is not None:
            data["modified_at"] = self.modified_at
        if self.annotations:
            data["annotations"] = _thaw(self.annotations)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SourceRef:
        return cls(
            role=data["role"],
            uri=data.get("uri"),
            media_type=data.get("media_type"),
            byte_size=data.get("byte_size"),
            modified_at=data.get("modified_at"),
            annotations=data.get("annotations", {}),
        )


@dataclass(frozen=True)
class RunSources:
    signal: SourceRef
    events: SourceRef
    confounds: SourceRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal", _copy_source_ref(self.signal, "signal"))
        object.__setattr__(self, "events", _copy_source_ref(self.events, "events"))
        if self.confounds is not None:
            object.__setattr__(
                self,
                "confounds",
                _copy_source_ref(self.confounds, "confounds"),
            )

    def to_dict(self) -> dict[str, object]:
        data = {
            "signal": self.signal.to_dict(),
            "events": self.events.to_dict(),
        }
        if self.confounds is not None:
            data["confounds"] = self.confounds.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunSources:
        return cls(
            signal=SourceRef.from_dict(_require_mapping(data.get("signal"), "signal")),
            events=SourceRef.from_dict(_require_mapping(data.get("events"), "events")),
            confounds=_optional_source_ref(data.get("confounds")),
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    execution_id: str
    sources: Sequence[RunSources] = ()
    activities: Sequence[Mapping[str, object]] = ()
    events: Sequence[Mapping[str, object]] = ()
    warnings: Sequence[Mapping[str, object]] = ()
    schema: str = SCHEMA_ID
    _extra: Mapping[str, object] = field(default_factory=dict, repr=False)
    _metadata_fingerprint: str | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _validate_schema(self.schema))
        object.__setattr__(self, "execution_id", _validate_uuid(self.execution_id))
        object.__setattr__(
            self,
            "sources",
            tuple(_copy_run_sources(source) for source in self.sources),
        )
        object.__setattr__(
            self,
            "activities",
            _freeze_mapping_sequence(self.activities),
        )
        object.__setattr__(
            self,
            "events",
            _freeze_mapping_sequence(self.events),
        )
        object.__setattr__(
            self,
            "warnings",
            _augment_warnings(_freeze_mapping_sequence(self.warnings), self.sources),
        )
        object.__setattr__(self, "_extra", _freeze_mapping(self._extra, path_safe=True))
        object.__setattr__(
            self,
            "_metadata_fingerprint",
            _metadata_fingerprint(self.sources),
        )

    @property
    def metadata_fingerprint(self) -> str | None:
        return self._metadata_fingerprint

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema": self.schema,
            "execution_id": self.execution_id,
            "sources": [source.to_dict() for source in self.sources],
            "activities": [_thaw(activity) for activity in self.activities],
            "events": [_thaw(event) for event in self.events],
            "warnings": [_thaw(warning) for warning in self.warnings],
            "metadata_fingerprint": self.metadata_fingerprint,
        }
        data.update(_thaw(_reject_forbidden_top_level_fields(self._extra)))
        return data

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
            schema=data.get("schema", SCHEMA_ID),
            execution_id=data["execution_id"],
            sources=tuple(
                RunSources.from_dict(_require_mapping(item, "sources item"))
                for item in _require_sequence(data.get("sources", ()), "sources")
            ),
            activities=_mapping_items(data.get("activities", ()), "activities"),
            events=_mapping_items(data.get("events", ()), "events"),
            warnings=_mapping_items(data.get("warnings", ()), "warnings"),
            _extra=_extra_fields(data, known),
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


def _validate_modified_at(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("modified_at must be a UTC ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("modified_at must be a UTC ISO-8601 string") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("modified_at must be a UTC ISO-8601 string")
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
    return _validate_relative_uri(value)


def _validate_bids_uri(value: str) -> str:
    parts = value.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError("uri must be a valid BIDS URI")
    _validate_relative_path(parts[2])
    return value


def _validate_relative_uri(value: str) -> str:
    if value.startswith("/"):
        raise ValueError("uri must be dataset-relative")
    _validate_relative_path(value)
    return value


def _validate_relative_path(value: str) -> None:
    if "\\" in value:
        raise ValueError("uri must be dataset-relative POSIX text")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("uri must be dataset-relative")
    if any(part in {"..", "."} for part in path.parts):
        raise ValueError("uri must not contain traversal")
    if not path.parts:
        raise ValueError("uri must be dataset-relative")


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
    return tuple(_freeze_mapping(value, path_safe=True) for value in values)


def _freeze_mapping(value: object, *, path_safe: bool) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("annotations must be a mapping")
    frozen = {
        _freeze_key(key): _freeze_json(item, path_safe=path_safe)
        for key, item in value.items()
    }
    return MappingProxyType(frozen)


def _freeze_key(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("JSON-safe mappings require string keys")
    return value


def _freeze_json(value: object, *, path_safe: bool) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("values must be JSON-safe")
        return value
    if isinstance(value, str):
        if path_safe and _looks_path_like(value):
            raise ValueError("path-like values are not allowed")
        return value
    if isinstance(value, Path):
        raise ValueError("path-like values are not allowed")
    if isinstance(value, Mapping):
        return _freeze_mapping(value, path_safe=path_safe)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, path_safe=path_safe) for item in value)
    raise ValueError("values must be JSON-safe")


def _looks_path_like(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("~", "/", "../", "./")):
        return True
    if "\\" in value:
        return True
    return value.startswith("bids:") or "/../" in value or "/./" in value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _augment_warnings(
    warnings: tuple[Mapping[str, object], ...],
    sources: Sequence[RunSources],
) -> tuple[Mapping[str, object], ...]:
    if _metadata_fingerprint(sources) is not None:
        return warnings
    if any(warning.get("code") == _QUALITY_WARNING["code"] for warning in warnings):
        return warnings
    return warnings + (_freeze_mapping(_QUALITY_WARNING, path_safe=True),)


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


def _extra_fields(
    data: Mapping[str, object], known: set[str]
) -> Mapping[str, object]:
    extra = {key: value for key, value in data.items() if key not in known}
    return _reject_forbidden_top_level_fields(extra)


def _reject_forbidden_top_level_fields(
    data: Mapping[str, object],
) -> Mapping[str, object]:
    for key in data:
        if key.lower() in _FORBIDDEN_TOP_LEVEL_FIELDS:
            raise ValueError(f"forbidden top-level field: {key}")
    return data
