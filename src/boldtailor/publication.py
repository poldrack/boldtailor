from __future__ import annotations

from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass, field
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4

from filelock import FileLock, Timeout

from boldtailor.logging import _error_code

_CONTROL_DIRECTORY = ".boldtailor"
_FAILURE_LOG = "publication_failures.jsonl"


class PublicationError(RuntimeError):
    """Publication failed; inspect the cause and any retained rollback backups."""

    def __init__(
        self,
        message: str,
        *,
        recovery_directory: Path | None = None,
        rollback_errors: Sequence[Exception] = (),
    ) -> None:
        super().__init__(message)
        self.recovery_directory = recovery_directory
        self.rollback_errors = tuple(rollback_errors)


@dataclass(frozen=True, slots=True)
class Artifact:
    """One immutable in-memory artifact addressed relative to a destination."""

    path: str
    payload: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_artifact_path(self.path))
        object.__setattr__(self, "payload", _own_bytes(self.payload))


@dataclass(slots=True)
class _Transaction:
    root: Path
    backed_up: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    created: bool = False

    @property
    def stage(self) -> Path:
        return self.root / "stage"

    @property
    def backups(self) -> Path:
        return self.root / "backups"


def publish_artifact_set(
    destination: str | os.PathLike[str],
    artifacts: Iterable[Artifact],
    *,
    source_paths: Iterable[str | os.PathLike[str]] = (),
    overwrite: bool = False,
    lock_timeout: float = 30.0,
) -> tuple[Path, ...]:
    """Publish with a writer lock, per-file replacement, and failure rollback.

    The complete set is not an atomic snapshot for concurrent readers.
    """
    requested = _prepare_artifacts(artifacts)
    sources = tuple(Path(path) for path in source_paths)
    _validate_lock_timeout(lock_timeout)
    root = Path(destination).absolute()
    _require_safe_destination(root)
    root = root.resolve(strict=False)
    _preflight(root, requested, sources, overwrite=overwrite)
    control = _prepare_control_directory(root)
    lock_path = control / "publication.lock"
    _reject_symlink(lock_path)
    lock = FileLock(lock_path, timeout=lock_timeout)
    try:
        with lock:
            return _publish_locked(
                root,
                control,
                requested,
                sources,
                overwrite=overwrite,
            )
    except Timeout as error:
        publication_error = PublicationError(
            f"publication lock timed out for destination {root}"
        )
        _record_failure(control, str(uuid4()), error)
        raise publication_error from error


def _publish_locked(
    destination: Path,
    control: Path,
    artifacts: tuple[Artifact, ...],
    source_paths: tuple[Path, ...],
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    _preflight(destination, artifacts, source_paths, overwrite=overwrite)
    transaction = _Transaction(control / "transactions" / str(uuid4()))
    try:
        _safe_mkdir(transaction.root.parent, control)
        _stage_artifacts(transaction, artifacts)
        if overwrite:
            _backup_existing(destination, transaction, artifacts)
        _promote_staged(destination, transaction, artifacts)
    except Exception as error:
        raise _handle_failure(destination, control, transaction, error) from error
    _remove_transaction(transaction)
    _fsync_directory(destination)
    return tuple(destination / artifact.path for artifact in artifacts)


def _prepare_artifacts(artifacts: Iterable[Artifact]) -> tuple[Artifact, ...]:
    requested = tuple(artifacts)
    if not requested:
        raise ValueError("publication requires at least one artifact")
    if not all(isinstance(artifact, Artifact) for artifact in requested):
        raise TypeError("artifacts must contain only Artifact values")
    _reject_duplicate_paths(requested)
    for artifact in requested:
        if PurePosixPath(artifact.path).parts[0].casefold() == _CONTROL_DIRECTORY:
            raise ValueError("artifact path is reserved for publication control data")
        _validate_metadata(artifact)
    return requested


def _validate_artifact_path(path: object) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("artifact path must be a relative POSIX path")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or path.endswith("/")
        or "//" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or parsed.as_posix() != path
    ):
        raise ValueError("artifact path must be a relative POSIX path")
    return path


def _own_bytes(payload: object) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("artifact payload must be bytes-like")
    return bytes(payload)


def _reject_duplicate_paths(artifacts: Sequence[Artifact]) -> None:
    folded: list[str] = []
    for artifact in artifacts:
        key = artifact.path.casefold()
        if any(
            key == path or key.startswith(f"{path}/") or path.startswith(f"{key}/")
            for path in folded
        ):
            raise ValueError(f"duplicate artifact path: {artifact.path}")
        folded.append(key)


def _validate_metadata(artifact: Artifact) -> None:
    suffix = PurePosixPath(artifact.path).suffix.casefold()
    try:
        if suffix == ".json":
            json.loads(artifact.payload, parse_constant=_reject_json_constant)
        elif suffix == ".jsonl":
            _validate_json_lines(artifact.payload)
        elif suffix == ".tsv":
            _validate_tsv(artifact.payload)
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as error:
        raise ValueError(f"invalid metadata payload: {artifact.path}") from error


def _validate_json_lines(payload: bytes) -> None:
    text = payload.decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            raise json.JSONDecodeError("blank JSONL record", line, 0)
        json.loads(line, parse_constant=_reject_json_constant)


def _reject_json_constant(value: str) -> None:
    raise json.JSONDecodeError(f"invalid JSON constant: {value}", value, 0)


def _validate_tsv(payload: bytes) -> None:
    text = payload.decode("utf-8")
    if "\x00" in text:
        raise csv.Error("TSV contains a null byte")
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True))
    if not rows or not rows[0] or any(not column for column in rows[0]):
        raise csv.Error("TSV header is required")
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        raise csv.Error("TSV rows must have the header width")


def _validate_lock_timeout(timeout: float) -> None:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout < 0
    ):
        raise ValueError("lock_timeout must be a finite non-negative number")


def _preflight(
    destination: Path,
    artifacts: Sequence[Artifact],
    source_paths: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    _require_safe_destination(destination)
    targets = tuple(destination / artifact.path for artifact in artifacts)
    _reject_source_overlap((destination / _CONTROL_DIRECTORY, *targets), source_paths)
    _reject_target_collisions(destination, targets, overwrite=overwrite)


def _require_safe_destination(destination: Path) -> None:
    _reject_symlink_components(destination)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"publication destination is not a directory: {destination}")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for component in reversed((absolute, *absolute.parents)):
        if component.is_symlink():
            raise ValueError(f"destination contains a symlink: {component}")


def _reject_source_overlap(targets: Sequence[Path], sources: Sequence[Path]) -> None:
    normalized_sources = tuple(source.resolve(strict=False) for source in sources)
    for target in targets:
        normalized_target = target.absolute()
        for source in normalized_sources:
            if _paths_overlap(normalized_target, source):
                raise ValueError(f"source and output paths overlap: {target}")


def _paths_overlap(first: Path, second: Path) -> bool:
    first_parts = tuple(part.casefold() for part in first.parts)
    second_parts = tuple(part.casefold() for part in second.parts)
    width = min(len(first_parts), len(second_parts))
    return first_parts[:width] == second_parts[:width]


def _reject_target_collisions(
    destination: Path,
    targets: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    for target in targets:
        _reject_symlink_target_path(destination, target)
        _reject_casefold_collision(destination, target)
        if target.exists() and (not overwrite or not target.is_file()):
            raise FileExistsError(f"artifact destination already exists: {target}")


def _reject_casefold_collision(destination: Path, target: Path) -> None:
    current = destination
    for part in target.relative_to(destination).parts:
        if not current.exists():
            return
        if not current.is_dir():
            raise FileExistsError(f"artifact parent is not a directory: {current}")
        names = {child.name.casefold(): child.name for child in current.iterdir()}
        actual = names.get(part.casefold())
        if actual is None:
            return
        if actual != part:
            raise FileExistsError(f"case-folded artifact collision: {current / actual}")
        current /= actual


def _reject_symlink_target_path(destination: Path, target: Path) -> None:
    current = destination
    for part in target.relative_to(destination).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"artifact destination contains a symlink: {current}")
        if current.exists() and current != target and not current.is_dir():
            raise FileExistsError(f"artifact parent is not a directory: {current}")


def _prepare_control_directory(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    _require_safe_destination(destination)
    control = destination / _CONTROL_DIRECTORY
    _safe_mkdir(control, destination)
    return control


def _safe_mkdir(path: Path, destination: Path) -> None:
    _reject_internal_symlinks(path, destination)
    path.mkdir(parents=True, exist_ok=True)
    _reject_internal_symlinks(path, destination)
    if not path.is_dir():
        raise ValueError(f"publication control path is not a directory: {path}")


def _reject_internal_symlinks(path: Path, destination: Path) -> None:
    current = destination
    for part in path.relative_to(destination).parts:
        current /= part
        _reject_symlink(current)


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"publication path is a symlink: {path}")


def _stage_artifacts(transaction: _Transaction, artifacts: Sequence[Artifact]) -> None:
    transaction.root.mkdir(exist_ok=False)
    transaction.created = True
    transaction.stage.mkdir()
    transaction.backups.mkdir()
    for artifact in artifacts:
        staged = transaction.stage / artifact.path
        staged.parent.mkdir(parents=True, exist_ok=True)
        with staged.open("xb") as stream:
            stream.write(artifact.payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staged.parent)


def _backup_existing(
    destination: Path,
    transaction: _Transaction,
    artifacts: Sequence[Artifact],
) -> None:
    for artifact in artifacts:
        target = destination / artifact.path
        if not target.exists():
            continue
        backup = transaction.backups / artifact.path
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.replace(target, backup)
        transaction.backed_up.append(artifact.path)
        _fsync_directory(target.parent)
        _fsync_directory(backup.parent)


def _promote_staged(
    destination: Path,
    transaction: _Transaction,
    artifacts: Sequence[Artifact],
) -> None:
    for artifact in artifacts:
        target = destination / artifact.path
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(transaction.stage / artifact.path, target)
        transaction.promoted.append(artifact.path)
        _fsync_directory(target.parent)


def _rollback(destination: Path, transaction: _Transaction) -> tuple[Exception, ...]:
    errors = []
    for path in reversed(transaction.promoted):
        if path in transaction.backed_up:
            continue
        target = destination / path
        try:
            target.unlink(missing_ok=True)
            _fsync_directory(target.parent)
        except Exception as error:
            errors.append(error)
    for path in reversed(transaction.backed_up):
        target = destination / path
        try:
            os.replace(transaction.backups / path, target)
            _fsync_directory(target.parent)
        except Exception as error:
            errors.append(error)
    return tuple(errors)


def _handle_failure(
    destination: Path,
    control: Path,
    transaction: _Transaction,
    error: Exception,
) -> PublicationError:
    rollback_errors = _rollback(destination, transaction)
    recovery = transaction.root if rollback_errors else None
    _record_failure(
        control,
        transaction.root.name,
        error,
        rollback_errors=rollback_errors,
        recovery_directory=(
            recovery.relative_to(destination).as_posix()
            if recovery is not None
            else None
        ),
    )
    if recovery is None:
        _remove_transaction(transaction)
    message = "artifact publication failed"
    if recovery is not None:
        message += f"; rollback incomplete, recovery files retained at {recovery}"
    return PublicationError(
        message, recovery_directory=recovery, rollback_errors=rollback_errors
    )


def _record_failure(
    control: Path,
    identifier: str,
    error: Exception,
    *,
    rollback_errors: Sequence[Exception] = (),
    recovery_directory: str | None = None,
) -> None:
    record = {
        "rollback_failed": bool(rollback_errors),
        "error_code": _error_code(error),
        "execution_id": identifier,
        "published": False,
        "status": "failed",
    }
    if recovery_directory is not None:
        record["recovery_directory"] = recovery_directory
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    log_path = control / _FAILURE_LOG
    if log_path.is_symlink():
        return
    try:
        with log_path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(control)
    except OSError:
        pass


def _remove_transaction(transaction: _Transaction) -> None:
    if not transaction.created:
        return
    try:
        shutil.rmtree(transaction.root)
        parent = transaction.root.parent
        if not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
