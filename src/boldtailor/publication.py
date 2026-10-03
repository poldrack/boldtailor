from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4

from filelock import FileLock, Timeout

from boldtailor.logging import _error_code

_CONTROL_SUFFIX = ".boldtailor"
_FAILURE_LOG = "failures.jsonl"


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
    control: Path
    identifier: str = field(default_factory=lambda: str(uuid4()))
    created: list[Path] = field(default_factory=list)
    backed_up: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)

    @property
    def stage(self) -> Path:
        return self.control / f"stage-{self.identifier}"

    @property
    def backups(self) -> Path:
        return self.control / f"backup-{self.identifier}"


def publish_artifact_set(
    destination: str | os.PathLike[str],
    artifacts: Iterable[Artifact],
    *,
    source_paths: Iterable[str | os.PathLike[str]] = (),
    overwrite: bool = False,
    lock_timeout: float = 30.0,
) -> tuple[Path, ...]:
    """Publish a file set under one writer lock, rolling back on failure.

    Lock, staging, backups, and the failure ledger live in the sibling
    ``<destination>.boldtailor/`` directory, not in the dataset. Concurrent
    readers may observe a partially replaced set.
    """
    requested = _prepare_artifacts(artifacts)
    _validate_lock_timeout(lock_timeout)
    root = _resolve_destination(destination)
    control = root.with_name(root.name + _CONTROL_SUFFIX)
    sources = tuple(Path(path).resolve(strict=False) for path in source_paths)
    targets = (control, *(root / artifact.path for artifact in requested))
    _reject_source_overlap(targets, sources)
    control.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(control / "lock", timeout=lock_timeout):
            return _publish_locked(root, control, requested, overwrite=overwrite)
    except Timeout as error:
        _record_failure(control, str(uuid4()), error)
        message = f"publication lock timed out for destination {root}"
        raise PublicationError(message) from error


def _publish_locked(
    destination: Path,
    control: Path,
    artifacts: tuple[Artifact, ...],
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    for artifact in artifacts:
        _check_target(destination, artifact.path, overwrite=overwrite)
    transaction = _Transaction(control)
    try:
        _stage_artifacts(transaction, artifacts)
        if overwrite:
            _backup_existing(destination, transaction, artifacts)
        _promote_staged(destination, transaction, artifacts)
    except Exception as error:
        raise _handle_failure(destination, transaction, error) from error
    _remove_transaction(transaction)
    return tuple(destination / artifact.path for artifact in artifacts)


def _prepare_artifacts(artifacts: Iterable[Artifact]) -> tuple[Artifact, ...]:
    requested = tuple(artifacts)
    if not requested:
        raise ValueError("publication requires at least one artifact")
    if not all(isinstance(artifact, Artifact) for artifact in requested):
        raise TypeError("artifacts must contain only Artifact values")
    _reject_duplicate_paths(requested)
    return requested


def _validate_artifact_path(path: object) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("artifact path must be a relative POSIX path")
    if PurePosixPath(path).is_absolute() or any(
        part in {"", ".", ".."} for part in path.split("/")
    ):
        raise ValueError("artifact path must be a relative POSIX path")
    return path


def _own_bytes(payload: object) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("artifact payload must be bytes-like")
    return bytes(payload)


def _reject_duplicate_paths(artifacts: Sequence[Artifact]) -> None:
    seen: list[str] = []
    for artifact in artifacts:
        key = artifact.path
        if any(
            key == path or key.startswith(f"{path}/") or path.startswith(f"{key}/")
            for path in seen
        ):
            raise ValueError(f"duplicate artifact path: {artifact.path}")
        seen.append(key)


def _validate_lock_timeout(timeout: float) -> None:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout < 0
    ):
        raise ValueError("lock_timeout must be a finite non-negative number")


def _resolve_destination(destination: str | os.PathLike[str]) -> Path:
    """Resolve the destination (and links such as macOS /tmp) before any checks."""
    root = Path(destination).resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise ValueError(f"publication destination is not a directory: {root}")
    return root


def _reject_source_overlap(targets: Sequence[Path], sources: Sequence[Path]) -> None:
    for target in targets:
        for source in sources:
            width = min(len(target.parts), len(source.parts))
            if target.parts[:width] == source.parts[:width]:
                raise ValueError(f"source and output paths overlap: {target}")


def _check_target(destination: Path, path: str, *, overwrite: bool) -> None:
    """Refuse symlinks inside the destination and unrequested overwrites."""
    current = destination
    parts = PurePosixPath(path).parts
    for index, part in enumerate(parts, start=1):
        current /= part
        if current.is_symlink():
            raise ValueError(f"artifact destination contains a symlink: {current}")
        if index < len(parts) and current.exists() and not current.is_dir():
            raise FileExistsError(f"artifact parent is not a directory: {current}")
    if current.exists() and (not overwrite or not current.is_file()):
        raise FileExistsError(f"artifact destination already exists: {current}")


def _make_transaction_directory(transaction: _Transaction, path: Path) -> None:
    path.mkdir(exist_ok=False)
    transaction.created.append(path)


def _stage_artifacts(transaction: _Transaction, artifacts: Sequence[Artifact]) -> None:
    _make_transaction_directory(transaction, transaction.stage)
    for artifact in artifacts:
        staged = transaction.stage / artifact.path
        staged.parent.mkdir(parents=True, exist_ok=True)
        with staged.open("xb") as stream:
            stream.write(artifact.payload)
            stream.flush()
            os.fsync(stream.fileno())


def _backup_existing(
    destination: Path,
    transaction: _Transaction,
    artifacts: Sequence[Artifact],
) -> None:
    for artifact in artifacts:
        target = destination / artifact.path
        if not target.exists():
            continue
        if transaction.backups not in transaction.created:
            _make_transaction_directory(transaction, transaction.backups)
        backup = transaction.backups / artifact.path
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.replace(target, backup)
        transaction.backed_up.append(artifact.path)


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


def _rollback(destination: Path, transaction: _Transaction) -> tuple[Exception, ...]:
    errors = []
    for path in reversed(transaction.promoted):
        if path not in transaction.backed_up:
            try:
                (destination / path).unlink(missing_ok=True)
            except Exception as error:
                errors.append(error)
    for path in reversed(transaction.backed_up):
        try:
            os.replace(transaction.backups / path, destination / path)
        except Exception as error:
            errors.append(error)
    return tuple(errors)


def _handle_failure(
    destination: Path,
    transaction: _Transaction,
    error: Exception,
) -> PublicationError:
    rollback_errors = _rollback(destination, transaction)
    recovery = _recovery_directory(transaction) if rollback_errors else None
    _record_failure(
        transaction.control,
        transaction.identifier,
        error,
        rollback_errors=rollback_errors,
        recovery_directory=recovery.name if recovery is not None else None,
    )
    message = "artifact publication failed"
    if recovery is None:
        _remove_transaction(transaction)
    else:
        message += f"; rollback incomplete, recovery files retained at {recovery}"
    return PublicationError(
        message, recovery_directory=recovery, rollback_errors=rollback_errors
    )


def _recovery_directory(transaction: _Transaction) -> Path | None:
    """The retained directory holding originals, else the staged outputs."""
    for path in (transaction.backups, transaction.stage):
        if path in transaction.created:
            return path
    return None


def _record_failure(
    control: Path,
    identifier: str,
    error: Exception,
    *,
    rollback_errors: Sequence[Exception] = (),
    recovery_directory: str | None = None,
) -> None:
    """Append one fixed-field line to the best-effort failure ledger."""
    record = {
        "rollback_failed": bool(rollback_errors),
        "error_code": _error_code(error),
        "execution_id": identifier,
        "published": False,
        "status": "failed",
    }
    if recovery_directory is not None:
        record["recovery_directory"] = recovery_directory
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with (control / _FAILURE_LOG).open("ab") as stream:
            stream.write(line.encode())
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        pass


def _remove_transaction(transaction: _Transaction) -> None:
    for path in transaction.created:
        try:
            shutil.rmtree(path)
        except OSError:
            pass
