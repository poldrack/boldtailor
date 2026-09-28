from __future__ import annotations

from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from uuid import uuid4

try:
    import fcntl
except ImportError:  # pragma: no cover - conservative non-POSIX fallback
    fcntl = None

from filelock import FileLock, Timeout

_CONTROL_DIRECTORY = ".boldtailor"
_FAILURE_LOG = "publication_failures.jsonl"
_PROVENANCE_PATH = "logs/boldtailor_provenance.json"


class PublicationError(RuntimeError):
    """A transactional publication attempt could not be completed."""


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
    identifier: str
    root: Path
    stage: Path
    backups: Path
    backed_up: list[_AnchoredEntry]
    promoted: list[_AnchoredEntry]
    unrestored: list[str]


@dataclass(slots=True)
class _AnchoredEntry:
    root: Path
    path: str
    descriptor: int
    identity: tuple[int, int]

    @property
    def name(self) -> str:
        return PurePosixPath(self.path).name


@dataclass(frozen=True, slots=True)
class _AnchoredPath(os.PathLike[str]):
    entry: _AnchoredEntry

    def __fspath__(self) -> str:
        return str(_descriptor_current_path(self.entry.descriptor) / self.entry.name)

    def __str__(self) -> str:
        return self.__fspath__()


def publish_artifact_set(
    destination: str | os.PathLike[str],
    artifacts: Iterable[Artifact],
    *,
    source_paths: Iterable[str | os.PathLike[str]] = (),
    overwrite: bool = False,
    retain_incomplete: bool = False,
    lock_timeout: float = 30.0,
) -> tuple[Path, ...]:
    """Publish with a writer lock, per-file replacement, and failure rollback.

    The complete set is not an atomic snapshot for concurrent readers.
    """
    requested = _prepare_artifacts(artifacts)
    sources = tuple(Path(path) for path in source_paths)
    root = Path(destination).absolute()
    _validate_lock_timeout(lock_timeout)
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
                retain_incomplete=retain_incomplete,
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
    retain_incomplete: bool,
) -> tuple[Path, ...]:
    _preflight(destination, artifacts, source_paths, overwrite=overwrite)
    transaction = _new_transaction(control)
    try:
        _stage_artifacts(transaction, artifacts)
        _promote_artifacts(destination, transaction, artifacts, overwrite=overwrite)
    except Exception as error:
        _handle_failure(control, transaction, artifacts, error, retain_incomplete)
        raise PublicationError("artifact publication failed") from error
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
        if PurePosixPath(artifact.path).parts[0] == _CONTROL_DIRECTORY:
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
        or timeout < 0
    ):
        raise ValueError("lock_timeout must be a non-negative number")


def _preflight(
    destination: Path,
    artifacts: Sequence[Artifact],
    source_paths: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    _require_safe_destination(destination)
    targets = tuple(destination / artifact.path for artifact in artifacts)
    _reject_source_overlap(targets, source_paths)
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


def _new_transaction(control: Path) -> _Transaction:
    identifier = str(uuid4())
    transactions = control / "transactions"
    _safe_mkdir(transactions, control)
    root = transactions / identifier
    root.mkdir(exist_ok=False)
    stage = root / "stage"
    backups = root / "backups"
    stage.mkdir()
    backups.mkdir()
    return _Transaction(identifier, root, stage, backups, [], [], [])


def _stage_artifacts(transaction: _Transaction, artifacts: Sequence[Artifact]) -> None:
    for artifact in artifacts:
        staged = transaction.stage / artifact.path
        staged.parent.mkdir(parents=True, exist_ok=True)
        with staged.open("xb") as stream:
            stream.write(artifact.payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staged.parent)


def _require_anchored_operations() -> None:
    required = (os.open, os.mkdir, os.stat, os.unlink)
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or not all(operation in os.supports_dir_fd for operation in required)
        or not _has_descriptor_path_support()
    ):
        raise PublicationError("platform lacks safe no-follow publication support")


def _has_descriptor_path_support() -> bool:
    return _has_f_getpath() or Path("/proc/self/fd").is_dir()


def _has_f_getpath() -> bool:
    return fcntl is not None and hasattr(fcntl, "F_GETPATH")


def _descriptor_current_path(descriptor: int) -> Path:
    if _has_f_getpath():
        buffer = fcntl.fcntl(descriptor, fcntl.F_GETPATH, b"\0" * 1024)
        return Path(os.fsdecode(buffer.split(b"\0", 1)[0]))
    proc_path = Path("/proc/self/fd") / str(descriptor)
    if proc_path.is_dir():
        return proc_path
    raise PublicationError("directory descriptor paths are unavailable")


def _open_anchored_entry(
    root: Path,
    path: str,
    *,
    create: bool = False,
) -> _AnchoredEntry:
    parts = PurePosixPath(path).parts
    descriptor = _open_directory(root)
    try:
        for part in parts[:-1]:
            child = _open_child_directory(descriptor, part, create=create)
            os.close(descriptor)
            descriptor = child
        identity = _descriptor_identity(descriptor)
        return _AnchoredEntry(root, path, descriptor, identity)
    except Exception:
        os.close(descriptor)
        raise


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"publication path is not a directory: {path}")
    return descriptor


def _open_child_directory(parent: int, name: str, *, create: bool) -> int:
    if create:
        try:
            os.mkdir(name, dir_fd=parent)
        except FileExistsError:
            pass
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent)


def _descriptor_identity(descriptor: int) -> tuple[int, int]:
    details = os.fstat(descriptor)
    return details.st_dev, details.st_ino


def _entry_is_current(entry: _AnchoredEntry) -> bool:
    try:
        current = _open_anchored_entry(entry.root, entry.path)
    except OSError:
        return False
    try:
        return current.identity == entry.identity
    finally:
        _close_entry(current)


def _anchored_path(entry: _AnchoredEntry) -> _AnchoredPath:
    return _AnchoredPath(entry)


def _anchored_replace(source: _AnchoredEntry, target: _AnchoredEntry) -> None:
    if not _entry_is_current(target):
        raise PublicationError("destination identity changed before replacement")
    os.replace(_anchored_path(source), _anchored_path(target))
    _fsync_descriptor(target.descriptor)


def _anchored_rename(source: _AnchoredEntry, target: _AnchoredEntry) -> None:
    if not _entry_is_current(target):
        raise OSError("destination identity changed before restore fallback")
    os.rename(_anchored_path(source), _anchored_path(target))
    _fsync_descriptor(target.descriptor)


def _close_entry(entry: _AnchoredEntry | None) -> None:
    if entry is not None and entry.descriptor >= 0:
        os.close(entry.descriptor)
        entry.descriptor = -1


def _entry_exists(entry: _AnchoredEntry) -> bool:
    try:
        os.stat(entry.name, dir_fd=entry.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _existing_destination_entry(root: Path, path: str) -> _AnchoredEntry | None:
    try:
        entry = _open_anchored_entry(root, path)
    except FileNotFoundError:
        return None
    if not _entry_exists(entry):
        _close_entry(entry)
        return None
    details = os.stat(entry.name, dir_fd=entry.descriptor, follow_symlinks=False)
    if not stat.S_ISREG(details.st_mode):
        _close_entry(entry)
        raise ValueError(f"artifact destination is not a regular file: {path}")
    return entry


def _promote_artifacts(
    destination: Path,
    transaction: _Transaction,
    artifacts: Sequence[Artifact],
    *,
    overwrite: bool,
) -> None:
    try:
        _require_anchored_operations()
        if overwrite:
            _backup_existing(destination, transaction, artifacts)
        _promote_staged(destination, transaction, artifacts)
    except Exception:
        _rollback(transaction)
        raise


def _backup_existing(
    destination: Path,
    transaction: _Transaction,
    artifacts: Sequence[Artifact],
) -> None:
    for artifact in artifacts:
        target_entry = _existing_destination_entry(destination, artifact.path)
        if target_entry is None:
            continue
        backup_entry = _open_anchored_entry(
            transaction.backups,
            artifact.path,
            create=True,
        )
        try:
            _anchored_replace(target_entry, backup_entry)
            transaction.backed_up.append(target_entry)
            target_entry = None
            if not _entry_is_current(transaction.backed_up[-1]):
                raise PublicationError("destination identity changed during backup")
        finally:
            _close_entry(target_entry)
            _close_entry(backup_entry)


def _promote_staged(
    destination: Path,
    transaction: _Transaction,
    artifacts: Sequence[Artifact],
) -> None:
    for artifact in artifacts:
        staged_entry = _open_anchored_entry(transaction.stage, artifact.path)
        target_entry = _open_anchored_entry(
            destination,
            artifact.path,
            create=True,
        )
        try:
            _anchored_replace(staged_entry, target_entry)
            transaction.promoted.append(target_entry)
            target_entry = None
            if not _entry_is_current(transaction.promoted[-1]):
                raise PublicationError("destination identity changed during promotion")
        finally:
            _close_entry(staged_entry)
            _close_entry(target_entry)


def _rollback(transaction: _Transaction) -> None:
    rollback_errors = []
    for promoted in reversed(transaction.promoted):
        try:
            _unlink_anchored(promoted)
        except OSError as error:
            rollback_errors.append(error)
        finally:
            _close_entry(promoted)
    for target in reversed(transaction.backed_up):
        try:
            _restore_backup(transaction, target)
        except Exception as error:
            rollback_errors.append(error)
            transaction.unrestored.append(target.path)
        finally:
            _close_entry(target)
    if rollback_errors:
        raise PublicationError("artifact rollback failed") from rollback_errors[0]


def _unlink_anchored(entry: _AnchoredEntry) -> None:
    try:
        os.unlink(entry.name, dir_fd=entry.descriptor)
    except FileNotFoundError:
        return
    _fsync_descriptor(entry.descriptor)


def _restore_backup(transaction: _Transaction, target: _AnchoredEntry) -> None:
    backup = _open_anchored_entry(transaction.backups, target.path)
    try:
        if _entry_exists(target):
            raise FileExistsError(f"rollback target is occupied: {target.path}")
        try:
            _anchored_replace(backup, target)
        except OSError:
            _anchored_rename(backup, target)
    finally:
        _close_entry(backup)


def _handle_failure(
    control: Path,
    transaction: _Transaction,
    artifacts: Sequence[Artifact],
    error: Exception,
    retain_incomplete: bool,
) -> None:
    recovery_safe = True
    if transaction.unrestored:
        recovery_safe = _retain_recovery(control, transaction)
    if retain_incomplete:
        _retain_failed_artifacts(control, transaction.identifier, artifacts)
    _record_failure(control, transaction.identifier, error)
    if recovery_safe:
        _remove_transaction(transaction)


def _retain_recovery(control: Path, transaction: _Transaction) -> bool:
    failed = control / "failed" / transaction.identifier
    recovery = failed / "recovery"
    try:
        _safe_mkdir(recovery, control)
        for path in transaction.unrestored:
            source = transaction.backups / path
            target = recovery / path
            _copy_recovery_file(source, target)
        _write_recovery_record(failed, transaction)
        _fsync_directory(recovery)
    except (OSError, ValueError):
        return False
    return True


def _copy_recovery_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    source_descriptor = os.open(source, flags)
    try:
        with os.fdopen(source_descriptor, "rb", closefd=False) as source_stream:
            with target.open("xb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
                target_stream.flush()
                os.fsync(target_stream.fileno())
    finally:
        os.close(source_descriptor)


def _write_recovery_record(failed: Path, transaction: _Transaction) -> None:
    record = {
        "execution_id": transaction.identifier,
        "published": False,
        "status": "failed",
    }
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    with (failed / "recovery.json").open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _retain_failed_artifacts(
    control: Path,
    identifier: str,
    artifacts: Sequence[Artifact],
) -> None:
    root = control / "failed" / identifier / "artifacts"
    _safe_mkdir(root, control)
    for artifact in artifacts:
        target = root / artifact.path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _failed_payload(artifact, identifier)
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    _fsync_directory(root)


def _failed_payload(artifact: Artifact, identifier: str) -> bytes:
    if artifact.path != _PROVENANCE_PATH:
        return artifact.payload
    provenance = json.loads(artifact.payload)
    if not isinstance(provenance, dict):
        return artifact.payload
    provenance["publication"] = {
        "execution_id": identifier,
        "published": False,
        "status": "failed",
    }
    return (
        json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _record_failure(control: Path, identifier: str, error: Exception) -> None:
    record = {
        "error_type": type(error).__name__,
        "execution_id": identifier,
        "message": _sanitized_message(error),
        "published": False,
        "status": "failed",
    }
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


def _sanitized_message(error: Exception) -> str:
    if isinstance(error, Timeout):
        return "publication lock acquisition timed out"
    if isinstance(error, OSError):
        return "operating system publication failure"
    return "publication operation failed"


def _remove_transaction(transaction: _Transaction) -> None:
    for entry in (*transaction.promoted, *transaction.backed_up):
        _close_entry(entry)
    if transaction.root.is_symlink():
        return
    shutil.rmtree(transaction.root, ignore_errors=True)
    parent = transaction.root.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


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


def _fsync_descriptor(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError:
        pass
