# Local Publication Simplification Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` with the preserved inline execution method and one independent review after this increment. Steps use checkbox (`- [x]`) syntax.

**Goal:** Make scientific output publication understandable using ordinary paths, one writer lock, staging, replacement, and explicit rollback.

**Architecture:** Validate the requested artifact set before writing and again under the destination lock. Stage files on the destination filesystem, move existing outputs into a backup directory, and promote staged files with `os.replace`. Record completed moves so ordinary exceptions can restore prior files. If rollback fails, preserve the original transaction directory and report its location directly; do not copy recovery data into a second hierarchy.

**Tech Stack:** Python >=3.12, pathlib, tempfile-style staging directories, os.replace, existing filelock, pytest, uv. No new dependencies.

**Spec:** [Approved scientific-readability design](../specs/2026-09-28-scientific-readability-design.md).

**Status:** Implemented after `a9f659b`; final verification is recorded in `docs/validation/scientific-readability-publication-2026-09-28.md`. The preceding lifecycle increment passed 856 tests.

## Global constraints

- Continue on `refactor/scientific-readability`. Keep main at `57acff5`, stash `c4cfbec5fccd5397ea650c550f5b485ea588c2cd`, and unrelated untracked files unchanged.
- Use `uv run` for Python, tests, and formatting. Keep every `__init__.py` empty.
- Write, observe, and commit failing tests before implementation. Alter existing tests only for the contract changes stated here or demonstrated test errors.
- Preserve artifact bytes, returned artifact ordering, source files, and scientific numerical results. Never delete the last recoverable original during cleanup.
- Scope: a local destination used by cooperating writers. Existing symlinks, unsafe relative paths, collisions, and source overlap are rejected. Another process actively swapping directories or bypassing the writer lock is outside this contract.
- Individual file replacements are atomic; an artifact set is not an atomic snapshot. Do not promise crash recovery, power-loss durability, or consistent concurrent-reader snapshots.
- Keep BIDS projection unchanged. Publication treats bytes as artifacts and validates existing JSON/JSONL/TSV formats; it does not reinterpret scientific provenance.
- Keep case-folded artifact and existing-path collision checks: they prevent ordinary mistakes across filesystem case conventions, independently of descriptor-based race defenses.

## Concrete API and behavior

Preserve `Artifact(path, payload)` and:

```python
publish_artifact_set(
    destination, artifacts, *, source_paths=(), overwrite=False, lock_timeout=30.0
) -> tuple[Path, ...]
```

Remove `retain_incomplete`. No repository application uses it; only publication tests and documentation mention it. The removed option regenerated the complete failed artifact set and rewrote canonical provenance. Successful rollback now discards staged outputs. Failed rollback always retains its transaction directory, irrespective of user options.

`PublicationError` remains a `RuntimeError` and gains two public attributes:

```python
class PublicationError(RuntimeError):
    def __init__(self, message, *, recovery_directory=None, rollback_errors=()):
        super().__init__(message)
        self.recovery_directory = recovery_directory  # Path or None
        self.rollback_errors = tuple(rollback_errors)
```

A staging, backup, or promotion failure raises `PublicationError` chained from the original exception. If rollback also fails, `recovery_directory` names the retained absolute transaction directory and `rollback_errors` preserves the rollback exceptions. The caller-facing message includes that directory. Original files remain in `backups/<artifact path>` when restoration fails; a failed rollback can leave partial new outputs at the destination. There is no automatic recovery command or second recovery-copy tree.

Invalid arguments and preflight collisions continue to raise `ValueError`, `TypeError`, or `FileExistsError` before publication. Lock timeouts raise `PublicationError` with no recovery directory. `lock_timeout` must be finite and non-negative; booleans remain invalid. Destination aliases containing `..` are normalized for source-overlap checks, after rejecting existing symlink components.

The best-effort `.boldtailor/publication_failures.jsonl` ledger keeps `execution_id`, `status="failed"`, and `published=false`. Replace `error_type` and `message` with the existing logging boundary's fixed `error_code`. Add `rollback_failed` and an optional destination-relative `recovery_directory`; do not export raw exception text, custom class names, or absolute recovery paths. Detailed causes remain on exceptions. A failure to append diagnostics must not remove backups or mask the original publication failure.

Successful publication and successful rollback clean up their transaction directory on a best-effort basis. Empty output directories may remain. Orphan transaction directories from interrupted processes are left for manual inspection, never automatically deleted by a subsequent writer.

## File map and interfaces

- `src/boldtailor/publication.py`: keep artifact/metadata validation and FileLock; simplify transaction state and promotion/rollback; remove descriptor and recovery-copy machinery.
- `src/boldtailor/logging.py`: reuse `_error_code(error)` without changing its category contract.
- `tests/test_publication.py`: ordinary containment, source protection, boundary-failure injection, retained recovery directory, categorical failure records, and real-process writers.
- `examples/NSD/test_nsd_single_trial.py`, `examples/NSD/test_nsd_hrf_selection.py`: retain real workflow rollback tests unchanged unless their existing injection assumes obsolete internal move counts.
- `docs/api.md`, `docs/development.md`, new `docs/publication-migration.md`, `docs/README.md`: current signature, recovery procedure, guarantees, and removed option.
- Roadmap/spec and new `docs/validation/scientific-readability-publication-2026-09-28.md`: implementation status and evidence.

Private transaction interface:

```python
@dataclass
class _Transaction:
    root: Path
    backed_up: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)

    @property
    def stage(self):
        return self.root / "stage"

    @property
    def backups(self):
        return self.root / "backups"
```

`_stage_artifacts(transaction, artifacts)` creates/writes/fsyncs staged files. `_backup_existing(destination, transaction, artifacts)` moves existing files to backups and records each successful move. `_promote_staged(destination, transaction, artifacts)` records each successful promotion. `_rollback(destination, transaction)` returns a tuple of exceptions after attempting every necessary removal/restoration. `_record_failure(control, identifier, error, *, rollback_errors=(), recovery_directory=None)` writes only fixed metadata; `recovery_directory` is a relative string in that record.

## Review focus

1. Source/output aliases and case-folded control paths must fail before changing inputs (Task 1).
2. Failure at every backup and promotion boundary must restore originals and remove newly created artifact files (Task 2).
3. Rollback failure, diagnostic-write failure, and cleanup failure must not discard the last original or replace the original error cause (Task 2).
4. Two real cooperating processes must serialize complete attempts, recheck collisions under lock, and honor finite timeouts without deadlock (Tasks 1–2).
5. Exported failure records must not reveal arbitrary exception text or custom class names; detailed exceptions and recovery location must remain useful to callers (Task 3).

## Task 1: Pin ordinary path and timeout safety

- [x] Add these tests to `tests/test_publication.py`; retain the existing artifact, symlink, collision, and source-read-only tests.

```python
@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf")])
def test_publication_rejects_nonfinite_timeout_before_writing(tmp_path, timeout):
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="lock_timeout"):
        publish_artifact_set(destination, [Artifact("a.bin", b"a")], lock_timeout=timeout)
    assert not destination.exists()


def test_source_overlap_through_destination_parent_alias_is_rejected(tmp_path):
    (tmp_path / "unused").mkdir()
    source = tmp_path / "signal.bin"
    source.write_bytes(b"original-input")
    with pytest.raises(ValueError, match="overlap"):
        publish_artifact_set(
            tmp_path / "unused" / "..", [Artifact("signal.bin", b"replacement")],
            source_paths=[source], overwrite=True,
        )
    assert source.read_bytes() == b"original-input"


@pytest.mark.parametrize("name", [".boldtailor", ".BOLDTAILOR", ".BoldTailor"])
def test_control_directory_names_are_reserved_case_insensitively(tmp_path, name):
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="reserved"):
        publish_artifact_set(destination, [Artifact(f"{name}/publication.lock", b"bad")])
    assert not destination.exists()
```

- [x] Run `uv run pytest tests/test_publication.py -q -W error`; inspect failures and commit tests before source edits. Avoid actually waiting on NaN/infinite lock values: validation tests use uncontended nonexistent destinations, so current code completes or rejects rather than blocking.
- [x] Check `math.isfinite(timeout)` in `_validate_lock_timeout`. Compare the artifact's first path component using `casefold()`. Normalize the destination with `resolve(strict=False)` only after checking existing symlink components; keep both unlocked and locked preflight checks. Normalize target paths consistently for source overlap.

```python
root = Path(destination).absolute()
_require_safe_destination(root)
root = root.resolve(strict=False)
```

- [x] Run the publication suite, format, document finite-timeout and alias semantics, and commit GREEN. Include the real-process writer and timeout tests in this run.

## Task 2: Ordinary-path publication and retained rollback backups

- [x] Add a test proving publication works without descriptor-relative platform capabilities:

```python
def test_publication_works_without_descriptor_relative_operations(tmp_path, monkeypatch):
    import boldtailor.publication as publication

    monkeypatch.setattr(publication.os, "supports_dir_fd", set())
    destination = tmp_path / "output"
    paths = publish_artifact_set(destination, [Artifact("nested/a.bin", b"a")])
    assert paths == (destination / "nested/a.bin",)
    assert paths[0].read_bytes() == b"a"
    _assert_no_transaction_debris(destination)
```

- [x] Replace the two restore-fallback/recovery-copy tests with direct retained-backup tests under the new contract. Use this main case and parameterize `block_diagnostics` to cover a ledger that cannot be written:

```python
@pytest.mark.parametrize("block_diagnostics", [False, True])
def test_failed_restore_retains_original_and_reports_recovery(
    tmp_path, monkeypatch, block_diagnostics
):
    import boldtailor.publication as publication

    destination = tmp_path / "output"
    destination.mkdir()
    original = destination / "old.bin"
    original.write_bytes(b"only-original")
    promotion_error = OSError("private promotion detail")
    restore_error = OSError("private restore detail")
    real_replace = publication.os.replace

    def fail_promotion_and_restore(source, target):
        source, target = Path(source), Path(target)
        if "backups" in source.parts:
            raise restore_error
        if "stage" in source.parts and target.name == "new.bin":
            raise promotion_error
        return real_replace(source, target)

    monkeypatch.setattr(publication.os, "replace", fail_promotion_and_restore)
    if block_diagnostics:
        (destination / ".boldtailor" / "publication_failures.jsonl").mkdir(parents=True)
    with pytest.raises(PublicationError) as caught:
        publish_artifact_set(destination, [Artifact("old.bin", b"new-old"),
                                          Artifact("new.bin", b"new")], overwrite=True)
    error = caught.value
    assert error.__cause__ is promotion_error
    assert error.rollback_errors == (restore_error,)
    recovery = error.recovery_directory
    assert recovery is not None and recovery.is_dir()
    assert str(recovery) in str(error)
    assert (recovery / "backups" / "old.bin").read_bytes() == b"only-original"
    assert not (destination / "new.bin").exists()
```

Import `Path` from pathlib in the test file. This injection identifies staging/backup operations by their paths instead of requiring a particular number of syscalls.

- [x] Retain the five existing backup/promotion boundary failures and the real-process tests. Replace the old `retain_incomplete` test with the explicit removed-argument behavior below. Remove the active adversarial parent-swap test: the approved local/cooperating-writer scope no longer promises protection against a swap inside `os.replace`. Keep all static symlink rejection tests.

```python
def test_removed_retention_option_is_rejected_before_writing(tmp_path):
    destination = tmp_path / "output"
    with pytest.raises(TypeError, match="retain_incomplete"):
        publish_artifact_set(destination, [Artifact("a.bin", b"a")], retain_incomplete=True)
    assert not destination.exists()
```

- [x] Add staging-write and cleanup-failure cases. For staging, patch `Path.open` only for a transaction `stage` path to raise a sentinel `OSError`, verify all originals untouched, `PublicationError.__cause__` is the sentinel, and no promoted file. For cleanup, patch `shutil.rmtree` to leave the scratch directory, verify a successful publication still returns correct artifact bytes; restore the patch before pytest fixture cleanup. Reuse the current real artifact set rather than mocking publication itself.

```python
@pytest.mark.parametrize("boundary", ["write", "stage_directory"])
def test_staging_failure_preserves_original_and_cause(tmp_path, monkeypatch, boundary):
    destination = tmp_path / "output"
    destination.mkdir()
    original = destination / "old.bin"
    original.write_bytes(b"original")
    failure = OSError("private staging failure")
    method = "open" if boundary == "write" else "mkdir"
    real_method = getattr(Path, method)

    def fail_stage(path, *args, **kwargs):
        if "transactions" in path.parts and "stage" in path.parts:
            raise failure
        return real_method(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, fail_stage)
    with pytest.raises(PublicationError) as caught:
        publish_artifact_set(destination, [Artifact("old.bin", b"new")], overwrite=True)
    assert caught.value.__cause__ is failure
    assert caught.value.recovery_directory is None
    assert not caught.value.rollback_errors
    assert original.read_bytes() == b"original"
    _assert_no_transaction_debris(destination)


@pytest.mark.parametrize("fail_publication", [False, True])
def test_cleanup_failure_does_not_change_publication_outcome(
    tmp_path, monkeypatch, fail_publication
):
    import boldtailor.publication as publication

    destination = tmp_path / "output"
    failure = OSError("original operation failure")

    def reject_cleanup(*args, **kwargs):
        raise OSError("cleanup failure")

    def reject_replace(*args, **kwargs):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(publication.shutil, "rmtree", reject_cleanup)
        if fail_publication:
            patch.setattr(publication.os, "replace", reject_replace)
            with pytest.raises(PublicationError) as caught:
                publish_artifact_set(destination, [Artifact("a.bin", b"a")])
            assert caught.value.__cause__ is failure
            assert not (destination / "a.bin").exists()
        else:
            paths = publish_artifact_set(destination, [Artifact("a.bin", b"a")])
            assert paths[0].read_bytes() == b"a"
```

- [x] Run the publication suite, inspect RED failures, commit tests.
- [x] Replace `_AnchoredEntry`, `_AnchoredPath`, capability probing, descriptor opening/identity checks, anchored replace/unlink, and descriptor fsync with the transaction interface above and ordinary `Path` operations. Remove `fcntl`, `stat`, `_require_anchored_operations`, descriptor-path probing, restore-rename fallback, recovery copying, failed-artifact regeneration, and `_failed_payload`.

```python
# Backup: record each successful move before attempting the next file.
backup = transaction.backups / artifact.path
backup.parent.mkdir(parents=True, exist_ok=True)
os.replace(destination / artifact.path, backup)
transaction.backed_up.append(artifact.path)

# Promotion: stage and destination share a filesystem.
target = destination / artifact.path
target.parent.mkdir(parents=True, exist_ok=True)
os.replace(transaction.stage / artifact.path, target)
transaction.promoted.append(artifact.path)
```

- [x] Implement rollback without a second fallback protocol:

```python
def _rollback(destination, transaction):
    errors = []
    for path in reversed(transaction.promoted):
        if path in transaction.backed_up:
            continue
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
```

Keep best-effort directory fsync after successful moves/removals. When constructing a transaction, choose its UUID/root in memory first and create stage/backups inside the protected try block, so staging-directory failures are handled too. Reject existing symlinks in the control/transaction directories before using them. Never scan/delete old transaction directories.

- [x] In `_publish_locked`, catch an operation failure once, attempt rollback once, append diagnostic metadata, clean up only if rollback succeeded, and raise `PublicationError` from the original error. Retain the transaction in place if any rollback operation failed. Diagnostics and cleanup are best effort; ordinary filesystem errors in them do not mask the operation error. On success, clean up best effort and return only artifact paths, in input order.
- [x] Preserve preflight, metadata validation, one persistent `FileLock`, lock timeout behavior, per-file fsync, and existing payload bytes. Extend `_record_failure` with recovery metadata, leaving the error-field migration to Task 3.
- [x] Run `uv run pytest tests/test_publication.py examples/NSD/test_nsd_single_trial.py examples/NSD/test_nsd_hrf_selection.py -q -W error`; run real-process tests with the existing required permissions. Update API/developer/migration documentation, format, and commit GREEN.

## Task 3: Fixed publication failure categories

- [x] Update the existing failure-ledger expectation from `error_type="OSError"` to `error_code="io_failure"`. Add a custom exception with a raising `__str__` so diagnostics cannot inspect its message/class name:

```python
def test_publication_failure_ledger_omits_exception_names_and_text(tmp_path, monkeypatch):
    import boldtailor.publication as publication

    class PrivatePatientError(ValueError):
        def __str__(self):
            raise AssertionError("do not format this exception")

    failure = PrivatePatientError()

    def reject_replace(*args, **kwargs):
        raise failure

    monkeypatch.setattr(publication.os, "replace", reject_replace)
    destination = tmp_path / "output"
    with pytest.raises(PublicationError) as caught:
        publish_artifact_set(destination, [Artifact("a.bin", b"a")])
    assert caught.value.__cause__ is failure
    records = _failure_records(destination)
    assert len(records) == 1
    assert records[0]["error_code"] == "invalid_input"
    assert records[0]["rollback_failed"] is False
    assert "error_type" not in records[0] and "message" not in records[0]
    assert "PrivatePatientError" not in json.dumps(records)
```

- [x] Extend the retained-recovery test with these assertions after its recovery-file checks. Keep its raw exceptions and absolute caller-facing recovery path intact.

```python
if not block_diagnostics:
    record = _failure_records(destination)[0]
    assert record["rollback_failed"] is True
    assert record["recovery_directory"] == recovery.relative_to(destination).as_posix()
    assert not Path(record["recovery_directory"]).is_absolute()
    assert str(tmp_path) not in json.dumps(record)
```
- [x] Run publication tests, observe RED, commit tests. Replace `_sanitized_message` and `type(error).__name__` with `boldtailor.logging._error_code(error)`. Keep the fixed status fields; do not put raw rollback exceptions into JSON.
- [x] Run publication tests including lock timeout and concurrent writers, update the migration guide's ledger example, format, and commit GREEN.

## Verification and handoff

Update the documentation index, roadmap, and spec status. Validate current local documentation links and execute the API's publication example. Record RED/GREEN commits and any execution decisions in the publication validation document.

Run default `uv run pytest -q -W error`, standard Black scope (`src tests examples/NSD examples/stop_signal_demo.py`), `git diff --check`, wheel construction, and isolated installed-wheel smoke. Request one fresh reviewer of the increment, resolve Important/Critical findings with committed RED-first tests and a final full suite. Keep this branch unmerged.

After this increment, the remaining architecture work is packaged imaging helpers, simpler notebooks, and the final review of the whole branch. The original review's phrase “atomic sets” is not adopted: the approved design explicitly promises individual replacements and recoverable-exception rollback only.
