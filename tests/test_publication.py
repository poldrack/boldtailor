import json
import multiprocessing
from pathlib import Path
import stat
import time
from uuid import UUID

import pytest

from boldtailor.publication import Artifact, PublicationError, publish_artifact_set


def _publication_worker(destination, marker, start, results):
    import boldtailor.publication as publication

    original_replace = publication.os.replace

    def delayed_replace(source, target):
        original_replace(source, target)
        time.sleep(0.08)

    publication.os.replace = delayed_replace
    start.wait()
    try:
        published = publication.publish_artifact_set(
            destination,
            (
                publication.Artifact("first.bin", marker.encode()),
                publication.Artifact("nested/second.bin", marker.encode()),
            ),
            overwrite=True,
        )
    except Exception as error:  # pragma: no cover - reported in the parent
        results.put(("error", type(error).__name__, str(error)))
    else:
        results.put(("ok", tuple(str(path) for path in published)))


def _lock_holder_worker(destination, acquired, results):
    import boldtailor.publication as publication

    original_replace = publication.os.replace

    def delayed_replace(source, target):
        acquired.set()
        time.sleep(0.5)
        original_replace(source, target)

    publication.os.replace = delayed_replace
    try:
        publication.publish_artifact_set(
            destination,
            (publication.Artifact("held.bin", b"holder"),),
        )
    except Exception as error:  # pragma: no cover - reported in the parent
        results.put(("error", type(error).__name__, str(error)))
    else:
        results.put(("ok",))


@pytest.fixture
def artifact_set():
    return (
        Artifact("dataset_description.json", b'{"Name":"example"}\n'),
        Artifact("logs/events.jsonl", b'{"event":"started"}\n'),
        Artifact("tables/results.tsv", b"name\tvalue\na\t1\n"),
        Artifact("arrays/effect.bin", b"\x00\x01\x02"),
    )


def _assert_no_transaction_debris(destination):
    transactions = destination / ".boldtailor" / "transactions"
    assert not transactions.exists() or not tuple(transactions.iterdir())


def _failure_records(destination):
    logs = tuple((destination / ".boldtailor").glob("*.jsonl"))
    assert len(logs) == 1
    return [json.loads(line) for line in logs[0].read_text().splitlines()]


def _join_process_or_fail(process, timeout=10):
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail(f"worker process {process.pid} did not exit within {timeout}s")


def test_artifact_owns_payload_bytes_and_validates_relative_posix_path():
    mutable = bytearray(b"initial")

    artifact = Artifact("nested/result.bin", mutable)
    mutable[:] = b"changed"

    assert artifact.path == "nested/result.bin"
    assert artifact.payload == b"initial"
    assert isinstance(artifact.payload, bytes)


@pytest.mark.parametrize(
    "path",
    (
        "",
        ".",
        "..",
        "../escape.bin",
        "nested/../escape.bin",
        "/absolute.bin",
        "nested//result.bin",
        "nested\\result.bin",
        "nested/./result.bin",
        "nested/result.bin/",
        "nested/\x00result.bin",
    ),
)
def test_artifact_rejects_paths_that_are_not_relative_normalized_posix(path):
    with pytest.raises(ValueError, match="relative POSIX path"):
        Artifact(path, b"payload")


def _artifacts(*paths):
    return tuple(Artifact(path, b"{}\n") for path in paths)


@pytest.mark.parametrize(
    ("artifacts", "kwargs", "message"),
    (
        ((), {}, "at least one artifact"),
        (_artifacts("result.json", "result.json"), {}, "duplicate artifact path"),
        (
            _artifacts("Sub-01/result.json", "sub-01/RESULT.JSON"),
            {},
            "duplicate artifact path",
        ),
        (_artifacts("a.bin"), {"lock_timeout": float("nan")}, "lock_timeout"),
        (_artifacts("a.bin"), {"lock_timeout": float("inf")}, "lock_timeout"),
        (_artifacts("a.bin"), {"lock_timeout": -float("inf")}, "lock_timeout"),
        (_artifacts(".boldtailor/publication.lock"), {}, "reserved"),
        (_artifacts(".BOLDTAILOR/publication.lock"), {}, "reserved"),
        (_artifacts(".BoldTailor/publication.lock"), {}, "reserved"),
    ),
)
def test_preflight_rejects_invalid_requests_before_writing(
    tmp_path, artifacts, kwargs, message
):
    destination = tmp_path / "derivatives"

    with pytest.raises(ValueError, match=message):
        publish_artifact_set(destination, artifacts, **kwargs)

    assert not destination.exists()


@pytest.mark.parametrize(
    ("path", "payload"),
    (
        ("broken.json", b'{"missing":'),
        ("broken.json", b"\xff"),
        ("broken.jsonl", b'{"valid":true}\nnot-json\n'),
        ("broken.tsv", b"name\tvalue\na\n"),
        ("broken.tsv", b"name\tvalue\na\t\xff\n"),
    ),
)
def test_preflight_rejects_invalid_metadata_before_writing(tmp_path, path, payload):
    destination = tmp_path / "derivatives"

    with pytest.raises(ValueError, match="metadata"):
        publish_artifact_set(destination, (Artifact(path, payload),))

    assert not (destination / path).exists()
    _assert_no_transaction_debris(destination)


def test_publication_writes_real_files_and_returns_only_artifact_paths(
    tmp_path,
    artifact_set,
):
    destination = tmp_path / "derivatives"

    published = publish_artifact_set(destination, artifact_set)

    assert set(published) == {destination / artifact.path for artifact in artifact_set}
    assert all(path.is_file() for path in published)
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in published
    } == {artifact.path: artifact.payload for artifact in artifact_set}
    assert all(".boldtailor" not in path.parts for path in published)
    _assert_no_transaction_debris(destination)


def test_existing_collision_is_refused_before_any_artifact_is_replaced(tmp_path):
    destination = tmp_path / "derivatives"
    destination.mkdir()
    existing = destination / "existing.bin"
    existing.write_bytes(b"original")
    artifacts = (
        Artifact("new.bin", b"must-not-appear"),
        Artifact("existing.bin", b"replacement"),
    )

    with pytest.raises(FileExistsError, match="existing.bin"):
        publish_artifact_set(destination, artifacts)

    assert existing.read_bytes() == b"original"
    assert not (destination / "new.bin").exists()
    _assert_no_transaction_debris(destination)


@pytest.mark.parametrize("symlink_location", ("destination", "parent", "artifact"))
def test_preflight_never_follows_destination_symlinks(tmp_path, symlink_location):
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "derivatives"
    artifact_path = "nested/result.bin"
    if symlink_location == "destination":
        destination.symlink_to(outside, target_is_directory=True)
        artifact_path = "result.bin"
    elif symlink_location == "parent":
        destination.mkdir()
        (destination / "nested").symlink_to(outside, target_is_directory=True)
    else:
        (destination / "nested").mkdir(parents=True)
        (destination / artifact_path).symlink_to(outside / "result.bin")

    with pytest.raises(ValueError, match="symlink"):
        publish_artifact_set(
            destination,
            (Artifact(artifact_path, b"must-not-escape"),),
            overwrite=True,
        )

    assert not (outside / "result.bin").exists()


def test_preflight_rejects_source_output_overlap_without_modifying_source(tmp_path):
    destination = tmp_path / "derivatives"
    destination.mkdir()
    source = destination / "source.tsv"
    source.write_bytes(b"source\tdata\n")
    before = source.stat()

    with pytest.raises(ValueError, match="source.*overlap"):
        publish_artifact_set(
            destination,
            (Artifact("source.tsv", b"replacement\tdata\n"),),
            source_paths=(source,),
            overwrite=True,
        )

    after = source.stat()
    assert source.read_bytes() == b"source\tdata\n"
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert after.st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("failure_at", range(1, 6))
def test_failure_at_every_replace_boundary_restores_original_set(
    tmp_path,
    monkeypatch,
    failure_at,
):
    import boldtailor.publication as publication

    destination = tmp_path / "derivatives"
    destination.mkdir()
    first = destination / "first.bin"
    second = destination / "nested" / "second.bin"
    second.parent.mkdir()
    first.write_bytes(b"original-first")
    second.write_bytes(b"original-second")
    original_replace = publication.os.replace
    calls = 0

    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("promotion failed: secret=/Users/alice/private-input.tsv")
        original_replace(source, target)

    monkeypatch.setattr(publication.os, "replace", fail_once)
    artifacts = (
        Artifact("first.bin", b"replacement-first"),
        Artifact("nested/second.bin", b"replacement-second"),
        Artifact("new.bin", b"must-be-rolled-back"),
    )

    with pytest.raises(PublicationError, match="publication failed"):
        publish_artifact_set(destination, artifacts, overwrite=True)

    assert first.read_bytes() == b"original-first"
    assert second.read_bytes() == b"original-second"
    assert not (destination / "new.bin").exists()
    _assert_no_transaction_debris(destination)


def test_successful_overwrite_replaces_the_complete_set_and_cleans_debris(tmp_path):
    destination = tmp_path / "derivatives"
    (destination / "nested").mkdir(parents=True)
    (destination / "first.bin").write_bytes(b"old-first")
    (destination / "nested" / "second.bin").write_bytes(b"old-second")
    artifacts = (
        Artifact("first.bin", b"new-first"),
        Artifact("nested/second.bin", b"new-second"),
        Artifact("new.bin", b"new-file"),
    )

    published = publish_artifact_set(destination, artifacts, overwrite=True)

    assert {path.read_bytes() for path in published} == {
        b"new-first",
        b"new-second",
        b"new-file",
    }
    _assert_no_transaction_debris(destination)
    assert not tuple(destination.rglob("*.backup"))


def test_failure_log_is_single_jsonl_stream_and_redacts_sensitive_context(
    tmp_path,
    monkeypatch,
):
    import boldtailor.publication as publication

    destination = tmp_path / "derivatives-secret"
    source = tmp_path / "private-source.tsv"
    source.write_bytes(b"source")

    def fail_replace(source_path, target_path):
        raise OSError(
            f"cannot publish {source_path} to {target_path}; token=private-secret"
        )

    monkeypatch.setattr(publication.os, "replace", fail_replace)

    with pytest.raises(PublicationError):
        publish_artifact_set(
            destination,
            (Artifact("result.bin", b"result"),),
            source_paths=(source,),
        )

    records = _failure_records(destination)
    assert len(records) == 1
    assert records[0]["error_code"] == "io_failure"
    assert records[0]["status"] == "failed"
    assert records[0]["published"] is False
    serialized = json.dumps(records[0]).lower()
    for forbidden in (
        str(tmp_path).lower(),
        "private-source",
        "derivatives-secret",
        "private-secret",
        "traceback",
        "environment",
        "cwd",
    ):
        assert forbidden not in serialized


def test_source_files_are_read_only_inputs_on_success_and_failure(
    tmp_path, monkeypatch
):
    import boldtailor.publication as publication

    source = tmp_path / "raw" / "source.tsv"
    source.parent.mkdir()
    source.write_bytes(b"source\tdata\n")
    source.chmod(0o444)
    before = source.stat()
    success_destination = tmp_path / "success"

    publish_artifact_set(
        success_destination,
        (Artifact("result.bin", b"result"),),
        source_paths=(source,),
    )

    original_replace = publication.os.replace

    def fail_replace(source_path, target_path):
        raise OSError("injected failure")

    monkeypatch.setattr(publication.os, "replace", fail_replace)
    with pytest.raises(PublicationError):
        publish_artifact_set(
            tmp_path / "failure",
            (Artifact("result.bin", b"result"),),
            source_paths=(source,),
        )
    monkeypatch.setattr(publication.os, "replace", original_replace)

    after = source.stat()
    assert source.read_bytes() == b"source\tdata\n"
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode) == 0o444
    assert after.st_mtime_ns == before.st_mtime_ns


def test_concurrent_writers_serialize_complete_transactions(tmp_path):
    destination = tmp_path / "derivatives"
    destination.mkdir()
    (destination / "nested").mkdir()
    (destination / "first.bin").write_bytes(b"initial")
    (destination / "nested" / "second.bin").write_bytes(b"initial")
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_publication_worker,
            args=(destination, marker, start, results),
        )
        for marker in ("writer-a", "writer-b")
    ]

    for process in processes:
        process.start()
    start.set()
    for process in processes:
        _join_process_or_fail(process)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(results.get(timeout=1)[0] for _ in processes) == ["ok", "ok"]
    assert (destination / "first.bin").read_bytes() in {b"writer-a", b"writer-b"}
    assert (destination / "first.bin").read_bytes() == (
        destination / "nested" / "second.bin"
    ).read_bytes()
    _assert_no_transaction_debris(destination)


def test_lock_timeout_is_contextual_and_leaves_no_partial_artifact(tmp_path):
    destination = tmp_path / "derivatives"
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    results = context.Queue()
    holder = context.Process(
        target=_lock_holder_worker,
        args=(destination, acquired, results),
    )
    holder.start()
    assert acquired.wait(timeout=5)

    with pytest.raises(PublicationError) as error:
        publish_artifact_set(
            destination,
            (Artifact("blocked.bin", b"blocked"),),
            lock_timeout=0.05,
        )

    assert "timed out" in str(error.value).lower()
    assert "derivatives" in str(error.value)
    assert not (destination / "blocked.bin").exists()
    _join_process_or_fail(holder)
    assert holder.exitcode == 0
    assert results.get(timeout=1)[0] == "ok"
    assert (destination / "held.bin").read_bytes() == b"holder"
    _assert_no_transaction_debris(destination)


def test_source_overlap_through_destination_parent_alias_is_rejected(tmp_path):
    (tmp_path / "unused").mkdir()
    source = tmp_path / "signal.bin"
    source.write_bytes(b"original-input")
    with pytest.raises(ValueError, match="overlap"):
        publish_artifact_set(
            tmp_path / "unused" / "..",
            [Artifact("signal.bin", b"replacement")],
            source_paths=[source],
            overwrite=True,
        )
    assert source.read_bytes() == b"original-input"


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
        publish_artifact_set(
            destination,
            [Artifact("old.bin", b"new-old"), Artifact("new.bin", b"new")],
            overwrite=True,
        )
    error = caught.value
    assert error.__cause__ is promotion_error
    assert error.rollback_errors == (restore_error,)
    recovery = error.recovery_directory
    assert recovery is not None and recovery.is_dir()
    assert str(recovery) in str(error)
    assert (recovery / "backups" / "old.bin").read_bytes() == b"only-original"
    if not block_diagnostics:
        record = _failure_records(destination)[0]
        assert record["rollback_failed"] is True
        assert (
            record["recovery_directory"] == recovery.relative_to(destination).as_posix()
        )
        assert not Path(record["recovery_directory"]).is_absolute()
        assert str(tmp_path) not in json.dumps(record)

    assert not (destination / "new.bin").exists()


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


def test_existing_transaction_is_never_removed_on_identifier_collision(
    tmp_path, monkeypatch
):
    import boldtailor.publication as publication

    identifier = UUID("123e4567-e89b-12d3-a456-426614174000")
    destination = tmp_path / "output"
    orphan = destination / ".boldtailor" / "transactions" / str(identifier)
    orphan.mkdir(parents=True)
    backup = orphan / "only-original.bin"
    backup.write_bytes(b"preserve-me")
    monkeypatch.setattr(publication, "uuid4", lambda: identifier)
    with pytest.raises(PublicationError) as caught:
        publish_artifact_set(destination, [Artifact("result.bin", b"result")])
    assert isinstance(caught.value.__cause__, FileExistsError)
    assert backup.read_bytes() == b"preserve-me"
    assert not (destination / "result.bin").exists()


def test_publication_failure_ledger_omits_exception_names_and_text(
    tmp_path, monkeypatch
):
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


@pytest.mark.parametrize(
    "source_name", ["publication.lock", "publication_failures.jsonl"]
)
def test_control_files_cannot_overwrite_protected_sources(
    tmp_path, monkeypatch, source_name
):
    import boldtailor.publication as publication

    destination = tmp_path / "output"
    control = destination / ".boldtailor"
    control.mkdir(parents=True)
    source = control / source_name
    source.write_bytes(b"irreplaceable-input")

    def reject_promotion(*args, **kwargs):
        raise OSError("injected publication failure")

    if source_name.endswith("jsonl"):
        monkeypatch.setattr(publication.os, "replace", reject_promotion)
    with pytest.raises(ValueError, match="overlap"):
        publish_artifact_set(
            destination, [Artifact("result.bin", b"result")], source_paths=[source]
        )
    assert source.read_bytes() == b"irreplaceable-input"
    assert not (destination / "result.bin").exists()
    assert not (control / "transactions").exists()
