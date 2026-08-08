import json
import multiprocessing
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


def test_publish_rejects_empty_artifact_set(tmp_path):
    with pytest.raises(ValueError, match="at least one artifact"):
        publish_artifact_set(tmp_path / "derivatives", ())


@pytest.mark.parametrize(
    "paths",
    (
        ("result.json", "result.json"),
        ("Sub-01/result.json", "sub-01/RESULT.JSON"),
    ),
)
def test_preflight_rejects_exact_and_casefolded_duplicate_paths(tmp_path, paths):
    artifacts = tuple(Artifact(path, b"{}\n") for path in paths)

    with pytest.raises(ValueError, match="duplicate artifact path"):
        publish_artifact_set(tmp_path / "derivatives", artifacts)


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
        path.relative_to(destination).as_posix(): path.read_bytes() for path in published
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
    assert records[0]["error_type"] == "OSError"
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


def test_retain_incomplete_keeps_staged_artifacts_and_failed_provenance(
    tmp_path,
    monkeypatch,
):
    import boldtailor.publication as publication

    destination = tmp_path / "derivatives"
    destination.mkdir()
    existing = destination / "existing.bin"
    existing.write_bytes(b"original")
    provenance = {
        "schema": "boldtailor.provenance/1",
        "execution_id": "12345678-1234-5678-1234-567812345678",
        "events": [],
    }
    artifacts = (
        Artifact("existing.bin", b"replacement"),
        Artifact("new.bin", b"new"),
        Artifact(
            "logs/boldtailor_provenance.json",
            (json.dumps(provenance) + "\n").encode(),
        ),
    )
    original_replace = publication.os.replace
    calls = 0

    def fail_second_promotion(source, target):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected failure")
        original_replace(source, target)

    monkeypatch.setattr(publication.os, "replace", fail_second_promotion)

    with pytest.raises(PublicationError):
        publish_artifact_set(
            destination,
            artifacts,
            overwrite=True,
            retain_incomplete=True,
        )

    failed_root = destination / ".boldtailor" / "failed"
    retained = tuple(failed_root.iterdir())
    assert len(retained) == 1
    UUID(retained[0].name)
    retained_artifacts = retained[0] / "artifacts"
    assert (retained_artifacts / "existing.bin").read_bytes() == b"replacement"
    assert (retained_artifacts / "new.bin").read_bytes() == b"new"
    retained_provenance = json.loads(
        (retained_artifacts / "logs/boldtailor_provenance.json").read_bytes()
    )
    assert retained_provenance["publication"] == {
        "execution_id": retained[0].name,
        "published": False,
        "status": "failed",
    }
    assert existing.read_bytes() == b"original"
    assert not (destination / "new.bin").exists()
    _assert_no_transaction_debris(destination)


def test_source_files_are_read_only_inputs_on_success_and_failure(tmp_path, monkeypatch):
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
