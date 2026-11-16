"""Manifest-driven removal is part of the locked publication transaction."""

import json
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from boldtailor import publication
from boldtailor.publication import Artifact, PublicationError, publish_artifact_set


@pytest.fixture
def prior_outputs(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    for name in ("old-a.bin", "old-b.bin", "keep.bin"):
        (root / name).write_bytes(name.encode())
    (root / "manifest.json").write_text(
        json.dumps(
            {"artifacts": ["old-a.bin", "old-b.bin", "keep.bin", "manifest.json"]}
        )
    )
    return root


def replacement():
    return [
        Artifact("new.bin", b"new"),
        Artifact("manifest.json", b'{"artifacts":["new.bin","manifest.json"]}'),
    ]


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def publish(root, **kwargs):
    return publish_artifact_set(
        root, replacement(), overwrite=True, manifest_path="manifest.json", **kwargs
    )


@pytest.mark.integration
def test_manifest_removal_preserves_sources_and_unlisted_files(prior_outputs):
    root = prior_outputs
    (root / "unlisted.bin").write_bytes(b"untouched")
    paths = publish(root, source_paths=[root / "keep.bin"])
    assert {p.name for p in paths} == {"new.bin", "manifest.json"}
    assert snapshot(root) == {
        "new.bin": b"new",
        "manifest.json": replacement()[1].payload,
        "keep.bin": b"keep.bin",
        "unlisted.bin": b"untouched",
    }


@pytest.mark.parametrize("failure_at", ["second_removal", "promotion"])
@pytest.mark.integration
def test_removal_and_promotion_failures_restore_entire_previous_set(
    prior_outputs, monkeypatch, failure_at
):
    root = prior_outputs
    before = snapshot(root)
    original = publication.os.replace
    removed = []
    failed = False

    def fail_once(source, target):
        nonlocal failed
        source, target = Path(source), Path(target)
        removing = source.parent == root and source.name.startswith("old-")
        promoting = target == root / "new.bin"
        trigger = (
            failure_at == "second_removal" and removing and len(removed) == 1
        ) or (failure_at == "promotion" and promoting)
        if trigger and not failed:
            failed = True
            assert removed, "removal must be included in rollback before promotion"
            raise OSError("injected transaction failure")
        original(source, target)
        if removing:
            removed.append(source.name)

    monkeypatch.setattr(publication.os, "replace", fail_once)
    with pytest.raises(PublicationError):
        publish(root)
    assert failed
    assert snapshot(root) == before


def test_previous_manifest_is_read_after_lock_acquisition(prior_outputs, monkeypatch):
    root = prior_outputs
    original = publication.FileLock

    class RacingLock:
        def __init__(self, *args, **kwargs):
            self.lock = original(*args, **kwargs)

        def __enter__(self):
            # Simulate the preceding writer completing before we acquire its lock.
            (root / "late.bin").write_bytes(b"late")
            (root / "manifest.json").write_text(json.dumps({"artifacts": ["late.bin"]}))
            return self.lock.__enter__()

        def __exit__(self, *exc):
            return self.lock.__exit__(*exc)

    monkeypatch.setattr(publication, "FileLock", RacingLock)
    publish(root)
    assert not (root / "late.bin").exists()
    assert (root / "old-a.bin").exists()  # No longer belongs to the preceding manifest.


def test_stale_files_are_removed_while_writer_lock_is_held(prior_outputs, monkeypatch):
    root = prior_outputs
    original = publication.os.replace
    removals = []

    def check_lock(source, target):
        if Path(source).parent == root and Path(source).name.startswith("old-"):
            with pytest.raises(Timeout):
                with FileLock(
                    root.with_name(root.name + ".boldtailor") / "lock", timeout=0
                ):
                    pass
            removals.append(Path(source).name)
        return original(source, target)

    monkeypatch.setattr(publication.os, "replace", check_lock)
    publish(root)
    assert set(removals) == {"old-a.bin", "old-b.bin"}


@pytest.mark.parametrize(
    "contents", ["not json", "[]", '{"artifacts": "bad"}', '{"artifacts": [12]}']
)
def test_corrupt_manifest_stops_before_modification(prior_outputs, contents):
    root = prior_outputs
    (root / "manifest.json").write_text(contents)
    before = snapshot(root)
    with pytest.raises(ValueError, match="manifest"):
        publish(root)
    assert snapshot(root) == before


def test_unreadable_manifest_stops_before_modification(prior_outputs, monkeypatch):
    root = prior_outputs
    before = snapshot(root)
    original = Path.read_text

    def deny_manifest(path, *args, **kwargs):
        if path == root / "manifest.json":
            raise PermissionError("unreadable manifest")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_manifest)
    with pytest.raises(PermissionError):
        publish(root)
    assert snapshot(root) == before


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_or_legacy_manifest_does_not_infer_files_to_remove(
    prior_outputs, legacy
):
    root = prior_outputs
    if legacy:
        (root / "manifest.json").write_text("{}")
    else:
        (root / "manifest.json").unlink()
    publish(root)
    assert (root / "old-a.bin").exists() and (root / "old-b.bin").exists()


def test_stale_paths_cannot_follow_parent_symlinks(prior_outputs, tmp_path):
    root = prior_outputs
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.bin"
    victim.write_bytes(b"keep")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / "manifest.json").write_text(
        json.dumps({"artifacts": ["linked/victim.bin"]})
    )
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="symlink"):
        publish(root)
    assert victim.read_bytes() == b"keep"
    assert (root / "manifest.json").read_bytes() == before
    assert not (root / "new.bin").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"overwrite": False, "manifest_path": "manifest.json"},
        {"overwrite": True, "manifest_path": "../manifest.json"},
        {"overwrite": True, "manifest_path": "absent.json"},
        {
            "overwrite": True,
            "manifest_path": "manifest.json",
            "keep_existing": ["manifest.json"],
        },
    ],
)
def test_manifest_cleanup_requires_an_overwritten_manifest_artifact(
    prior_outputs, options
):
    before = snapshot(prior_outputs)
    with pytest.raises(ValueError, match="manifest"):
        publish_artifact_set(prior_outputs, replacement(), **options)
    assert snapshot(prior_outputs) == before


def test_stale_manifest_entries_cannot_delete_protected_source_ancestors(prior_outputs):
    root = prior_outputs
    folder = root / "source-data"
    folder.mkdir()
    source = folder / "input.bin"
    source.write_bytes(b"input")
    (root / "manifest.json").write_text(json.dumps({"artifacts": ["source-data"]}))
    publish(root, source_paths=[source])
    assert source.read_bytes() == b"input"


def test_invalid_relative_manifest_entries_are_never_deletion_targets(
    prior_outputs, tmp_path
):
    root = prior_outputs
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"untouched")
    (root / "manifest.json").write_text(
        json.dumps({"artifacts": ["../outside.bin", str(outside), "old-a.bin"]})
    )
    publish(root)
    assert outside.read_bytes() == b"untouched"
    assert not (root / "old-a.bin").exists()
