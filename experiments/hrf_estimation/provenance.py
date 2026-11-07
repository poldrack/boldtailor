"""Input/content identities and verified completion for experiment outputs."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import boldtailor.hrf_library


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(digest, values):
    values = np.asarray(values)
    digest.update(str((values.shape, values.dtype.str)).encode())
    digest.update(values.tobytes())


def identities(runs, brain_axis):
    digest = sha256()
    for run in runs:
        digest.update(run.run_id.encode())
        for values in (run.signals, run.frame_times, run.feature_ids):
            _array_hash(digest, values)
        digest.update(run.events.to_csv(index=False).encode())
        digest.update(run.confounds.to_csv(index=False).encode())
    for values in (brain_axis.name, brain_axis.vertex):
        _array_hash(digest, values)
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    return dict(
        input_fingerprint=digest.hexdigest(),
        implementation_fingerprint=implementation_fingerprint(
            Path(__file__).parent, Path(boldtailor.hrf_library.__file__).parent
        ),
        environment_fingerprint=file_hash(lock),
    )


def implementation_fingerprint(experiment_dir, package_dir):
    digest = sha256()
    for label, directory in (("experiment", experiment_dir), ("package", package_dir)):
        for path in sorted(Path(directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(directory)
            if label == "experiment" and (
                len(relative.parts) != 1
                or path.suffix != ".py"
                or path.name.startswith("test_")
            ):
                continue
            digest.update(f"{label}/{relative}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cached_result(destination, manifest):
    previous = destination / "manifest.json"
    # JSON turns tuple run lists into arrays; compare the serialized representation.
    if previous.exists() and json.loads(previous.read_text()) != json.loads(
        json.dumps(manifest)
    ):
        raise ValueError(
            "analysis directory contains different inputs/settings/code; use a new directory"
        )
    complete = destination / "COMPLETE.json"
    if not complete.exists():
        return None
    try:
        hashes = json.loads(complete.read_text())["artifact_hashes"]
        if "scores.tsv" not in hashes:
            return None
        for name, expected in hashes.items():
            path = destination / name
            if (
                Path(name).name != name
                or not path.is_file()
                or file_hash(path) != expected
            ):
                return None
        return pd.read_csv(destination / "scores.tsv", sep="\t")
    except (KeyError, ValueError, OSError):
        return None


def completion_record(destination, rows):
    hashes = {
        p.name: file_hash(p)
        for p in destination.iterdir()
        if p.is_file() and p.name != "COMPLETE.json" and not p.name.startswith(".")
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return dict(
        rows=rows,
        artifact_hashes=hashes,
        git_commit=revision,
        git_dirty=bool(dirty.strip()),
    )
