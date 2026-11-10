"""Atomic, fingerprinted artifacts for resumable experiments."""

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def file_hash(path):
    digest = sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, "w") as out:
            out.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(
        path, json.dumps(value, default=json_value, sort_keys=True, indent=2) + "\n"
    )


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as out:
            np.savez_compressed(out, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def complete(folder, files):
    folder = Path(folder)
    write_json(
        folder / "COMPLETE.json", {name: file_hash(folder / name) for name in files}
    )


def verify(folder):
    folder = Path(folder)
    record = json.loads((folder / "COMPLETE.json").read_text())
    for name, digest in record.items():
        if not (folder / name).is_file() or file_hash(folder / name) != digest:
            raise ValueError(f"corrupt artifact: {folder/name}")
    return record
