"""Common on-disk beta format shared by boldtailor and released betas."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.nsd_replication.trials import TRIAL_COLUMNS


def fit_dir(output_dir, source, subject, session, level):
    return Path(output_dir) / "fits" / source / subject / session / level


def write_fit(path, betas, trials, metadata, extras=None):
    if len(betas) != len(trials):
        raise ValueError(f"betas rows ({len(betas)}) must match trials ({len(trials)})")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").unlink(missing_ok=True)
    np.save(path / "betas.npy", np.asarray(betas, dtype=np.float32))
    trials.loc[:, list(TRIAL_COLUMNS)].to_csv(
        path / "trials.tsv", sep="\t", index=False
    )
    for name, values in (extras or {}).items():
        np.save(path / f"{name}.npy", np.asarray(values))
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _discarded(path):
    meta = json.loads((path / "metadata.json").read_text())
    return bool(meta.get("betas_discarded", False))


def is_complete(path):
    path = Path(path)
    if not all((path / name).is_file() for name in ("trials.tsv", "metadata.json")):
        return False
    return (path / "betas.npy").is_file() or _discarded(path)


def read_fit(path):
    path = Path(path)
    if not is_complete(path):
        raise FileNotFoundError(f"incomplete fit: {path}")
    if not (path / "betas.npy").is_file():
        raise ValueError(f"{path}: betas were discarded after metrics")
    trials = pd.read_csv(path / "trials.tsv", sep="\t")
    metadata = json.loads((path / "metadata.json").read_text())
    return np.load(path / "betas.npy"), trials, metadata


def read_extra(path, name):
    return np.load(Path(path) / f"{name}.npy")


def zscore(betas):
    values = np.asarray(betas, dtype=float)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values).all(axis=0)
    ptp = np.ptp(values, axis=0)
    ok = finite & (ptp > 0)
    sd = values[:, ok].std(axis=0)
    out[:, ok] = (values[:, ok] - values[:, ok].mean(axis=0)) / sd
    return out


def inputs_digest(session):
    digest = hashlib.sha256()
    digest.update(json.dumps([session.source, list(session.labels)]).encode())
    for times, events, confounds in zip(
        session.frame_times, session.events, session.confounds, strict=True
    ):
        digest.update(np.asarray(times, dtype="<f8").tobytes())
        digest.update(events.to_csv(index=False).encode())
        digest.update(confounds.to_csv(index=False).encode())
    return digest.hexdigest()
