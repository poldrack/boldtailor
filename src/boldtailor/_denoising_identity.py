"""Content fingerprints that tie a denoising result to its analysis data.

Anonymous arrays carry no source metadata, so identity is checked on content:
each run's signal, time grid, baseline confounds, and events. A changed
feature order, run order, or confound table changes a fingerprint.
"""

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunIdentity:
    """Fingerprints of one run of the analysis a result was selected on."""

    n_scans: int
    signal: str
    frame_times: str
    confounds: str
    confound_columns: tuple[str, ...]
    events: str

    def to_dict(self) -> dict[str, object]:
        return dict(
            n_scans=self.n_scans,
            signal=self.signal,
            frame_times=self.frame_times,
            confounds=self.confounds,
            confound_columns=list(self.confound_columns),
            events=self.events,
        )


def array_digest(values) -> str:
    """sha256 of an array's shape and little-endian float64 values."""
    array = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(list(array.shape)).encode()
    return sha256(header + array.tobytes()).hexdigest()


def table_digest(frame: pd.DataFrame) -> str:
    """sha256 of a table's row count, column names, dtypes, and values."""
    digest = sha256(
        json.dumps(
            [len(frame), [[str(c), str(t)] for c, t in frame.dtypes.items()]]
        ).encode()
    )
    for column in frame.columns:
        values = pd.util.hash_pandas_object(frame[column], index=False)
        digest.update(values.to_numpy().tobytes())
    return digest.hexdigest()


def run_identities(data) -> tuple[RunIdentity, ...]:
    return tuple(
        RunIdentity(
            n_scans=len(signal),
            signal=array_digest(signal),
            frame_times=array_digest(times),
            confounds=table_digest(confounds),
            confound_columns=tuple(str(c) for c in confounds.columns),
            events=table_digest(events),
        )
        for signal, times, confounds, events in zip(
            data.signals, data.frame_times, data.confounds, data.events, strict=True
        )
    )


def _check_runs(expected, actual, n_features, data):
    if len(actual) != len(expected):
        raise ValueError(
            f"data has {len(actual)} runs; the denoising result was selected "
            f"on {len(expected)} runs"
        )
    if data.n_features != n_features:
        raise ValueError(
            f"data has {data.n_features} features; the denoising result was "
            f"selected on {n_features} features"
        )
    signals = [run.signal for run in expected]
    observed = [run.signal for run in actual]
    if observed != signals and sorted(observed) == sorted(signals):
        raise ValueError(
            "runs are not in the order the denoising result was selected on"
        )


_FIELDS = (
    ("frame_times", "time grid of run {label} differs"),
    ("signal", "signals of run {label} differ (features reordered or data changed)"),
    ("confounds", "baseline confounds of run {label} differ"),
    ("events", "events of run {label} differ"),
)


def _check_run(expected, actual, label):
    if actual.n_scans != expected.n_scans:
        raise ValueError(
            f"run {label} has {actual.n_scans} rows; the denoising components "
            f"have {expected.n_scans}"
        )
    for name, message in _FIELDS:
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(
                message.format(label=label) + " from the analysis the denoising "
                "result was selected on"
            )


def check_identity(expected, n_features, data, labels) -> None:
    """Raise unless ``data`` is the analysis the result was selected on."""
    actual = run_identities(data)
    _check_runs(expected, actual, n_features, data)
    for want, have, label in zip(expected, actual, labels, strict=True):
        _check_run(want, have, f"'{label}'")
