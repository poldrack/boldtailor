"""Input preparation shared by the cells in the NSD workflow notebook."""

from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.model import ModelSpec
from .nsd_cifti import discover_runs, _sources
from .nsd_single_trial import _load_runs

REGRESSORS = ("task", "response_time", "trial_type")


@dataclass(frozen=True)
class WorkflowRun:
    inputs: object
    image: object
    events: pd.DataFrame
    confounds: pd.DataFrame
    frame_times: np.ndarray
    label: str
    number: int
    retained_frames: np.ndarray


def _rt_amplitudes(values):
    observed = np.isfinite(values) & (values > 0)
    if not observed.any():
        raise ValueError(
            "response_time needs positive finite observations for the RT effect"
        )
    centered = np.zeros(len(values))
    centered[observed] = values[observed] - values[observed].mean()
    amplitudes = {"response_time": centered}
    if not observed.all():
        amplitudes["missing_response_time"] = (~observed).astype(float)
    return amplitudes


def glm_events(events):
    """Retain all stimuli, with centered RT and an indicator for unavailable RT."""
    amplitudes = {"task": np.ones(len(events))}
    for name in REGRESSORS[1:]:
        if name not in events:
            raise ValueError(f"Missing {name}")
        try:
            values = pd.to_numeric(events[name], errors="raise").to_numpy(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numeric values") from error
        if name == "response_time":
            amplitudes.update(_rt_amplitudes(values))
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite for every trial")
        if name == "trial_type" and set(values) != {0, 1}:
            raise ValueError("trial_type must contain both binary codes 0 and 1")
        amplitudes[name] = values - values.mean()
    return pd.concat(
        [
            events[["onset", "duration"]].assign(trial_type=k, modulation=v)
            for k, v in amplitudes.items()
        ],
        ignore_index=True,
    )


def _trim(run, *, hrf_only=False):
    flags = run.confounds.filter(like="non_steady_state_outlier")
    if not np.isin(flags.to_numpy(), [0, 1]).all():
        raise ValueError("Nonsteady flags must be binary")
    dropped = np.flatnonzero(flags.to_numpy().any(axis=1))
    if not np.array_equal(dropped, np.arange(len(dropped))):
        raise ValueError("Nonsteady flags must mark contiguous leading volumes")
    retained = np.arange(len(dropped), len(run.frame_times))
    if not len(retained):
        raise ValueError("No scans remain after trimming")
    if not hrf_only:
        glm_events(run.events)  # Validate the conventional GLM encoding.
    return WorkflowRun(
        run.inputs,
        run.image,
        run.events,
        run.confounds.drop(columns=flags.columns).iloc[retained].reset_index(drop=True),
        run.frame_times[retained],
        run.label,
        run.number,
        retained,
    )


def load_session(root, prep, *, subject="sub-07", session="ses-nsd10", hrf_only=False):
    """Keep original event onsets and acquisition times when dropping NSS scans."""
    inputs = discover_runs(Path(root), Path(prep), subject=subject, session=session)
    raw_runs, _ = _load_runs(inputs)
    runs = [_trim(run, hrf_only=hrf_only) for run in raw_runs]
    if len({tuple(r.confounds.columns) for r in runs}) != 1:
        raise ValueError("Retained confound names must match across runs")
    if len(runs) < 2:
        raise ValueError("HRF selection needs at least two runs")
    if not hrf_only and any(
        sum(r.number % 2 == parity for r in runs) < 2 for parity in (0, 1)
    ):
        raise ValueError("The full notebook needs at least two odd and two even runs")
    return runs


def block_signals(runs, indices):
    indices = np.asarray(indices, dtype=int)
    start, stop = int(indices.min()), int(indices.max()) + 1
    return [
        np.asarray(r.image.dataobj[r.retained_frames[0] :, start:stop], dtype=float)[
            :, indices - start
        ]
        for r in runs
    ]


def _trimmed_sources(run, root, indices):
    sources = _sources(run, Path(root), np.asarray(indices))
    return replace(
        sources,
        signal=replace(
            sources.signal,
            annotations={
                **sources.signal.annotations,
                "retained_frame_indices": run.retained_frames.tolist(),
            },
        ),
    )


def load_block(runs, root, indices, *, glm=False):
    """Use raw trial rows for selection/beta series; expand rows only for GLMs."""
    return from_arrays(
        block_signals(runs, indices),
        [glm_events(r.events) if glm else r.events for r in runs],
        frame_times=[r.frame_times for r in runs],
        confounds=[r.confounds for r in runs],
        sources=[_trimmed_sources(r, root, indices) for r in runs],
        provenance_metadata={
            "event_encoding": (
                "centered_joint_modulators" if glm else "one_row_per_trial"
            )
        },
    )


def glm_model(runs):
    return ModelSpec(
        contrasts={name: {name: 1} for name in REGRESSORS},
        confounds=tuple(runs[0].confounds.columns),
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
    )


def make_blocks(runs, *, block_size=4096, max_grayordinates=None):
    """Skip constant signals; exports restore their positions as NaN."""
    for name, value in (
        ("block_size", block_size),
        ("max_grayordinates", max_grayordinates),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    count = len(runs[0].image.header.get_axis(1))
    limit = count if max_grayordinates is None else min(count, max_grayordinates)
    blocks = []
    for start in range(0, limit, block_size):
        indices = np.arange(start, min(start + block_size, limit))
        signals = block_signals(runs, indices)
        if not all(np.isfinite(y).all() for y in signals):
            raise ValueError("CIFTI signals must be finite")
        sst = sum(np.sum((y - y.mean(axis=0)) ** 2, axis=0) for y in signals)
        if np.any(sst > 0):
            blocks.append(indices[sst > 0])
    if not blocks:
        raise ValueError("No nonconstant grayordinates in the requested subset")
    return blocks


def run_summary(runs):
    return pd.DataFrame(
        [
            dict(
                run=r.label,
                trials=len(r.events),
                scans=r.image.shape[0],
                retained_scans=len(r.frame_times),
                dropped_scans=len(r.retained_frames) and int(r.retained_frames[0]),
                first_frame_seconds=r.frame_times[0],
                mean_rt_seconds=r.events.response_time.mean(),
                type_0=int((r.events.trial_type == 0).sum()),
                type_1=int((r.events.trial_type == 1).sum()),
            )
            for r in runs
        ]
    )
