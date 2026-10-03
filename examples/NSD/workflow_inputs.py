"""Input preparation shared by the cells in the NSD workflow notebook."""

from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.model import ModelSpec, Modulator, TaskModel
from .workflow_files import discover_runs, load_runs, odd_even_parity, run_sources

REGRESSORS = ("task", "response_time", "trial_type")


def selection_task_model(include_rt=True):
    """The GLM task model, or trial type alone when RT must not drive HRF selection."""
    if not isinstance(include_rt, bool):
        raise ValueError("include_rt must be a boolean")
    if include_rt:
        return NSD_TASK_MODEL
    return TaskModel(
        tuple(m for m in NSD_TASK_MODEL.modulators if m.column != "response_time")
    )


NSD_TASK_MODEL = TaskModel(
    (
        Modulator("response_time", center=True, missing="indicator"),
        Modulator("trial_type", center=False),
    )
)


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


def _numeric_column(events, name):
    if name not in events:
        raise ValueError(f"Missing {name}")
    try:
        return pd.to_numeric(events[name], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error


def validate_glm_events(events):
    """Require an observed positive RT and both binary trial_type codes."""
    rt = _numeric_column(events, "response_time")
    if not (np.isfinite(rt) & (rt > 0)).any():
        raise ValueError(
            "response_time needs positive finite observations for the RT effect"
        )
    trial_type = _numeric_column(events, "trial_type")
    if not np.isfinite(trial_type).all() or set(trial_type) != {0, 1}:
        raise ValueError("trial_type must contain both binary codes 0 and 1")


def _missing_nonpositive_rt(events):
    """Package semantics: missing means non-finite, so nonpositive RTs become NaN."""
    rt = _numeric_column(events, "response_time")
    return events.assign(response_time=np.where(np.isfinite(rt) & (rt > 0), rt, np.nan))


def _trim(run):
    flags = run.confounds.filter(like="non_steady_state_outlier")
    if not np.isin(flags.to_numpy(), [0, 1]).all():
        raise ValueError("Nonsteady flags must be binary")
    dropped = np.flatnonzero(flags.to_numpy().any(axis=1))
    if not np.array_equal(dropped, np.arange(len(dropped))):
        raise ValueError("Nonsteady flags must mark contiguous leading volumes")
    retained = np.arange(len(dropped), len(run.frame_times))
    if not len(retained):
        raise ValueError("No scans remain after trimming")
    validate_glm_events(run.events)
    return WorkflowRun(
        run.inputs,
        run.image,
        _missing_nonpositive_rt(run.events),
        run.confounds.drop(columns=flags.columns).iloc[retained].reset_index(drop=True),
        run.frame_times[retained],
        run.label,
        run.number,
        retained,
    )


def load_session(root, prep, *, subject="sub-07", session="ses-nsd10", hrf_only=False):
    """Keep original event onsets and acquisition times when dropping NSS scans."""
    inputs = discover_runs(Path(root), Path(prep), subject=subject, session=session)
    raw_runs, _ = load_runs(inputs)
    runs = [_trim(run) for run in raw_runs]
    if len({tuple(r.confounds.columns) for r in runs}) != 1:
        raise ValueError("Retained confound names must match across runs")
    if len(runs) < 2:
        raise ValueError("HRF selection needs at least two runs")
    if not hrf_only and any(len(half) < 2 for half in odd_even_parity(runs).values()):
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
    sources = run_sources(run, Path(root), np.asarray(indices))
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


def load_block(runs, root, indices):
    """Raw trial rows for every analysis; the task model expands them."""
    return from_arrays(
        block_signals(runs, indices),
        [r.events for r in runs],
        frame_times=[r.frame_times for r in runs],
        confounds=[r.confounds for r in runs],
        sources=[_trimmed_sources(r, root, indices) for r in runs],
        provenance_metadata={
            "event_encoding": "raw_trials_with_task_model",
            "task_model": NSD_TASK_MODEL.to_dict(),
        },
    )


def glm_model(runs):
    return ModelSpec(
        contrasts={name: {name: 1} for name in REGRESSORS},
        confounds=tuple(runs[0].confounds.columns),
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
        task_model=NSD_TASK_MODEL,
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
