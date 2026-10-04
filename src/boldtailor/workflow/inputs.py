"""Session loading, modulator detection, and the GLM model for any task."""

from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor._task_design import categorical_names
from boldtailor.model import ModelSpec, Modulator, TaskModel, level_name
from boldtailor.workflow.files import (
    discover_runs,
    load_runs,
    odd_even_parity,
    run_sources,
)


class InputError(ValueError):
    """Inputs are missing or malformed; the CLI maps this to exit code 2."""


TRIAL_TYPE_NOTE = "trial_type has fewer than two levels; not used as a modulator"


def _table_levels(table, column, label):
    try:
        return {level_name(v) for v in table[column]}
    except ValueError as error:
        raise InputError(f"{label}: {column!r}: {error}") from error


def observed_levels(tables, column, labels=None):
    """Canonical non-missing level names of ``column`` across every table."""
    labels = labels or [f"run {i}" for i in range(1, len(tables) + 1)]
    names = set()
    for table, label in zip(tables, labels, strict=True):
        names |= _table_levels(table, column, label)
    names.discard(None)
    return names


def _everywhere(column, tables):
    return bool(tables) and all(column in t.columns for t in tables)


def _automatic(tables, labels):
    modulators = []
    if _everywhere("response_time", tables):
        modulators.append(Modulator("response_time", missing="indicator"))
    if _everywhere("trial_type", tables):
        levels = observed_levels(tables, "trial_type", labels)
        if len(levels) >= 2:
            modulators.append(
                Modulator("trial_type", kind="categorical", levels=tuple(levels))
            )
    return tuple(modulators)


def _resolved(modulator, tables, labels):
    if modulator.resolved:
        return modulator
    try:
        levels = observed_levels(tables, modulator.column, labels)
        return modulator.with_levels(levels)
    except ValueError as error:
        raise InputError(f"modulator {modulator.column!r}: {error}") from error


def detect_task_model(events_tables, modulators=None, labels=None):
    """The task model from event columns, or explicit modulators checked per run.

    Detection takes ``response_time`` and a categorical ``trial_type`` with at
    least two levels; unresolved categorical modulators take their levels from
    every run. ``labels`` (BIDS run labels) name the run in errors.
    """
    tables = list(events_tables)
    labels = labels or [f"run {i}" for i in range(1, len(tables) + 1)]
    if modulators is None:
        modulators = _automatic(tables, labels)
    for label, table in zip(labels, tables, strict=True):
        missing = [m.column for m in modulators if m.column not in table.columns]
        if missing:
            raise InputError(f"{label} events lack modulator column(s) {missing}")
    return TaskModel(tuple(_resolved(m, tables, labels) for m in modulators))


def task_model_notes(events_tables, modulators=None):
    """Why automatic detection left out a column that every run has."""
    tables = list(events_tables)
    if modulators is None and _everywhere("trial_type", tables):
        if len(observed_levels(tables, "trial_type")) < 2:
            return [TRIAL_TYPE_NOTE]
    return []


def selection_task_model(task_model, include_rt=True):
    """The GLM task model, or without RT when RT must not drive HRF selection."""
    if not isinstance(include_rt, bool):
        raise ValueError("include_rt must be a boolean")
    if include_rt:
        return task_model
    return TaskModel(
        tuple(m for m in task_model.modulators if m.column != "response_time")
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
        raise ValueError(
            f"{name} must contain numeric values; for text levels use "
            f"--modulator {name}:categorical"
        ) from error


def _columns(task_model):
    return {m.column for m in task_model.modulators}


def _check_response_time(events):
    rt = _numeric_column(events, "response_time")
    if not (np.isfinite(rt) & (rt > 0)).any():
        raise ValueError(
            "response_time needs positive finite observations for the RT effect"
        )


def validate_glm_events(events, task_model):
    """Require an observed positive RT and every categorical level, if modeled.

    Other numeric modulators must parse as numbers.
    """
    for modulator in task_model.modulators:
        if modulator.column not in events:
            raise ValueError(f"Missing {modulator.column}")
        if modulator.kind == "categorical":
            categorical_names(events[modulator.column], modulator)
        elif modulator.column == "response_time":
            _check_response_time(events)
        else:
            _numeric_column(events, modulator.column)


def _missing_nonpositive_rt(events):
    """Package semantics: missing means non-finite, so nonpositive RTs become NaN."""
    if "response_time" not in events:
        return events
    rt = _numeric_column(events, "response_time")
    return events.assign(response_time=np.where(np.isfinite(rt) & (rt > 0), rt, np.nan))


def _checked(run, task_model):
    """Nonsteady columns, retained frames, and events with RT missingness."""
    flags = run.confounds.filter(like="non_steady_state_outlier")
    if not np.isin(flags.to_numpy(), [0, 1]).all():
        raise ValueError("Nonsteady flags must be binary")
    dropped = np.flatnonzero(flags.to_numpy().any(axis=1))
    if not np.array_equal(dropped, np.arange(len(dropped))):
        raise ValueError("Nonsteady flags must mark contiguous leading volumes")
    retained = np.arange(len(dropped), len(run.frame_times))
    if not len(retained):
        raise ValueError("No scans remain after trimming")
    validate_glm_events(run.events, task_model)
    return flags.columns, retained, _missing_nonpositive_rt(run.events)


AUTOMATIC_HINT = (
    "; trial_type was detected automatically: to code missing values pass "
    "--modulator trial_type:categorical,indicator (add --modulator "
    "response_time:indicator to keep RT), list modulators without trial_type, "
    "or use --no-modulators"
)


def _categorical_failure(error, task_model):
    return any(
        m.kind == "categorical" and repr(m.column) in str(error)
        for m in task_model.modulators
    )


def _trim(run, task_model, automatic=False):
    try:
        flags, retained, events = _checked(run, task_model)
    except ValueError as error:
        hint = (
            AUTOMATIC_HINT
            if automatic and _categorical_failure(error, task_model)
            else ""
        )
        raise InputError(f"{run.label}: {error}{hint}") from error
    return WorkflowRun(
        run.inputs,
        run.image,
        events,
        run.confounds.drop(columns=flags).iloc[retained].reset_index(drop=True),
        run.frame_times[retained],
        run.label,
        run.number,
        retained,
    )


def load_session(settings, *, hrf_only=False):
    """Keep original event onsets and acquisition times when dropping NSS scans."""
    raw_runs, _ = load_runs(discover_runs(settings))
    task_model = detect_task_model(
        [r.events for r in raw_runs],
        settings.modulators,
        labels=[r.label for r in raw_runs],
    )
    automatic = settings.modulators is None
    runs = [_trim(run, task_model, automatic) for run in raw_runs]
    if len({tuple(r.confounds.columns) for r in runs}) != 1:
        raise InputError("Retained confound names must match across runs")
    if len(runs) < 2:
        raise InputError("HRF selection needs at least two runs")
    if not hrf_only and any(len(half) < 2 for half in odd_even_parity(runs).values()):
        raise InputError("The full workflow needs at least two odd and two even runs")
    return tuple(runs)


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


def load_block(runs, root, indices, task_model):
    """Raw trial rows for every analysis; the task model expands them."""
    return from_arrays(
        block_signals(runs, indices),
        [r.events for r in runs],
        frame_times=[r.frame_times for r in runs],
        confounds=[r.confounds for r in runs],
        sources=[_trimmed_sources(r, root, indices) for r in runs],
        provenance_metadata={
            "event_encoding": "raw_trials_with_task_model",
            "task_model": task_model.to_dict(),
        },
    )


def glm_model(runs, task_model):
    return ModelSpec(
        contrasts={name: {name: 1} for name in task_model.regressor_names},
        confounds=tuple(runs[0].confounds.columns),
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
        task_model=task_model,
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


def _level_counts(run, modulator):
    names = [level_name(v) for v in run.events[modulator.column]]
    return {f"n_{modulator.column}_{lv}": names.count(lv) for lv in modulator.levels}


def _run_row(run, task_model):
    row = dict(
        run=run.label,
        trials=len(run.events),
        scans=run.image.shape[0],
        retained_scans=len(run.frame_times),
        dropped_scans=len(run.retained_frames) and int(run.retained_frames[0]),
        first_frame_seconds=run.frame_times[0],
    )
    if "response_time" in _columns(task_model):
        row["mean_rt_seconds"] = run.events.response_time.mean()
    for modulator in task_model.modulators:
        if modulator.kind == "categorical":
            row.update(_level_counts(run, modulator))
    return row


def run_summary(runs, task_model):
    return pd.DataFrame([_run_row(r, task_model) for r in runs])
