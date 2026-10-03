from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import warnings

import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

from boldtailor._hrf_design import (  # event_response_scales, hrf_model: public
    event_response_scales,
    hrf_model,
    resolve_hrf,
    scale_event_amplitudes,
)
from boldtailor._task_design import (  # expand_events, task_columns: public
    expand_events,
    run_task_columns,
    task_columns,
)
from boldtailor.data import AnalysisData
from boldtailor.model import ModelSpec


@dataclass(frozen=True)
class CompiledDesign:
    matrix: pd.DataFrame
    excluded_event_count: int
    min_onset_cutoff: float


def compile_designs(data: AnalysisData, model: ModelSpec) -> tuple[CompiledDesign, ...]:
    events = data.events
    confounds = data.confounds
    return tuple(
        _compile_run(data.frame_times[run], events[run], confounds[run], model, run)
        for run in range(data.n_runs)
    )


def compile_nuisance_designs(
    data: AnalysisData, model: ModelSpec
) -> tuple[CompiledDesign, ...]:
    times, confounds = data.frame_times, data.confounds
    return tuple(
        _compile_nuisance_run(times[run], confounds[run], model, run)
        for run in range(data.n_runs)
    )


def _compile_run(
    frame_times: np.ndarray,
    events: pd.DataFrame,
    confounds: pd.DataFrame,
    model: ModelSpec,
    run: int,
) -> CompiledDesign:
    selected = _select_confounds(confounds, model.confounds, run)
    modeled_events, excluded_count, cutoff = _select_modeled_events(
        events, frame_times, model.min_onset, run
    )
    if model.task_model is None:
        design = _make_design_matrix(frame_times, modeled_events, selected, model, run)
    else:
        design = _make_task_model_design(
            frame_times, modeled_events, selected, model, run
        )
    _validate_design_matrix(design, run)
    return CompiledDesign(
        matrix=design,
        excluded_event_count=excluded_count,
        min_onset_cutoff=cutoff,
    )


def _make_task_model_design(
    frame_times: np.ndarray,
    events: pd.DataFrame,
    confounds: pd.DataFrame | None,
    model: ModelSpec,
    run: int,
) -> pd.DataFrame:
    task = run_task_columns(
        events,
        model.task_model,
        frame_times,
        model.hrf_model,
        run=run,
        min_onset=model.min_onset,
        oversampling=model.oversampling,
    )
    nuisance = _make_nuisance_matrix(frame_times, confounds, model, run)
    return pd.concat([task, nuisance], axis=1)


def _compile_nuisance_run(
    frame_times: np.ndarray,
    confounds: pd.DataFrame,
    model: ModelSpec,
    run: int,
) -> CompiledDesign:
    selected = _select_confounds(confounds, model.confounds, run)
    matrix = _make_nuisance_matrix(frame_times, selected, model, run)
    _validate_design_matrix(matrix, run)
    return CompiledDesign(
        matrix=matrix,
        excluded_event_count=0,
        min_onset_cutoff=float(frame_times[0] + model.min_onset),
    )


def _make_design_matrix(
    frame_times: np.ndarray,
    events: pd.DataFrame,
    confounds: pd.DataFrame | None,
    model: ModelSpec,
    run: int,
) -> pd.DataFrame:
    hrf = resolve_hrf(model.hrf_model)
    if hrf is not model.hrf_model:
        task = kernel_task_columns(frame_times, events, hrf, model, run)
        nuisance = _make_nuisance_matrix(frame_times, confounds, model, run)
        return pd.concat([task, nuisance], axis=1)
    try:
        return make_first_level_design_matrix(
            frame_times,
            events=events,
            hrf_model=hrf,
            drift_model=model.drift_model,
            high_pass=model.high_pass,
            drift_order=model.drift_order,
            add_regs=confounds,
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
    except (NotImplementedError, ValueError) as error:
        raise ValueError(f"run {run} design compilation failed: {error}") from error


def kernel_task_columns(frame_times, events, kernel, model, run):
    """Nilearn condition columns for one peak-one kernel, in Nilearn's order.

    Each event's amplitude is scaled so its response peaks at one. Task
    columns are built apart from nuisance columns: Nilearn appends the
    callable's name, which must neither rename contrasts nor hit a confound.
    """
    try:
        scaled = scale_event_amplitudes(
            events, kernel, frame_times, model.oversampling, model.min_onset
        )
        with redirect_stdout(io.StringIO()):  # Nilearn's modulation notice
            task = make_first_level_design_matrix(
                frame_times,
                events=scaled,
                hrf_model=kernel,
                drift_model=None,
                min_onset=model.min_onset,
                oversampling=model.oversampling,
            ).drop(columns="constant")
    except (NotImplementedError, ValueError) as error:
        raise ValueError(f"run {run} design compilation failed: {error}") from error
    task.columns = [name.removesuffix("_kernel") for name in task.columns]
    return task


def _make_nuisance_matrix(
    frame_times: np.ndarray,
    confounds: pd.DataFrame | None,
    model: ModelSpec,
    run: int,
) -> pd.DataFrame:
    try:
        return make_first_level_design_matrix(
            frame_times,
            events=None,
            hrf_model=None,
            drift_model=model.drift_model,
            high_pass=model.high_pass,
            drift_order=model.drift_order,
            add_regs=confounds,
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
    except (NotImplementedError, ValueError) as error:
        raise ValueError(
            f"run {run} nuisance design compilation failed: {error}"
        ) from error


def _validate_design_matrix(design: pd.DataFrame, run: int) -> None:
    if design.columns.has_duplicates:
        duplicates = design.columns[design.columns.duplicated()].tolist()
        raise ValueError(f"run {run} design has duplicate columns: {duplicates}")
    if not np.isfinite(design.to_numpy()).all():
        raise ValueError(f"run {run} design must be finite")


def _select_modeled_events(
    events: pd.DataFrame,
    frame_times: np.ndarray,
    min_onset: float,
    run: int,
) -> tuple[pd.DataFrame, int, float]:
    cutoff = float(frame_times[0] + min_onset)
    excluded = events["onset"].to_numpy(dtype=float) < cutoff
    count = int(excluded.sum())
    if count:
        label = "event" if count == 1 else "events"
        warnings.warn(
            f"run {run}: excluding {count} {label} earlier than "
            f"the min_onset cutoff {cutoff:g}",
            UserWarning,
            stacklevel=2,
        )
    modeled = events.loc[~excluded].reset_index(drop=True)
    return modeled, count, cutoff


def _select_confounds(
    confounds: pd.DataFrame, names: tuple[str, ...], run: int
) -> pd.DataFrame | None:
    if not names:
        return None
    missing = [name for name in names if name not in confounds]
    if missing:
        raise ValueError(f"run {run} missing confounds: {', '.join(missing)}")
    raw = confounds.loc[:, list(names)]
    selected = raw.apply(pd.to_numeric, errors="coerce")
    bad = [n for n in names if selected[n].isna().sum() > raw[n].isna().sum()]
    if bad:
        raise ValueError(f"run {run} non-numeric confound columns: {', '.join(bad)}")
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError(f"run {run} selected confounds must be finite")
    return selected
