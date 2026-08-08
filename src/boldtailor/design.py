from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

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
    design = make_first_level_design_matrix(
        frame_times,
        events=modeled_events,
        hrf_model=model.hrf_model,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        add_regs=selected,
        min_onset=model.min_onset,
        oversampling=model.oversampling,
    )
    if design.columns.has_duplicates:
        duplicates = design.columns[design.columns.duplicated()].tolist()
        raise ValueError(f"run {run} design has duplicate columns: {duplicates}")
    if not np.isfinite(design.to_numpy()).all():
        raise ValueError(f"run {run} design must be finite")
    return CompiledDesign(
        matrix=design,
        excluded_event_count=excluded_count,
        min_onset_cutoff=cutoff,
    )


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
    selected = confounds.loc[:, list(names)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError(f"run {run} selected confounds must be finite")
    return selected
