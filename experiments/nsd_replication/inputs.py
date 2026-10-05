"""Load one NSD ppdata session as cortical-only arrays for boldtailor."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.model import TaskModel
from boldtailor.workflow.inputs import detect_task_model

from experiments.nsd_replication.confounds import glmsingle_polynomials

CORTEX = ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")
_PPDATA = (
    "subj{nn}/func1pt8mm/timeseries/{subject}_{session}_task-{task}_{run}"
    "_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii"
)


@dataclass(frozen=True, kw_only=True)
class Session:
    subject: str
    session: str
    labels: tuple[str, ...]
    signals: tuple[np.ndarray, ...]
    events: tuple[pd.DataFrame, ...]
    confounds: tuple[pd.DataFrame, ...]
    frame_times: tuple[np.ndarray, ...]
    brain: nib.cifti2.BrainModelAxis
    task_model: TaskModel | None
    source: str


def cortical_indices(brain):
    present = {name for name, _, _ in brain.iter_structures()}
    missing = [name for name in CORTEX if name not in present]
    if missing:
        raise ValueError(f"CIFTI lacks cortical surface structures: {missing}")
    return np.flatnonzero(brain.surface_mask & np.isin(brain.name, CORTEX))


def same_grayordinates(a, b):
    return (
        len(a) == len(b)
        and np.array_equal(a.name, b.name)
        and np.array_equal(a.vertex, b.vertex)
    )


def analysis_data(session, indices, confounds=None):
    confounds = session.confounds if confounds is None else confounds
    return from_arrays(
        [np.asarray(y[:, indices], dtype=float) for y in session.signals],
        list(session.events),
        frame_times=list(session.frame_times),
        confounds=list(confounds),
    )


def _events_files(config, subject, session):
    func = config.bids_dir / subject / session / "func"
    files = sorted(
        func.glob(f"{subject}_{session}_task-{config.task}_run-*_events.tsv")
    )
    if not files:
        raise FileNotFoundError(f"{subject} {session}: no events in {func}")
    return files


def ppdata_path(config, subject, session, run):
    if config.ppdata_dir is None:
        raise ValueError("ppdata_dir is not configured")
    name = _PPDATA.format(
        nn=subject.removeprefix("sub-"),
        subject=subject,
        session=session,
        task=config.task,
        run=run,
    )
    return Path(config.ppdata_dir) / name


def _existing_path(config, subject, session, run):
    path = ppdata_path(config, subject, session, run)
    if not path.is_file():
        raise FileNotFoundError(f"{subject} {session} {run}: missing {path}")
    return path


def _load_run(config, subject, session, events_path):
    run = events_path.name.split("_")[3]
    image = nib.load(_existing_path(config, subject, session, run))
    series = image.header.get_axis(0)
    n = image.shape[0]
    times = config.onset_offset + series.start + np.arange(n) * series.step
    events = pd.read_csv(events_path, sep="\t")
    _check_onsets(events, times, series.step, f"{subject} {session} {run}")
    return run, image, times, events


def _check_onsets(events, times, tr, name):
    end = events["onset"] + events["duration"]
    if events["onset"].min() < times[0] - tr or end.max() > times[-1] + tr:
        raise ValueError(f"{name}: onsets fall outside the ppdata time base")


def _cortex(runs, name):
    brain = runs[0][1].header.get_axis(1)
    for run in runs[1:]:
        if not same_grayordinates(run[1].header.get_axis(1), brain):
            raise ValueError(f"{name}: ppdata runs differ in grayordinates")
    return brain, cortical_indices(brain)


def load_ppdata(config, subject, session):
    files = _events_files(config, subject, session)
    runs = [_load_run(config, subject, session, f) for f in files]
    brain, cortex = _cortex(runs, f"{subject} {session}")
    labels = [r[0] for r in runs]
    events = [r[3] for r in runs]
    return Session(
        subject=subject,
        session=session,
        labels=tuple(labels),
        signals=tuple(
            np.asarray(r[1].dataobj, dtype=np.float32)[:, cortex] for r in runs
        ),
        events=tuple(events),
        confounds=tuple(
            glmsingle_polynomials(len(r[2]), float(r[1].header.get_axis(0).step))
            for r in runs
        ),
        frame_times=tuple(r[2] for r in runs),
        brain=brain[cortex],
        task_model=detect_task_model(events, labels=labels),
        source="ppdata",
    )


def ppdata_brain(config, subject, session):
    run = _events_files(config, subject, session)[0].name.split("_")[3]
    brain = nib.load(_existing_path(config, subject, session, run)).header.get_axis(1)
    return brain[cortical_indices(brain)]
