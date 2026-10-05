"""NSD's released GLMsingle betas, converted to fsLR 91k CIFTI, in the common format."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from experiments.nsd_replication.betas import fit_dir, is_complete, write_fit
from experiments.nsd_replication.inputs import (
    cortical_indices,
    ppdata_brain,
    same_grayordinates,
)
from experiments.nsd_replication.metrics import columnwise_corr
from experiments.nsd_replication.trials import trial_table

RELEASED = {"b1": "assumehrf", "b2": "fithrf", "b4": "fithrfGLMdenoiseRR"}
_NAME = (
    "{subject}_{session}_task-{task}_space-fsLR_den-91k_desc-{version}"
    "_stat-effect_statmap.dscalar.nii"
)


def released_path(config, subject, session, level):
    name = _NAME.format(
        subject=subject, session=session, task=config.task, version=RELEASED[level]
    )
    return Path(config.released_dir) / subject / session / "func" / name


def _check_maps(axis, n_trials, name):
    maps = list(axis.name)
    if len(maps) != n_trials:
        raise ValueError(f"{name}: {len(maps)} trials, but the events have {n_trials}")
    if maps != [f"trial-{i:03d}" for i in range(1, n_trials + 1)]:
        raise ValueError(f"{name}: maps are not trial-001... in presentation order")


def load_released(path, brain, n_trials, name):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name}: missing {path}")
    image = nib.load(path)
    axis = image.header.get_axis(1)
    cortex = cortical_indices(axis)
    if not same_grayordinates(axis[cortex], brain):
        raise ValueError(f"{name}: grayordinates differ from the ppdata cortex")
    _check_maps(image.header.get_axis(0), n_trials, name)
    values = np.asarray(image.dataobj, dtype=np.float32)[:, cortex]
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite cortical betas")
    return values


def bids_trials(config, subject, session):
    func = config.bids_dir / subject / session / "func"
    files = sorted(
        func.glob(f"{subject}_{session}_task-{config.task}_run-*_events.tsv")
    )
    if not files:
        raise FileNotFoundError(f"{subject} {session}: no events in {func}")
    events = [pd.read_csv(f, sep="\t") for f in files]
    labels = [f.name.split("_")[3] for f in files]
    return trial_table(events, labels, session, config.image_column)


def _stat(path):
    info = Path(path).stat()
    return {
        "file": str(path),
        "file_size": info.st_size,
        "file_mtime_ns": info.st_mtime_ns,
    }


def _current(target, stat):
    if not is_complete(target):
        return False
    meta = json.loads((target / "metadata.json").read_text())
    return all(meta.get(k) == v for k, v in stat.items())


def _index_level(config, subject, session, level, brain, trials):
    version = RELEASED[level]
    source = released_path(config, subject, session, level)
    target = fit_dir(config.output_dir, "released", subject, session, level)
    stat = _stat(source) if source.is_file() else {}
    if not stat or not _current(target, stat):
        name = f"{subject} {session} {version}"
        betas = load_released(source, brain, len(trials), name)
        meta = {"level": level, "version": version, **stat}
        write_fit(target, betas, trials, meta)
    return target


def index_released(config, subject, session):
    brain = ppdata_brain(config, subject, session)
    trials = bids_trials(config, subject, session)
    return [
        _index_level(config, subject, session, level, brain, trials)
        for level in RELEASED
    ]


def alignment_check(ours_b1, released_b1, run_length, roi):
    ours, released = ours_b1[:, roi], released_b1[:, roi]
    true = columnwise_corr(ours, released)
    null = columnwise_corr(ours, np.roll(released, run_length, axis=0))
    return {
        "true_median": float(np.nanmedian(true)),
        "null_median": float(np.nanmedian(null)),
    }
