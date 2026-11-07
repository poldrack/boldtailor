"""Entity-matched, read-only NSD-preprocessed run loading."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from experiments.hrf_estimation.timing import build_nuisance


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    bold_path: Path
    events_path: Path
    confounds_path: Path


@dataclass(frozen=True)
class RunData:
    run_id: str
    signals: np.ndarray
    events: pd.DataFrame
    frame_times: np.ndarray
    confounds: pd.DataFrame
    feature_ids: np.ndarray
    brain_axis: object = None


def discover_runs(root, subject, sessions):
    root = Path(root)
    runs = []
    for session in sessions:
        ses = f'ses-nsd{session:02}'
        folder = root / 'derivatives/nsdpreproc' / subject / ses / 'func'
        pattern = f'{subject}_{ses}_task-nsdcore_run-*_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii'
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f'no layerB2 CIFTIs for {subject} {ses}')
        runs.extend(_spec(root, path) for path in files)
    return tuple(runs)


def _spec(root, bold):
    run_id = bold.name.split('_space-')[0]
    subject, session = run_id.split('_')[:2]
    events = root / subject / session / 'func' / f'{run_id}_events.tsv'
    confounds = bold.parent / f'{run_id}_desc-confounds_timeseries.tsv'
    for path in (events, confounds):
        if not path.is_file():
            raise FileNotFoundError(f'missing matched file: {path}')
    return RunSpec(run_id, bold, events, confounds)


def _validate_vertices(vertices, size):
    values = np.asarray(vertices)
    if (values.ndim != 1 or len(values) == 0 or values.dtype.kind not in 'iu'
            or len(np.unique(values)) != len(values)
            or np.any(values < 0) or np.any(values >= size)):
        raise ValueError('vertices must be unique valid integer column indices')
    return values.astype(int)


def load_run(spec, vertices, cutoff=80.0):
    image = nib.load(spec.bold_path)
    series, brain = image.header.get_axis(0), image.header.get_axis(1)
    if not isinstance(series, nib.cifti2.SeriesAxis) or not isinstance(brain, nib.cifti2.BrainModelAxis):
        raise ValueError('expected CIFTI time series with a BrainModelAxis')
    vertices = _validate_vertices(vertices, len(brain))
    times = series.start + np.arange(series.size) * series.step
    motion = pd.read_csv(spec.confounds_path, sep='\t')
    nuisance = build_nuisance(motion, times, cutoff)
    # Slice a memory-mapped array; never retain all session signals in RAM.
    signals = np.asarray(np.asanyarray(image.dataobj)[:, vertices], dtype=float)
    if not np.isfinite(signals).all():
        raise ValueError(f'{spec.run_id}: nonfinite BOLD values')
    events = pd.read_csv(spec.events_path, sep='\t')
    return RunData(spec.run_id, signals, events, times, nuisance, vertices, brain[vertices])
