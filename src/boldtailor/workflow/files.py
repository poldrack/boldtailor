"""BIDS/fMRIPrep discovery, CIFTI loading, nuisance selection, and sources."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.provenance import RunSources, SourceRef

MOTION = tuple(
    axis + suffix
    for axis in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
    for suffix in ("", "_derivative1", "_power2", "_derivative1_power2")
)


def bids_label(value, entity):
    """Whether value is a BIDS label such as ``sub-07`` for entity ``sub``."""
    return re.fullmatch(rf"{entity}-[A-Za-z0-9]+", value) is not None


@dataclass(frozen=True)
class RunInputs:
    stem: str
    bold: Path
    events: Path
    confounds: Path
    confounds_json: Path
    bold_json: Path


@dataclass(frozen=True)
class RawRun:
    inputs: RunInputs
    image: nib.Cifti2Image
    events: pd.DataFrame
    confounds: pd.DataFrame
    frame_times: np.ndarray
    label: str
    number: int


def input_paths(run):
    return run.bold, run.events, run.confounds, run.confounds_json, run.bold_json


def _run_inputs(event, derivative, space_entity):
    stem = event.name.removesuffix("_events.tsv")
    bold = derivative / f"{stem}_{space_entity}_bold.dtseries.nii"
    item = RunInputs(
        stem,
        bold,
        event,
        derivative / f"{stem}_desc-confounds_timeseries.tsv",
        derivative / f"{stem}_desc-confounds_timeseries.json",
        bold.with_name(bold.name.replace(".dtseries.nii", ".json")),
    )
    for path in input_paths(item):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input for {stem}: {path}")
    return item


def discover_runs(settings):
    """Require a unique CIFTI in the settings' space and complete metadata per run."""
    func = Path(settings.subject) / settings.session / "func"
    raw = settings.bids_dir / func
    derivative = settings.fmriprep_dir / func
    prefix = f"{settings.subject}_{settings.session}_task-{settings.task}_run-*"
    pattern = f"{prefix}_events.tsv"
    events = sorted(raw.glob(pattern))
    if not events:
        raise FileNotFoundError(f"No events matching {pattern} found in {raw}")
    runs = [_run_inputs(event, derivative, settings.space_entity) for event in events]
    found = set(derivative.glob(f"{prefix}_{settings.space_entity}_bold.dtseries.nii"))
    if found != {run.bold for run in runs}:
        raise ValueError("CIFTI runs and events runs do not match")
    return tuple(runs)


def _acompcor(table, metadata):
    components = [
        name
        for name, info in metadata.items()
        if name.startswith("a_comp_cor_")
        and info.get("Retained") is True
        and info.get("Mask") == "combined"
        and name in table
    ]
    components.sort(
        key=lambda name: (-float(metadata[name]["VarianceExplained"]), name)
    )
    if len(components) < 6:
        raise ValueError("Require six retained combined-mask aCompCor components")
    return components[:6]


def _confound_names(table, metadata):
    cosine = sorted(name for name in table if name.startswith("cosine"))
    if not cosine:
        raise ValueError("Missing fMRIPrep cosine high-pass regressors")
    spikes = sorted(
        name for name in table if name.startswith("non_steady_state_outlier")
    )
    names = list(MOTION) + _acompcor(table, metadata) + cosine + spikes
    missing = set(names) - set(table)
    if missing:
        raise ValueError(f"Missing confounds: {sorted(missing)}")
    return names


def select_confounds(table: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Select motion24, top six combined-mask aCompCor, cosines, and NSS spikes."""
    names = _confound_names(table, metadata)
    selected = table.loc[:, names].apply(pd.to_numeric, errors="raise").copy()
    if selected.empty:
        raise ValueError("Confounds must have rows")
    # Only temporal derivatives are undefined at the first acquired volume.
    for name in MOTION:
        if "derivative1" in name and pd.isna(selected.iloc[0][name]):
            selected.loc[selected.index[0], name] = 0.0
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError("Selected confounds must be finite beyond initial derivatives")
    return selected


def _frame_times(inputs, image):
    series, brain = (image.header.get_axis(i) for i in (0, 1))
    if not isinstance(series, nib.cifti2.SeriesAxis) or series.unit != "SECOND":
        raise ValueError("CIFTI time axis must be a SeriesAxis in seconds")
    if not isinstance(brain, nib.cifti2.BrainModelAxis):
        raise ValueError("CIFTI spatial axis must be a BrainModelAxis")
    timing = json.loads(inputs.bold_json.read_text())
    if not np.isclose(timing["RepetitionTime"], series.step):
        raise ValueError(f"{inputs.stem}: inconsistent repetition time")
    if timing.get("SliceTimingCorrected") and "StartTime" not in timing:
        raise ValueError(f"{inputs.stem}: slice-timing correction needs StartTime")
    offset = float(timing.get("StartTime", series.start))
    return offset + np.arange(image.shape[0]) * series.step


def load_inputs(inputs: RunInputs):
    """Load image, raw events, selected nuisances, and sidecar-corrected frame times."""
    image = nib.load(inputs.bold)
    if not isinstance(image, nib.Cifti2Image):
        raise ValueError(f"{inputs.stem}: expected CIFTI image")
    times = _frame_times(inputs, image)
    confounds = select_confounds(
        pd.read_csv(inputs.confounds, sep="\t"),
        json.loads(inputs.confounds_json.read_text()),
    )
    if len(confounds) != len(times):
        raise ValueError(f"{inputs.stem}: confound rows must match CIFTI volumes")
    return image, pd.read_csv(inputs.events, sep="\t"), confounds, times


def _raw_run(item):
    image, events, confounds, times = load_inputs(item)
    label = item.stem.rsplit("_", 1)[-1]
    number = int(label.removeprefix("run-"))
    return RawRun(item, image, events, confounds, times, label, number)


def load_runs(inputs):
    """Load runs in BIDS run order; every run must share one BrainModel axis."""
    runs = sorted((_raw_run(item) for item in inputs), key=lambda run: run.number)
    if len({r.number for r in runs}) != len(runs):
        raise ValueError("BIDS run numbers must be unique")
    brain = runs[0].image.header.get_axis(1)
    if any(r.image.header.get_axis(1) != brain for r in runs):
        raise ValueError("All runs must have identical grayordinate BrainModel axes")
    return tuple(runs), brain


def source_ref(path, root, role, **annotations):
    stat = path.stat()
    return SourceRef(
        role=role,
        uri=path.relative_to(root).as_posix(),
        byte_size=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        annotations=annotations,
    )


def run_sources(run, root, indices):
    """Source records for one run's BOLD (with grayordinates), events, confounds."""
    inputs = run.inputs
    return RunSources(
        signal=source_ref(
            inputs.bold,
            root,
            "signal",
            grayordinate_indices=indices.tolist(),
            sidecar=source_ref(inputs.bold_json, root, "signal").to_dict(),
        ),
        events=source_ref(inputs.events, root, "events"),
        confounds=source_ref(
            inputs.confounds,
            root,
            "confounds",
            sidecar=source_ref(inputs.confounds_json, root, "confounds").to_dict(),
        ),
    )


def odd_even_parity(runs):
    """Run positions with odd and even BIDS run numbers, in input order."""
    return dict(
        odd=[i for i, r in enumerate(runs) if r.number % 2],
        even=[i for i, r in enumerate(runs) if not r.number % 2],
    )


def reaction_times(runs, *, missing_ok=False):
    """Per-run RT arrays; unavailable (nonpositive or missing) RTs are NaN.

    With ``missing_ok``, raw event tables may lack ``response_time`` or hold
    non-numeric entries; those trials get NaN.
    """
    if not missing_ok:
        return [r.events.response_time.to_numpy() for r in runs]
    return [
        pd.to_numeric(
            r.events.get("response_time", pd.Series(np.nan, index=r.events.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        for r in runs
    ]
