"""Read-only, run-at-a-time checks for the portable simulation inputs."""

from io import BytesIO

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from experiments.hrf_estimation.design import partition_events
from experiments.hrf_estimation.timing import build_nuisance, sampling_interval
from experiments.hrf_simulations.bundle import _check, _read_json


def _global_checks(handle, manifest):
    ids = handle["feature_ids"][:]
    brain = nib.cifti2.BrainModelAxis(**_read_json(handle, "brain"))
    regions = {name: data[:] for name, data in handle["rois"].items()}
    valid_ids = (
        ids.ndim == 1
        and ids.dtype.kind in "iu"
        and len(ids) > 0
        and np.all(ids >= 0)
        and len(np.unique(ids)) == len(ids)
    )
    checks = dict(
        unique_nonnegative_feature_ids=valid_ids,
        matching_brain_axis=len(brain) == len(ids),
        three_named_rois=set(regions)
        == {"fusiform", "lateraloccipital", "inferiortemporal"},
    )
    for name, members in regions.items():
        checks[f"{name}_membership"] = (
            members.ndim == 1
            and members.dtype.kind in "iu"
            and len(members) > 0
            and len(np.unique(members)) == len(members)
            and np.isin(members, ids).all()
        )
    combined = np.concatenate(list(regions.values())) if regions else np.array([])
    checks["roi_union_matches_export"] = np.array_equal(
        np.unique(combined), np.sort(ids)
    )
    checks["rois_do_not_overlap"] = len(combined) == len(np.unique(combined))
    count = sum(
        len(session)
        for subject in handle["subjects"].values()
        for session in subject.values()
    )
    checks["manifest_run_count"] = count == manifest["runs"]
    return pd.DataFrame([dict(check=k, ok=bool(v)) for k, v in checks.items()])


def _run_check(group, n_vertices, full_signal_scan):
    times = group["frame_times"][:]
    tr = sampling_interval(times)
    signals = group["signals"]
    if signals.shape != (len(times), n_vertices):
        raise ValueError("signal shape does not match timing/vertex counts")
    events = partition_events(
        pd.read_csv(BytesIO(group["events"][:].tobytes()), sep="\t")
    )
    motion = pd.read_csv(BytesIO(group["motion"][:].tobytes()), sep="\t")
    nuisance = build_nuisance(motion, times, cutoff=80.0)
    if not np.isfinite(nuisance.to_numpy()).all():
        raise ValueError("nonfinite nuisance design")
    columns = (
        np.arange(n_vertices)
        if full_signal_scan
        else np.unique(np.linspace(0, n_vertices - 1, min(3, n_vertices)).astype(int))
    )
    values = signals[:] if full_signal_scan else signals[:, columns]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite signal values")
    valid = events[events.category == "valid"]
    return dict(
        n_frames=len(times),
        tr=tr,
        n_vertices=n_vertices,
        checked_vertices=len(columns),
        constant_vertices=int((values.std(axis=0) == 0).sum()),
        n_events=len(events),
        n_valid=len(valid),
        n_missing=int((events.category == "missing").sum()),
        n_incorrect=int((events.category == "incorrect").sum()),
        rt_median=float(valid.response_time.median()),
        n_nuisance=nuisance.shape[1],
        events_outside_acquisition=int(
            (
                (events.onset < times[0])
                | (events.onset + events.duration > times[-1] + tr)
            ).sum()
        ),
    )


def validate_bundle(path, full_signal_scan=False):
    """Check every run's metadata; sample three columns or read all BOLD values.

    HDF5 checksums apply to chunks read. This does not rehash original NSD files
    or validate anatomical registration against the unavailable source surfaces.
    """
    rows = []
    with h5py.File(path, "r") as handle:
        _check(handle)
        manifest = _read_json(handle, "manifest")
        checks = _global_checks(handle, manifest)
        n_vertices = len(handle["feature_ids"])
        for subject, sessions in handle["subjects"].items():
            for session, groups in sessions.items():
                for run_id, group in groups.items():
                    row = dict(
                        subject=subject,
                        session=int(session.removeprefix("ses-nsd")),
                        run_id=run_id,
                    )
                    try:
                        row.update(
                            _run_check(group, n_vertices, full_signal_scan),
                            ok=True,
                            error="",
                        )
                    except (ValueError, KeyError, OSError, TypeError) as error:
                        row.update(ok=False, error=str(error))
                    rows.append(row)
    return dict(
        global_checks=checks,
        runs=pd.DataFrame(rows),
        skipped=pd.DataFrame(manifest.get("skipped", [])),
        manifest=manifest,
        full_signal_scan=full_signal_scan,
    )
