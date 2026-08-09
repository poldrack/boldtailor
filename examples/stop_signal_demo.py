from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import mimetypes
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.provenance import RunSources, SourceRef
from boldtailor.publication import Artifact
from boldtailor.results import AnalysisResult


@dataclass(frozen=True, slots=True)
class RunInputs:
    session: str
    events: Path
    bold: Path
    mask: Path
    confounds: Path


@dataclass(frozen=True, slots=True)
class LoadedRun:
    inputs: RunInputs
    signals: np.ndarray
    events: pd.DataFrame
    confounds: pd.DataFrame
    frame_times: np.ndarray
    voxel_indices: np.ndarray
    spatial_shape: tuple[int, int, int]
    affine: np.ndarray
    tr: float


def discover_run_inputs(
    bids_root: Path,
    fmriprep_root: Path,
    *,
    subject: str,
    session: str,
    task: str,
    run: str,
    space: str,
    resolution: int,
) -> RunInputs:
    raw_func = Path(bids_root) / subject / session / "func"
    derivative_func = Path(fmriprep_root) / subject / session / "func"
    stem = f"{subject}_{session}_task-{task}_{run}"
    return RunInputs(
        session=session,
        events=_one(raw_func.glob(f"{stem}*_events.tsv"), "events"),
        bold=_one(
            derivative_func.glob(
                f"{stem}_space-{space}_res-{resolution}_desc-preproc_bold.nii.gz"
            ),
            "BOLD",
        ),
        mask=_one(
            derivative_func.glob(
                f"{stem}_space-{space}_res-{resolution}_desc-brain_mask.nii.gz"
            ),
            "mask",
        ),
        confounds=_one(
            derivative_func.glob(f"{stem}_desc-confounds_timeseries.tsv"),
            "confounds",
        ),
    )


def _one(paths, role: str) -> Path:
    matches = tuple(sorted(paths))
    if not matches:
        raise FileNotFoundError(f"{role} discovery expected exactly one file")
    if len(matches) != 1:
        raise ValueError(f"{role} discovery expected exactly one file")
    return matches[0]


def common_roi_voxels(
    inputs: Sequence[RunInputs],
    *,
    center_mni: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    if not inputs:
        raise ValueError("ROI requires at least one run")
    masks = [nib.load(item.mask) for item in inputs]
    shape = masks[0].shape
    affine = masks[0].affine
    if any(
        mask.shape != shape or not np.allclose(mask.affine, affine) for mask in masks
    ):
        raise ValueError("run masks must share shape and affine")
    indices = np.indices(shape).reshape(3, -1).T
    world = nib.affines.apply_affine(affine, indices)
    in_sphere = np.linalg.norm(world - np.asarray(center_mni), axis=1) <= radius_mm
    in_all_masks = np.logical_and.reduce(
        [np.asarray(mask.dataobj).reshape(-1) > 0 for mask in masks]
    )
    voxels = indices[in_sphere & in_all_masks]
    if not len(voxels):
        raise ValueError("ROI does not overlap every run mask")
    return _immutable_array(voxels, dtype=int)


def load_run(
    inputs: RunInputs,
    voxel_indices: np.ndarray,
    *,
    trial_types: Sequence[str],
    confound_names: Sequence[str],
) -> LoadedRun:
    image = nib.load(inputs.bold)
    voxels = _validated_voxels(voxel_indices, image.shape[:3])
    n_scans = image.shape[3]
    tr = float(image.header.get_zooms()[3])
    events = _model_events(inputs.events, trial_types, n_scans, tr)
    confounds = _selected_confounds(inputs.confounds, confound_names, n_scans)
    return LoadedRun(
        inputs=inputs,
        signals=_extract_signals(image, voxels),
        events=events,
        confounds=confounds,
        frame_times=_frame_times(n_scans, tr),
        voxel_indices=voxels,
        spatial_shape=tuple(int(length) for length in image.shape[:3]),
        affine=_immutable_array(image.affine, dtype=float),
        tr=tr,
    )


def roi_image(values: np.ndarray, loaded: LoadedRun) -> nib.Nifti1Image:
    restored = np.asarray(values, dtype=float)
    if restored.ndim != 1 or len(restored) != len(loaded.voxel_indices):
        raise ValueError("ROI values must be one-dimensional and match voxel count")
    if not np.isfinite(restored).all():
        raise ValueError("ROI values must be finite")
    data = np.zeros(loaded.spatial_shape, dtype=float)
    data[tuple(loaded.voxel_indices.T)] = restored
    return nib.Nifti1Image(data, loaded.affine)


def run_sources(inputs: RunInputs, bids_root: Path) -> RunSources:
    mask = _source_metadata(inputs.mask, bids_root)
    signal = _source_ref(
        inputs.bold, bids_root, role="signal", annotations={"mask": mask}
    )
    return RunSources(
        signal=signal,
        events=_source_ref(inputs.events, bids_root, role="events"),
        confounds=_source_ref(inputs.confounds, bids_root, role="confounds"),
    )


def result_artifacts(
    result: AnalysisResult,
    *,
    subject: str,
    task: str,
    sessions: Sequence[str],
    configuration: Mapping[str, object],
) -> tuple[Artifact, ...]:
    designs = tuple(
        Artifact(
            f"reports/{subject}_{session}_task-{task}_desc-design_matrix.tsv",
            _tsv_bytes(design.rename_axis("frame_time").reset_index()),
        )
        for session, design in zip(sessions, result.design_matrices, strict=True)
    )
    reports = (
        Artifact(
            f"reports/{subject}_task-{task}_desc-roi_contrasts.tsv",
            _tsv_bytes(_contrast_summary(result)),
        ),
        Artifact(
            f"reports/{subject}_task-{task}_desc-example_config.json",
            _json_bytes(configuration),
        ),
    )
    return designs + reports


def publication_destination(
    bids_root: Path,
    *,
    persistent: bool,
    requested: Path | None = None,
    temporary_parent: Path | None = None,
) -> Path:
    root = Path(bids_root).resolve()
    if not persistent:
        parent = None if temporary_parent is None else Path(temporary_parent)
        return Path(tempfile.mkdtemp(prefix="boldtailor-", dir=parent)).resolve()
    expected = root / "derivatives" / "boldtailor"
    if expected.parent.is_symlink() or expected.is_symlink():
        raise ValueError("persistent output must not contain a symlink")
    expected = expected.resolve()
    destination = expected if requested is None else Path(requested).resolve()
    if destination != expected:
        raise ValueError("persistent output must be dataset derivatives/boldtailor")
    return destination


def protected_source_paths(inputs: Sequence[RunInputs]) -> tuple[Path, ...]:
    return tuple(
        path
        for item in inputs
        for path in (item.events, item.bold, item.mask, item.confounds)
    )


def _contrast_summary(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for name in result.contrast_names:
        effect = result.effect(name)
        z_score = result.z_score(name)
        rows.append(
            {
                "contrast": name,
                "n_features": len(effect),
                "mean_effect": np.mean(effect),
                "mean_z": np.mean(z_score),
                "max_abs_z": np.max(np.abs(z_score)),
                "min_one_sided_p": np.min(result.one_sided_p_value(name)),
            }
        )
    return pd.DataFrame(rows)


def _tsv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        sep="\t", index=False, float_format="%.10g", lineterminator="\n"
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _immutable_array(values: np.ndarray, *, dtype: type) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _validated_voxels(voxels: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    indices = np.asarray(voxels, dtype=int)
    if indices.ndim != 2 or indices.shape[1] != 3 or not len(indices):
        raise ValueError("ROI voxel indices must have shape (n, 3)")
    limits = np.asarray(shape)
    if np.any(indices < 0) or np.any(indices >= limits):
        raise ValueError("ROI voxel indices exceed image bounds")
    return _immutable_array(indices, dtype=int)


def _extract_signals(image: nib.Nifti1Image, voxels: np.ndarray) -> np.ndarray:
    lower = voxels.min(axis=0)
    upper = voxels.max(axis=0) + 1
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    block = np.asarray(image.dataobj[slices + (slice(None),)], dtype=float)
    local = voxels - lower
    signals = block[tuple(local.T) + (slice(None),)].T
    if not np.isfinite(signals).all():
        raise ValueError("ROI signals must be finite")
    return _immutable_array(signals, dtype=float)


def _model_events(
    path: Path, trial_types: Sequence[str], n_scans: int, tr: float
) -> pd.DataFrame:
    events = pd.read_csv(path, sep="\t")
    columns = ["onset", "duration", "trial_type"]
    missing = set(columns).difference(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")
    modeled = events.loc[events.trial_type.isin(trial_types), columns].copy()
    if modeled.empty:
        raise ValueError("no modeled events remain")
    timing = modeled[["onset", "duration"]].to_numpy(dtype=float)
    if not np.isfinite(timing).all():
        raise ValueError("event timing must be finite")
    if np.any(timing.sum(axis=1) > n_scans * tr):
        raise ValueError("event timing exceeds acquisition")
    return modeled.copy(deep=True)


def _selected_confounds(
    path: Path, confound_names: Sequence[str], n_scans: int
) -> pd.DataFrame:
    confounds = pd.read_csv(path, sep="\t")
    names = list(confound_names)
    missing = set(names).difference(confounds.columns)
    if missing:
        raise ValueError(f"confounds missing required columns: {sorted(missing)}")
    if len(confounds) != n_scans:
        raise ValueError(f"confounds must contain exactly {n_scans} rows")
    selected = confounds.loc[:, names].copy()
    if "framewise_displacement" in selected and pd.isna(
        selected.loc[selected.index[0], "framewise_displacement"]
    ):
        selected.loc[selected.index[0], "framewise_displacement"] = 0.0
    values = selected.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.all():
        column = selected.columns[np.flatnonzero(~finite.all(axis=0))[0]]
        raise ValueError(f"confounds contain non-finite values in {column}")
    return pd.DataFrame(values, columns=names, index=selected.index).copy(deep=True)


def _frame_times(n_scans: int, tr: float) -> np.ndarray:
    return _immutable_array(np.arange(n_scans, dtype=float) * tr, dtype=float)


def _source_ref(
    path: Path,
    bids_root: Path,
    *,
    role: str,
    annotations: dict[str, object] | None = None,
) -> SourceRef:
    return SourceRef(
        role=role,
        annotations={} if annotations is None else annotations,
        **_source_metadata(path, bids_root),
    )


def _source_metadata(path: Path, bids_root: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "uri": path.relative_to(bids_root).as_posix(),
        "media_type": _media_type(path),
        "byte_size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _media_type(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return "application/gzip"
    if path.suffix == ".tsv":
        return "text/tab-separated-values"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
