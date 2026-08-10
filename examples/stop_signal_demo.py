from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import mimetypes
from pathlib import Path
import tempfile
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from boldtailor.provenance import RunSources, SourceRef
from boldtailor.publication import Artifact
from boldtailor.results import AnalysisResult, TaskDeltaR2Result

_CONTRAST_LABELS = (
    ("successful_inhibition", "successfulInhibition"),
    ("stop_vs_go", "stopVsGo"),
    ("go_success_vs_baseline", "goSuccessVsBaseline"),
)


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


def common_brain_mask(inputs: Sequence[RunInputs]) -> nib.Nifti1Image:
    if not inputs:
        raise ValueError("common brain mask requires at least one run")
    masks = tuple(nib.load(item.mask) for item in inputs)
    _require_same_geometry(masks, "run masks must share shape and affine")
    common = np.logical_and.reduce(
        [np.asarray(mask.dataobj, dtype=bool) for mask in masks]
    )
    if not common.any():
        raise ValueError("run mask intersection must contain at least one voxel")
    header = masks[0].header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(common.astype(np.uint8), masks[0].affine, header)


def make_masker(mask_image: nib.Nifti1Image) -> NiftiMasker:
    return NiftiMasker(
        mask_img=mask_image,
        standardize=False,
        detrend=False,
        smoothing_fwhm=None,
        low_pass=None,
        high_pass=None,
        reports=False,
    ).fit()


def load_run(
    inputs: RunInputs,
    masker: NiftiMasker,
    *,
    trial_types: Sequence[str],
    confound_names: Sequence[str],
) -> LoadedRun:
    image = nib.load(inputs.bold)
    _require_bold_geometry(image, masker)
    n_scans = image.shape[3]
    tr = float(image.header.get_zooms()[3])
    events = _model_events(inputs.events, trial_types, n_scans, tr)
    confounds = _selected_confounds(inputs.confounds, confound_names, n_scans)
    return LoadedRun(
        inputs=inputs,
        signals=_transformed_signals(image, masker, n_scans),
        events=events,
        confounds=confounds,
        frame_times=_frame_times(n_scans, tr),
        tr=tr,
    )


def whole_brain_image(values: np.ndarray, masker: NiftiMasker) -> nib.Nifti1Image:
    restored = np.asarray(values, dtype=float)
    voxel_count = _mask_voxel_count(masker)
    if restored.ndim != 1 or len(restored) != voxel_count:
        raise ValueError(
            "whole-brain values must be one-dimensional and match mask voxel count"
        )
    if not np.isfinite(restored).all():
        raise ValueError("whole-brain values must be finite")
    return masker.inverse_transform(restored)


def estimate_signal_memory_gib(scan_counts: Sequence[int], voxel_count: int) -> float:
    counts = tuple(scan_counts)
    if not counts or any(count <= 0 for count in counts) or voxel_count <= 0:
        raise ValueError("memory estimate requires positive scan and voxel counts")
    return float(sum(counts) * voxel_count * 8 / 2**30)


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
    masker: NiftiMasker,
    common_mask: nib.Nifti1Image,
    *,
    subject: str,
    task: str,
    sessions: Sequence[str],
    space: str,
    resolution: int,
    configuration: Mapping[str, object],
    task_delta: TaskDeltaR2Result,
) -> tuple[Artifact, ...]:
    _require_spatial_context(masker, common_mask, space, resolution)
    _require_task_delta_dimensions(task_delta, masker)
    designs = tuple(
        Artifact(
            f"reports/{subject}_{session}_task-{task}_desc-design_matrix.tsv",
            _tsv_bytes(design.rename_axis("frame_time").reset_index()),
        )
        for session, design in zip(sessions, result.design_matrices, strict=True)
    )
    reports = _result_reports(
        result,
        subject=subject,
        task=task,
        configuration=configuration,
    )
    images = _result_images(
        result,
        masker,
        common_mask,
        subject=subject,
        task=task,
        sessions=sessions,
        space=space,
        resolution=resolution,
        task_delta=task_delta,
    )
    return (
        designs
        + reports
        + images
        + (_image_manifest(images, subject=subject, task=task),)
    )


def _require_spatial_context(
    masker: NiftiMasker,
    common_mask: nib.Nifti1Image,
    space: str,
    resolution: int,
) -> None:
    if any(value is None for value in (masker, common_mask, space, resolution)):
        raise ValueError(
            "image artifacts require masker, common mask, space, and resolution"
        )
    _require_common_mask_matches_masker(common_mask, masker)


def _require_common_mask_matches_masker(
    common_mask: nib.Nifti1Image, masker: NiftiMasker
) -> None:
    masker_mask = masker.mask_img_
    common_values = np.asarray(common_mask.dataobj, dtype=bool)
    masker_values = np.asarray(masker_mask.dataobj, dtype=bool)
    if (
        common_mask.shape != masker_mask.shape
        or not np.allclose(common_mask.affine, masker_mask.affine)
        or not np.array_equal(common_values, masker_values)
    ):
        raise ValueError("common mask must match fitted masker")


def _require_task_delta_dimensions(
    task_delta: TaskDeltaR2Result, masker: NiftiMasker
) -> None:
    values = task_delta.delta_r2
    if values.ndim != 1 or values.size != _mask_voxel_count(masker):
        raise ValueError("task delta r-squared values must match mask voxel count")


def _result_reports(
    result: AnalysisResult,
    *,
    subject: str,
    task: str,
    configuration: Mapping[str, object],
) -> tuple[Artifact, ...]:
    return (
        Artifact(
            f"reports/{subject}_task-{task}_desc-wholebrain_contrasts.tsv",
            _tsv_bytes(_contrast_summary(result)),
        ),
        Artifact(
            f"reports/{subject}_task-{task}_desc-example_config.json",
            _json_bytes(configuration),
        ),
    )


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


def _result_images(
    result: AnalysisResult,
    masker: NiftiMasker,
    common_mask: nib.Nifti1Image,
    *,
    subject: str,
    task: str,
    sessions: Sequence[str],
    space: str,
    resolution: int,
    task_delta: TaskDeltaR2Result,
) -> tuple[Artifact, ...]:
    stem = _image_stem(subject, task, space, resolution)
    mask = _image_artifact(f"{stem}_desc-common_mask.nii.gz", common_mask)
    contrasts = _contrast_images(result, masker, stem)
    run_r2 = _run_r2_images(
        result,
        masker,
        subject=subject,
        task=task,
        sessions=sessions,
        space=space,
        resolution=resolution,
    )
    aggregate = _image_artifact(
        f"{stem}_desc-aggregate_stat-r2_statmap.nii.gz",
        whole_brain_image(result.r2, masker),
    )
    delta = _image_artifact(
        f"{stem}_desc-taskDelta_stat-r2_statmap.nii.gz",
        whole_brain_image(task_delta.raw_delta_r2, masker),
    )
    return (mask,) + contrasts + run_r2 + (aggregate, delta)


def _contrast_images(
    result: AnalysisResult, masker: NiftiMasker, stem: str
) -> tuple[Artifact, ...]:
    images = []
    for name, label in _CONTRAST_LABELS:
        images.extend(
            (
                _image_artifact(
                    f"{stem}_contrast-{label}_stat-effect_statmap.nii.gz",
                    whole_brain_image(result.effect(name), masker),
                ),
                _image_artifact(
                    f"{stem}_contrast-{label}_stat-z_statmap.nii.gz",
                    whole_brain_image(result.z_score(name), masker),
                ),
            )
        )
    return tuple(images)


def _run_r2_images(
    result: AnalysisResult,
    masker: NiftiMasker,
    *,
    subject: str,
    task: str,
    sessions: Sequence[str],
    space: str,
    resolution: int,
) -> tuple[Artifact, ...]:
    return tuple(
        _image_artifact(
            f"{_image_stem(subject, task, space, resolution, session)}_stat-r2_statmap.nii.gz",
            whole_brain_image(values, masker),
        )
        for session, values in zip(sessions, result.run_r2, strict=True)
    )


def _image_stem(
    subject: str,
    task: str,
    space: str,
    resolution: int,
    session: str | None = None,
) -> str:
    subject_entities = subject if session is None else f"{subject}_{session}"
    return f"images/{subject_entities}_task-{task}_space-{space}_res-{resolution}"


def _nifti_bytes(image: nib.Nifti1Image) -> bytes:
    return gzip.compress(image.to_bytes(), compresslevel=9, mtime=0)


def _image_artifact(path: str, image: nib.Nifti1Image) -> Artifact:
    return Artifact(path, _nifti_bytes(image))


def _image_manifest(images: Sequence[Artifact], *, subject: str, task: str) -> Artifact:
    frame = pd.DataFrame(
        [
            {
                "relative_path": image.path,
                "media_type": "application/gzip",
                "byte_size": len(image.payload),
                "sha256": hashlib.sha256(image.payload).hexdigest(),
            }
            for image in images
        ]
    )
    return Artifact(
        f"reports/{subject}_task-{task}_desc-image_manifest.tsv",
        _tsv_bytes(frame),
    )


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


def _require_same_geometry(images: Sequence[nib.Nifti1Image], message: str) -> None:
    reference = images[0]
    if any(
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine)
        for image in images[1:]
    ):
        raise ValueError(message)


def _require_bold_geometry(image: nib.Nifti1Image, masker: NiftiMasker) -> None:
    mask_image = masker.mask_img_
    if image.shape[:3] != mask_image.shape or not np.allclose(
        image.affine, mask_image.affine
    ):
        raise ValueError("BOLD image must match common mask geometry")


def _mask_voxel_count(masker: NiftiMasker) -> int:
    return int(np.asarray(masker.mask_img_.dataobj, dtype=bool).sum())


def _transformed_signals(
    image: nib.Nifti1Image, masker: NiftiMasker, n_scans: int
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="boolean values for 'standardize' will be deprecated.*",
            category=FutureWarning,
        )
        signals = np.asarray(masker.transform(image), dtype=float)
    if signals.ndim != 2 or signals.shape[0] != n_scans:
        raise ValueError("transformed signals must have one row per scan")
    if signals.shape[1] != _mask_voxel_count(masker):
        raise ValueError("transformed signals must match common mask voxel count")
    if not np.isfinite(signals).all():
        raise ValueError("whole-brain signals must be finite")
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
