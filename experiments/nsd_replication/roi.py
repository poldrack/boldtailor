"""Subject nsdgeneral, resampled to fsLR 32k exactly as the released betas were."""

import json
import os
from pathlib import Path
import subprocess
import tempfile

import nibabel as nib
import numpy as np

FSLR_VERTICES = 32492
_HEMI = {"CIFTI_STRUCTURE_CORTEX_LEFT": "L", "CIFTI_STRUCTURE_CORTEX_RIGHT": "R"}
_KEYS = {
    "source_sphere": "SourceSpheres",
    "target_sphere": "TargetSpheres",
    "source_area": "SourceAreaMetrics",
    "target_area": "TargetAreaMetrics",
}
_SIDECAR = (
    "{subject}/{session}/func/{subject}_{session}_task-{task}_space-fsLR_den-91k"
    "_desc-assumehrf_stat-effect_statmap.json"
)


def label_paths(config, subject):
    label = config.freesurfer_dir / f"subj{subject.removeprefix('sub-')}" / "label"
    return {"L": label / "lh.nsdgeneral.mgz", "R": label / "rh.nsdgeneral.mgz"}


def resampling_inputs(sidecar):
    missing = [key for key in _KEYS.values() if key not in sidecar]
    if missing:
        raise ValueError(f"beta sidecar lacks resampling inputs: {missing}")
    return {
        hemi: {name: Path(sidecar[key][i]) for name, key in _KEYS.items()}
        for i, hemi in enumerate(("L", "R"))
    }


def resample_command(metric_in, metric_out, inputs):
    return [
        "wb_command",
        "-metric-resample",
        str(metric_in),
        str(inputs["source_sphere"]),
        str(inputs["target_sphere"]),
        "ADAP_BARY_AREA",
        str(metric_out),
        "-area-metrics",
        str(inputs["source_area"]),
        str(inputs["target_area"]),
    ]


def _save_metric(values, path):
    array = nib.gifti.GiftiDataArray(np.asarray(values, dtype=np.float32))
    nib.save(nib.gifti.GiftiImage(darrays=[array]), path)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr}")


def _resample(source, target, inputs):
    """wb_command writes beside ``target``; the finished file is moved into
    place so the cache never holds a partial file."""
    partial = target.with_name(f".partial-{target.name}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            metric = Path(tmp) / "native.func.gii"
            _save_metric(np.asarray(nib.load(source).dataobj).ravel(), metric)
            _run(resample_command(metric, partial, inputs))
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def fslr_labels(config, subject, sidecar):
    inputs = resampling_inputs(sidecar)
    out = config.output_dir / "roi"
    out.mkdir(parents=True, exist_ok=True)
    labels = {}
    for hemi, source in label_paths(config, subject).items():
        target = out / f"{subject}_hemi-{hemi}_nsdgeneral.func.gii"
        if not target.is_file():
            _resample(source, target, inputs[hemi])
        labels[hemi] = nib.load(target).agg_data().astype(float)
    return labels


def roi_mask(brain, labels, threshold=0.5):
    for hemi, values in labels.items():
        if len(values) != FSLR_VERTICES:
            raise ValueError(f"{hemi} labels must have {FSLR_VERTICES} vertices")
    hemis = np.array([_HEMI[name] for name in brain.name])
    mask = np.zeros(len(brain), bool)
    for hemi in ("L", "R"):
        rows = hemis == hemi
        mask[rows] = labels[hemi][brain.vertex[rows]] >= threshold
    if not mask.any():
        raise ValueError("nsdgeneral ROI is empty on these grayordinates")
    return mask


def load_roi(config, subject, brain):
    if config.released_dir is None:
        raise ValueError("released_dir is required to read the ROI resampling inputs")
    path = config.released_dir / _SIDECAR.format(
        subject=subject, session=config.sessions[0], task=config.task
    )
    sidecar = json.loads(path.read_text())
    return roi_mask(brain, fslr_labels(config, subject, sidecar))
