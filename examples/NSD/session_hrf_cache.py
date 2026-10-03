"""Fingerprint and publish reusable session HRFs, retaining exact libraries."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from boldtailor.cifti import scalar_artifact, spatial_signature
from boldtailor.model import HRF_NORMALIZATION
from boldtailor.publication import publish_artifact_set
from boldtailor.reliability import library_indices
from .workflow_artifacts import (
    json_artifact,
    npz_artifact,
    parameter_artifact,
    table_artifact,
)
from boldtailor.workflow.files import input_paths
from .workflow_inputs import NSD_TASK_MODEL, _trimmed_sources

MAP_NAMES = ("hrf_id", "selected_cv_r2", "canonical_cv_r2", "delta_cv_r2")


def request_metadata(runs, root, library, limit, task_model=NSD_TASK_MODEL):
    records = []
    for run in runs:
        digest = sha256(run.events.to_csv(index=False).encode())
        digest.update(run.confounds.to_numpy(dtype="<f8").tobytes())
        digest.update(np.asarray(run.frame_times, dtype="<f8").tobytes())
        records.append(
            dict(
                label=run.label,
                sources=_trimmed_sources(run, root, np.array([], dtype=int)).to_dict(),
                confound_columns=list(run.confounds),
                design_inputs=digest.hexdigest(),
            )
        )
    brain = runs[0].image.header.get_axis(1)
    return dict(
        schema="boldtailor.session_hrf/1",
        bids_root=str(Path(root).resolve()),
        library_fingerprint=library.fingerprint,
        grayordinate_limit=limit,
        spatial_fingerprint=spatial_signature(brain, np.arange(len(brain))),
        runs=records,
        model="task_model_leave_one_run_out",
        task_model=task_model.to_dict(),
        hrf_normalization=HRF_NORMALIZATION,
        oversampling=50,
        nuisance="motion24_top6_combined_acompcor_fmriprep_cosines_intercept",
        trimming="leading_nonsteady_volumes_original_acquisition_times",
        software={
            name: version(name)
            for name in ("boldtailor", "nilearn", "numpy", "scipy", "pandas", "nibabel")
        },
    )


def request_id(request):
    return sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cache_paths(subject, session, identity):
    stem = f"{subject}/{session}/func/{subject}_{session}_task-nsdcore"
    desc = f"desc-sessionHRF{identity[:16]}"
    return dict(
        selection=f"{stem}_space-fsLR_den-91k_{desc}_stat-selection.dscalar.nii",
        parameters=f"{stem}_space-fsLR_den-91k_{desc}_stat-hrfparameters.dscalar.nii",
        **{
            name: f"{stem}_{desc}_{name}.{ext}"
            for name, ext in (
                ("metadata", "json"),
                ("provenance", "json"),
                ("library", "tsv"),
                ("curves", "npz"),
            )
        },
    )


def read_selection(path, brain, library, limit):
    image = nib.load(path)
    if (
        image.shape != (4, len(brain))
        or image.header.get_axis(1) != brain
        or list(image.header.get_axis(0).name) != list(MAP_NAMES)
    ):
        raise ValueError(
            "Cached selection has incompatible grayordinate axis or map names"
        )
    maps = image.get_fdata()
    library_indices(maps[0], len(library.candidates))
    if np.isfinite(maps[:, limit:]).any():
        raise ValueError("Cached selection exceeds its requested grayordinate coverage")
    return maps


def load_cache(root, paths, request, brain, library):
    root = Path(root)
    if not (root / paths["metadata"]).is_file():
        return None
    try:
        metadata = json.loads((root / paths["metadata"]).read_text())
        if metadata["request"] != request or metadata["request_id"] != request_id(
            request
        ):
            return None
        for name, path in paths.items():
            if (
                name != "metadata"
                and sha256((root / path).read_bytes()).hexdigest()
                != metadata["sha256"][name]
            ):
                return None
        return read_selection(
            root / paths["selection"], brain, library, request["grayordinate_limit"]
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        nib.filebasedimages.ImageFileError,
    ):
        return None


def save_cache(
    root, paths, request, runs, library, maps, provenance, imported_from=None
):
    brain = runs[0].image.header.get_axis(1)
    artifacts = [
        scalar_artifact(paths["selection"], brain, maps, MAP_NAMES),
        parameter_artifact(brain, library, maps[0], paths["parameters"]),
        table_artifact(paths["library"], library.parameter_table),
        npz_artifact(paths["curves"], times=library.times, curves=library.curves),
        json_artifact(paths["provenance"], provenance),
    ]
    by_path = {a.path: a.payload for a in artifacts}
    metadata = dict(
        request_id=request_id(request),
        request=request,
        imported_from=imported_from,
        sha256={
            k: sha256(by_path[p]).hexdigest()
            for k, p in paths.items()
            if k != "metadata"
        },
    )
    artifacts.append(json_artifact(paths["metadata"], metadata))
    publish_artifact_set(
        root,
        artifacts,
        source_paths=[p for r in runs for p in input_paths(r.inputs)],
        overwrite=True,
    )
