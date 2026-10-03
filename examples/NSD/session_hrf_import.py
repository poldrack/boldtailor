"""Conservatively import matching HRFs exported by the full NSD notebook."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.cifti import spatial_signature
from boldtailor.model import HRF_NORMALIZATION
from boldtailor.hrf_library import HrfLibrary
from .session_hrf_cache import read_selection
from .workflow_outputs import _input_artifacts, _metadata, _stem
from .workflow_inputs import _trimmed_sources, make_blocks, selection_task_model


def _same_library(root, stem, library):
    table = pd.read_csv(
        root / f"{stem}_desc-notebookHRF_library.tsv",
        sep="\t",
        float_precision="round_trip",
    )
    recovered = HrfLibrary.from_table(table)
    with np.load(
        root / f"{stem}_desc-notebookHRF_library.npz", allow_pickle=False
    ) as curves:
        return (
            recovered.fingerprint == library.fingerprint
            and np.array_equal(curves["curves"], library.curves)
            and np.array_equal(curves["times"], library.times)
        )


def _matching_provenance(records, runs, root, library, maps, limit, task_model):
    covered = set()
    brain = runs[0].image.header.get_axis(1)
    for block in records:
        indices = np.asarray(block["grayordinate_indices"])
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= limit)
            or len(set(indices)) != len(indices)
            or covered.intersection(indices)
        ):
            return False
        covered.update(indices)
        selection = block["scopes"]["all"]
        provenance = selection["selection_provenance"]
        current = [_trimmed_sources(r, root, indices).to_dict() for r in runs]
        if provenance["sources"] != current:
            return False
        activity = provenance["activities"][-1]
        if (
            activity["name"] != "hrf_selection"
            or activity["library_fingerprint"] != library.fingerprint
            or activity["run_labels"] != [r.label for r in runs]
            or activity["oversampling"] != 50
            or activity["feature_signature"] != spatial_signature(brain, indices)
            or activity.get("task_model_fingerprint") != task_model.fingerprint
            or activity.get("hrf_normalization") != HRF_NORMALIZATION
        ):
            return False
        ids = np.asarray(selection["hrf_indices"])
        if not np.array_equal(
            np.where(ids < 0, np.nan, ids), maps[0, indices], equal_nan=True
        ):
            return False
    expected = set(np.concatenate(make_blocks(runs, max_grayordinates=limit)))
    return covered == expected and set(np.flatnonzero(np.isfinite(maps[0]))).issubset(
        covered
    )


def find_workflow_estimate(roots, runs, root, library, request, subject, session):
    """Return maps and provenance only when input identities and settings agree."""
    stem = _stem(subject, session)
    brain = runs[0].image.header.get_axis(1)
    limit = request["grayordinate_limit"]
    include_rt = "response_time" in request["task_model"]["regressors"]
    task_model = selection_task_model(include_rt)
    expected = _metadata(runs, library, {"hrf_selection_rt": include_rt})
    for directory in dict.fromkeys(Path(p) for p in roots):
        metadata_path = directory / f"{stem}_desc-notebook_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
            keys = (
                "library_fingerprint",
                "retained_scans",
                "trimming",
                "nuisance_columns",
                "high_pass",
                "hrf_selection",
            )
            if any(metadata[k] != expected[k] for k in keys):
                continue
            saved_limit = metadata["settings"].get("max_grayordinates")
            if (
                len(brain) if saved_limit is None else min(saved_limit, len(brain))
            ) != limit:
                continue
            if not _same_library(directory, stem, library):
                continue
            if any(
                (directory / a.path).read_bytes() != a.payload
                for a in _input_artifacts(stem, runs)
            ):
                continue
            path = (
                directory
                / f"{stem}_space-fsLR_den-91k_desc-notebookHRFAll_stat-selection.dscalar.nii"
            )
            maps = read_selection(path, brain, library, limit)
            records = json.loads(
                (directory / f"{stem}_desc-notebookHRF_provenance.json").read_text()
            )
            if _matching_provenance(
                records, runs, root, library, maps, limit, task_model
            ):
                return maps, records, str(path)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            nib.filebasedimages.ImageFileError,
        ):
            continue
    return None
