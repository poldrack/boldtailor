"""Conservatively import matching HRFs exported by the boldtailor workflow."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.cifti import spatial_signature
from boldtailor.model import HRF_NORMALIZATION
from boldtailor.hrf_library import HrfLibrary
from boldtailor.workflow.inputs import (
    _trimmed_sources,
    make_blocks,
    selection_task_model,
)
from boldtailor.workflow.outputs import input_artifacts, metadata
from .session_hrf_cache import read_selection

METADATA_KEYS = (
    "library_fingerprint",
    "retained_scans",
    "trimming",
    "nuisance_columns",
    "high_pass",
    "hrf_selection",
)
_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    IndexError,
    nib.filebasedimages.ImageFileError,
)


def _same_library(root, stem, library):
    table = pd.read_csv(
        root / f"{stem}_desc-HRF_library.tsv",
        sep="\t",
        float_precision="round_trip",
    )
    recovered = HrfLibrary.from_table(table)
    with np.load(root / f"{stem}_desc-HRF_library.npz", allow_pickle=False) as curves:
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


def _expected_metadata(runs, library, settings, task_model):
    """The settings-file fields a selection-only analysis would have written."""
    return metadata(
        runs,
        library,
        settings,
        task_model,
        beta_models={},
        activation=None,
        selections=None,
        skipped=(),
        report=None,
    )


def _compatible_export(directory, settings, expected, limit, n_features):
    path = directory / f"{settings.stem}_desc-boldtailor_metadata.json"
    if not path.is_file():
        return False
    saved = json.loads(path.read_text())
    if any(saved[k] != expected[k] for k in METADATA_KEYS):
        return False
    saved_limit = saved["settings"].get("max_grayordinates")
    covered = n_features if saved_limit is None else min(saved_limit, n_features)
    return covered == limit


def _same_inputs(directory, artifacts):
    return all((directory / a.path).read_bytes() == a.payload for a in artifacts)


def _read_export(directory, settings, brain, library, limit):
    stem = settings.stem
    path = (
        directory
        / f"{stem}_{settings.space_entity}_desc-HRFAll_stat-selection.dscalar.nii"
    )
    maps = read_selection(path, brain, library, limit)
    records = json.loads((directory / f"{stem}_desc-HRF_provenance.json").read_text())
    return maps, records, str(path)


def _import_from(directory, runs, settings, library, expected, inputs, task_model):
    brain = runs[0].image.header.get_axis(1)
    limit = inputs["limit"]
    if not _compatible_export(directory, settings, expected, limit, len(brain)):
        return None
    stem = settings.stem
    if not _same_library(directory, stem, library):
        return None
    if not _same_inputs(directory, inputs["artifacts"]):
        return None
    maps, records, path = _read_export(directory, settings, brain, library, limit)
    root = settings.bids_dir
    if _matching_provenance(records, runs, root, library, maps, limit, task_model):
        return maps, records, path
    return None


def find_workflow_estimate(roots, runs, settings, library, request, task_model):
    """Return maps and provenance only when input identities and settings agree.

    ``task_model`` is the session's full GLM model; HRF selection used its
    ``settings.hrf_selection_rt`` subset, recorded in ``request``.
    """
    selection_model = selection_task_model(task_model, settings.hrf_selection_rt)
    expected = _expected_metadata(runs, library, settings, task_model)
    inputs = dict(
        limit=request["grayordinate_limit"],
        artifacts=input_artifacts(settings, runs, task_model),
    )
    for directory in dict.fromkeys(Path(p) for p in roots):
        try:
            imported = _import_from(
                directory, runs, settings, library, expected, inputs, selection_model
            )
        except _ERRORS:
            continue
        if imported is not None:
            return imported
    return None
