"""In-memory JSON, TSV, NPZ, PNG, and CIFTI artifacts for workflow outputs."""

from importlib.metadata import version
from io import BytesIO
import json

import numpy as np

from boldtailor.cifti import scalar_artifact
from boldtailor.hrf_library import PARAMETER_NAMES
from boldtailor.publication import Artifact


def json_artifact(path, value):
    """Indented, strict (no NaN) JSON with a trailing newline."""
    return Artifact(
        path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    )


def table_artifact(path, frame):
    """Tab-separated table without the index; missing values as ``n/a``."""
    return Artifact(path, frame.to_csv(sep="\t", index=False, na_rep="n/a").encode())


def npz_artifact(path, **arrays):
    """Compressed NPZ of named arrays (load with ``allow_pickle=False``)."""
    with BytesIO() as stream:
        np.savez_compressed(stream, **arrays)
        return Artifact(path, stream.getvalue())


def figure_artifact(path, figure):
    """PNG at 130 dpi; the figure is cleared afterwards."""
    try:
        with BytesIO() as stream:
            figure.savefig(stream, format="png", dpi=130)
            return Artifact(path, stream.getvalue())
    finally:
        figure.clear()


def dataset_description(name):
    """BIDS ``dataset_description.json`` for a boldtailor derivative."""
    return json_artifact(
        "dataset_description.json",
        {
            "Name": name,
            "BIDSVersion": "1.11.1",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "boldtailor", "Version": version("boldtailor")}],
        },
    )


def parameter_artifact(brain, library, ids, path):
    """Look up the exact selected library entry, preserving undefined features."""
    names = [*PARAMETER_NAMES, "peak_time"]
    parameters = np.full((len(names), len(brain)), np.nan)
    valid = np.isfinite(ids)
    parameters[:, valid] = (
        library.parameter_table[names].to_numpy()[ids[valid].astype(int)].T
    )
    return scalar_artifact(path, brain, parameters, names)


def scalar_map(stem, space_entity, brain, descriptor, statistic, values, names):
    """A ``<space>_desc-<descriptor>_stat-<statistic>`` dense scalar map."""
    path = f"{stem}_{space_entity}_desc-{descriptor}_stat-{statistic}.dscalar.nii"
    return scalar_artifact(path, brain, values, names)
