"""In-memory JSON, TSV, NPZ, PNG, and CIFTI artifacts for the NSD notebooks."""

from io import BytesIO
import json

import numpy as np

from boldtailor.cifti import scalar_artifact
from boldtailor.hrf_library import PARAMETER_NAMES
from boldtailor.publication import Artifact


def json_artifact(path, value):
    return Artifact(
        path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    )


def table_artifact(path, frame):
    return Artifact(path, frame.to_csv(sep="\t", index=False, na_rep="n/a").encode())


def npz_artifact(path, **arrays):
    with BytesIO() as stream:
        np.savez_compressed(stream, **arrays)
        return Artifact(path, stream.getvalue())


def figure_artifact(path, figure):
    try:
        with BytesIO() as stream:
            figure.savefig(stream, format="png", dpi=130)
            return Artifact(path, stream.getvalue())
    finally:
        figure.clear()


def parameter_artifact(brain, library, ids, path):
    """Look up the exact selected library entry, preserving undefined features."""
    names = [*PARAMETER_NAMES, "peak_time"]
    parameters = np.full((len(names), len(brain)), np.nan)
    valid = np.isfinite(ids)
    parameters[:, valid] = (
        library.parameter_table[names].to_numpy()[ids[valid].astype(int)].T
    )
    return scalar_artifact(path, brain, parameters, names)


def notebook_map(stem, brain, descriptor, statistic, values, names):
    """A ``desc-notebook<descriptor>_stat-<statistic>`` dense scalar map."""
    path = f"{stem}_space-fsLR_den-91k_desc-notebook{descriptor}_stat-{statistic}.dscalar.nii"
    return scalar_artifact(path, brain, values, names)
