"""Local data locations for the NSD notebooks; never read .env or create files."""

import os
from pathlib import Path


def notebook_paths(overrides=None):
    """Resolve explicit paths, then environment variables, then derivative defaults.

    Set NSD_BIDS_ROOT or supply bids_root in the notebook configuration.
    NSD_FMRIPREP_ROOT and NSD_OUTPUT_ROOT optionally override derivative paths.
    Existence and BIDS containment are checked by the input/output helpers.
    """
    overrides = {} if overrides is None else overrides
    root = overrides.get("bids_root", os.environ.get("NSD_BIDS_ROOT"))
    if root is None or not str(root).strip():
        raise ValueError(
            "Set NSD_BIDS_ROOT or supply bids_root in the notebook configuration"
        )
    root = Path(root).expanduser()
    defaults = {
        "bids_root": root,
        "fmriprep_root": root / "derivatives/fmriprep-25.2.5",
        "output_root": root / "derivatives/boldtailor",
    }
    paths = {}
    for key, default in defaults.items():
        value = overrides.get(key, os.environ.get(f"NSD_{key.upper()}", default))
        if value is None or not str(value).strip():
            raise ValueError(f"Supply a nonempty {key} in the notebook configuration")
        paths[key] = str(Path(value).expanduser())
    return paths
