"""NSD notebooks: build WorkflowSettings from the notebook configuration dictionary."""

import os

from boldtailor.workflow.settings import WorkflowSettings

_RENAMED = {
    "bids_root": "bids_dir",
    "fmriprep_root": "fmriprep_dir",
    "output_root": "output_dir",
}


def nsd_settings(config, **overrides):
    values = {_RENAMED.get(k, k): v for k, v in dict(config).items()}
    values.setdefault("task", "nsdcore")
    values.update(overrides)
    return WorkflowSettings.from_dict(values)


def nsd_paths(config):
    """Configured paths, else NSD_BIDS_ROOT / NSD_FMRIPREP_ROOT / NSD_OUTPUT_ROOT.

    Only bids_root is required; WorkflowSettings derives the others when absent.
    """
    paths = {}
    for key in _RENAMED:
        value = config.get(key) or os.environ.get(f"NSD_{key.upper()}")
        if value:
            paths[key] = str(value)
    if "bids_root" not in paths:
        raise ValueError(
            "Set NSD_BIDS_ROOT or supply bids_root in the notebook configuration"
        )
    return paths
