"""Run the existing single-session notebook only for missing result sets."""

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pandas as pd

from .multisession_inputs import (
    _compatible,
    load_one_session,
    summary_paths,
    validate_sessions,
)
from .notebook_paths import notebook_paths
from .workflow_reuse import ANALYSIS_SETTINGS

SCIENCE_SETTINGS = tuple(
    k
    for k in ANALYSIS_SETTINGS
    if k not in ("bids_root", "fmriprep_root", "subject", "session")
)


def _execute_workflow(config):
    """A separate kernel per session releases its arrays before the next fit."""
    path = Path(__file__).with_name("nsd_workflow.ipynb")
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    notebook.cells.insert(0, nbformat.v4.new_code_cell(f"NSD_CONFIG = {config!r}"))
    NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parents[2])}},
    ).execute()


def _inherit_settings(config, sessions, estimators):
    """Use an existing session as the scientific template for missing sessions."""
    config = {**config, **notebook_paths(config)}
    subject = config.get("subject", "sub-07")
    for session in sessions:
        path = summary_paths(config["output_root"], subject, session, estimators)[
            "metadata"
        ]
        if path.is_file():
            saved = json.loads(path.read_text())["settings"]
            inherited = {key: saved[key] for key in SCIENCE_SETTINGS if key in saved}
            return {**inherited, **config}
    return config


def _check_requested(path, config):
    if not path.is_file():
        return
    saved = json.loads(path.read_text())["settings"]
    changed = [
        key
        for key in SCIENCE_SETTINGS
        if key in config and saved.get(key) != config[key]
    ]
    if changed:
        raise ValueError(
            f"Saved settings conflict in {path}: {', '.join(changed)}; "
            "use matching settings or a separate output_root"
        )


def _producer_config(config, estimators):
    modes = {"Ridge": "fixed", "RidgeCV": "cv", "FractionalCV": "fractional_cv"}
    requested = [e for e in estimators if e != "OLS"]
    if len(requested) > 1:
        raise ValueError("estimators can include OLS and at most one ridge estimator")
    implied = modes[requested[0]] if requested else "off"
    mode = config.get("ridge_mode", implied)
    if mode not in ("off", *modes.values()) or (requested and mode != implied):
        raise ValueError(
            "Requested estimators conflict with ridge_mode; use matching settings"
        )
    return dict(config, ridge_mode=mode)


def _preflight(config, subject, sessions, estimators, fit_missing):
    requests, first = [], None
    for session in sessions:
        paths = summary_paths(config["output_root"], subject, session, estimators)
        _check_requested(paths["metadata"], config)
        missing = [p for p in paths.values() if not p.is_file()]
        if missing and not fit_missing:
            raise FileNotFoundError(f"Missing results for {session}: {missing[0]}")
        if not missing:
            record = load_one_session(
                config["output_root"], subject, session, estimators
            )
            if first is None:
                first = record
            else:
                _compatible(first, record, estimators)
        requests.append((session, missing))
    return requests


def ensure_session_outputs(
    config, sessions, *, estimators=("OLS", "FractionalCV"), fit_missing=True
):
    """Preflight all sessions, then reuse complete summaries or fit missing ones.

    Scientific settings inherit from the first existing session unless explicitly
    supplied. Completed outputs are read only. Incomplete sessions are rerun with
    the single-session workflow's transactional overwrite behavior.
    """
    subject = config.get("subject", "sub-07")
    validate_sessions(subject, sessions, estimators)
    config = _producer_config(
        _inherit_settings(config, sessions, estimators), estimators
    )
    output = config["output_root"]
    requests = _preflight(config, subject, sessions, estimators, fit_missing)
    rows = []
    for session, missing in requests:
        if missing:
            print(f"Fitting missing session outputs: {session}", flush=True)
            _execute_workflow(
                dict(
                    config,
                    subject=subject,
                    session=session,
                    surface_maps=False,
                    existing_results="overwrite",
                )
            )
        record = load_one_session(output, subject, session, estimators)
        status = "fitted" if missing else "reused"
        print(f"{session}: {status}", flush=True)
        rows.append(
            dict(
                session=session,
                status=status,
                library_fingerprint=record["library"].fingerprint,
                metadata=str(
                    summary_paths(output, subject, session, estimators)["metadata"]
                ),
            )
        )
    return pd.DataFrame(rows)
