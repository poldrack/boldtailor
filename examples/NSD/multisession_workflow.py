"""Run the package workflow only for missing single-session result sets."""

import json

import pandas as pd

from boldtailor.workflow.run import run_workflow
from .multisession_inputs import (
    _compatible,
    load_one_session,
    summary_paths,
    validate_sessions,
)
from .nsd_settings import nsd_paths, nsd_settings

SCIENCE_SETTINGS = (
    "task",
    "modulators",
    "max_grayordinates",
    "encoding_mode",
    "ridge_mode",
    "ridge_fractions",
    "ridge_alphas",
    "ridge_percentile",
    "ridge_alpha",
    "hrf_library",
    "hrf_selection_rt",
    "hrf_n_samples",
    "hrf_seed",
)


def _with_paths(config, sessions):
    """Explicit or NSD_* paths; the output root defaults to the workflow's own."""
    config = {**config, **nsd_paths(config)}
    if "output_root" not in config:
        config.setdefault("subject", "sub-07")
        default = nsd_settings(config, session=sessions[0]).output_dir
        config["output_root"] = str(default)
    return config


def _inherit_settings(config, sessions, estimators):
    """Use an existing session as the scientific template for missing sessions."""
    config = _with_paths(config, sessions)
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


def _workflow_settings(config, subject, session):
    return nsd_settings(
        dict(config, subject=subject),
        session=session,
        surface_maps=False,
        existing_results="overwrite",
    )


def _preflight(config, subject, sessions, estimators, fit_missing):
    """Check every session, and build settings for the missing ones, before fitting."""
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
    return [
        (s, _workflow_settings(config, subject, s) if missing else None)
        for s, missing in requests
    ]


def ensure_session_outputs(
    config, sessions, *, estimators=("OLS", "FractionalCV"), fit_missing=True
):
    """Preflight all sessions, then reuse complete summaries or fit missing ones.

    Scientific settings inherit from the first existing session unless explicitly
    supplied. Completed outputs are read only. Incomplete sessions are rerun with
    ``run_workflow`` and ``existing_results="overwrite"``.
    """
    subject = config.get("subject", "sub-07")
    validate_sessions(subject, sessions, estimators)
    config = _producer_config(
        _inherit_settings(config, sessions, estimators), estimators
    )
    output = config["output_root"]
    requests = _preflight(config, subject, sessions, estimators, fit_missing)
    rows = []
    for session, settings in requests:
        if settings is not None:
            print(f"Fitting missing session outputs: {session}", flush=True)
            run_workflow(settings)
        record = load_one_session(output, subject, session, estimators)
        status = "fitted" if settings is not None else "reused"
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
