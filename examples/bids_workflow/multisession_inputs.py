"""Load compact per-session HRF and beta summaries in a common spatial order."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.hrf_library import HrfLibrary
from boldtailor.cifti import read_scalar
from boldtailor.diagnostics import ONE_SAMPLE_T_NAMES as ACTIVATION_NAMES
from boldtailor.reliability import library_indices
from boldtailor.workflow.files import bids_label
from boldtailor.workflow.outputs import R2_NAMES
from .session_hrf_cache import MAP_NAMES as HRF_NAMES

METRICS = ("mean_beta", "task_t", "task_delta_r2", "rt_r", "rt_abs_r")
GLM_METRICS = ("task", "response_time")
ESTIMATORS = ("OLS", "FractionalCV", "RidgeCV", "Ridge")
TASK = "nsdcore"


def validate_sessions(subject, sessions, estimators):
    if not bids_label(subject, "sub"):
        raise ValueError("subject must be a BIDS subject label")
    if (
        len(sessions) < 2
        or len(set(sessions)) != len(sessions)
        or any(not bids_label(s, "ses") for s in sessions)
    ):
        raise ValueError("Provide at least two unique BIDS sessions")
    if (
        not estimators
        or len(set(estimators)) != len(estimators)
        or any(e not in ESTIMATORS for e in estimators)
    ):
        raise ValueError(f"estimators must be unique entries from {ESTIMATORS}")


def summary_paths(output, subject, session, estimators):
    base = Path(output) / f"{subject}/{session}/func/{subject}_{session}_task-{TASK}"
    paths = dict(
        metadata=Path(f"{base}_desc-boldtailor_metadata.json"),
        parameters=Path(f"{base}_desc-HRF_library.tsv"),
        curves=Path(f"{base}_desc-HRF_library.npz"),
    )
    descriptors = {
        "HRFAll": ("selection",),
        "CanonicalGLM": ("effects",),
        "OptimizedGLM": ("effects",),
    }
    descriptors.update(
        {
            prefix + "Trial" + e: ("activation", "rsquared", "rtcorrelation")
            for e in estimators
            for prefix in ("Canonical", "Optimized")
        }
    )
    for descriptor, stats in descriptors.items():
        for stat in stats:
            paths[descriptor, stat] = Path(
                f"{base}_space-fsLR_den-91k_desc-{descriptor}_stat-{stat}.dscalar.nii"
            )
    return paths


def _library(paths, metadata):
    table = pd.read_csv(paths["parameters"], sep="\t", float_precision="round_trip")
    library = HrfLibrary.from_table(table)
    with np.load(paths["curves"], allow_pickle=False) as archive:
        same = np.array_equal(library.curves, archive["curves"]) and np.array_equal(
            library.times, archive["times"]
        )
    if not same or metadata["library_fingerprint"] != library.fingerprint:
        raise ValueError(f"Saved HRF library is inconsistent: {paths['curves']}")
    return library


def _beta(paths, brain, descriptor):
    activation = read_scalar(paths[descriptor, "activation"], brain, ACTIVATION_NAMES)
    r2 = read_scalar(paths[descriptor, "rsquared"], brain, R2_NAMES)
    rt = read_scalar(
        paths[descriptor, "rtcorrelation"], brain, ["all_runs", "odd_runs", "even_runs"]
    )
    return np.stack([activation[0], activation[1], r2[2], rt[0], np.abs(rt[0])])


def _glm_effects(path, brain, metadata):
    """Task and RT effect maps, read with the session's saved regressor names."""
    regressors = list(metadata["regressors"])
    effects = read_scalar(path, brain, regressors)
    return effects[[regressors.index(name) for name in GLM_METRICS]]


def load_one_session(output, subject, session, estimators):
    paths = summary_paths(output, subject, session, estimators)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing result for {session}: {path}")
    metadata = json.loads(paths["metadata"].read_text())
    library = _library(paths, metadata)
    brain = nib.load(paths["HRFAll", "selection"]).header.get_axis(1)
    ids = read_scalar(paths["HRFAll", "selection"], brain, HRF_NAMES)[0]
    library_indices(ids, len(library.curves))
    return dict(
        session=session,
        brain=brain,
        library=library,
        metadata=metadata,
        hrf_indices=ids,
        sources=list(paths.values()),
        glm={
            prefix: _glm_effects(paths[prefix + "GLM", "effects"], brain, metadata)
            for prefix in ("Canonical", "Optimized")
        },
        beta={
            e: {
                prefix: _beta(paths, brain, prefix + "Trial" + e)
                for prefix in ("Canonical", "Optimized")
            }
            for e in estimators
        },
    )


def _compatible(first, other, estimators):
    if other["brain"] != first["brain"]:
        raise ValueError("Sessions have different grayordinate axes")
    if other["library"].fingerprint != first["library"].fingerprint:
        raise ValueError("Sessions must use the same HRF library")
    keys = (
        "noise_model",
        "trimming",
        "high_pass",
        "hrf_selection",
        "task_model_fingerprint",
        "hrf_normalization",
    )
    for key in keys:
        if other["metadata"].get(key) != first["metadata"].get(key):
            raise ValueError(f"Sessions use different {key}")
    if any(e != "OLS" for e in estimators):
        for key in (
            "ridge_mode",
            "ridge_fractions",
            "ridge_alphas",
            "ridge_alpha",
            "ridge_percentile",
            "encoding_mode",
        ):
            a, b = (r["metadata"]["settings"].get(key) for r in (first, other))
            if a != b:
                raise ValueError(f"Sessions use different {key}")


def load_sessions(output, subject, sessions, *, estimators=("OLS", "FractionalCV")):
    """Read existing summaries; no raw BOLD reads or model fitting occur here."""
    sessions, estimators = list(sessions), list(estimators)
    validate_sessions(subject, sessions, estimators)
    records = [load_one_session(output, subject, s, estimators) for s in sessions]
    for record in records[1:]:
        _compatible(records[0], record, estimators)
    return dict(
        subject=subject,
        sessions=sessions,
        estimators=estimators,
        brain=records[0]["brain"],
        library=records[0]["library"],
        records=records,
        metrics=METRICS,
        glm_metrics=GLM_METRICS,
    )
