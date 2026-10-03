"""Export session agreement maps with an explicit canonical comparison."""

from hashlib import sha256
import json

import numpy as np
import pandas as pd

from boldtailor.publication import publish_artifact_set
from .hrf_artifacts import figure_artifact
from boldtailor.reliability import SUMMARY_NAMES, finite_mean
from .single_trial_artifacts import json_artifact, scalar_artifact, table_artifact


def session_table(estimates):
    return pd.DataFrame(
        [
            dict(
                session=e.session,
                status="reused" if e.reused_from else "fitted",
                grayordinates=int(np.isfinite(e.maps[0]).sum()),
                mean_selected_cv_r2=float(finite_mean(e.maps[1])),
                mean_canonical_cv_r2=float(finite_mean(e.maps[2])),
                request_id=e.request_id,
                metadata=str(e.cache_path),
            )
            for e in estimates
        ]
    )


def save_reliability(
    output, estimates, library, result, *, subject="sub-07", settings=None, figures=None
):
    sessions = [e.session for e in estimates]
    brain = estimates[0].brain
    if sessions != result["sessions"] or any(e.brain != brain for e in estimates):
        raise ValueError(
            "Comparison sessions and grayordinate axes must match estimates"
        )
    identity = dict(
        schema="boldtailor.session_hrf_reliability/1",
        sessions=sessions,
        requests=[e.request_id for e in estimates],
        library_fingerprint=library.fingerprint,
    )
    digest = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    stem = f"{subject}/func/{subject}_task-nsdcore_desc-sessionHRFReliability{digest}"
    maps = (
        ("pairwise", result["pairwise"], result["pair_names"]),
        ("pairbaseline", result["pair_baseline"], result["pair_names"]),
        ("pairdelta", result["pair_delta"], result["pair_names"]),
        ("canonical", result["canonical"], sessions),
        ("summary", result["summary"], SUMMARY_NAMES),
        ("parametersd", result["parameter_sd"], result["parameter_names"]),
        ("hrfindex", np.stack([e.maps[0] for e in estimates]), sessions),
    )
    artifacts = [
        scalar_artifact(stem + f"_stat-{name}.dscalar.nii", brain, values, labels)
        for name, values, labels in maps
    ]
    pairs = pd.DataFrame(
        dict(
            pair=result["pair_names"],
            grayordinates=np.isfinite(result["pairwise"]).sum(axis=1),
            mean_r=finite_mean(result["pairwise"], axis=1),
            mean_canonical_r=finite_mean(result["pair_baseline"], axis=1),
            mean_delta_r=finite_mean(result["pair_delta"], axis=1),
        )
    )
    metadata = dict(
        **identity,
        settings=settings or {},
        correlation="Pearson over the full stored 0.1-second HRF grid, no temporal shifting",
        canonical_baseline="For each pair and grayordinate: mean(r(session A, SPM), r(session B, SPM))",
        summary="Arithmetic means over valid session pairs at each grayordinate; delta uses those same pairs",
        summary_map_order=list(SUMMARY_NAMES),
        parameter_sd="Sample SD across valid sessions; requires at least two estimates",
        missing="NaN for undefined correlations or SD; availability counts are zero where no estimates exist",
        interpretation="Descriptive curve-shape agreement, not an ICC or independent BOLD prediction score; pairs share sessions",
        time_range_seconds=[float(library.times[0]), float(library.times[-1])],
        caches=[str(e.cache_path) for e in estimates],
    )
    artifacts.extend(
        [
            table_artifact(stem + "_sessions.tsv", session_table(estimates)),
            table_artifact(stem + "_pairs.tsv", pairs),
            json_artifact(stem + "_metadata.json", metadata),
        ]
    )
    artifacts.extend(
        figure_artifact(stem + f"_{name}.png", fig)
        for name, fig in (figures or {}).items()
    )
    return publish_artifact_set(
        output,
        artifacts,
        source_paths=[e.cache_path for e in estimates],
        overwrite=True,
    )
