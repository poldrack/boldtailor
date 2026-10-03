"""Publish paired session summaries with source identities and explicit units."""

from hashlib import sha256

import numpy as np
import pandas as pd

from boldtailor.publication import publish_artifact_set
from boldtailor.reliability import SUMMARY_NAMES, finite_mean
from boldtailor.cifti import scalar_artifact
from .workflow_artifacts import figure_artifact, json_artifact, table_artifact

SUMMARY_STATS = (
    "canonical_mean",
    "optimized_mean",
    "difference_mean",
    "difference_sd",
    "positive_fraction",
    "valid_sessions",
)


def beta_summary_table(result):
    rows = []
    for estimator, values in result["beta"].items():
        for i, metric in enumerate(result["metrics"]):
            change = values["difference_mean"][i]
            valid = np.isfinite(change)
            rows.append(
                dict(
                    estimator=estimator,
                    metric=metric,
                    grayordinates=int(valid.sum()),
                    median_change=(
                        float(np.median(change[valid])) if valid.any() else np.nan
                    ),
                    p10_change=(
                        float(np.percentile(change[valid], 10))
                        if valid.any()
                        else np.nan
                    ),
                    p90_change=(
                        float(np.percentile(change[valid], 90))
                        if valid.any()
                        else np.nan
                    ),
                )
            )
    return pd.DataFrame(rows)


def _hrf_artifacts(base, brain, result):
    hrf = result["hrf"]
    return [
        scalar_artifact(f"{base}HRF_stat-{stat}.dscalar.nii", brain, values, names)
        for stat, values, names in (
            ("pairwise", hrf["pairwise"], hrf["pair_names"]),
            ("canonical", hrf["canonical"], hrf["sessions"]),
            ("summary", hrf["summary"], SUMMARY_NAMES),
            (
                "peaktime",
                np.stack([hrf["peak_time_mean"], hrf["summary"][3]]),
                ["mean_peak_time_seconds", "valid_sessions"],
            ),
        )
    ]


def _beta_artifacts(base, brain, loaded, result):
    artifacts = []
    for estimator, values in result["beta"].items():
        for stat in SUMMARY_STATS:
            artifacts.append(
                scalar_artifact(
                    f"{base}{estimator}_stat-{stat}.dscalar.nii",
                    brain,
                    values[stat],
                    result["metrics"],
                )
            )
        delta = values["session_differences"]
        names = [
            f"{session}:{metric}"
            for session in loaded["sessions"]
            for metric in result["metrics"]
        ]
        artifacts.append(
            scalar_artifact(
                f"{base}{estimator}_stat-sessionchanges.dscalar.nii",
                brain,
                delta.reshape(-1, delta.shape[-1]),
                names,
            )
        )
    return artifacts


def _glm_artifacts(base, brain, result):
    return [
        scalar_artifact(
            f"{base}GLM_stat-{stat}.dscalar.nii",
            brain,
            result["glm"][stat],
            result["glm_metrics"],
        )
        for stat in SUMMARY_STATS
    ]


def _metadata(loaded, sources):
    return dict(
        subject=loaded["subject"],
        sessions=loaded["sessions"],
        estimators=loaded["estimators"],
        library_fingerprint=loaded["library"].fingerprint,
        metrics=list(loaded["metrics"]),
        glm_metrics=list(loaded["glm_metrics"]),
        glm_units=dict(
            task="native signal units", response_time="native signal units per second"
        ),
        glm_interpretation="Equal-session means of conventional GLM coefficients, not t statistics or significance maps. Optimized minus canonical is a signed coefficient change, not a measure of prediction accuracy.",
        direction="optimized minus canonical within each session",
        aggregation="equal session weights on matched finite model values; at least two sessions",
        hrf_correlation="Pearson over the full stored HRF time grid without temporal shifting",
        hrf_pairs="descriptive dependent pairs, not independent replicates for inference",
        hrf_peak_time=dict(
            units="seconds",
            minimum_sessions=1,
            definition="Time of the maximum of each selected full HRF on the library time grid",
            aggregation="Equal-weight mean of valid session peak times, not the peak of the mean HRF; no valid sessions gives NaN",
        ),
        units=dict(
            mean_beta="native signal units",
            task_t="descriptive t statistic; differences are not t tests",
            task_delta_r2="full minus confound-only BOLD R2",
            rt_r="within-run centered Pearson r",
            rt_abs_r="absolute within-run centered Pearson r",
        ),
        interpretation="In-sample descriptive comparisons; raw beta means depend on session signal scale. "
        "OLS compares HRFs without ridge. CV models tune their own penalties using RT/type; "
        "larger RT associations are not independent evidence of improved prediction.",
        sources=[
            dict(path=str(p), sha256=sha256(p.read_bytes()).hexdigest())
            for p in sources
        ],
    )


def save_multisession(output, loaded, result, *, figures=None):
    """Overwrite only aggregate files, preserving all per-session sources."""
    subject = loaded["subject"]
    base = f"{subject}/func/{subject}_task-nsdcore_desc-multisession"
    brain = loaded["brain"]
    sources = [p for record in loaded["records"] for p in record["sources"]]
    artifacts = _hrf_artifacts(base, brain, result) + _beta_artifacts(
        base, brain, loaded, result
    )
    artifacts.extend(_glm_artifacts(base, brain, result))
    hrf = result["hrf"]
    pairs = pd.DataFrame(
        dict(
            pair=hrf["pair_names"],
            mean_r=finite_mean(hrf["pairwise"], axis=1),
            grayordinates=np.isfinite(hrf["pairwise"]).sum(axis=1),
        )
    )
    artifacts.extend(
        [
            table_artifact(base + "_betasummary.tsv", beta_summary_table(result)),
            table_artifact(base + "_hrfpairs.tsv", pairs),
            json_artifact(base + "_metadata.json", _metadata(loaded, sources)),
        ]
    )
    for name, figure in (figures or {}).items():
        artifacts.append(figure_artifact(base + f"_{name}.png", figure))
    return publish_artifact_set(output, artifacts, source_paths=sources, overwrite=True)
