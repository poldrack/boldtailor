"""Equal-session-weight comparisons of HRF shapes and paired beta summaries."""

import numpy as np

from boldtailor.reliability import compare_hrfs


def paired_summary(canonical, optimized):
    """Match finite observations before averaging; require two paired sessions.

    Axis zero indexes sessions. Remaining axes can index metrics/grayordinates.
    SD describes variation in within-session changes, not uncertainty estimated
    from independent trials. The difference of t statistics is not itself a t.
    """
    a, b = np.asarray(canonical, dtype=float), np.asarray(optimized, dtype=float)
    if a.shape != b.shape or a.ndim < 2 or a.shape[0] < 2:
        raise ValueError("Matching arrays with at least two sessions are required")
    matched = np.isfinite(a) & np.isfinite(b)
    count = matched.sum(axis=0)
    a, b = np.where(matched, a, 0), np.where(matched, b, 0)
    delta = b - a

    def mean(values):
        return np.divide(
            values.sum(axis=0),
            count,
            out=np.full(count.shape, np.nan),
            where=count >= 2,
        )

    difference = mean(delta)
    squared = np.where(matched, delta - difference, 0) ** 2
    variance = np.divide(
        squared.sum(axis=0),
        count - 1,
        out=np.full(count.shape, np.nan),
        where=count >= 2,
    )
    return dict(
        canonical_mean=mean(a),
        optimized_mean=mean(b),
        difference_mean=difference,
        difference_sd=np.sqrt(variance),
        positive_fraction=mean((delta > 0).astype(float)),
        valid_sessions=count,
        session_differences=np.where(matched, delta, np.nan),
    )


def analyze_sessions(loaded):
    records = loaded["records"]
    ids = np.stack([r["hrf_indices"] for r in records])
    hrf = compare_hrfs(loaded["library"], ids, loaded["sessions"])
    beta = {
        e: paired_summary(
            np.stack([r["beta"][e]["Canonical"] for r in records]),
            np.stack([r["beta"][e]["Optimized"] for r in records]),
        )
        for e in loaded["estimators"]
    }
    glm = paired_summary(
        np.stack([r["glm"]["Canonical"] for r in records]),
        np.stack([r["glm"]["Optimized"] for r in records]),
    )
    return dict(
        hrf=hrf,
        beta=beta,
        metrics=loaded["metrics"],
        glm=glm,
        glm_metrics=loaded["glm_metrics"],
    )
