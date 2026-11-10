"""Scale-invariant HRF recovery metrics; failures remain explicit."""

import numpy as np

TIMES = np.arange(501) * 0.1


def curve(candidate):
    values = candidate.kernel(0.1, 1)
    values = np.interp(TIMES, np.arange(len(values)) * 0.1, values, right=0.0)
    return values / values.max()


def recovery(candidate, truth):
    if candidate is None:
        return dict(
            failed=True,
            curve_rmse=np.nan,
            curve_correlation=np.nan,
            peak_bias_seconds=np.nan,
            peak_absolute_error=np.nan,
            candidate_id=-1,
        )
    a, b = curve(candidate), curve(truth)
    bias = float(TIMES[np.argmax(a)] - TIMES[np.argmax(b)])
    return dict(
        failed=False,
        curve_rmse=float(np.sqrt(np.mean((a - b) ** 2))),
        curve_correlation=float(np.corrcoef(a, b)[0, 1]),
        peak_bias_seconds=bias,
        peak_absolute_error=abs(bias),
        candidate_id=candidate.id,
    )


def library_floor(truth, library):
    rows = [recovery(k, truth) for k in library.candidates]
    return dict(
        min_curve_rmse=min(r["curve_rmse"] for r in rows),
        max_curve_correlation=max(r["curve_correlation"] for r in rows),
        min_peak_absolute_error=min(r["peak_absolute_error"] for r in rows),
    )
