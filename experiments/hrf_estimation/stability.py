"""Within/between-session comparison of normalized HRF shapes."""

import numpy as np
import pandas as pd


def _curve(kernel):
    times = np.arange(360) * 0.1
    samples = kernel.kernel(0.1, 1)
    curve = np.interp(times, np.arange(len(samples)) * 0.1, samples, right=0)
    return curve / curve.max()


def _width(values, peak):
    threshold = values[peak] / 2
    left = right = peak
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    while right < len(values) - 1 and values[right + 1] >= threshold:
        right += 1
    if left == 0 or right == len(values) - 1:
        return np.nan
    start = (
        left - 1 + (threshold - values[left - 1]) / (values[left] - values[left - 1])
    )
    stop = right + (values[right] - threshold) / (values[right] - values[right + 1])
    return 0.1 * (stop - start)


def _metrics(curve):
    peak, trough = int(curve.argmax()), int(curve.argmin())
    valid = peak < trough < len(curve) - 1 and curve[trough] < -1e-6
    return np.array(
        [
            peak * 0.1,
            _width(curve, peak),
            trough * 0.1 if valid else np.nan,
            _width(-curve, trough) if valid else np.nan,
            -curve[trough] if valid else np.nan,
        ]
    )


def compare_hrfs(left, right):
    if not np.array_equal(left.feature_ids, right.feature_ids):
        raise ValueError("feature identities/order differ")
    if not np.array_equal(left.brain_axis, right.brain_axis):
        raise ValueError("anatomical identities/order differ")
    rows = []
    names = (
        "peak_difference",
        "fwhm_difference",
        "trough_difference",
        "undershoot_fwhm_difference",
        "depth_difference",
    )
    for feature, a, b in zip(
        left.feature_ids, left.kernels, right.kernels, strict=True
    ):
        row = dict(
            feature_id=int(feature),
            curve_correlation=np.nan,
            curve_rmse=np.nan,
            **dict.fromkeys(names, np.nan),
        )
        if a is not None and b is not None:
            ca, cb = _curve(a), _curve(b)
            row.update(
                curve_correlation=float(np.corrcoef(ca, cb)[0, 1]),
                curve_rmse=float(np.sqrt(np.mean((ca - cb) ** 2))),
            )
            row.update(zip(names, np.abs(_metrics(ca) - _metrics(cb)), strict=True))
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature_id")
