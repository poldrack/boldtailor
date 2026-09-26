"""Descriptive reaction-time checks independent of trial-beta estimation."""

from io import BytesIO
from numbers import Integral

from matplotlib.figure import Figure
import numpy as np

from boldtailor.publication import Artifact


def _validate_runs(beta_runs, rt_runs, run_numbers):
    if (
        not beta_runs
        or len(beta_runs) != len(rt_runs)
        or len(beta_runs) != len(run_numbers)
    ):
        raise ValueError("beta, RT, and run lists must have equal nonzero lengths")
    if not all(
        isinstance(n, Integral) and not isinstance(n, bool) and n > 0
        for n in run_numbers
    ):
        raise ValueError("run numbers must be positive integers")
    if len(set(run_numbers)) != len(run_numbers):
        raise ValueError("run numbers must be unique")
    features = np.asarray(beta_runs[0]).shape[-1]
    for beta, rt in zip(beta_runs, rt_runs, strict=True):
        if (
            np.ndim(beta) != 2
            or np.shape(beta)[1] != features
            or np.shape(rt) != (len(beta),)
        ):
            raise ValueError("RT and beta dimensions must agree for every run")


def _centered(beta, rt):
    beta, rt = np.asarray(beta, dtype=float), np.asarray(rt, dtype=float)[:, None]
    valid = np.isfinite(beta) & np.isfinite(rt) & (rt > 0)
    counts = valid.sum(axis=0)
    divisor = np.maximum(counts, 1)
    b = np.where(valid, beta, 0.0)
    r = np.where(valid, rt, 0.0)
    b = np.where(valid, b - b.sum(axis=0) / divisor, 0.0)
    r = np.where(valid, r - r.sum(axis=0) / divisor, 0.0)
    return b, r, counts, valid


def _correlation(cross, beta_ss, rt_ss, counts):
    r = np.full(np.shape(cross), np.nan)
    denominator = np.sqrt(beta_ss * rt_ss)
    np.divide(cross, denominator, out=r, where=(counts >= 3) & (denominator > 0))
    return np.clip(r, -1.0, 1.0)


def correlate_rt(beta_runs, rt_runs, *, run_numbers):
    """Pearson r after matched-mask within-run centering; no p-values."""
    _validate_runs(beta_runs, rt_runs, run_numbers)
    moments, counts = [], []
    for beta, rt in zip(beta_runs, rt_runs, strict=True):
        b, r, count, _ = _centered(beta, rt)
        moments.append(
            np.stack([(b * r).sum(axis=0), (b * b).sum(axis=0), (r * r).sum(axis=0)])
        )
        counts.append(count)
    moments, counts = np.stack(moments), np.stack(counts)
    result = {
        "per_run": _correlation(*np.moveaxis(moments, 1, 0), counts),
        "counts": {"per_run": counts},
    }
    numbers = np.asarray(run_numbers)
    for name, mask in (
        ("all", numbers > 0),
        ("odd", numbers % 2 == 1),
        ("even", numbers % 2 == 0),
    ):
        count = counts[mask].sum(axis=0)
        result[name] = _correlation(*moments[mask].sum(axis=0), count)
        result["counts"][name] = count
    return result


def select_vertices(odd_r, cortical_mask, *, count=5):
    """Rank cortical candidates by absolute odd-run OLS r, then index."""
    r, cortex = np.asarray(odd_r), np.asarray(cortical_mask, dtype=bool)
    if (
        r.ndim != 1
        or cortex.shape != r.shape
        or not isinstance(count, Integral)
        or count < 0
    ):
        raise ValueError(
            "selection requires matching feature arrays and a nonnegative count"
        )
    indices = np.flatnonzero(cortex & np.isfinite(r))
    return indices[np.lexsort((indices, -np.abs(r[indices])))][:count]


def even_run_points(beta_runs, rt_runs, run_numbers, vertex):
    """Return matched-mask, within-run centered points from even runs only."""
    points = []
    for beta, rt, number in zip(beta_runs, rt_runs, run_numbers, strict=True):
        if number % 2 == 0:
            b, r, _, valid = _centered(np.asarray(beta)[:, [vertex]], rt)
            points.append((r[valid], b[valid]))
    if not points:
        return np.array([]), np.array([])
    return tuple(np.concatenate([p[i] for p in points]) for i in (0, 1))


def scatter_artifact(beta_runs_by_model, rt_runs, run_numbers, vertices, path):
    """Plot held-out even runs at vertices chosen using odd-run OLS only."""
    figure = Figure(
        figsize=(5 * len(beta_runs_by_model), 2.7 * max(1, len(vertices))),
        layout="constrained",
    )
    axes = figure.subplots(
        max(1, len(vertices)), len(beta_runs_by_model), squeeze=False
    )
    for column, (model, beta) in enumerate(beta_runs_by_model.items()):
        for row, vertex in enumerate(vertices):
            x, y = even_run_points(beta, rt_runs, run_numbers, vertex)
            axis = axes[row, column]
            axis.scatter(x, y, s=9, alpha=0.45, edgecolors="none")
            r = _correlation(np.sum(x * y), np.sum(y * y), np.sum(x * x), len(x))
            axis.set(
                title=f"{model} · grayordinate {vertex} · even r={float(r):.3f}",
                xlabel="RT deviation within run (s)",
                ylabel="Beta deviation (native units)",
            )
        if not len(vertices):
            axes[0, column].text(0.5, 0.5, "No eligible cortical vertices", ha="center")
            axes[0, column].set_axis_off()
    figure.suptitle("Vertices selected by odd-run OLS | Even-run descriptive check")
    stream = BytesIO()
    try:
        figure.savefig(stream, format="png", dpi=130)
        return Artifact(path, stream.getvalue())
    finally:
        figure.clear()
        stream.close()
