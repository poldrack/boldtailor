"""Script-only RT vertex selection and scatter plots; numerics in boldtailor."""

from io import BytesIO
from numbers import Integral

from matplotlib.figure import Figure
import numpy as np

from boldtailor.diagnostics import even_run_points, pearson_correlation
from boldtailor.publication import Artifact


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


def scatter_artifact(
    beta_runs_by_model, rt_runs, run_numbers, vertices, path, *, vertex_labels=None
):
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
            r = pearson_correlation(np.sum(x * y), np.sum(y * y), np.sum(x * x), len(x))
            label = vertex if vertex_labels is None else vertex_labels[row]
            axis.set(
                title=f"{model} · grayordinate {label} · even r={float(r):.3f}",
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
