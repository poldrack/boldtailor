"""Inspect grayordinate-level changes and full HRF curves across sessions."""

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from boldtailor.workflow.surfaces import surface_figure

METRIC_LABELS = dict(
    mean_beta="Mean trial beta (native units)",
    task_t="Task t (descriptive)",
    task_delta_r2="Task-added BOLD ΔR²",
    rt_r="Signed beta–RT r",
    rt_abs_r="Absolute beta–RT r",
)


def beta_change_figure(result):
    estimators, metrics = list(result["beta"]), result["metrics"]
    fig = Figure(
        figsize=(6 * len(estimators), 2.6 * len(metrics)), layout="constrained"
    )
    axes = fig.subplots(len(metrics), len(estimators), squeeze=False)
    for col, estimator in enumerate(estimators):
        for row, metric in enumerate(metrics):
            values = result["beta"][estimator]["difference_mean"][row]
            ax = axes[row, col]
            ax.hist(values[np.isfinite(values)], bins=50, color="darkorange")
            ax.axvline(0, color="black", linestyle="--")
            ax.set(
                title=f"{estimator}: {METRIC_LABELS[metric]}",
                xlabel="Mean within-session change (optimized − canonical)",
                ylabel="Grayordinates",
            )
    return fig


def comparison_surface(maps, brain, meshes, *, label, title, t_scale=False):
    fig = surface_figure(
        maps, brain, meshes, statistic="t" if t_scale else "rt", title=title
    )
    fig.axes[-1].set_ylabel(label)
    return fig


def peak_time_surface(result, brain, meshes):
    """Display the mean of valid session HRF peak times at each grayordinate."""
    fig = surface_figure(
        {"Mean peak time": result["hrf"]["peak_time_mean"]},
        brain,
        meshes,
        statistic="peak_time",
        title="Mean HRF time-to-peak across sessions",
    )
    fig.axes[-1].yaxis.set_label_position("left")
    return fig


def glm_effect_figures(result, brain, meshes):
    """Plot matched session means and signed coefficient changes on shared scales."""
    definitions = (
        ("task", "GLMTask", "Mean task response", "Native signal units"),
        (
            "response_time",
            "GLMResponseTime",
            "Mean RT effect",
            "Native signal units per second",
        ),
    )
    figures = {}
    for metric, key, title, units in definitions:
        row = result["glm_metrics"].index(metric)
        maps = {
            label: result["glm"][stat][row]
            for label, stat in (
                ("CanonicalGLM", "canonical_mean"),
                ("OptimizedGLM", "optimized_mean"),
                ("Difference\n(opt. − canon.)", "difference_mean"),
            )
        }
        fig = comparison_surface(
            maps,
            brain,
            meshes,
            label=units,
            title=f"{title} across sessions: GLM coefficients",
        )
        fig.axes[-1].yaxis.set_label_position("left")
        figures[key] = fig
    return figures


def choose_grayordinate(loaded, result, requested=None):
    if requested is not None:
        if (
            isinstance(requested, bool)
            or not isinstance(requested, (int, np.integer))
            or not 0 <= requested < len(loaded["brain"])
        ):
            raise ValueError("grayordinate must be a valid zero-based integer index")
        return int(requested)
    available = result["hrf"]["summary"][3] >= 2
    cortex = loaded["brain"].vertex >= 0
    candidates = np.flatnonzero(available & cortex)
    if not len(candidates):
        candidates = np.flatnonzero(available)
    return int(candidates[0]) if len(candidates) else None


def grayordinate_hrf_figure(loaded, result, grayordinate):
    fig = Figure(figsize=(12, 4.5), layout="constrained")
    curves_ax, matrix_ax = fig.subplots(1, 2)
    library, sessions = loaded["library"], loaded["sessions"]
    matrix = np.full((len(sessions), len(sessions)), np.nan)
    for i, record in enumerate(loaded["records"]):
        cid = record["hrf_indices"][grayordinate]
        if np.isfinite(cid) and cid >= 0:
            curves_ax.plot(library.times, library.curves[int(cid)], label=sessions[i])
            matrix[i, i] = 1
    curves_ax.plot(
        library.times,
        library.curves[0],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Canonical",
    )
    for row, (a, b) in enumerate(result["hrf"]["pairs"]):
        matrix[a, b] = matrix[b, a] = result["hrf"]["pairwise"][row, grayordinate]
    curves_ax.set(
        xlabel="Time (s)",
        ylabel="HRF amplitude",
        title=f"Grayordinate {grayordinate}: full HRFs",
    )
    curves_ax.legend(fontsize=7, ncol=2)
    image = matrix_ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    matrix_ax.set(
        xticks=range(len(sessions)),
        yticks=range(len(sessions)),
        xticklabels=sessions,
        yticklabels=sessions,
        title="Full-curve Pearson r",
    )
    matrix_ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    matrix_ax.tick_params(axis="y", labelsize=8)
    fig.colorbar(image, ax=matrix_ax, label="Pearson r")
    return fig


def grayordinate_beta_figure(loaded, grayordinate):
    metrics, estimators, sessions = (
        loaded["metrics"],
        loaded["estimators"],
        loaded["sessions"],
    )
    fig = Figure(
        figsize=(6 * len(estimators), 2.7 * len(metrics)), layout="constrained"
    )
    axes = fig.subplots(len(metrics), len(estimators), squeeze=False)
    rows = []
    for col, estimator in enumerate(estimators):
        for i, metric in enumerate(metrics):
            ax = axes[i, col]
            for prefix, color, style in (
                ("Canonical", "black", "o--"),
                ("Optimized", "darkorange", "s-"),
            ):
                values = [
                    r["beta"][estimator][prefix][i, grayordinate]
                    for r in loaded["records"]
                ]
                ax.plot(range(len(sessions)), values, style, color=color, label=prefix)
                rows.extend(
                    dict(
                        session=s,
                        estimator=estimator,
                        model=prefix,
                        metric=metric,
                        value=v,
                    )
                    for s, v in zip(sessions, values, strict=True)
                )
            ax.set(
                title=f"{estimator}: {METRIC_LABELS[metric]}",
                xticks=range(len(sessions)),
                xticklabels=sessions,
                ylabel=METRIC_LABELS[metric],
            )
            ax.tick_params(axis="x", labelrotation=60, labelsize=8)
            ax.legend(fontsize=8)
    fig.suptitle(f"Grayordinate {grayordinate}: matched session results")
    return fig, pd.DataFrame(rows)
