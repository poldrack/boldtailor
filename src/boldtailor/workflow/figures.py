"""Tuning-curve figures for workflow reports."""

from matplotlib.figure import Figure

from boldtailor.workflow.tables import tuned_models, tuning_rows


def plot_tuning(axis, mode, tuning):
    for scope, tuned in tuning.items():
        rows = tuning_rows(mode, scope, tuned)
        axis.plot(
            [r["fraction"] for r in rows],
            [r["summary_r2"] for r in rows],
            marker="o",
            label=scope,
        )
    axis.set(
        xlabel="Ridge fraction",
        ylabel=f"{rows[0]['percentile']:g}th percentile R² (summary)",
        title=f"{mode}: per-grayordinate CV selection",
    )
    axis.legend()


def _plot_alpha_tuning(axis, mode, tuning):
    positive = []
    for scope, tuned in tuning.items():
        selected = tuned["selection"]
        axis.plot(
            selected.alphas,
            selected.objective_scores,
            marker="o",
            label=f"{scope}: alpha={selected.ridge_alpha:g}",
        )
        positive.extend(a for a in selected.alphas if a > 0)
    if positive:
        axis.set_xscale("symlog", linthresh=min(positive))
    axis.set(
        title=f"{mode}: inner CV selection",
        xlabel="Ridge alpha",
        ylabel=f"{selected.percentile:g}th percentile of encoding R²",
    )
    axis.legend()


def tuning_figure(beta_models):
    """One tuning-curve panel per tuned model."""
    tuned = tuned_models(beta_models)
    figure = Figure(figsize=(6 * len(tuned), 4), layout="constrained")
    axes = figure.subplots(1, len(tuned), squeeze=False)[0]
    for axis, model in zip(axes, tuned, strict=True):
        plot = plot_tuning if model.fractional else _plot_alpha_tuning
        plot(axis, model.hrf.capitalize(), model.tuning)
    return figure
