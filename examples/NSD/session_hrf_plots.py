"""Figures from already computed session HRF comparisons."""

import matplotlib.pyplot as plt
import numpy as np

from boldtailor.reliability import finite_mean


def _correlation_matrix(comparison):
    count = len(comparison["sessions"])
    matrix = np.full((count + 1, count + 1), np.nan)
    for row, (a, b) in enumerate(comparison["pairs"]):
        matrix[a, b] = matrix[b, a] = finite_mean(comparison["pairwise"][row])
    for i in range(count):
        matrix[i, -1] = matrix[-1, i] = finite_mean(comparison["canonical"][i])
        if np.isfinite(comparison["canonical"][i]).any():
            matrix[i, i] = 1
    matrix[-1, -1] = 1
    return matrix


def agreement_figure(comparison):
    """Plot mean correlations and matched within-grayordinate comparisons."""
    labels = [*comparison["sessions"], "Canonical"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    image = axes[0].imshow(
        _correlation_matrix(comparison), vmin=-1, vmax=1, cmap="coolwarm"
    )
    axes[0].set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title="Mean HRF correlation across grayordinates",
    )
    plt.setp(axes[0].get_xticklabels(), rotation=60, ha="right", fontsize=8)
    plt.setp(axes[0].get_yticklabels(), fontsize=8)
    fig.colorbar(image, ax=axes[0], label="Pearson r")
    valid = np.isfinite(comparison["summary"][:3]).all(axis=0)
    for row, label in ((0, "Between sessions"), (1, "Canonical baseline")):
        axes[1].hist(
            comparison["summary"][row, valid],
            bins=np.linspace(-1, 1, 51),
            histtype="step",
            linewidth=2,
            label=label,
        )
    axes[1].set(
        xlabel="Mean r at each grayordinate",
        ylabel="Grayordinates",
        title="Matched comparison",
    )
    axes[1].legend()
    axes[2].hist(comparison["summary"][2, valid], bins=50)
    axes[2].axvline(0, color="black", linestyle="--")
    axes[2].set(
        xlabel="Between-session r minus canonical baseline",
        ylabel="Grayordinates",
        title="Agreement above baseline",
    )
    return fig


def parameter_variability_figure(comparison):
    """Plot across-session sample SD for response delay and time to peak."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, name in zip(axes, ("response_delay", "peak_time")):
        values = comparison["parameter_sd"][comparison["parameter_names"].index(name)]
        ax.hist(values[np.isfinite(values)], bins=40)
        ax.set(
            xlabel="Across-session SD (seconds)",
            ylabel="Grayordinates",
            title=name.replace("_", " ").title(),
        )
    return fig
