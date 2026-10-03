"""Presentation for the NSD tutorial; inputs are already fitted arrays and tables.

Functions return figures without displaying or saving them. They never refit a
model or change the inputs. Scientific choices remain in the notebook.
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from boldtailor.hrf_library import PARAMETER_NAMES
import pandas as pd

from boldtailor.reliability import library_indices


def design_figure(frame_times, design, regressors):
    """Plot named task regressors against original acquisition times."""
    fig, ax = plt.subplots(figsize=(11, 3), layout="constrained")
    for name in regressors:
        ax.plot(frame_times, design[name], label=name)
    ax.set(
        xlabel="Original acquisition time (s)",
        ylabel="Convolved regressor",
        title="First run: canonical task design",
    )
    ax.legend()
    return fig


def library_figure(library):
    """Plot every stored HRF, coloring noncanonical curves by time to peak."""
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    if len(library.candidates) > 1:
        segments = [
            np.column_stack([library.times, curve]) for curve in library.curves[1:]
        ]
        lines = LineCollection(segments, cmap="viridis", linewidths=0.75, alpha=0.3)
        lines.set_array(library.parameter_table.peak_time.to_numpy()[1:])
        ax.add_collection(lines)
        fig.colorbar(lines, ax=ax, label="Time to peak (seconds)")
    ax.plot(
        library.times,
        library.curves[0],
        color="black",
        linewidth=2.6,
        label="Canonical SPM",
    )
    ax.axhline(0, color="0.65", linewidth=0.7)
    ax.autoscale_view()
    ax.set(
        xlim=(library.times[0], library.times[-1]),
        xlabel="Time after impulse (seconds)",
        ylabel="HRF amplitude (discrete sum = 1)",
        title=f"HRF library: {len(library.candidates)} candidates",
    )
    ax.legend()
    return fig


def glm_comparison(glms):
    """Summarize finite R² values and plot optimized minus canonical full R²."""
    summary = []
    for name, result in glms.items():
        for statistic, values in zip(
            ("full R²", "confounds R²", "task ΔR²"), result["r2"], strict=True
        ):
            finite = values[np.isfinite(values)]
            summary.append(
                dict(
                    model=name,
                    statistic=statistic,
                    n=len(finite),
                    median=np.median(finite) if len(finite) else np.nan,
                )
            )
    canonical = glms["CanonicalGLM"]["r2"][0]
    optimized = glms["OptimizedGLM"]["r2"][0]
    valid = np.isfinite(canonical) & np.isfinite(optimized)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3), layout="constrained")
    axes[0].scatter(canonical[valid], optimized[valid], s=3, alpha=0.2)
    axes[0].plot([0, 1], [0, 1], color="black", lw=1)
    axes[0].set(xlabel="Canonical full R²", ylabel="Optimized full R²")
    axes[1].hist(optimized[valid] - canonical[valid], bins=50)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set(xlabel="Optimized − canonical full R²", ylabel="Grayordinates")
    return pd.DataFrame(summary), fig


def _selected_parameters(library, ids, names):
    indices = library_indices(ids, len(library.candidates))
    values = library.parameter_table[names].to_numpy()[np.maximum(indices, 0)].copy()
    values[indices < 0] = np.nan
    return values


def parameter_agreement(library, odd_ids, even_ids):
    """Compare selected parameters at grayordinates defined in both halves."""
    names = [*PARAMETER_NAMES, "peak_time"]  # the 2 x 4 panel grid
    odd = _selected_parameters(library, odd_ids, names)
    even = _selected_parameters(library, even_ids, names)
    agreement = []
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for i, (name, ax) in enumerate(zip(names, axes.flat, strict=True)):
        valid = np.isfinite(odd[:, i]) & np.isfinite(even[:, i])
        x, y = odd[valid, i], even[valid, i]
        r = (
            np.corrcoef(x, y)[0, 1]
            if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0
            else np.nan
        )
        agreement.append(dict(parameter=name, grayordinates=len(x), pearson_r=r))
        ax.scatter(x, y, s=4, alpha=0.1)
        ax.set(title=name, xlabel="Odd runs", ylabel="Even runs")
    return pd.DataFrame(agreement), fig


def _curve_scatter(fig, axis, x, y, label, reference):
    if len(x):
        counts = axis.hexbin(x, y, gridsize=35, mincnt=1, cmap="Blues")
        lower = max(-1.02, min(x.min(), y.min()) - 0.03)
        axis.plot([lower, 1.02], [lower, 1.02], color="black", linewidth=1)
        axis.set(xlim=(lower, 1.02), ylim=(lower, 1.02))
        fig.colorbar(counts, ax=axis, label="Grayordinates")
    else:
        axis.text(0.5, 0.5, "No paired HRFs", ha="center")
    axis.set(xlabel=label + " r", ylabel="Odd vs even r", title=reference)


def curve_agreement(curve_r):
    """Show three curve correlations on the same jointly finite grayordinates."""
    paired = np.isfinite(curve_r).all(axis=0)
    labels = ("Odd vs even", "Odd vs canonical", "Even vs canonical")
    summary = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    for label, values in zip(labels, curve_r, strict=True):
        q25, median, q75 = (
            np.quantile(values[paired], [0.25, 0.5, 0.75])
            if paired.any()
            else (np.nan,) * 3
        )
        summary.append(
            dict(
                comparison=label,
                grayordinates=int(paired.sum()),
                q25=q25,
                median=median,
                q75=q75,
            )
        )
        axes[0].hist(
            values[paired], bins=np.linspace(-1, 1, 51), histtype="step", label=label
        )
    axes[0].set(
        xlabel="Full-HRF Pearson r",
        ylabel="Grayordinates",
        title="Same grayordinates in each comparison",
    )
    axes[0].legend()
    for axis, row, reference in zip(
        axes[1:], (1, 2), ("Odd-run reference", "Even-run reference")
    ):
        _curve_scatter(
            fig, axis, curve_r[row, paired], curve_r[0, paired], labels[row], reference
        )
    return pd.DataFrame(summary), fig
