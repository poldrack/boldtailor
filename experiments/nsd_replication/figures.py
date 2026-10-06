"""Figures for the NSD replication experiments; each returns a Figure."""

import matplotlib.pyplot as plt
import numpy as np

from boldtailor.cifti import cortical_values

THRESHOLD_LABEL = "Voxel reliability threshold (r)"
DIFFERENCE_LABEL = "Reliability: difference from composite (r)"
TRIAL_SPACING = 4.0  # seconds between trial onsets in NSD


def _lines(ax, table, x, y, group="version"):
    for name, rows in table.groupby(group, sort=False):
        rows = rows.sort_values(x)
        ax.plot(rows[x], rows[y], marker="o", label=str(name))
    ax.legend()


def _curves(table):
    fig, ax = plt.subplots()
    _lines(ax, table, "threshold", "mean_difference")
    ax.set_xlabel(THRESHOLD_LABEL)
    ax.set_ylabel(DIFFERENCE_LABEL)
    return fig


def fig_r1_curves(curves):
    return _curves(curves)


def fig_r3_lss(curves):
    """b4 against the LSS variants (lss-assume, lss-fit)."""
    return _curves(curves)


def _flat_panel(ax, values, title):
    ax.plot(values, marker=".", markersize=1, linestyle="none")
    ax.set_title(title)
    ax.set_xlabel("Vertex index")
    ax.set_ylabel("Reliability difference (r)")


def fig_r1_maps(values, brain, meshes=None):
    """Per-hemisphere values against vertex index.

    Cortical surface rendering (``boldtailor.workflow.surfaces.surface_figure``)
    is not supported: its colour scales exist only for r2, delta_r2, rt, t and
    peak_time, not for reliability differences.
    """
    if meshes is not None:
        raise NotImplementedError(
            "surface rendering of reliability differences is not supported: "
            "boldtailor.workflow.surfaces has no matching colour scale"
        )
    shown = cortical_values(values, brain)
    fig, axes = plt.subplots(1, 2, sharey=True)
    for ax, hemi in zip(axes, ("left", "right")):
        _flat_panel(ax, shown.get(hemi, np.array([])), hemi.title())
    return fig


def fig_r4_lag(tables):
    fig, ax = plt.subplots()
    for version, table in tables.items():
        ax.plot(
            table["lag"] * TRIAL_SPACING, table["mean_r"], marker="o", label=version
        )
    ax.set_xlabel("Time between trials (s)")
    ax.set_ylabel("Mean pattern correlation")
    ax.legend()
    return fig


def fig_r5_rsa(rsa):
    fig, axes = plt.subplots(1, len(rsa), sharey=True, squeeze=False)
    for ax, (level, table) in zip(axes[0], rsa.items()):
        by_t = table.groupby("threshold")["r"].mean()
        ax.plot(by_t.index, by_t.values, marker="o")
        ax.set_title(level)
        ax.set_xlabel(THRESHOLD_LABEL)
    axes[0][0].set_ylabel("Between-subject RDM correlation (r)")
    return fig


def fig_r6_decoding(table):
    fig, ax = plt.subplots()
    _lines(ax, table, "threshold", "accuracy")
    ax.plot(table["threshold"], table["chance"], "k--", label="chance")
    ax.set_xlabel(THRESHOLD_LABEL)
    ax.set_ylabel("Decoding accuracy")
    ax.legend()
    return fig


def fig_r2_hrf(consistency):
    fig, ax = plt.subplots()
    for column in ("mean_pairwise_r", "mean_canonical_baseline"):
        ax.plot(consistency[column].to_numpy(), marker="o", label=column)
    ax.set_xlabel("Session subset")
    ax.set_ylabel("Mean HRF-choice consistency (r)")
    ax.legend()
    return fig


def _tost_title(level, tost):
    result = (tost or {}).get(level)
    if result is None:
        return level
    verdict = "equivalent" if result["equivalent"] else "not equivalent"
    return f"{level}: TOST {verdict} (p={result['p']:.3g})"


def _parity_panel(ax, rows, title):
    ax.scatter(rows["released"], rows["boldtailor"])
    low = min(rows["released"].min(), rows["boldtailor"].min())
    high = max(rows["released"].max(), rows["boldtailor"].max())
    ax.plot([low, high], [low, high], "k--")
    ax.set_title(title)
    ax.set_xlabel("Released median reliability (r)")


def fig_parity(table, tost=None):
    """Per-subject boldtailor vs released medians, one panel per level."""
    levels = list(dict.fromkeys(table["level"]))
    fig, axes = plt.subplots(1, len(levels), squeeze=False)
    for ax, level in zip(axes[0], levels):
        _parity_panel(ax, table[table["level"] == level], _tost_title(level, tost))
    axes[0][0].set_ylabel("boldtailor median reliability (r)")
    return fig


def fig_gate(table):
    fig, ax = plt.subplots()
    pivot = table.groupby(["condition", "gate"])["n_components"].mean().unstack()
    pivot.plot.bar(ax=ax, rot=0)
    ax.set_xlabel("")
    ax.set_ylabel("Mean number of noise components")
    ax.legend(title="gate")
    return fig


def fig_modulators(table):
    fig, ax = plt.subplots()
    groups = [rows["r"].to_numpy() for _, rows in table.groupby("version", sort=False)]
    ax.boxplot(groups, tick_labels=list(dict.fromkeys(table["version"])))
    ax.set_ylabel("Correlation of mean ROI beta with RT (r)")
    return fig
