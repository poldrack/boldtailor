"""Offline cortical views of existing NSD grayordinate statistics."""

from pathlib import Path

from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from nilearn.plotting import plot_surf
from nilearn.surface import load_surf_mesh

from boldtailor.cifti import CORTEX_STRUCTURES, cortical_values

VIEWS = (
    ("left", "lateral"),
    ("left", "medial"),
    ("right", "medial"),
    ("right", "lateral"),
)


def find_surface_meshes(fmriprep_root, subject, *, paths=None):
    """Find subject fsLR 32k midthickness meshes, or use explicit left/right paths.

    Return None if either hemisphere is unavailable. Multiple matches require
    explicit paths, preventing an arbitrary choice across anatomical sessions.
    Explicit meshes must use the same fsLR vertex ordering as the CIFTI data.
    """
    if paths is not None:
        if set(paths) != {"left", "right"}:
            raise ValueError("surface_meshes must provide both left and right paths")
        result = {hemi: Path(path).expanduser() for hemi, path in paths.items()}
        for path in result.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        return result
    folder = Path(fmriprep_root) / subject
    result = {}
    for hemi, code in (("left", "L"), ("right", "R")):
        matches = sorted(
            folder.rglob(f"*_hemi-{code}_space-fsLR_den-32k_midthickness.surf.gii")
        )
        if len(matches) > 1:
            raise ValueError(
                "Found multiple fsLR surfaces; set surface_meshes explicitly"
            )
        if matches:
            result[hemi] = matches[0]
    return result if len(result) == 2 else None


def _load_meshes(paths, brain):
    if set(paths) != {"left", "right"}:
        raise ValueError("meshes must contain left and right surfaces")
    meshes = {hemi: load_surf_mesh(path) for hemi, path in paths.items()}
    for hemi, mesh in meshes.items():
        expected = brain.nvertices.get(CORTEX_STRUCTURES[hemi])
        if expected is not None and len(mesh.coordinates) != expected:
            raise ValueError(
                f"{hemi} mesh has {len(mesh.coordinates)} vertices; CIFTI expects {expected}"
            )
    return meshes


def _color_scale(projected, statistic):
    if statistic not in ("r2", "delta_r2", "rt", "t", "peak_time"):
        raise ValueError("statistic must be r2, delta_r2, rt, t, or peak_time")
    arrays = [
        a[np.isfinite(a)] for mapping in projected.values() for a in mapping.values()
    ]
    finite = np.concatenate(arrays) if arrays else np.array([])
    if statistic == "peak_time":
        lower, upper = (
            (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        )
        if lower == upper:
            lower, upper = max(0.0, lower - 0.5), upper + 0.5
        return (lower, upper), "viridis", "Mean time-to-peak (seconds)"
    if statistic == "t":
        return (-10.0, 10.0), "RdBu_r", "Beta-series t versus zero (independent trials)"
    if statistic == "rt":
        limit = float(np.max(np.abs(finite))) if len(finite) else 1.0
        limit = limit or 1.0
        label = "Within-run-centered beta–RT Pearson r"
        return (-limit, limit), "RdBu_r", label
    lower = min(0.0, float(np.min(finite))) if len(finite) else 0.0
    label = (
        "Pooled BOLD ΔR² (full − confound-only)"
        if statistic == "delta_r2"
        else "Pooled BOLD full-model R²"
    )
    cmap = (
        ListedColormap(colormaps["afmhot"](np.sqrt(np.linspace(0, 1, 1024))))
        if statistic == "delta_r2"
        else "viridis"
    )
    return (lower, 1.0), cmap, label


def _label(name):
    for prefix in ("Canonical", "Optimized"):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            suffix = {
                "GLM": "GLM",
                "TrialOLS": "trial OLS",
                "TrialRidge": "trial ridge",
                "TrialRidgeCV": "trial alpha CV",
                "TrialFractionalCV": "trial fractional CV",
            }.get(suffix, suffix)
            return prefix + "\n" + suffix
    return name


def _panel(axis, mesh, values, hemi, view, limits, cmap):
    # A triangle touching an undefined vertex is gray; no zero-filling or
    # averaging across missing vertices at the medial wall or a subset boundary.
    plot_surf(
        mesh,
        surf_map=values,
        hemi=hemi,
        view=view,
        engine="matplotlib",
        colorbar=False,
        avg_method="mean",
        alpha=1,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        axes=axis,
    )
    axis.set_box_aspect(None, zoom=1.25)


def surface_figure(maps, brain, meshes, *, statistic, title=None):
    """Compare maps in four cortical views with a shared scale and no threshold.

    R² and ΔR² span 0–1 (extended below zero when needed). ΔR² uses a square-root
    heat-color progression to emphasize small values, with ticks in original units.
    RT uses a symmetric range covering all finite cortical values; t uses ±10.
    Peak time uses a sequential scale spanning finite cortical values in seconds.
    Volumetric structures
    never affect the plots or color limits. Input maps and meshes are unchanged.
    """
    if not maps:
        raise ValueError("provide at least one map")
    projected = {name: cortical_values(values, brain) for name, values in maps.items()}
    loaded = _load_meshes(meshes, brain)
    limits, cmap, colorbar_label = _color_scale(projected, statistic)
    # An unmanaged Figure displays only when explicitly requested by the
    # notebook; pyplot registration would also display it at the cell boundary.
    fig = Figure(figsize=(13, 2.6 * len(maps) + 0.8))
    axes = fig.subplots(
        len(maps),
        4,
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.15, right=0.91, bottom=0.05, top=0.90, wspace=0, hspace=0.05
    )
    for row, (name, values) in enumerate(projected.items()):
        for col, (hemi, view) in enumerate(VIEWS):
            mesh = loaded[hemi]
            data = values.get(hemi, np.full(len(mesh.coordinates), np.nan))
            _panel(axes[row, col], mesh, data, hemi, view, limits, cmap)
            if row == 0:
                axes[row, col].set_title(f"{hemi.title()} {view}", fontsize=10)
        axes[row, 0].text2D(
            -0.08,
            0.5,
            _label(name),
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
    cax = fig.add_axes([0.94, 0.2, 0.014, 0.58])
    fig.colorbar(
        ScalarMappable(norm=Normalize(*limits), cmap=cmap),
        cax=cax,
        label=colorbar_label,
    )
    fig.suptitle(title or colorbar_label, fontsize=13, y=0.99)
    return fig
