"""Surface summaries preserve CIFTI vertex identity and signed statistics."""

import importlib

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest


def surfaces():
    try:
        return importlib.import_module("examples.NSD.workflow_surfaces")
    except ModuleNotFoundError as error:
        pytest.fail(f"Cortical surface summaries are not implemented: {error}")


@pytest.fixture
def cortical_axis():
    axis = nib.cifti2.BrainModelAxis
    right = axis.from_surface([5, 0, 2], 6, name="CortexRight")
    volume = axis.from_mask(np.ones((1, 1, 1), bool), name="ThalamusLeft")
    left = axis.from_surface([4, 1, 3, 0], 6, name="CortexLeft")
    return right + volume + left


@pytest.fixture
def surface_files(tmp_path):
    """Two real, small GIFTI meshes for offline rendering and discovery."""
    coords = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    folder = tmp_path / "fmriprep/sub-07/ses-nsdanat/anat"
    folder.mkdir(parents=True)
    paths = {}
    for hemi, code in (("left", "L"), ("right", "R")):
        path = (
            folder
            / f"sub-07_ses-nsdanat_hemi-{code}_space-fsLR_den-32k_midthickness.surf.gii"
        )
        nib.save(
            nib.gifti.GiftiImage(
                darrays=[
                    nib.gifti.GiftiDataArray(coords, intent="NIFTI_INTENT_POINTSET"),
                    nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"),
                ]
            ),
            path,
        )
        paths[hemi] = path
    return tmp_path / "fmriprep", paths


def test_surface_values_use_vertex_ids_and_exclude_volume(cortical_axis):
    values = np.array([0.1, -0.2, np.nan, 99, 0.7, 0, -0.6, 0.3])
    original = values.copy()
    projected = surfaces().cortical_values(values, cortical_axis)
    np.testing.assert_allclose(projected["left"], [0.3, 0, np.nan, -0.6, 0.7, np.nan])
    np.testing.assert_allclose(
        projected["right"], [-0.2, np.nan, np.nan, np.nan, np.nan, 0.1]
    )
    np.testing.assert_allclose(values, original)
    with pytest.raises(ValueError, match="grayordinate"):
        surfaces().cortical_values(values[:-1], cortical_axis)


def test_multisession_hrf_surface_shows_matched_delta(cortical_axis, surface_files):
    import nbformat
    from pathlib import Path
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from examples.NSD.multisession_plots import comparison_surface
    from examples.NSD.session_hrf_plots import agreement_figure

    notebook = nbformat.read(
        Path(__file__).with_name("nsd_multisession.ipynb"), as_version=4
    )
    cell = next(c for c in notebook.cells if 'figures["HRFSurface"] =' in c.source)
    between = np.full(8, 0.8)
    baseline = np.array([0.6, 0.9, np.nan, 0.8, 0.7, 0.8, 0.9, 0.7])
    hrf = dict(
        sessions=["ses-nsd10", "ses-nsd11"],
        pairs=[(0, 1)],
        pair_names=["ses-nsd10_vs_ses-nsd11"],
        pairwise=between[None],
        canonical=np.stack([baseline, baseline]),
        summary=np.stack([between, baseline, between - baseline]),
    )
    figures = {}
    try:
        exec(
            cell.source,
            dict(
                result={"hrf": hrf},
                brain=cortical_axis,
                meshes=surface_files[1],
                figures=figures,
                comparison_surface=comparison_surface,
                agreement_figure=agreement_figure,
                display=lambda _: None,
                np=np,
                pd=pd,
                plt=plt,
            ),
        )
        fig = figures["HRFSurface"]
        assert len([ax for ax in fig.axes if ax.name == "3d"]) == 4
        np.testing.assert_allclose(fig.axes[-1].get_ylim(), [-0.2, 0.2])
        assert "Δr" in fig.axes[-1].get_ylabel()
        assert "between-session" in fig.axes[-1].get_ylabel().lower()
        assert "canonical" in fig.axes[-1].get_ylabel().lower()
        FigureCanvasAgg(fig).draw()
        renderer = fig.canvas.get_renderer()
        for label in [fig.axes[-1].yaxis.label, *fig.axes[0].texts]:
            bounds = label.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width
            assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.height
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_discovery_uses_fslr_meshes_and_supports_explicit_paths(surface_files):
    root, paths = surface_files
    native = paths["left"].with_name("sub-07_hemi-L_inflated.surf.gii")
    native.write_bytes(paths["left"].read_bytes())
    assert surfaces().find_surface_meshes(root, "sub-07") == paths
    assert surfaces().find_surface_meshes(root, "sub-99") is None
    assert surfaces().find_surface_meshes(root, "sub-99", paths=paths) == paths
    with pytest.raises(ValueError, match="left.*right"):
        surfaces().find_surface_meshes(root, "sub-07", paths={"left": paths["left"]})
    duplicate = paths["left"].with_name(
        paths["left"].name.replace("sub-07_", "sub-07_copy_")
    )
    duplicate.write_bytes(paths["left"].read_bytes())
    with pytest.raises(ValueError, match="multiple|ambiguous"):
        surfaces().find_surface_meshes(root, "sub-07")


def test_multisession_glm_cell_renders_absolute_and_delta_maps(
    cortical_axis, surface_files
):
    from pathlib import Path
    import nbformat
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    notebook = nbformat.read(Path(__file__).with_name("nsd_multisession.ipynb"), 4)
    cells = {c.id: c for c in notebook.cells}
    assert "glm-session-surfaces" in cells
    from examples.NSD.multisession_plots import glm_effect_figures

    canonical = np.stack([np.full(8, 3.0), np.full(8, -0.4)])
    optimized = np.stack([np.full(8, 5.0), np.full(8, -0.2)])
    result = dict(
        glm_metrics=("task", "response_time"),
        glm=dict(
            canonical_mean=canonical,
            optimized_mean=optimized,
            difference_mean=optimized - canonical,
        ),
    )
    figures = {}
    try:
        exec(
            cells["glm-session-surfaces"].source,
            dict(
                result=result,
                brain=cortical_axis,
                meshes=surface_files[1],
                glm_effect_figures=glm_effect_figures,
                figures=figures,
                display=lambda _: None,
            ),
        )
        assert set(figures) == {"GLMTask", "GLMResponseTime"}
        for name, limit in (("GLMTask", 5), ("GLMResponseTime", 0.4)):
            fig = figures[name]
            assert len([a for a in fig.axes if a.name == "3d"]) == 12
            np.testing.assert_allclose(fig.axes[-1].get_ylim(), [-limit, limit])
            assert "signal units" in fig.axes[-1].get_ylabel().lower()
            if name == "GLMResponseTime":
                assert "second" in fig.axes[-1].get_ylabel().lower()
            FigureCanvasAgg(fig).draw()
            expected = (3, 5, 2) if name == "GLMTask" else (-0.4, -0.2, 0.2)
            colorbar = fig.axes[-1]._colorbar
            for axis_index, value in zip((0, 4, 8), expected, strict=True):
                colors = np.asarray(
                    fig.axes[axis_index].collections[0].get_facecolors()
                )
                target = colorbar.cmap(colorbar.norm(value))
                # Triangle averaging can shift a boundary value by one color bin.
                assert np.any(
                    np.all(np.isclose(colors, target, atol=0.02, rtol=0), axis=1)
                )
            assert (
                fig.axes[-1].yaxis.label.get_window_extent(fig.canvas.get_renderer()).x1
                <= fig.bbox.width
            )
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_surface_figure_keeps_signed_shared_scale_and_missing_data(
    cortical_axis, surface_files, tmp_path
):
    _, meshes = surface_files
    maps = {
        "Canonical": np.array([0.1, -0.2, np.nan, 999, 0.6, 0, -0.4, 0.3]),
        "Optimized": np.array([0.2, -0.3, 0.1, -999, 0.5, 0, -0.2, 0.4]),
    }
    fig = surfaces().surface_figure(maps, cortical_axis, meshes, statistic="rt")
    try:
        assert len([a for a in fig.axes if a.name == "3d"]) == 8
        np.testing.assert_allclose(fig.axes[-1].get_ylim(), [-0.6, 0.6])
        assert "Pearson" in fig.axes[-1].get_ylabel()
        path = tmp_path / "rt.png"
        fig.savefig(path)
        assert path.stat().st_size > 1000
    finally:
        plt.close(fig)


def test_empty_cortex_renders_without_warnings_and_preserves_negative_r2(
    cortical_axis, surface_files
):
    _, meshes = surface_files
    module = surfaces()
    for values, expected in (
        (np.full(8, np.nan), [0, 1]),
        (np.full(8, -0.2), [-0.2, 1]),
    ):
        fig = module.surface_figure(
            {"Model": values}, cortical_axis, meshes, statistic="r2"
        )
        try:
            np.testing.assert_allclose(fig.axes[-1].get_ylim(), expected)
            fig.canvas.draw()
        finally:
            plt.close(fig)


def test_wrong_surface_density_is_rejected(cortical_axis, surface_files):
    _, meshes = surface_files
    left = nib.cifti2.BrainModelAxis.from_surface([0, 1], 8, name="CortexLeft")
    with pytest.raises(ValueError, match="vertices|vertex"):
        surfaces().surface_figure({"Model": np.ones(2)}, left, meshes, statistic="r2")


@pytest.mark.parametrize(
    "values,limits",
    [
        ([2.5, 5.1, np.nan, 999, 4, 3, 4.5, 4], [2.5, 5.1]),
        ([4] * 8, [3.5, 4.5]),
        ([np.nan] * 8, [0, 1]),
    ],
)
def test_peak_time_surface_uses_sequential_seconds_scale(
    cortical_axis, surface_files, values, limits
):
    from pathlib import Path
    import nbformat
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    notebook = nbformat.read(Path(__file__).with_name("nsd_multisession.ipynb"), 4)
    cells = {c.id: c for c in notebook.cells}
    assert "hrf-peak-time-surface" in cells
    from examples.NSD.multisession_plots import peak_time_surface

    figures = {}
    try:
        exec(
            cells["hrf-peak-time-surface"].source,
            dict(
                result={"hrf": {"peak_time_mean": np.array(values)}},
                brain=cortical_axis,
                meshes=surface_files[1],
                figures=figures,
                peak_time_surface=peak_time_surface,
                display=lambda _: None,
            ),
        )
        fig = figures["HRFPeakTime"]
        assert len([a for a in fig.axes if a.name == "3d"]) == 4
        np.testing.assert_allclose(fig.axes[-1].get_ylim(), limits)
        assert "seconds" in fig.axes[-1].get_ylabel()
        assert fig.axes[-1]._colorbar.cmap.name == "viridis"
        FigureCanvasAgg(fig).draw()
        label = fig.axes[-1].yaxis.label.get_window_extent(fig.canvas.get_renderer())
        assert label.x0 >= 0 and label.x1 <= fig.bbox.width
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_delta_r2_surface_labels_task_contribution(cortical_axis, surface_files):
    _, meshes = surface_files
    fig = surfaces().surface_figure(
        {"Model": np.full(8, 0.1)}, cortical_axis, meshes, statistic="delta_r2"
    )
    try:
        assert "ΔR²" in fig.axes[-1].get_ylabel()
        assert "confound-only" in fig.axes[-1].get_ylabel()
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_delta_r2_heat_colors_emphasize_small_values(cortical_axis, surface_files):
    from matplotlib import colormaps

    _, meshes = surface_files
    fig = surfaces().surface_figure(
        {"Model": np.full(8, 0.1)}, cortical_axis, meshes, statistic="delta_r2"
    )
    try:
        colorbar = fig.axes[-1]._colorbar
        # A quarter of the variance range gets half of the color progression.
        np.testing.assert_allclose(
            colorbar.cmap(colorbar.norm(0.25)), colormaps["afmhot"](0.5), atol=0.01
        )
        np.testing.assert_allclose(colorbar.cmap(0.0), [0, 0, 0, 1])
        np.testing.assert_allclose(colorbar.cmap(1.0), [1, 1, 1, 1])
        np.testing.assert_allclose(colorbar.norm([0, 1]), [0, 1])
    finally:
        plt.close(fig)


def test_beta_surface_cell_plots_full_minus_confound_r2():
    import nbformat
    from pathlib import Path

    notebook = nbformat.read(
        Path(__file__).with_name("nsd_workflow.ipynb"), as_version=4
    )
    cell = next(c for c in notebook.cells if c.id == "beta-surface-maps")
    full = np.array([0.9, 0.7, np.nan])
    nuisance = np.array([0.8, 0.65, np.nan])
    expected = full - nuisance
    captured = {}

    def capture(maps, brain, meshes, **options):
        captured.update(maps=maps, **options)

    exec(
        cell.source,
        dict(
            surface_meshes={},
            brain=None,
            surface_figure=capture,
            beta_models={"OLS": {"r2": np.stack([full, nuisance, expected])}},
            figures={},
            display=lambda figure: None,
        ),
    )
    np.testing.assert_allclose(captured["maps"]["OLS"], expected)
    assert captured["statistic"] == "delta_r2"
    assert "ΔR²" in captured["title"]


@pytest.mark.parametrize("magnitude", [5, 50])
def test_activation_surface_uses_fixed_t_scale(cortical_axis, surface_files, magnitude):
    _, meshes = surface_files
    values = np.array([2, -magnitude, np.nan, 999, 3, 0, -1, magnitude])
    fig = surfaces().surface_figure(
        {"OLS": values}, cortical_axis, meshes, statistic="t"
    )
    try:
        np.testing.assert_allclose(fig.axes[-1].get_ylim(), [-10, 10])
        assert "t" in fig.axes[-1].get_ylabel().lower()
        assert "Pearson" not in fig.axes[-1].get_ylabel()
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_notebook_surface_cells_render_existing_results_and_register_exports(
    cortical_axis, surface_files, tmp_path
):
    """Execute actual cells; no fits or network access are needed for these plots."""
    import nbformat
    from pathlib import Path

    root, meshes = surface_files
    notebook = nbformat.read(
        Path(__file__).with_name("nsd_workflow.ipynb"), as_version=4
    )
    cells = {c.id: c for c in notebook.cells}
    required = ("glm-surface-maps", "beta-surface-maps", "rt-surface-maps")
    assert all(
        key in cells for key in required
    ), "Notebook needs executable surface-map cells"
    model = dict(r2=np.full((3, 8), 0.2), rt={"all": np.linspace(-0.2, 0.3, 8)})
    context = dict(
        settings={"subject": "sub-07", "surface_maps": True, "surface_meshes": meshes},
        prep=root,
        brain=cortical_axis,
        glms={"CanonicalGLM": model, "OptimizedGLM": model},
        beta_models={"CanonicalTrialOLS": model, "OptimizedTrialFractionalCV": model},
        figures={},
        display=lambda figure: None,
    )
    try:
        for key in required:
            exec(cells[key].source, context)
        assert {"GLMR2Surface", "BetaR2Surface", "RTSurface"} <= context[
            "figures"
        ].keys()
        from examples.NSD.hrf_artifacts import figure_artifact
        from boldtailor.publication import publish_artifact_set

        artifacts = [
            figure_artifact(name + ".png", fig)
            for name, fig in context["figures"].items()
        ]
        saved = publish_artifact_set(tmp_path / "output", artifacts)
        assert len(saved) == 3
        assert all(p.stat().st_size > 1000 for p in saved)
    finally:
        plt.close("all")


def test_inline_notebook_displays_each_surface_figure_once(
    cortical_axis, surface_files, tmp_path
):
    import nbformat
    from nbclient import NotebookClient
    from pathlib import Path

    root, meshes = surface_files
    path = Path(__file__).with_name("nsd_workflow.ipynb")
    cells = {c.id: c for c in nbformat.read(path, as_version=4).cells}
    image_path = tmp_path / "axis.dscalar.nii"
    nib.save(
        nib.Cifti2Image(
            np.zeros((1, len(cortical_axis))),
            header=nib.Cifti2Header.from_axes(
                (nib.cifti2.ScalarAxis(["test"]), cortical_axis)
            ),
        ),
        image_path,
    )
    setup = f"""%matplotlib inline
from pathlib import Path
import sys
sys.path.insert(0, {str(path.parents[2])!r})
import numpy as np
import nibabel as nib
from IPython.display import display
settings = {{"subject": "sub-07", "surface_meshes": {str({k: str(v) for k, v in meshes.items()})}}}
prep = Path({str(root)!r})
brain = nib.load({str(image_path)!r}).header.get_axis(1)
model = dict(r2=np.full((3, 8), .2), rt={{"all": np.linspace(-.2, .3, 8)}})
glms = {{"CanonicalGLM": model, "OptimizedGLM": model}}
beta_models = {{"CanonicalTrialOLS": model, "OptimizedTrialFractionalCV": model}}
figures = {{}}
"""
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(setup),
            *[
                cells[key]
                for key in ("glm-surface-maps", "beta-surface-maps", "rt-surface-maps")
            ],
        ]
    )
    executed = NotebookClient(notebook, timeout=60, kernel_name="python3").execute()
    for cell in executed.cells[1:]:
        images = [o for o in cell.outputs if "image/png" in o.get("data", {})]
        assert (
            len(images) == 1
        ), f"{cell.id} should display its surface figure exactly once"
