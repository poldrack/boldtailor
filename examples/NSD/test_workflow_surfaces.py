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
