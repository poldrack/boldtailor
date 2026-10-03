"""Surface summaries preserve CIFTI vertex identity and signed statistics."""

import importlib

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest


def surfaces():
    try:
        return importlib.import_module("boldtailor.workflow.surfaces")
    except ModuleNotFoundError as error:
        pytest.fail(f"Workflow surface module is missing: {error}")


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
