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


@pytest.mark.notebook
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
