"""CIFTI scalar exports, reloads, and cortical projections keep grayordinate identity."""

from hashlib import sha256

import nibabel as nib
import numpy as np
import pytest

from boldtailor import cifti


@pytest.fixture
def cortical_axis():
    axis = nib.cifti2.BrainModelAxis
    right = axis.from_surface([5, 0, 2], 6, name="CortexRight")
    volume = axis.from_mask(np.ones((1, 1, 1), bool), name="ThalamusLeft")
    left = axis.from_surface([4, 1, 3, 0], 6, name="CortexLeft")
    return right + volume + left


def test_surface_values_use_vertex_ids_and_exclude_volume(cortical_axis):
    values = np.array([0.1, -0.2, np.nan, 99, 0.7, 0, -0.6, 0.3])
    original = values.copy()
    projected = cifti.cortical_values(values, cortical_axis)
    np.testing.assert_allclose(projected["left"], [0.3, 0, np.nan, -0.6, 0.7, np.nan])
    np.testing.assert_allclose(
        projected["right"], [-0.2, np.nan, np.nan, np.nan, np.nan, 0.1]
    )
    np.testing.assert_allclose(values, original)
    with pytest.raises(ValueError, match="grayordinate"):
        cifti.cortical_values(values[:-1], cortical_axis)


def test_scalar_artifact_round_trips_names_axis_and_float32_values(
    cortical_axis, tmp_path
):
    values = np.arange(16, dtype=float).reshape(2, 8) / 3
    values[1, 2] = np.nan
    artifact = cifti.scalar_artifact(
        "a/b.dscalar.nii", cortical_axis, values, ["x", "y"]
    )
    assert artifact.path == "a/b.dscalar.nii"
    path = tmp_path / "b.dscalar.nii"
    path.write_bytes(artifact.payload)
    image = nib.load(path)
    assert image.header.get_axis(0).name.tolist() == ["x", "y"]
    assert image.header.get_axis(1) == cortical_axis
    assert image.nifti_header.get_intent()[0] == "dense scalar"
    np.testing.assert_array_equal(image.get_fdata(), values.astype(np.float32))


def test_read_scalar_checks_the_spatial_axis_and_map_names(cortical_axis, tmp_path):
    path = tmp_path / "map.dscalar.nii"
    values = np.ones((1, 8))
    path.write_bytes(
        cifti.scalar_artifact(path.name, cortical_axis, values, ["r"]).payload
    )
    np.testing.assert_array_equal(cifti.read_scalar(path, cortical_axis), values)
    np.testing.assert_array_equal(cifti.read_scalar(path, cortical_axis, ["r"]), values)
    with pytest.raises(ValueError, match="map names"):
        cifti.read_scalar(path, cortical_axis, ["other"])
    with pytest.raises(ValueError, match="axis"):
        cifti.read_scalar(path, cortical_axis[::-1])


def test_spatial_signature_identifies_axis_and_ordered_indices(cortical_axis):
    header = nib.Cifti2Header.from_axes(
        (nib.cifti2.ScalarAxis(["spatial_identity"]), cortical_axis)
    )
    expected = sha256(
        header.to_xml() + np.array([0, 3], dtype="<i8").tobytes()
    ).hexdigest()
    assert cifti.spatial_signature(cortical_axis, [0, 3]) == expected
    assert cifti.spatial_signature(cortical_axis, [3, 0]) != expected
    assert cifti.spatial_signature(cortical_axis[::-1], [0, 3]) != expected
