import nibabel as nib
import numpy as np
import pytest

from experiments.hrf_estimation.test_inputs import bids as bids  # noqa: F401


def test_three_anatomical_labels_map_by_surface_identity():
    from experiments.hrf_simulations.rois import map_regions

    left = nib.cifti2.BrainModelAxis.from_surface([2, 0, 1], 3, "CortexLeft")
    right = nib.cifti2.BrainModelAxis.from_surface([1, 2, 0], 3, "CortexRight")
    sphere = np.eye(3)
    annotations = {
        s: np.array(["fusiform", "lateraloccipital", "inferiortemporal"])
        for s in ("L", "R")
    }
    spheres = {s: sphere for s in ("L", "R")}
    regions = map_regions(left + right, annotations, spheres, spheres)
    np.testing.assert_array_equal(regions["fusiform"], [1, 5])
    np.testing.assert_array_equal(regions["lateraloccipital"], [2, 3])
    np.testing.assert_array_equal(regions["inferiortemporal"], [0, 4])


def test_roi_bundle_selects_without_duplicate_data(bids, tmp_path):
    from experiments.hrf_simulations.bundle import (
        export_bundle,
        load_session,
        roi_inventory,
    )

    path = tmp_path / "three.h5"
    regions = {
        "fusiform": np.array([0]),
        "lateraloccipital": np.array([2]),
        "inferiortemporal": np.array([1]),
    }
    export_bundle(bids, path, np.array([0, 1, 2]), rois=regions)
    assert roi_inventory(path) == {name: 1 for name in regions}
    for name, ids in regions.items():
        runs = load_session(path, "sub-07", 11, roi=name)
        np.testing.assert_array_equal(runs[0].feature_ids, ids)
        np.testing.assert_array_equal(runs[0].signals[:, 0], np.arange(8) * 3 + ids[0])
    with pytest.raises(ValueError, match="ROI"):
        load_session(path, "sub-07", 11, roi="missing")


def test_calibration_roi_cannot_be_reused_for_another(tmp_path):
    import json
    from argparse import Namespace
    from experiments.hrf_simulations.cli import _check_calibration_subject

    (tmp_path / "provenance.json").write_text(
        json.dumps({"subject": "sub-07", "roi": "fusiform"})
    )
    with pytest.raises(ValueError, match="ROI"):
        _check_calibration_subject(
            Namespace(calibration=tmp_path, subject="sub-07", roi="lateraloccipital")
        )
