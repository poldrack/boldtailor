import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from experiments.nsd_replication.roi import (
    resample_command,
    resampling_inputs,
    roi_mask,
)

DATA = Path("/Volumes/extdata1/NSD/BIDS")


def _brain():
    left = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 3]), 32492, "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    right = nib.cifti2.BrainModelAxis.from_surface(
        np.array([5]), 32492, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    return left + right


def _labels():
    return {"L": np.zeros(32492), "R": np.zeros(32492)}


def test_roi_threshold_at_half():
    labels = _labels()
    labels["L"][[0, 3]] = [0.6, 0.4]
    labels["R"][5] = 1.0
    assert roi_mask(_brain(), labels).tolist() == [True, False, True]


def test_empty_roi_raises():
    with pytest.raises(ValueError, match="empty"):
        roi_mask(_brain(), _labels())


def test_wrong_label_length_raises():
    labels = _labels()
    labels["L"] = np.zeros(10)
    with pytest.raises(ValueError, match="32492"):
        roi_mask(_brain(), labels)


def test_resampling_inputs_and_command():
    sidecar = {
        "SourceSpheres": ["/s/lh.sphere", "/s/rh.sphere"],
        "TargetSpheres": ["/t/L.sphere", "/t/R.sphere"],
        "SourceAreaMetrics": ["/s/lh.area", "/s/rh.area"],
        "TargetAreaMetrics": ["/t/L.area", "/t/R.area"],
    }
    inputs = resampling_inputs(sidecar)
    assert inputs["R"]["source_sphere"] == Path("/s/rh.sphere")
    command = resample_command("in.gii", "out.gii", inputs["L"])
    assert command == [
        "wb_command",
        "-metric-resample",
        "in.gii",
        "/s/lh.sphere",
        "/t/L.sphere",
        "ADAP_BARY_AREA",
        "out.gii",
        "-area-metrics",
        "/s/lh.area",
        "/t/L.area",
    ]


def test_resampling_inputs_missing_key_raises():
    with pytest.raises(ValueError, match="TargetAreaMetrics"):
        resampling_inputs(
            {
                "SourceSpheres": [1, 2],
                "TargetSpheres": [1, 2],
                "SourceAreaMetrics": [1, 2],
            }
        )


@pytest.mark.skipif(not DATA.is_dir(), reason="NSD data volume not mounted")
def test_pilot_roi_on_ppdata_cortex(tmp_path):
    import shutil

    if shutil.which("wb_command") is None:
        pytest.skip("Connectome Workbench not installed")
    from experiments.nsd_replication.config import ExperimentConfig
    from experiments.nsd_replication.inputs import ppdata_brain
    from experiments.nsd_replication.roi import load_roi

    config = ExperimentConfig(
        bids_dir=DATA,
        output_dir=tmp_path,
        freesurfer_dir=DATA / "derivatives/freesurfer-NSD",
        released_dir=DATA / "derivatives/betas-fsLR",
        ppdata_dir=DATA / "derivatives/ppdata",
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
    )
    mask = load_roi(config, "sub-07", ppdata_brain(config, "sub-07", "ses-nsd10"))
    assert mask.shape == (59412,)
    assert 0.05 < mask.mean() < 0.4
