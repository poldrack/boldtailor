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


@pytest.fixture
def label_source(tmp_path):
    path = tmp_path / "src" / "lh.nsdgeneral.mgz"
    path.parent.mkdir()
    nib.save(nib.MGHImage(np.zeros((10, 1, 1), np.float32), np.eye(4)), path)
    return path


def _inputs():
    return {
        k: Path(k)
        for k in ("source_sphere", "target_sphere", "source_area", "target_area")
    }


def test_failed_resample_leaves_no_cache_file(tmp_path, label_source, monkeypatch):
    from experiments.nsd_replication import roi

    def partial(command):
        Path(command[6]).write_text("partial")
        raise RuntimeError("wb_command failed")

    monkeypatch.setattr(roi, "_run", partial)
    out = tmp_path / "roi"
    out.mkdir()
    with pytest.raises(RuntimeError):
        roi._resample(
            label_source, out / "sub-07_hemi-L_nsdgeneral.func.gii", _inputs()
        )
    assert list(out.iterdir()) == []


def test_resample_moves_finished_file_into_place(tmp_path, label_source, monkeypatch):
    from experiments.nsd_replication import roi

    def finish(command):
        assert Path(command[6]).parent == out
        roi._save_metric(np.ones(5), command[6])

    monkeypatch.setattr(roi, "_run", finish)
    out = tmp_path / "roi"
    out.mkdir()
    target = out / "sub-07_hemi-L_nsdgeneral.func.gii"
    roi._resample(label_source, target, _inputs())
    assert list(out.iterdir()) == [target]
    np.testing.assert_array_equal(nib.load(target).agg_data(), np.ones(5))


def test_load_roi_requires_released_dir(tmp_path):
    from experiments.nsd_replication.config import ExperimentConfig
    from experiments.nsd_replication.roi import load_roi

    config = ExperimentConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path,
        freesurfer_dir=tmp_path,
        subjects=("sub-07",),
        sessions=("ses-a",),
    )
    with pytest.raises(ValueError, match="released_dir is required"):
        load_roi(config, "sub-07", _brain())
