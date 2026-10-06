from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from experiments.nsd_replication.released import (
    alignment_check,
    bids_trials,
    load_released,
    released_path,
)

DATA = Path("/Volumes/extdata1/NSD/BIDS")


def _brain():
    left = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 1]), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    right = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0]), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    mask = np.zeros((2, 1, 1), bool)
    mask[0, 0, 0] = True
    thal = nib.cifti2.BrainModelAxis.from_mask(mask, "CIFTI_STRUCTURE_THALAMUS_LEFT")
    return left + right + thal


def _write(path, values, names=None):
    names = names or [f"trial-{i:03d}" for i in range(1, len(values) + 1)]
    image = nib.Cifti2Image(values, header=(nib.cifti2.ScalarAxis(names), _brain()))
    nib.save(image, path)
    return path


def test_load_released_keeps_cortex_in_order(tmp_path):
    values = np.array([[1, 2, 3, np.nan], [4, 5, 6, np.nan]], dtype=np.float32)
    path = _write(tmp_path / "b.dscalar.nii", values)
    cortex = _brain()[[0, 1, 2]]
    got = load_released(path, cortex, 2, "sub-07 ses-nsd10 assumehrf")
    np.testing.assert_array_equal(got, values[:, :3])


def test_released_trial_count_mismatch_names_session(tmp_path):
    path = _write(tmp_path / "b.dscalar.nii", np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError, match="sub-07 ses-nsd10 assumehrf.*3 trials.*750"):
        load_released(path, _brain()[[0, 1, 2]], 750, "sub-07 ses-nsd10 assumehrf")


def test_released_axis_mismatch_raises(tmp_path):
    path = _write(tmp_path / "b.dscalar.nii", np.zeros((2, 4), np.float32))
    with pytest.raises(ValueError, match="grayordinates"):
        load_released(path, _brain()[[0, 1]], 2, "x")


def test_released_map_order_checked(tmp_path):
    path = _write(
        tmp_path / "b.dscalar.nii",
        np.zeros((2, 4), np.float32),
        ["trial-002", "trial-001"],
    )
    with pytest.raises(ValueError, match="order"):
        load_released(path, _brain()[[0, 1, 2]], 2, "x")


def test_missing_released_file_names_version(tmp_path):
    with pytest.raises(FileNotFoundError, match="fithrf"):
        load_released(
            tmp_path / "missing.dscalar.nii", _brain(), 2, "sub-07 ses-nsd10 fithrf"
        )


def test_alignment_check_detects_shift():
    rng = np.random.default_rng(0)
    released = rng.normal(size=(120, 5))
    ours = released + rng.normal(scale=0.5, size=released.shape)
    result = alignment_check(ours, released, 60, np.ones(5, bool))
    assert result["true_median"] > 0.8 and abs(result["null_median"]) < 0.3


@pytest.mark.skipif(not DATA.is_dir(), reason="NSD data volume not mounted")
def test_pilot_released_matches_ppdata_axis():
    from experiments.nsd_replication.config import ExperimentConfig
    from experiments.nsd_replication.inputs import ppdata_brain

    config = ExperimentConfig(
        bids_dir=DATA,
        output_dir=DATA / "derivatives/nsd-replication",
        freesurfer_dir=DATA / "derivatives/freesurfer-NSD",
        ppdata_dir=DATA / "derivatives/ppdata",
        released_dir=DATA / "derivatives/betas-fsLR",
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
    )
    brain = ppdata_brain(config, "sub-07", "ses-nsd10")
    trials = bids_trials(config, "sub-07", "ses-nsd10")
    path = released_path(config, "sub-07", "ses-nsd10", "b4")
    got = load_released(path, brain, len(trials), "sub-07 ses-nsd10 b4")
    assert got.shape == (750, 59412) and got.dtype == np.float32


def test_discarded_released_betas_are_not_current(tmp_path):
    import json

    import pandas as pd

    from experiments.nsd_replication import released
    from experiments.nsd_replication.betas import write_fit

    source = _write(tmp_path / "b.dscalar.nii", np.zeros((2, 4), np.float32))
    stat = released._stat(source)
    trials = pd.DataFrame(
        {
            "session": "s",
            "run": "r",
            "trial": [0, 1],
            "onset": [0.0, 4.0],
            "image": [1, 2],
        }
    )
    target = tmp_path / "fit"
    write_fit(target, np.zeros((2, 3)), trials, {"level": "b1", **stat})
    assert released._current(target, stat)
    meta = json.loads((target / "metadata.json").read_text())
    (target / "metadata.json").write_text(json.dumps(dict(meta, betas_discarded=True)))
    (target / "betas.npy").unlink()
    assert not released._current(target, stat)
