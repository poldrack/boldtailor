from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

SESSIONS = ("ses-02", "ses-04")
AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, 42.0],
        [0.0, 2.0, 0.0, 10.0],
        [0.0, 0.0, 2.0, 14.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _events(session):
    trial_types = [
        "go_success",
        "stop_success",
        "stop_failure",
        "go_failure",
        "go_success",
        "stop_success",
        "stop_failure",
        "go_success",
    ]
    if session == "ses-04":
        trial_types.remove("go_failure")
    return pd.DataFrame(
        {
            "onset": np.arange(5.0, 5.0 + 10.0 * len(trial_types), 10.0),
            "duration": np.ones(len(trial_types)),
            "trial_type": trial_types,
        }
    )


def _confounds(n_scans):
    rng = np.random.default_rng(20260808 + n_scans)
    frame = pd.DataFrame(
        rng.normal(0.0, 0.05, size=(n_scans, 6)),
        columns=("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"),
    )
    frame["framewise_displacement"] = np.abs(rng.normal(0.1, 0.02, size=n_scans))
    frame.loc[0, "framewise_displacement"] = np.nan
    return frame


@pytest.fixture
def stop_signal_bids_dataset(tmp_path):
    root = tmp_path / "rdoc_fmri"
    derivative = root / "derivatives" / "fmri_25.2.0"
    rng = np.random.default_rng(20260808)
    for index, session in enumerate(SESSIONS):
        n_scans = 80 + index * 8
        raw_func = root / "sub-s4" / session / "func"
        derivative_func = derivative / "sub-s4" / session / "func"
        raw_func.mkdir(parents=True)
        derivative_func.mkdir(parents=True)
        stem = f"sub-s4_{session}_task-stopSignal_run-01"
        _events(session).to_csv(raw_func / f"{stem}_events.tsv", sep="\t", index=False)
        shape = (7, 7, 7, n_scans)
        signal = rng.normal(1000.0, 3.0, shape).astype(np.float32)
        image = nib.Nifti1Image(signal, AFFINE)
        image.header.set_zooms((2.0, 2.0, 2.0, 1.5))
        nib.save(
            image,
            derivative_func
            / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        )
        mask = nib.Nifti1Image(np.ones(shape[:3], dtype=np.uint8), AFFINE)
        nib.save(
            mask,
            derivative_func
            / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        )
        _confounds(n_scans).to_csv(
            derivative_func / f"{stem}_desc-confounds_timeseries.tsv",
            sep="\t",
            index=False,
        )
    (root / "dataset_description.json").write_text(
        '{"Name":"fixture","BIDSVersion":"1.11.1"}\n'
    )
    return root
