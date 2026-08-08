import importlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from examples.stop_signal_demo import (
    common_roi_voxels,
    load_run,
    roi_image,
    run_sources,
)


TRIAL_TYPES = (
    "go_success",
    "go_failure",
    "stop_success",
    "stop_failure",
)
CONFOUNDS = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "framewise_displacement",
)


def _demo_module():
    return importlib.import_module("examples.stop_signal_demo")


def _discover(root, session="ses-02"):
    return _demo_module().discover_run_inputs(
        root,
        root / "derivatives" / "fmri_25.2.0",
        subject="sub-s4",
        session=session,
        task="stopSignal",
        run="run-01",
        space="MNI152NLin2009cAsym",
        resolution=2,
    )


def test_discover_run_inputs_matches_all_entities(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)

    assert isinstance(inputs, _demo_module().RunInputs)
    assert inputs.session == "ses-02"
    assert inputs.events.name.endswith("run-01_events.tsv")
    assert "space-MNI152NLin2009cAsym_res-2" in inputs.bold.name
    assert inputs.mask.name.endswith("desc-brain_mask.nii.gz")
    assert inputs.confounds.name.endswith("desc-confounds_timeseries.tsv")


def test_discover_run_inputs_rejects_missing_file(stop_signal_bids_dataset):
    _discover(stop_signal_bids_dataset).events.unlink()

    with pytest.raises(FileNotFoundError, match="events.*exactly one"):
        _discover(stop_signal_bids_dataset)


def test_discover_run_inputs_rejects_duplicate_file(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    duplicate = inputs.events.with_name(
        inputs.events.name.replace("events", "copy_events")
    )
    duplicate.write_bytes(inputs.events.read_bytes())

    with pytest.raises(ValueError, match="events.*exactly one"):
        _discover(stop_signal_bids_dataset)


def _loaded_runs(root):
    inputs = tuple(_discover(root, session) for session in ("ses-02", "ses-04"))
    voxels = common_roi_voxels(
        inputs, center_mni=(48.0, 16.0, 20.0), radius_mm=6.0
    )
    return inputs, voxels, tuple(
        load_run(
            item,
            voxels,
            trial_types=TRIAL_TYPES,
            confound_names=CONFOUNDS,
        )
        for item in inputs
    )


def test_load_run_extracts_common_bounded_roi(stop_signal_bids_dataset):
    _, voxels, runs = _loaded_runs(stop_signal_bids_dataset)

    assert voxels.ndim == 2 and voxels.shape[1] == 3
    assert len(voxels) > 1
    assert runs[0].signals.shape == (80, len(voxels))
    assert runs[1].signals.shape == (88, len(voxels))
    np.testing.assert_allclose(np.diff(runs[0].frame_times), 1.5)
    assert set(runs[0].events.trial_type) <= set(TRIAL_TYPES)
    assert tuple(runs[0].confounds) == CONFOUNDS
    assert runs[0].confounds.iloc[0].framewise_displacement == 0.0
    assert np.isfinite(runs[0].signals).all()


def test_roi_image_restores_values_to_spatial_coordinates(stop_signal_bids_dataset):
    _, voxels, runs = _loaded_runs(stop_signal_bids_dataset)
    values = np.arange(len(voxels), dtype=float)

    image = roi_image(values, runs[0])

    restored = image.get_fdata()[tuple(voxels.T)]
    np.testing.assert_array_equal(restored, values)
    assert image.shape == runs[0].spatial_shape


def test_load_run_rejects_confound_length_mismatch(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.confounds, sep="\t").iloc[:-1]
    frame.to_csv(inputs.confounds, sep="\t", index=False)
    voxels = common_roi_voxels(
        (inputs,), center_mni=(48.0, 16.0, 20.0), radius_mm=6.0
    )

    with pytest.raises(ValueError, match="confounds.*80 rows"):
        load_run(
            inputs, voxels, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS
        )


def test_load_run_rejects_unexpected_nonfinite_confound(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.confounds, sep="\t")
    frame.loc[3, "trans_x"] = np.nan
    frame.to_csv(inputs.confounds, sep="\t", index=False)
    voxels = common_roi_voxels(
        (inputs,), center_mni=(48.0, 16.0, 20.0), radius_mm=6.0
    )

    with pytest.raises(ValueError, match="non-finite.*trans_x"):
        load_run(
            inputs, voxels, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS
        )


def test_run_sources_records_dataset_relative_inputs(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)

    sources = run_sources(inputs, stop_signal_bids_dataset)

    assert sources.signal.uri.startswith("derivatives/fmri_25.2.0/")
    assert sources.signal.byte_size == inputs.bold.stat().st_size
    assert sources.signal.annotations["mask"]["uri"].endswith("brain_mask.nii.gz")
    assert sources.events.uri.startswith("sub-s4/ses-02/")
    assert sources.confounds.uri.endswith("desc-confounds_timeseries.tsv")
    assert sources.signal.modified_at.endswith("Z")


def test_common_roi_rejects_nonoverlap(stop_signal_bids_dataset):
    inputs = (_discover(stop_signal_bids_dataset),)

    with pytest.raises(ValueError, match="ROI does not overlap"):
        common_roi_voxels(
            inputs, center_mni=(500.0, 500.0, 500.0), radius_mm=1.0
        )


def test_load_run_rejects_event_beyond_acquisition(stop_signal_bids_dataset):
    inputs = _discover(stop_signal_bids_dataset)
    frame = pd.read_csv(inputs.events, sep="\t")
    frame.loc[0, ["onset", "duration"]] = [119.5, 1.0]
    frame.to_csv(inputs.events, sep="\t", index=False)
    voxels = common_roi_voxels(
        (inputs,), center_mni=(48.0, 16.0, 20.0), radius_mm=6.0
    )

    with pytest.raises(ValueError, match="event timing exceeds acquisition"):
        load_run(
            inputs, voxels, trial_types=TRIAL_TYPES, confound_names=CONFOUNDS
        )
