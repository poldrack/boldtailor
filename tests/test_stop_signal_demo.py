import importlib
from pathlib import Path

import pytest


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
