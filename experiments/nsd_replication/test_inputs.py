from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.inputs import (
    CORTEX,
    Session,
    analysis_data,
    cortical_indices,
    same_grayordinates,
)

DATA = Path("/Volumes/extdata1/NSD/BIDS")


def _brain(with_cortex=True):
    left = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 2]), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    right = nib.cifti2.BrainModelAxis.from_surface(
        np.array([1]), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, :] = True
    thal = nib.cifti2.BrainModelAxis.from_mask(mask, "CIFTI_STRUCTURE_THALAMUS_LEFT")
    return (left + thal + right) if with_cortex else (left + thal)


def _cortex_only():
    left = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 2]), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    right = nib.cifti2.BrainModelAxis.from_surface(
        np.array([1]), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    return left + right


def test_cortical_indices_keep_surface_in_order():
    assert cortical_indices(_brain()).tolist() == [0, 1, 4]


def test_missing_cortical_structure_raises():
    with pytest.raises(ValueError, match="CORTEX_RIGHT"):
        cortical_indices(_brain(with_cortex=False))


def test_same_grayordinates_ignores_volume_metadata():
    full = _brain()
    assert same_grayordinates(full[cortical_indices(full)], _cortex_only())
    assert not same_grayordinates(full[[0, 1]], _cortex_only())


def test_analysis_data_selects_columns_and_keeps_runs():
    signals = (np.arange(12, dtype=np.float32).reshape(4, 3),) * 2
    events = (pd.DataFrame({"onset": [0.0], "duration": [1.0]}),) * 2
    session = Session(
        subject="sub-07",
        session="ses-nsd10",
        labels=("run-01", "run-02"),
        signals=signals,
        events=events,
        confounds=(pd.DataFrame({"c": [0.0, 1, 0, 1]}),) * 2,
        frame_times=(np.arange(4) * 1.6,) * 2,
        brain=_cortex_only(),
        task_model=None,
        source="ppdata",
    )
    data = analysis_data(session, np.array([0, 2]))
    assert data.n_runs == 2
    np.testing.assert_array_equal(data.signals[0], signals[0][:, [0, 2]])


def _pilot_config():
    from experiments.nsd_replication.config import ExperimentConfig

    return ExperimentConfig(
        bids_dir=DATA,
        output_dir=DATA / "derivatives/nsd-replication",
        freesurfer_dir=DATA / "derivatives/freesurfer-NSD",
        ppdata_dir=DATA / "derivatives/ppdata",
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
    )


@pytest.mark.skipif(not DATA.is_dir(), reason="NSD data volume not mounted")
def test_ppdata_pilot_session_is_cortical_only():
    from experiments.nsd_replication.inputs import load_ppdata, ppdata_brain

    session = load_ppdata(_pilot_config(), "sub-07", "ses-nsd10")
    assert len(session.labels) == 12
    assert session.signals[0].shape == (226, 59412)
    assert set(session.brain.name) == set(CORTEX)
    assert "73k_id" in session.events[0]
    assert list(session.confounds[0].columns) == [
        "poly_0",
        "poly_1",
        "poly_2",
        "poly_3",
    ]
    np.testing.assert_allclose(np.diff(session.frame_times[0]), 4 / 3, rtol=1e-5)
    assert same_grayordinates(
        ppdata_brain(_pilot_config(), "sub-07", "ses-nsd10"), session.brain
    )
    assert session.task_model is not None
