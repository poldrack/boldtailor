import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.betas import (
    fit_dir,
    inputs_digest,
    is_complete,
    read_extra,
    read_fit,
    write_fit,
    zscore,
)
from experiments.nsd_replication.inputs import Session


def _trials():
    return pd.DataFrame(
        {
            "session": "ses-a",
            "run": "run-01",
            "trial": [0, 1],
            "onset": [4.0, 8.0],
            "image": [5, 6],
        }
    )


def test_round_trip(tmp_path):
    path = fit_dir(tmp_path, "ppdata", "sub-07", "ses-nsd10", "b1")
    betas = np.array([[1.0, 2.0], [3.0, 4.0]])
    write_fit(
        path,
        betas,
        _trials(),
        {"level": "b1"},
        extras={"hrf_indices": np.array([3, 4])},
    )
    got, trials, meta = read_fit(path)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, betas)
    pd.testing.assert_frame_equal(trials, _trials())
    assert meta == {"level": "b1"}
    assert read_extra(path, "hrf_indices").tolist() == [3, 4]


def test_incomplete_fit_is_not_complete(tmp_path):
    path = tmp_path / "fit"
    path.mkdir()
    np.save(path / "betas.npy", np.zeros((1, 1)))
    assert not is_complete(path)


def test_row_count_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="rows"):
        write_fit(tmp_path / "f", np.zeros((3, 1)), _trials(), {})


def test_zscore_columns():
    values = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    z = zscore(values)
    np.testing.assert_allclose(z[:, 0], [-1.224744871, 0, 1.224744871])


def test_zscore_constant_feature_is_nan_without_warning():
    values = np.array([[1.0, 5.0, np.nan], [3.0, 5.0, 1.0]])
    with np.errstate(all="raise"):
        z = zscore(values)
    assert np.isnan(z[:, 1]).all() and np.isnan(z[:, 2]).all()


def test_zscore_nonrepresentable_constant_is_nan():
    # Test both 0.1 (repeated, non-representable) and 0.3 (as float32 then upcast)
    values = np.hstack(
        [np.full((3, 2), 0.1), np.full((3, 1), np.float32(0.3)).astype(float)]
    )
    with np.errstate(all="raise"):
        z = zscore(values)
    assert np.isnan(z).all()


def _minimal_session(source="ppdata", confound_val=0.0):
    """Create a minimal Session for testing inputs_digest."""
    left = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0]), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    right = nib.cifti2.BrainModelAxis.from_surface(
        np.array([1]), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    brain = left + right

    return Session(
        subject="sub-07",
        session="ses-nsd10",
        labels=("run-01",),
        signals=(np.zeros((5, 2), dtype=np.float32),),
        events=(pd.DataFrame({"onset": [1.0, 2.0]}),),
        confounds=(pd.DataFrame({"c": [confound_val, confound_val]}),),
        frame_times=(np.array([0.0, 1.6, 3.2, 4.8, 6.4]),),
        brain=brain,
        task_model=None,
        source=source,
    )


def test_inputs_digest_same_sessions_equal():
    s1 = _minimal_session()
    s2 = _minimal_session()
    assert inputs_digest(s1) == inputs_digest(s2)


def test_inputs_digest_different_confound_differs():
    s1 = _minimal_session(confound_val=0.0)
    s2 = _minimal_session(confound_val=1.0)
    assert inputs_digest(s1) != inputs_digest(s2)
