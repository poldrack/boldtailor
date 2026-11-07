"""Real temporary CIFTIs exercise run matching and vertex identity."""

import importlib

import nibabel as nib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bids(tmp_path):
    brain = nib.cifti2.BrainModelAxis.from_surface([4, 1, 3], 6, 'CortexLeft')
    for run in (2, 1):
        stem = f'sub-07_ses-nsd11_task-nsdcore_run-{run:02}'
        derivative = tmp_path / 'derivatives/nsdpreproc/sub-07/ses-nsd11/func'
        raw = tmp_path / 'sub-07/ses-nsd11/func'
        derivative.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        image = nib.Cifti2Image(np.arange(24).reshape(8, 3).astype(float),
            header=nib.Cifti2Header.from_axes((nib.cifti2.SeriesAxis(0, 4/3, 8), brain)))
        nib.save(image, derivative / (stem + '_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii'))
        pd.DataFrame(np.zeros((8, 6)), columns=[
            'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']).to_csv(
                derivative / (stem + '_desc-confounds_timeseries.tsv'), sep='\t', index=False)
        pd.DataFrame({'onset': [1.0], 'duration': [3.0], 'trial_type': [0],
            'response_time': [0.8], 'is_correct': [1], 'missing_data': [0]}).to_csv(
                raw / (stem + '_events.tsv'), sep='\t', index=False)
    return tmp_path


def api():
    return importlib.import_module('experiments.hrf_estimation.inputs')


def test_discover_and_load_preserve_run_and_vertex_identity(bids):
    runs = api().discover_runs(bids, 'sub-07', (11,))
    assert [r.run_id for r in runs] == [
        'sub-07_ses-nsd11_task-nsdcore_run-01', 'sub-07_ses-nsd11_task-nsdcore_run-02']
    loaded = api().load_run(runs[0], np.array([2, 0]))
    np.testing.assert_array_equal(loaded.signals[:, 0], np.arange(8) * 3 + 2)
    np.testing.assert_array_equal(loaded.feature_ids, [2, 0])
    np.testing.assert_array_equal(loaded.brain_axis.vertex, [3, 4])
    assert loaded.confounds.shape == (8, 25)
    np.testing.assert_allclose(loaded.frame_times, np.arange(8) * 4/3)


def test_missing_matched_confounds_is_error(bids):
    next(bids.rglob('*run-01_desc-confounds_timeseries.tsv')).unlink()
    with pytest.raises(FileNotFoundError, match='confounds'):
        api().discover_runs(bids, 'sub-07', (11,))


def test_missing_requested_session_is_error(bids):
    with pytest.raises(FileNotFoundError, match='nsd12'):
        api().discover_runs(bids, 'sub-07', (11, 12))


def test_motion_row_mismatch_is_error(bids):
    run = api().discover_runs(bids, 'sub-07', (11,))[0]
    table = pd.read_csv(run.confounds_path, sep='\t').iloc[:-1]
    table.to_csv(run.confounds_path, sep='\t', index=False)
    with pytest.raises(ValueError, match='row'):
        api().load_run(run, np.array([0]))


@pytest.mark.parametrize('vertices', [[0, 0], [-1], [3], [0.5], []])
def test_invalid_vertex_selection_is_rejected(bids, vertices):
    run = api().discover_runs(bids, 'sub-07', (11,))[0]
    with pytest.raises(ValueError, match='vert'):
        api().load_run(run, np.array(vertices))
