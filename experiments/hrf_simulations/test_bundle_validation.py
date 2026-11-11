import h5py
import numpy as np

from experiments.hrf_estimation.test_inputs import bids as bids  # noqa: F401


def make_bundle(bids, path):
    from experiments.hrf_simulations.bundle import export_bundle
    export_bundle(bids, path, np.array([0, 1, 2]), rois={
        'fusiform': np.array([0]), 'lateraloccipital': np.array([1]),
        'inferiortemporal': np.array([2]),
    })


def test_validation_reports_all_runs_and_scan_scope(bids, tmp_path):
    from experiments.hrf_simulations.bundle_validation import validate_bundle
    path = tmp_path/'bundle.h5'
    make_bundle(bids, path)
    result = validate_bundle(path, full_signal_scan=True)
    assert len(result['runs']) == 2
    assert result['runs'].ok.all()
    assert result['runs'].checked_vertices.tolist() == [3, 3]
    assert result['global_checks'].ok.all()
    assert result['skipped'].empty


def test_validation_detects_nonfinite_signals_and_shape_mismatch(bids, tmp_path):
    from experiments.hrf_simulations.bundle_validation import validate_bundle
    path = tmp_path/'bad.h5'
    make_bundle(bids, path)
    with h5py.File(path, 'r+') as handle:
        groups = list(handle['subjects/sub-07/ses-nsd11'].values())
        groups[0]['signals'][0, 0] = np.nan
        del groups[1]['frame_times']
        groups[1].create_dataset('frame_times', data=np.arange(7))
    result = validate_bundle(path, full_signal_scan=True)
    assert not result['runs'].ok.any()
    assert result['runs'].error.str.contains('finite').any()
    assert result['runs'].error.str.contains('shape').any()


def test_validation_detects_invalid_roi_membership(bids, tmp_path):
    from experiments.hrf_simulations.bundle_validation import validate_bundle
    path = tmp_path/'bad_roi.h5'
    make_bundle(bids, path)
    with h5py.File(path, 'r+') as handle:
        handle['rois/fusiform'][0] = 99
    result = validate_bundle(path)
    assert not result['global_checks'].ok.all()
