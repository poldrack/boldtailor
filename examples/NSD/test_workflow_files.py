"""NSD discovery, CIFTI loading, and nuisance selection use small real files."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from examples.NSD import workflow_files


def load(root, prep):
    return workflow_files.load_runs(workflow_files.discover_runs(root, prep))


def test_confounds_use_24_motion_top_six_combined_components_and_cosines(confounds):
    table, metadata = confounds
    result = workflow_files.select_confounds(table, metadata)
    expected = list(table.columns[:30]) + ["cosine00", "non_steady_state_outlier00"]
    assert list(result.columns) == expected
    np.testing.assert_allclose(result, table[expected].fillna(0))
    assert not table.iloc[0].notna().all()  # input was not mutated


@pytest.mark.parametrize("column,row", [("trans_x", 0), ("rot_z_derivative1", 2)])
def test_only_initial_motion_derivative_nans_are_filled(confounds, column, row):
    table, metadata = confounds
    table.loc[row, column] = np.nan
    with pytest.raises(ValueError, match="finite|missing"):
        workflow_files.select_confounds(table, metadata)


def test_fewer_than_six_retained_combined_components_is_an_error(confounds):
    table, metadata = confounds
    for i in (5, 6, 7):
        metadata[f"a_comp_cor_{i:02d}"]["Retained"] = False
    with pytest.raises(ValueError, match="six|6"):
        workflow_files.select_confounds(table, metadata)


def test_discovery_requires_every_run_to_have_cifti(dataset):
    root, prep, *_ = dataset
    next(prep.rglob("*run-02*.dtseries.nii")).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="run-02|CIFTI"):
        workflow_files.discover_runs(root, prep)


def test_inconsistent_grayordinate_order_is_rejected(dataset):
    root, prep, *_ = dataset
    path = next(prep.rglob("*run-02*.dtseries.nii"))
    image = nib.load(path)
    axes = (image.header.get_axis(0), image.header.get_axis(1)[::-1])
    changed = nib.Cifti2Image(image.get_fdata(), nib.Cifti2Header.from_axes(axes))
    nib.save(changed, path)
    with pytest.raises(ValueError, match="grayordinate|BrainModel"):
        load(root, prep)


def test_confounds_length_must_match_cifti(dataset):
    root, prep, *_ = dataset
    path = next(prep.rglob("*run-01*confounds_timeseries.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.iloc[:-1].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="rows|volumes|length"):
        load(root, prep)


def test_frame_times_follow_the_sidecar_start_time(dataset):
    root, prep, *_ = dataset
    runs, brain = load(root, prep)
    assert [r.label for r in runs] == ["run-01", "run-02"]
    assert [r.number for r in runs] == [1, 2]
    np.testing.assert_allclose(runs[0].frame_times, 0.775 + np.arange(96) * 1.6)
    assert brain == runs[1].image.header.get_axis(1)


def test_slice_timing_correction_requires_a_start_time(dataset):
    root, prep, *_ = dataset
    sidecar = next(prep.rglob("*run-01*_bold.json"))
    sidecar.write_text('{"RepetitionTime": 1.6, "SliceTimingCorrected": true}')
    with pytest.raises(ValueError, match="StartTime"):
        load(root, prep)


def test_bids_labels_need_the_entity_prefix_and_alphanumerics():
    assert workflow_files.bids_label("sub-07", "sub")
    assert not workflow_files.bids_label("ses-07", "sub")
    assert not workflow_files.bids_label("sub-07_x", "sub")
