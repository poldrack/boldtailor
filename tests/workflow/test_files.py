"""Workflow discovery, CIFTI loading, and nuisance selection use small real files."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.workflow import files as workflow_files


def load(root, settings_for):
    return workflow_files.load_runs(workflow_files.discover_runs(settings_for(root)))


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


def test_discovery_requires_every_run_to_have_cifti(dataset, settings_for):
    root, prep, *_ = dataset
    next(prep.rglob("*run-02*.dtseries.nii")).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="run-02|CIFTI"):
        workflow_files.discover_runs(settings_for(root))


def test_inconsistent_grayordinate_order_is_rejected(dataset, settings_for):
    root, prep, *_ = dataset
    path = next(prep.rglob("*run-02*.dtseries.nii"))
    image = nib.load(path)
    axes = (image.header.get_axis(0), image.header.get_axis(1)[::-1])
    changed = nib.Cifti2Image(image.get_fdata(), nib.Cifti2Header.from_axes(axes))
    nib.save(changed, path)
    with pytest.raises(ValueError, match="grayordinate|BrainModel"):
        load(root, settings_for)


def test_confounds_length_must_match_cifti(dataset, settings_for):
    root, prep, *_ = dataset
    path = next(prep.rglob("*run-01*confounds_timeseries.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.iloc[:-1].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="rows|volumes|length"):
        load(root, settings_for)


def test_frame_times_follow_the_sidecar_start_time(dataset, settings_for):
    root, prep, *_ = dataset
    runs, brain = load(root, settings_for)
    assert [r.label for r in runs] == ["run-01", "run-02"]
    assert [r.number for r in runs] == [1, 2]
    np.testing.assert_allclose(runs[0].frame_times, 0.775 + np.arange(96) * 1.6)
    assert brain == runs[1].image.header.get_axis(1)


def test_slice_timing_correction_requires_a_start_time(dataset, settings_for):
    root, prep, *_ = dataset
    sidecar = next(prep.rglob("*run-01*_bold.json"))
    sidecar.write_text('{"RepetitionTime": 1.6, "SliceTimingCorrected": true}')
    with pytest.raises(ValueError, match="StartTime"):
        load(root, settings_for)


def test_bids_labels_need_the_entity_prefix_and_alphanumerics():
    assert workflow_files.bids_label("sub-07", "sub")
    assert not workflow_files.bids_label("ses-07", "sub")
    assert not workflow_files.bids_label("sub-07_x", "sub")


def test_discovery_uses_the_task_and_space_from_settings(dataset, settings_for):
    root, prep, *_ = dataset
    runs = workflow_files.discover_runs(settings_for(root))
    assert [r.stem for r in runs] == [
        "sub-07_ses-nsd10_task-nsdcore_run-01",
        "sub-07_ses-nsd10_task-nsdcore_run-02",
    ]
    assert runs[0].bold.name.endswith("_space-fsLR_den-91k_bold.dtseries.nii")
    with pytest.raises(FileNotFoundError, match="task-other"):
        workflow_files.discover_runs(settings_for(root, task="other"))


def _run(**columns):
    return SimpleNamespace(events=pd.DataFrame(columns, index=range(3)))


def test_reaction_times_keep_clean_columns_and_require_them_by_default():
    rt = workflow_files.reaction_times([_run(response_time=[0.5, np.nan, 1.0])])
    np.testing.assert_allclose(rt[0], [0.5, np.nan, 1.0])
    with pytest.raises(AttributeError):
        workflow_files.reaction_times([_run(onset=[0, 1, 2])])


def test_missing_ok_reaction_times_tolerate_absent_or_text_values():
    runs = [_run(onset=[0, 1, 2]), _run(response_time=["0.5", "n/a", 2])]
    rt = workflow_files.reaction_times(runs, missing_ok=True)
    np.testing.assert_array_equal(np.isnan(rt[0]), [True, True, True])
    np.testing.assert_allclose(rt[1], [0.5, np.nan, 2.0])
    assert all(values.dtype == float for values in rt)
