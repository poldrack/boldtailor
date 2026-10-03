"""Behavioral tests using small, real CIFTI files and independent OLS fits."""

import importlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor
from boldtailor._hrf_design import hrf_model


def example():
    try:
        return importlib.import_module("examples.NSD.nsd_cifti")
    except ModuleNotFoundError:
        pytest.fail("The NSD CIFTI example has not been implemented")


@pytest.fixture
def confounds():
    rng = np.random.default_rng(54)
    table = pd.DataFrame()
    for axis in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"):
        motion = rng.normal(size=96)
        derivative = np.r_[np.nan, np.diff(motion)]
        for suffix, values in {
            "": motion,
            "_derivative1": derivative,
            "_power2": motion**2,
            "_derivative1_power2": derivative**2,
        }.items():
            table[axis + suffix] = values
    metadata = {}
    for i in range(8):
        name = f"a_comp_cor_{i:02d}"
        table[name] = rng.normal(size=96)
        metadata[name] = {
            "Mask": "combined",
            "Retained": True,
            "VarianceExplained": (8 - i) / 100,
            "Method": "aCompCor",
        }
    # A separate mask must not displace the combined-mask components.
    table["a_comp_cor_08"] = rng.normal(size=96)
    metadata["a_comp_cor_08"] = {
        "Mask": "CSF",
        "Retained": True,
        "VarianceExplained": 0.8,
    }
    table["cosine00"] = np.cos(np.pi * (np.arange(96) + 0.5) / 96)
    table["non_steady_state_outlier00"] = np.r_[1.0, np.zeros(95)]
    return table, metadata


@pytest.fixture
def events():
    return pd.DataFrame(
        {
            "onset": [8.0, 22.0, 38.0, 60.0, 90.0, 112.0],
            "duration": [3.0] * 6,
            "trial_type": [0, 1, 0, 1, 1, 0],
            "response_time": [0.5, 1.0, 2.5, 1.5, 3.0, 0.5],
        }
    )


def test_confounds_use_24_motion_top_six_combined_components_and_cosines(confounds):
    table, metadata = confounds
    result = example().select_confounds(table, metadata)
    expected = list(table.columns[:30]) + ["cosine00", "non_steady_state_outlier00"]
    assert list(result.columns) == expected
    np.testing.assert_allclose(result, table[expected].fillna(0))
    assert not table.iloc[0].notna().all()  # input was not mutated


@pytest.mark.parametrize("column,row", [("trans_x", 0), ("rot_z_derivative1", 2)])
def test_only_initial_motion_derivative_nans_are_filled(confounds, column, row):
    table, metadata = confounds
    table.loc[row, column] = np.nan
    with pytest.raises(ValueError, match="finite|missing"):
        example().select_confounds(table, metadata)


def test_fewer_than_six_retained_combined_components_is_an_error(confounds):
    table, metadata = confounds
    for i in (5, 6, 7):
        metadata[f"a_comp_cor_{i:02d}"]["Retained"] = False
    with pytest.raises(ValueError, match="six|6"):
        example().select_confounds(table, metadata)


def test_rt_modulation_is_centered_and_preserves_stimulus_timing(events):
    times = 0.775 + np.arange(96) * 1.6
    design = example().task_regressors(events, times)
    expected_stim, _ = compute_regressor(
        np.vstack([events.onset, events.duration, np.ones(6)]),
        hrf_model("spm"),
        times,
    )
    expected_rt, _ = compute_regressor(
        np.vstack([events.onset, events.duration, [-1, -0.5, 1, 0, 1.5, -1]]),
        hrf_model("spm"),
        times,
    )
    assert list(design.columns) == ["stimulus", "response_time"]
    np.testing.assert_allclose(design.stimulus, expected_stim[:, 0])
    np.testing.assert_allclose(design.response_time, expected_rt[:, 0])


def test_missing_response_time_is_rejected_explicitly(events):
    events.loc[1, "response_time"] = np.nan
    with pytest.raises(ValueError, match="response_time"):
        example().task_regressors(events, np.arange(96) * 1.6)


@pytest.fixture
def dataset(tmp_path, confounds, events):
    root = tmp_path / "bids"
    prep = root / "derivatives" / "fmriprep"
    raw_func = root / "sub-07" / "ses-nsd10" / "func"
    prep_func = prep / "sub-07" / "ses-nsd10" / "func"
    raw_func.mkdir(parents=True)
    prep_func.mkdir(parents=True)
    brain = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 2, 3, 6]),
        8,
        name="CortexLeft",
    )
    table, metadata = confounds
    times = 0.775 + np.arange(96) * 1.6
    # The oracle design uses the peak-one kernel; simulated responses keep
    # their original Nilearn-scaled amplitudes.
    task, simulated_task = (
        np.column_stack(
            [
                compute_regressor(
                    np.vstack([events.onset, events.duration, amp]), hrf, times
                )[0]
                for amp in [np.ones(6), np.array([-1, -0.5, 1, 0, 1.5, -1])]
            ]
        )
        for hrf in (hrf_model("spm"), "spm")
    )
    nuisance = np.column_stack(
        [
            table.iloc[:, :30].fillna(0),
            table.cosine00,
            table.non_steady_state_outlier00,
            np.ones(96),
        ]
    )
    full = np.column_stack([task, nuisance])
    simulated = np.column_stack([simulated_task, nuisance])
    rng = np.random.default_rng(23)
    signal_runs = []
    # Unequal variances and run means expose erroneous arithmetic/global pooling.
    for run, scale in [(1, 1), (2, 5)]:
        stem = f"sub-07_ses-nsd10_task-nsdcore_run-{run:02d}"
        events.to_csv(raw_func / f"{stem}_events.tsv", sep="\t", index=False)
        table.to_csv(
            prep_func / f"{stem}_desc-confounds_timeseries.tsv", sep="\t", index=False
        )
        (prep_func / f"{stem}_desc-confounds_timeseries.json").write_text(
            json.dumps(metadata)
        )
        y = scale * (simulated @ rng.normal(size=(35, 4)) + rng.normal(size=(96, 4)))
        y += run * 100
        y[:, -1] = 0  # Undefined R² must retain its spatial position as NaN.
        signal_runs.append(y)
        axes = (nib.cifti2.SeriesAxis(0, 1.6, 96), brain)
        image = nib.Cifti2Image(y, header=nib.Cifti2Header.from_axes(axes))
        nib.save(image, prep_func / f"{stem}_space-fsLR_den-91k_bold.dtseries.nii")
        (prep_func / f"{stem}_space-fsLR_den-91k_bold.json").write_text(
            json.dumps(
                {
                    "StartTime": 0.775,
                    "RepetitionTime": 1.6,
                    "SliceTimingCorrected": True,
                }
            )
        )
    return root, prep, signal_runs, full, nuisance, brain


def test_complete_example_publishes_correct_pooled_maps_and_designs(dataset, tmp_path):
    root, prep, signals, full, nuisance, brain = dataset
    output = tmp_path / "output"
    paths = example().run_analysis(root, prep, output, block_size=2)
    assert all(path.is_file() for path in paths)
    denom = sum(np.sum((y - y.mean(axis=0)) ** 2, axis=0) for y in signals)
    expected = {}
    for label, matrix in [("full", full), ("confounds", nuisance)]:
        sse = sum(
            np.sum(
                (y - matrix @ np.linalg.lstsq(matrix, y, rcond=None)[0]) ** 2, axis=0
            )
            for y in signals
        )
        expected[label] = 1 - sse[:3] / denom[:3]
    expected["task"] = expected["full"] - expected["confounds"]
    for label, values in expected.items():
        image = nib.load(next(output.rglob(f"*desc-{label}_*.dscalar.nii")))
        assert image.header.get_axis(1) == brain
        assert image.shape == (1, 4)
        np.testing.assert_allclose(image.get_fdata()[0, :3], values, atol=2e-7)
        assert np.isnan(image.get_fdata()[0, -1])
    designs = sorted(output.rglob("*design.tsv"))
    assert len(designs) == 2
    saved = pd.read_csv(designs[0], sep="\t")
    np.testing.assert_allclose(saved.frame_time, 0.775 + np.arange(96) * 1.6)
    np.testing.assert_allclose(saved.drop(columns="frame_time"), full, atol=1e-12)
    description = json.loads((output / "dataset_description.json").read_text())
    assert description["DatasetType"] == "derivative"
    assert list(output.rglob("*provenance.json"))


def test_discovery_requires_every_run_to_have_cifti(dataset):
    root, prep, *_ = dataset
    next(prep.rglob("*run-02*.dtseries.nii")).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="run-02|CIFTI"):
        example().discover_runs(root, prep)


def test_inconsistent_grayordinate_order_is_rejected(dataset, tmp_path):
    root, prep, *_ = dataset
    path = next(prep.rglob("*run-02*.dtseries.nii"))
    image = nib.load(path)
    axes = (image.header.get_axis(0), image.header.get_axis(1)[::-1])
    changed = nib.Cifti2Image(image.get_fdata(), nib.Cifti2Header.from_axes(axes))
    nib.save(changed, path)
    with pytest.raises(ValueError, match="grayordinate|BrainModel"):
        example().run_analysis(root, prep, tmp_path / "output")


def test_confounds_length_must_match_cifti(dataset, tmp_path):
    root, prep, *_ = dataset
    path = next(prep.rglob("*run-01*confounds_timeseries.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.iloc[:-1].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="rows|volumes|length"):
        example().run_analysis(root, prep, tmp_path / "output")


def test_relative_fmriprep_override_works(dataset, tmp_path, monkeypatch):
    root, prep, *_ = dataset
    monkeypatch.chdir(tmp_path)
    paths = example().run_analysis(
        root, prep.relative_to(tmp_path), tmp_path / "output"
    )
    assert len([path for path in paths if path.name.endswith(".dscalar.nii")]) == 3


def test_external_fmriprep_root_has_explicit_provenance_error(dataset, tmp_path):
    root, *_ = dataset
    with pytest.raises(ValueError, match="inside bids_root"):
        example().run_analysis(root, tmp_path / "external", tmp_path / "output")


def test_conventional_model_metadata_records_peak_normalization():
    assert example().MODEL["hrf"] == "spm"
    assert example().MODEL["hrf_normalization"] == "peak_one"
