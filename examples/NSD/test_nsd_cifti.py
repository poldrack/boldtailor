"""Behavioral tests using small, real CIFTI files and independent OLS fits."""

import importlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor
from boldtailor.design import hrf_model
from tests.oracles import scaled_condition


def example():
    try:
        return importlib.import_module("examples.NSD.nsd_cifti")
    except ModuleNotFoundError:
        pytest.fail("The NSD CIFTI example has not been implemented")


def test_rt_modulation_is_centered_and_preserves_stimulus_timing(events):
    times = 0.775 + np.arange(96) * 1.6
    design = example().task_regressors(events, times)
    expected_stim, _ = compute_regressor(
        scaled_condition(events.onset, events.duration, 1.0, hrf_model("spm"), times),
        hrf_model("spm"),
        times,
    )
    expected_rt, _ = compute_regressor(
        scaled_condition(
            events.onset,
            events.duration,
            [-1, -0.5, 1, 0, 1.5, -1],
            hrf_model("spm"),
            times,
        ),
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
    assert example().MODEL["hrf_normalization"] == "peak_one_event_response"


def test_scripts_need_an_explicit_or_environment_bids_root(tmp_path, monkeypatch):
    from examples.NSD.nsd_single_trial import run_single_trial_analysis

    monkeypatch.delenv("NSD_BIDS_ROOT", raising=False)
    assert example().BIDS_ROOT is None
    with pytest.raises(ValueError, match="NSD_BIDS_ROOT"):
        example().run_analysis(output_root=tmp_path / "output")
    with pytest.raises(ValueError, match="NSD_BIDS_ROOT"):
        run_single_trial_analysis(output_root=tmp_path / "output")


def test_scripts_read_the_bids_root_from_the_environment(
    dataset, tmp_path, monkeypatch
):
    root, prep, *_ = dataset
    monkeypatch.setenv("NSD_BIDS_ROOT", str(root))
    monkeypatch.delenv("NSD_OUTPUT_ROOT", raising=False)
    paths = example().run_analysis(fmriprep_root=prep, output_root=tmp_path / "out")
    assert len([p for p in paths if p.name.endswith(".dscalar.nii")]) == 3
