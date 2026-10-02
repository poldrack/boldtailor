"""Saved notebook results can be replotted without fitting or changing data."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary
from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401
from examples.NSD.test_nsd_workflow import four_runs  # noqa: F401
from examples.NSD.workflow_inputs import NSD_TASK_MODEL, load_session
from examples.NSD.test_ridge_workflow import six_run_dataset  # noqa: F401
from examples.NSD import workflow_outputs


def execute_notebook(config):
    notebook = nbformat.read(Path(__file__).with_name("nsd_workflow.ipynb"), 4)
    context = {"NSD_CONFIG": config, "display": lambda value: None}
    plt.switch_backend("Agg")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="FigureCanvasAgg is non-interactive"
            )
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    exec(cell.source, context)
    finally:
        plt.close("all")
    return context


@pytest.mark.parametrize("mode", ["off", "cv", "fractional_cv"])
def test_notebook_reuses_saved_results_without_fitting(
    six_run_dataset, tmp_path, monkeypatch, mode  # noqa: F811
):
    from examples.NSD import workflow_analysis, ridge_workflow
    from examples.NSD.ridge_outputs import tuning_table

    root, prep = six_run_dataset
    config = dict(
        bids_root=str(root),
        fmriprep_root=str(prep),
        output_root=str(tmp_path / "output"),
        n_jobs=1,
        block_size=4,
        hrf_n_samples=1,
        ridge_mode=mode,
        ridge_fractions=[0.4, 1.0],
        ridge_alphas=[0.0, 0.1],
        surface_maps=False,
        existing_results="reuse",
    )
    original = execute_notebook(config)
    if mode == "off":
        marker = tmp_path / "output/unrelated.txt"
        marker.write_text("keep")
        previous = original["paths"][0].stat().st_mtime_ns
        original = execute_notebook({**config, "existing_results": "overwrite"})
        assert original["paths"][0].stat().st_mtime_ns != previous
        assert marker.read_text() == "keep"
    data_paths = [p for p in original["paths"] if p.suffix != ".png"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in data_paths}

    def forbidden(*args, **kwargs):
        pytest.fail("Reusing results must not fit models")

    for name in ("fit_glms", "select_hrfs", "fit_beta_series"):
        monkeypatch.setattr(workflow_analysis, name, forbidden)
    monkeypatch.setattr(ridge_workflow, "fit_cv_beta_series", forbidden)
    restored = execute_notebook(config)
    assert restored["reused_results"] is not None
    for name, result in original["beta_models"].items():
        actual = restored["beta_models"][name]
        np.testing.assert_allclose(actual["r2"], result["r2"], atol=1e-6)
        for saved, expected in zip(actual["betas"], result["betas"], strict=True):
            np.testing.assert_allclose(saved, expected, atol=1e-6)
    for name in original["hrf_maps"]:
        np.testing.assert_allclose(
            restored["hrf_maps"][name], original["hrf_maps"][name], atol=1e-6
        )
    if mode != "off":
        pd.testing.assert_frame_equal(
            tuning_table(restored["ridge_cv"]),
            tuning_table(original["ridge_cv"]),
            atol=1e-6,
            rtol=1e-5,
        )
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == v for p, v in before.items())


def test_output_policy_allows_overwrite_and_detects_reuse(tmp_path):
    directory = tmp_path / "sub-07/ses-nsd10/func"
    directory.mkdir(parents=True)
    (
        directory / "sub-07_ses-nsd10_task-nsdcore_desc-notebook_metadata.json"
    ).write_text("{}")
    with pytest.raises(FileExistsError):
        workflow_outputs.check_output(tmp_path)
    assert workflow_outputs.check_output(tmp_path, existing_results="reuse") is True
    assert (
        workflow_outputs.check_output(tmp_path, existing_results="overwrite") is False
    )
    with pytest.raises(ValueError, match="existing_results"):
        workflow_outputs.check_output(tmp_path, existing_results="typo")


def test_reuse_rejects_partial_outputs(tmp_path):
    directory = tmp_path / "sub-07/ses-nsd10/func"
    directory.mkdir(parents=True)
    (directory / "sub-07_ses-nsd10_task-nsdcore_desc-notebook_partial.tsv").write_text(
        "x\n1\n"
    )
    with pytest.raises(ValueError, match="incomplete"):
        workflow_outputs.check_output(tmp_path, existing_results="reuse")


def test_reuse_rejects_changed_analysis_settings(tmp_path):
    from examples.NSD.workflow_reuse import validate_saved_settings

    metadata = {
        "settings": {"ridge_mode": "fractional_cv", "hrf_seed": 0},
        "task_model_fingerprint": NSD_TASK_MODEL.fingerprint,
        "hrf_normalization": "peak_one",
    }
    with pytest.raises(ValueError, match="hrf_seed"):
        validate_saved_settings(
            metadata, {"ridge_mode": "fractional_cv", "hrf_seed": 1}
        )
    validate_saved_settings(metadata, {**metadata["settings"], "n_jobs": 8})


def test_saved_results_with_other_task_model_are_rejected():
    from examples.NSD.workflow_reuse import validate_saved_settings

    settings = {"ridge_mode": "off"}
    stale = {"settings": settings, "task_model_fingerprint": "0" * 64}
    with pytest.raises(ValueError, match="task_model_fingerprint"):
        validate_saved_settings(stale, settings)
    with pytest.raises(ValueError, match="task_model_fingerprint"):
        validate_saved_settings({"settings": settings}, settings)


def test_metadata_records_the_task_model_fingerprint(four_runs):  # noqa: F811
    library = HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])
    runs = load_session(*four_runs)
    published = workflow_outputs._metadata(runs, library, {})
    assert published["task_model_fingerprint"] == NSD_TASK_MODEL.fingerprint


def test_saved_results_with_other_hrf_normalization_are_rejected():
    from examples.NSD.workflow_reuse import validate_saved_settings

    settings = {"ridge_mode": "off"}
    current = {
        "settings": settings,
        "task_model_fingerprint": NSD_TASK_MODEL.fingerprint,
    }
    stale = {**current, "hrf_normalization": "discrete_sum_one"}
    with pytest.raises(ValueError, match="hrf_normalization"):
        validate_saved_settings(stale, settings)
    with pytest.raises(ValueError, match="hrf_normalization"):
        validate_saved_settings(current, settings)
    validate_saved_settings({**current, "hrf_normalization": "peak_one"}, settings)


def test_metadata_records_peak_hrf_normalization(four_runs):  # noqa: F811
    library = HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])
    runs = load_session(*four_runs)
    published = workflow_outputs._metadata(runs, library, {})
    assert published["hrf_normalization"] == "peak_one"
