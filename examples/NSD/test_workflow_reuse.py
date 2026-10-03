"""Saved notebook results can be replotted without fitting or changing data."""

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import nbformat
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary
from boldtailor.workflow.inputs import NSD_TASK_MODEL, load_session
from boldtailor.workflow import outputs as workflow_outputs


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


def _metadata(files):
    path = next(p for p in files if p.name.endswith("desc-notebook_metadata.json"))
    return json.loads(path.read_text())


def _assert_tuned_betas(files, suffix):
    betas = [p for p in files if f"Trial{suffix}_betas.dscalar.nii" in p.name]
    assert len(betas) == 12, "Both HRF modes need six tuned beta images"
    for image in map(nib.load, betas):
        assert image.shape == (6, 4)
        assert np.isnan(image.get_fdata()[:, -1]).all()


def _assert_off_exports(files):
    assert not any("TrialRidgeCV_betas" in p.name for p in files)
    assert not any(
        "TrialRidge_betas" in p.name for p in files
    ), "Disabled mode must not use the old default alpha"


def _assert_two_point_grid_boundary(metadata):
    # every winner of a two-value grid is an endpoint
    rows = metadata["ridge_cv"]["at_boundary_fraction"]
    assert {(r["mode"], r["scope"]) for r in rows} == {
        (mode, scope)
        for mode in ("Canonical", "Optimized")
        for scope in ("all", "odd", "even")
    }
    assert all(r["fraction"] == 1.0 for r in rows)
    scopes = {row["scope"] for row in metadata["hrf_boundary_summary"]}
    assert scopes == {"all", "odd", "even"}
    assert len(metadata["hrf_boundary_summary"]) == 36


def _assert_cv_exports(files):
    _assert_tuned_betas(files, "RidgeCV")
    scores = [
        p for p in files if p.name.endswith("_scores.tsv") and "RidgeCV" in p.name
    ]
    assert len(scores) == 6
    metadata = _metadata(files)
    assert metadata["ridge_cv"]["validation_target"] == "candidate_regularized_betas"
    assert metadata["ridge_cv"]["percentile"] == 90.0
    assert metadata["noise_model"] == "ols"
    assert metadata["ridge_cv"]["encoding_mode"] == "within_run"
    _assert_two_point_grid_boundary(metadata)


def _assert_tuning_links(files):
    for mode in ("Canonical", "Optimized"):
        decision = json.loads(
            next(
                p for p in files if f"{mode}FractionalCVAll_provenance.json" in p.name
            ).read_text()
        )
        final = json.loads(
            next(
                p for p in files if f"{mode}TrialFractionalCV_provenance.json" in p.name
            ).read_text()
        )
        for block in final:
            assert (
                block["record"]["activities"][-1]["tuning_analysis_fingerprint"]
                == decision["analysis_fingerprint"]
            )


def _assert_fractional_exports(files):
    _assert_tuned_betas(files, "FractionalCV")
    assert not any("TrialRidgeCV_betas" in p.name for p in files)
    activation = [p for p in files if "_stat-activation.dscalar.nii" in p.name]
    assert len(activation) == 4
    assert sum("TrialFractionalCV" in p.name for p in activation) == 2
    fractions = [
        p
        for p in files
        if "FractionalCV" in p.name and "_stat-ridgefraction." in p.name
    ]
    assert (
        len(fractions) == 12
    )  # Three tuning scopes, two outer splits, final per mode.
    metadata = _metadata(files)
    assert metadata["ridge_cv"]["validation_target"] == "fixed_ols_betas"
    assert metadata["ridge_cv"]["objective"] == "maximum_encoding_r2_per_grayordinate"
    assert metadata["ridge_cv"]["percentile_role"] == "descriptive_only"
    _assert_two_point_grid_boundary(metadata)
    _assert_tuning_links(files)


MODE_EXPORTS = dict(
    off=_assert_off_exports,
    cv=_assert_cv_exports,
    fractional_cv=_assert_fractional_exports,
)


@pytest.mark.parametrize("mode", ["off", "cv", "fractional_cv"])
def test_notebook_reuses_saved_results_without_fitting(
    six_run_dataset, tmp_path, monkeypatch, mode
):
    from boldtailor.workflow import analysis as workflow_analysis
    from boldtailor.workflow import beta_series as ridge_workflow
    from boldtailor.workflow.outputs import tuning_table

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
    MODE_EXPORTS[mode](list((tmp_path / "output").rglob("*")))
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
        "hrf_normalization": "peak_one_event_response",
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
    validate_saved_settings(
        {**current, "hrf_normalization": "peak_one_event_response"}, settings
    )
