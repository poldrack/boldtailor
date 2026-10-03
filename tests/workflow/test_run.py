"""run_workflow executes the enabled stages, skips what it cannot do, and writes the derivative."""

import json
import pytest

from boldtailor.workflow import run as workflow_run


def _func(settings):
    return settings.output_dir / "sub-07" / "ses-nsd10" / "func"


def test_full_run_writes_every_stage_and_the_report(
    six_run_dataset, settings_for, tmp_path
):
    root, _ = six_run_dataset
    settings = settings_for(
        root,
        output_dir=tmp_path / "out",
        hrf_library="sobol",
        hrf_n_samples=2,
        ridge_mode="fractional_cv",
        ridge_fractions=(0.4, 1.0),
        block_size=4,
    )
    result = workflow_run.run_workflow(settings)
    names = {p.name for p in result.paths}
    assert "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json" in names
    assert (
        "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-OptimizedGLM_stat-rsquared.dscalar.nii"
        in names
    )
    assert (
        "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-HRFOdd_stat-selection.dscalar.nii"
        in names
    )
    assert (
        "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-OptimizedTrialFractionalCV_stat-ridgefraction.dscalar.nii"
        in names
    )
    assert (
        "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-CanonicalTrialOLS_stat-activation.dscalar.nii"
        in names
    )
    assert (
        result.report_path
        == settings.output_dir / "sub-07_ses-nsd10_task-nsdcore_report.html"
    )
    assert result.report_path.exists() and result.skipped == ()
    metadata = json.loads(
        (
            _func(settings)
            / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json"
        ).read_text()
    )
    assert (
        metadata["settings"]["ridge_mode"] == "fractional_cv"
        and metadata["report"] == result.report_path.name
    )
    assert (settings.output_dir / "dataset_description.json").exists()
    assert not any("notebook" in n for n in names)


def test_disabled_stages_write_nothing_of_their_own(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    settings = settings_for(
        root,
        output_dir=tmp_path / "out",
        hrf_library="canonical",
        stages=frozenset({"glms"}),
    )
    result = workflow_run.run_workflow(settings)
    names = {p.name for p in result.paths}
    assert any("desc-OptimizedGLM" in n for n in names)
    assert not any("Trial" in n or "HRFOdd" in n or "activation" in n for n in names)


def test_reliability_is_skipped_with_a_reason_when_a_parity_is_short(
    dataset, settings_for, tmp_path
):
    root, *_ = dataset  # two runs: one odd, one even
    settings = settings_for(
        root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off"
    )
    result = workflow_run.run_workflow(settings)
    assert result.skipped == (("reliability", "fewer than two odd or two even runs"),)
    assert "fewer than two odd" in result.report_path.read_text()


def test_second_run_into_the_same_output_stops_before_loading(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    settings = settings_for(
        root,
        output_dir=tmp_path / "out",
        hrf_library="canonical",
        stages=frozenset({"glms"}),
    )
    workflow_run.run_workflow(settings)
    with pytest.raises(FileExistsError):
        workflow_run.run_workflow(settings)
    again = workflow_run.run_workflow(
        settings_for(
            root,
            output_dir=tmp_path / "out",
            hrf_library="canonical",
            stages=frozenset({"glms"}),
            existing_results="overwrite",
        )
    )
    assert again.paths


def test_describe_inputs_reports_runs_model_and_output(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    info = workflow_run.describe_inputs(
        settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical")
    )
    assert info["runs"] == ["run-01", "run-02", "run-03", "run-04"]
    assert info["task_model"]["regressors"] == ["task", "response_time", "trial_type"]
    assert (
        info["output_dir"] == str(tmp_path / "out") and info["library_candidates"] == 1
    )
