"""run_workflow executes the enabled stages, skips what it cannot do, and writes the derivative."""

import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from boldtailor.workflow import inputs, run as workflow_run


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
    open_before = set(plt.get_fignums())
    result = workflow_run.run_workflow(settings)
    assert set(plt.get_fignums()) <= open_before, "every figure is closed"
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
    split_descriptors = (
        "HRFOdd",
        "HRFEven",
        "HRFOddToEven",
        "HRFEvenToOdd",
        "HRFReliability",
    )
    names = {p.name for p in result.paths}
    assert not any(f"_desc-{d}_" in n for d in split_descriptors for n in names)


@pytest.mark.parametrize("session", ["dataset", "four_runs"])
def test_ridge_cv_with_a_short_parity_stops_before_any_fitting(
    session, request, settings_for, tmp_path, monkeypatch
):
    # Two runs (1 odd, 1 even) or four (2 and 2): optimized ridge CV needs 3 and 3.
    root = request.getfixturevalue(session)[0]

    def refuse(*args, **kwargs):
        raise AssertionError("HRF selection ran before the ridge CV check")

    monkeypatch.setattr(workflow_run.analysis, "select_hrfs", refuse)
    settings = settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical")
    assert settings.ridge_mode == "fractional_cv"
    with pytest.raises(inputs.InputError) as raised:
        workflow_run.run_workflow(settings)
    message = str(raised.value)
    assert (
        "ridge cross-validation needs at least three odd and three even runs" in message
    )
    assert "--ridge-mode off" in message and "--skip-stage betas" in message
    out = tmp_path / "out"
    assert not out.exists() or not any(out.iterdir())


def test_a_failed_stage_leaves_no_report(
    four_runs, settings_for, tmp_path, monkeypatch
):
    root, _ = four_runs

    def fail(*args, **kwargs):
        raise RuntimeError("beta stage failed")

    monkeypatch.setattr(workflow_run.beta_series, "fit_beta_models", fail)
    settings = settings_for(
        root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off"
    )
    with pytest.raises(RuntimeError, match="beta stage failed"):
        workflow_run.run_workflow(settings)
    assert not (
        settings.output_dir / "sub-07_ses-nsd10_task-nsdcore_report.html"
    ).exists()


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


def test_surfaces_are_looked_up_only_when_requested(
    four_runs, settings_for, tmp_path, monkeypatch
):
    root, _ = four_runs

    def refuse(*args, **kwargs):
        raise AssertionError("meshes searched although surface_maps is off")

    monkeypatch.setattr(workflow_run.surfaces, "find_surface_meshes", refuse)
    settings = settings_for(
        root,
        output_dir=tmp_path / "off",
        hrf_library="canonical",
        stages=frozenset({"glms"}),
    )
    assert workflow_run.run_workflow(settings).paths
    monkeypatch.setattr(
        workflow_run.surfaces, "find_surface_meshes", lambda *a, **k: None
    )
    settings = settings_for(
        root,
        output_dir=tmp_path / "missing",
        hrf_library="canonical",
        stages=frozenset({"glms"}),
        surface_maps=True,
    )
    result = workflow_run.run_workflow(settings)
    assert not any("Surface" in p.name for p in result.paths)


def test_task_only_session_runs_without_rt_or_trial_type(
    four_runs, settings_for, tmp_path
):
    import pandas as pd

    root, _ = four_runs
    for path in (root / "sub-07" / "ses-nsd10" / "func").glob("*_events.tsv"):
        frame = pd.read_csv(path, sep="\t")
        frame.drop(columns=["response_time", "trial_type"]).to_csv(
            path, sep="\t", index=False
        )
    settings = settings_for(
        root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off"
    )
    result = workflow_run.run_workflow(settings)
    names = {p.name for p in result.paths}
    assert not any("rtcorrelation" in n or "RTCheck" in n for n in names)
    text = result.report_path.read_text()
    assert "Task regressors: task" in text and "No reaction-time column" in text
    metadata = json.loads(
        (
            _func(settings)
            / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json"
        ).read_text()
    )
    absent = {"rt_check", "response_time", "missing_response_time"}
    assert not absent & set(metadata)
    assert metadata["task_model"]["regressors"] == ["task"]


TRIAL_TYPE_NOTE = "trial_type is not binary 0/1; not used as a modulator"


def test_string_trial_type_is_reported_and_left_out_of_the_model(
    four_runs, settings_for, tmp_path
):
    from tests.workflow.synthetic_bids import face_house_events, rewrite_events

    root, _ = four_runs
    rewrite_events(root, face_house_events)
    settings = settings_for(
        root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off"
    )
    info = workflow_run.describe_inputs(settings)
    assert info["task_model"]["regressors"] == ["task"]
    assert TRIAL_TYPE_NOTE in info["notes"]
    result = workflow_run.run_workflow(settings)
    assert TRIAL_TYPE_NOTE in result.report_path.read_text()
    metadata = json.loads(
        (
            _func(settings)
            / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json"
        ).read_text()
    )
    assert TRIAL_TYPE_NOTE in metadata["notes"]


def test_ridge_cv_without_modulators_stops_before_any_fitting(
    six_run_dataset, settings_for, tmp_path, monkeypatch
):
    from tests.workflow.synthetic_bids import rewrite_events, task_only_events

    root, _ = six_run_dataset
    rewrite_events(root, task_only_events)

    def refuse(*args, **kwargs):
        raise AssertionError("HRF selection ran before the ridge CV check")

    monkeypatch.setattr(workflow_run.analysis, "select_hrfs", refuse)
    settings = settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical")
    assert settings.ridge_mode == "fractional_cv"
    with pytest.raises(inputs.InputError) as raised:
        workflow_run.run_workflow(settings)
    message = str(raised.value)
    assert "modulator" in message
    assert "--ridge-mode off" in message and "--skip-stage betas" in message


def test_loading_errors_become_input_errors(four_runs, settings_for, tmp_path):
    root, prep = four_runs
    path = next(prep.rglob("*run-02*confounds_timeseries.tsv"))
    pd.read_csv(path, sep="\t").drop(columns="cosine00").to_csv(
        path, sep="\t", index=False
    )
    settings = settings_for(
        root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off"
    )
    for call in (workflow_run.describe_inputs, workflow_run.run_workflow):
        with pytest.raises(inputs.InputError, match="cosine"):
            call(settings)


def test_missing_explicit_surface_mesh_stops_before_any_fitting(
    four_runs, settings_for, tmp_path, monkeypatch
):
    root, _ = four_runs

    def refuse(*args, **kwargs):
        raise AssertionError("HRF selection ran before the mesh check")

    monkeypatch.setattr(workflow_run.analysis, "select_hrfs", refuse)
    settings = settings_for(
        root,
        output_dir=tmp_path / "out",
        hrf_library="canonical",
        ridge_mode="off",
        surface_maps=True,
        surface_meshes={"left": tmp_path / "l.gii", "right": tmp_path / "r.gii"},
    )
    with pytest.raises(inputs.InputError, match="l.gii"):
        workflow_run.run_workflow(settings)


@pytest.mark.parametrize("copies", [0, 2])
def test_missing_or_ambiguous_meshes_skip_surface_figures_with_a_note(
    four_runs, settings_for, tmp_path, copies
):
    root, prep = four_runs
    for session in range(copies):
        anat = prep / "sub-07" / f"ses-{session}" / "anat"
        anat.mkdir(parents=True)
        for hemi in "LR":
            name = f"sub-07_ses-{session}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii"
            (anat / name).write_text("")
    settings = settings_for(
        root,
        output_dir=tmp_path / "out",
        hrf_library="canonical",
        ridge_mode="off",
        stages=frozenset({"glms"}),
        surface_maps=True,
    )
    result = workflow_run.run_workflow(settings)
    assert not any("Surface" in p.name for p in result.paths)
    metadata = json.loads(
        (
            _func(settings)
            / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json"
        ).read_text()
    )
    notes = " ".join(metadata["notes"])
    assert "surface" in notes
    assert ("multiple" in notes) == bool(copies)
    assert "surface" in result.report_path.read_text()
