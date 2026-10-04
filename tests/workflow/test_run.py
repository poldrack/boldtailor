"""run_workflow executes the enabled stages, skips what it cannot do, and writes the derivative."""

import base64
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
    _assert_report_tables(result.report_path.read_text())
    png = (
        _func(settings) / "sub-07_ses-nsd10_task-nsdcore_desc-Library_plot.png"
    ).read_bytes()
    assert base64.b64encode(png).decode("ascii") in result.report_path.read_text()


def _section(html, name):
    start = html.index(f'<section id="{name}">')
    return html[start : html.index("</section>", start)]


def _assert_report_tables(html):
    inputs_section = _section(html, "inputs")
    assert "retained_scans" in inputs_section and "a_comp_cor_00" in inputs_section
    assert "peak_time" in inputs_section and "hrf_id" in inputs_section
    assert "selected_peak_time" in _section(html, "glms")
    assert "Odd vs even" in _section(html, "reliability")
    betas = _section(html, "betas")
    assert "odd_to_even" in betas and "median_encoding_r2" in betas
    assert "boundary_fraction" in betas


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


def test_describe_inputs_reports_the_problems_a_run_would_hit(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    out = tmp_path / "out"
    info = workflow_run.describe_inputs(
        settings_for(root, output_dir=out, hrf_library="canonical")
    )
    assert [p["kind"] for p in info["problems"]] == ["input"]
    assert "three odd and three even" in info["problems"][0]["message"]
    quick = dict(output_dir=out, hrf_library="canonical", ridge_mode="off")
    assert workflow_run.describe_inputs(settings_for(root, **quick))["problems"] == []
    workflow_run.run_workflow(settings_for(root, stages=frozenset({"glms"}), **quick))
    info = workflow_run.describe_inputs(settings_for(root, **quick))
    assert [p["kind"] for p in info["problems"]] == ["existing_results"]
    info = workflow_run.describe_inputs(settings_for(root, task="other", **quick))
    assert info["problems"][0]["kind"] == "input" and "runs" not in info


def test_describe_inputs_reports_runs_model_and_output(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    info = workflow_run.describe_inputs(
        settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical")
    )
    assert info["runs"] == ["run-01", "run-02", "run-03", "run-04"]
    assert info["task_model"]["regressors"] == [
        "task",
        "response_time",
        "trial_type[1]",
    ]
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


def test_string_trial_type_is_expanded_into_indicators(
    four_runs, settings_for, tmp_path
):
    from tests.workflow.synthetic_bids import face_house_events, rewrite_events

    root, _ = four_runs
    rewrite_events(root, face_house_events)
    settings = settings_for(
        root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off"
    )
    info = workflow_run.describe_inputs(settings)
    assert info["task_model"]["regressors"] == ["task", "trial_type[house]"]
    assert info["task_model"]["modulators"][0]["levels"] == ["face", "house"]
    assert not any("trial_type" in note for note in info["notes"])
    workflow_run.run_workflow(settings)
    metadata = json.loads(
        (
            _func(settings)
            / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json"
        ).read_text()
    )
    assert metadata["task_model"]["regressors"] == ["task", "trial_type[house]"]


def _three_level_events(table):
    labels = ["face", "house", "scrambled face"] * (len(table) // 3 + 1)
    return table.assign(trial_type=labels[: len(table)])


def test_string_trial_type_run_writes_indicator_maps_and_describes_them(
    four_runs, settings_for, tmp_path
):
    import nibabel as nib

    from tests.workflow.synthetic_bids import rewrite_events

    root, _ = four_runs
    rewrite_events(root, _three_level_events)
    settings = settings_for(
        root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off"
    )
    result = workflow_run.run_workflow(settings)
    name = "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json"
    meta = json.loads((_func(settings) / name).read_text())
    expected = [
        "task",
        "response_time",
        "trial_type[house]",
        "trial_type[scrambled face]",
    ]
    assert meta["task_model"]["regressors"] == expected
    assert meta["categorical_modulators"]["trial_type"]["reference"] == "face"
    effects = next(
        p for p in result.paths if "desc-OptimizedGLM_stat-effects" in p.name
    )
    assert list(nib.load(effects).header.get_axis(0).name) == expected
    html = result.report_path.read_text()
    assert "trial_type[scrambled face]" in html and "reference face" in html


def test_run_missing_a_level_is_an_input_error_naming_the_run_once(
    four_runs, settings_for, tmp_path
):
    import pandas as pd

    root, _ = four_runs
    path = next(root.rglob("*run-02_events.tsv"))
    pd.read_csv(path, sep="\t").assign(trial_type=0).to_csv(path, sep="\t", index=False)
    settings = settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical")
    with pytest.raises(inputs.InputError) as raised:
        workflow_run.run_workflow(settings)
    message = str(raised.value)
    assert "run-02" in message and "'1'" in message
    assert message.count("run") == 1


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
    with pytest.raises(inputs.InputError, match="cosine"):
        workflow_run.run_workflow(settings)
    (problem,) = workflow_run.describe_inputs(settings)["problems"]
    assert problem["kind"] == "input" and "cosine" in problem["message"]


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
