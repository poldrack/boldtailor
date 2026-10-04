"""One writer publishes every stage with boldtailor descriptors."""

from importlib.metadata import version
import json

from matplotlib.figure import Figure
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary, PARAMETER_NAMES
from boldtailor.model import Modulator
from boldtailor.publication import publish_artifact_set
from boldtailor.workflow import analysis, artifacts, beta_series, inputs, outputs

SPLIT_DESCRIPTORS = (
    "HRFOdd",
    "HRFEven",
    "HRFOddToEven",
    "HRFEvenToOdd",
    "HRFReliability",
)


def _session(settings):
    runs = inputs.load_session(settings)
    model = inputs.detect_task_model([r.events for r in runs], settings.modulators)
    return runs, model, runs[0].image.header.get_axis(1)


def _beta_models(settings, library, *, block_size=4):
    runs, model, brain = _session(settings)
    root = settings.bids_dir
    blocks = inputs.make_blocks(runs, block_size=block_size)
    selections = analysis.select_hrfs(runs, root, blocks, library, task_model=model)
    models = beta_series.fit_beta_models(
        runs, root, blocks, settings, library, selections, model
    )
    return runs, model, brain, selections, models


def _find(paths, fragment):
    return next(p for p in paths if fragment in p.name)


def test_dataset_description_is_a_boldtailor_derivative():
    artifact = artifacts.dataset_description("NSD single-trial models")
    assert artifact.path == "dataset_description.json"
    assert json.loads(artifact.payload) == {
        "Name": "NSD single-trial models",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "boldtailor", "Version": version("boldtailor")}],
    }


def test_scalar_map_names_the_space_descriptor_and_statistic():
    brain = nib.cifti2.BrainModelAxis.from_surface([0, 1], 4, name="CortexLeft")
    artifact = artifacts.scalar_map(
        "sub-01/ses-a/func/sub-01_ses-a_task-x",
        "space-fsLR_den-91k",
        brain,
        "CanonicalGLM",
        "t",
        np.ones((1, 2)),
        ["task"],
    )
    assert artifact.path == (
        "sub-01/ses-a/func/sub-01_ses-a_task-x_space-fsLR_den-91k"
        "_desc-CanonicalGLM_stat-t.dscalar.nii"
    )


def test_activation_exports_preserve_map_names_axis_and_values(
    dataset, settings_for, tmp_path
):
    root, *_ = dataset
    settings = settings_for(root)
    brain = nib.cifti2.BrainModelAxis.from_surface([2, 0, 1], 4, name="CortexLeft")
    maps = {
        "mean_beta": np.array([2, -2, np.nan]),
        "t": np.array([3, -3, np.nan]),
        "p_uncorrected": np.array([0.04, 0.04, np.nan]),
        "n_trials": np.array([5, 5, np.nan]),
        "df": np.array([4, 4, np.nan]),
    }
    written = outputs.activation_artifacts(settings, brain, {"CanonicalTrialOLS": maps})
    paths = publish_artifact_set(tmp_path / "out", written)
    image = nib.load(paths[0])
    assert image.header.get_axis(0).name.tolist() == list(maps)
    assert image.header.get_axis(1) == brain
    np.testing.assert_allclose(image.get_fdata(), np.stack(list(maps.values())))
    assert paths[0].name == (
        "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k"
        "_desc-CanonicalTrialOLS_stat-activation.dscalar.nii"
    )


def _assert_tuning_scope(paths, selected, base):
    image = nib.load(_find(paths, f"{base}_stat-encodingcvr2."))
    np.testing.assert_allclose(
        image.get_fdata(), selected["scores"].cv_r2, rtol=1e-6, atol=1e-6
    )
    scores = pd.read_csv(_find(paths, f"{base}_scores.tsv"), sep="\t")
    np.testing.assert_allclose(scores.objective, selected["selection"].objective_scores)
    assert (
        scores.loc[scores.selected, "alpha"].item() == selected["selection"].ridge_alpha
    )
    with np.load(_find(paths, f"{base}_folds.npz")) as arrays:
        np.testing.assert_allclose(arrays["sse"], selected["scores"].fold_sse)
        np.testing.assert_allclose(arrays["sst"], selected["scores"].fold_sst)
    record = json.loads(_find(paths, f"{base}_provenance.json").read_text())
    assert record["analysis_fingerprint"] == selected["provenance"].analysis_fingerprint
    assert (
        record["activities"][-1]["selected_alpha"] == selected["selection"].ridge_alpha
    )


def test_ridge_artifacts_match_numeric_results(
    six_run_dataset, cv_library, settings_for, tmp_path
):
    root, _ = six_run_dataset
    settings = settings_for(root, ridge_mode="cv", ridge_alphas=(0.0, 0.1))
    runs, task_model, brain, _, models = _beta_models(settings, cv_library)
    model = models["CanonicalTrialRidgeCV"]
    described = outputs.metadata(
        runs,
        cv_library,
        settings,
        task_model,
        beta_models=models,
        activation=None,
        selections=None,
        skipped=(),
        report=None,
    )
    assert described["ridge_cv"]["encoding_predictors"] == [
        "task",
        "response_time",
        "trial_type[1]",
    ]
    written = outputs.beta_model_artifacts(settings, brain, runs, model, cv_library)
    output = tmp_path / "published"
    sentinel = output / "old_fixed_ridge.txt"
    output.mkdir()
    sentinel.write_text("retain prior fixed-alpha outputs")
    paths = publish_artifact_set(output, written)
    assert not any("notebook" in p.name for p in paths)
    for scope in ("Odd", "Even", "All"):
        base = f"desc-CanonicalRidgeCV{scope}"
        _assert_tuning_scope(paths, model.tuning[scope.lower()], base)
        image = nib.load(_find(paths, f"{base}_stat-encodingcvr2."))
        assert image.header.get_axis(1) == brain
    for split in ("OddToEven", "EvenToOdd"):
        metadata = json.loads(
            _find(paths, f"desc-CanonicalRidgeCV{split}_metadata.json").read_text()
        )
        assert not set(metadata["train_run_labels"]) & set(metadata["test_run_labels"])
        assert metadata["validation_target"] == "selected_penalty_regularized_betas"
        assert metadata["predictor_names"] == [
            "task",
            *[m.column for m in task_model.modulators],
        ]
        coefficients = nib.load(
            _find(paths, f"desc-CanonicalRidgeCV{split}_stat-coefficients.")
        )
        assert coefficients.header.get_axis(0).name.tolist() == (
            metadata["predictor_names"]
        )
    predictors = pd.read_csv(
        _find(paths, "desc-CanonicalTrialRidgeCV_predictors.tsv"), sep="\t"
    )
    assert len(predictors) == 36 and predictors.trial_id.is_unique
    assert len([p for p in paths if p.name.endswith("_predictions.dscalar.nii")]) == 6
    assert len([p for p in paths if p.name.endswith("_targets.dscalar.nii")]) == 6
    assert outputs.tuning_figure(models).axes
    assert sentinel.read_text() == "retain prior fixed-alpha outputs"
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(FileExistsError):
        publish_artifact_set(output, written)
    assert all(p.read_bytes() == original for p, original in before.items())


def _assert_fraction_exports(paths, model, brain, runs):
    mode = model.hrf.capitalize()
    for scope in ("Odd", "Even", "All"):
        image = nib.load(_find(paths, f"{mode}FractionalCV{scope}_stat-ridgefraction."))
        assert image.header.get_axis(1) == brain
        np.testing.assert_allclose(
            image.get_fdata()[0],
            model.tuning[scope.lower()]["selection"].ridge_fraction,
        )
    for scope, outer in model.evaluation.items():
        descriptor = (
            mode + "FractionalCV" + "".join(w.title() for w in scope.split("_"))
        )
        for label, targets in zip(outer["test_run_labels"], outer["targets"]):
            run = next(r for r in runs if r.label == label)
            path = next(
                p
                for p in paths
                if run.inputs.stem in p.name
                and f"desc-{descriptor}_targets.dscalar.nii" in p.name
            )
            np.testing.assert_allclose(nib.load(path).get_fdata(), targets, atol=1e-6)
    alpha = _find(paths, f"{mode}TrialFractionalCV_stat-ridgealpha.")
    np.testing.assert_allclose(
        nib.load(alpha).get_fdata(), model.fit["run_ridge_alphas"], atol=1e-7
    )


def test_fraction_exports_match_the_fitted_models(
    six_run_dataset, cv_library, settings_for, tmp_path
):
    root, _ = six_run_dataset
    settings = settings_for(
        root, ridge_mode="fractional_cv", ridge_fractions=(0.3, 0.7, 1.0)
    )
    runs, _, brain, _, models = _beta_models(settings, cv_library, block_size=1)
    tuned = {k: m for k, m in models.items() if m.tuning}
    assert set(tuned) == {"CanonicalTrialFractionalCV", "OptimizedTrialFractionalCV"}
    for name, model in tuned.items():
        written = outputs.beta_model_artifacts(settings, brain, runs, model, cv_library)
        paths = publish_artifact_set(tmp_path / name, written)
        _assert_fraction_exports(paths, model, brain, runs)
        mode = model.hrf.capitalize()
        metadata = json.loads(
            _find(paths, f"{mode}FractionalCVAll_metadata.json").read_text()
        )
        assert metadata["selection_rule"] == "maximum_encoding_r2_per_grayordinate"
        assert metadata["validation_target"] == "fixed_ols_betas"
        assert metadata["percentile_role"] == "descriptive_only"
    table = outputs.tuning_table(models)
    assert set(table["mode"]) == {"Canonical", "Optimized"}
    for mode in ("Canonical", "Optimized"):
        rows = table.loc[table["mode"] == mode]
        assert rows.groupby("scope").selected_grayordinates.sum().tolist() == [3, 3, 3]
    assert "alpha" not in table
    assert outputs.tuning_figure(models).axes[0].get_xlabel() == "Ridge fraction"


def test_beta_model_writer_covers_every_estimator(
    six_run_dataset, cv_library, settings_for
):
    root, _ = six_run_dataset
    settings = settings_for(
        root, ridge_mode="fractional_cv", ridge_fractions=(0.4, 1.0), block_size=4
    )
    runs = inputs.load_session(settings)
    model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=4)
    brain = runs[0].image.header.get_axis(1)
    selections = analysis.select_hrfs(runs, root, blocks, cv_library, task_model=model)
    models = beta_series.fit_beta_models(
        runs, root, blocks, settings, cv_library, selections, model
    )
    names = {
        a.path
        for m in models.values()
        for a in outputs.beta_model_artifacts(settings, brain, runs, m, cv_library)
    }
    stem = settings.stem
    assert (
        f"{stem}_space-fsLR_den-91k_desc-OptimizedTrialFractionalCV_stat-ridgefraction.dscalar.nii"
        in names
    )
    assert (
        f"{stem}_space-fsLR_den-91k_desc-OptimizedFractionalCVAll_stat-encodingcvr2.dscalar.nii"
        in names
    )
    assert (
        f"{stem}_space-fsLR_den-91k_desc-CanonicalTrialOLS_stat-rtcorrelation.dscalar.nii"
        in names
    )
    assert not any("notebook" in n for n in names)


def test_beta_model_writer_omits_rt_correlations_without_rt(
    four_runs, small_library, settings_for
):
    root, _ = four_runs
    settings = settings_for(
        root, ridge_mode="off", modulators=(Modulator("trial_type"),)
    )
    runs, _, brain, _, models = _beta_models(settings, small_library)
    assert all(m.fit["rt"] is None for m in models.values())
    names = {
        a.path
        for m in models.values()
        for a in outputs.beta_model_artifacts(settings, brain, runs, m, small_library)
    }
    assert any("desc-CanonicalTrialOLS_betas" in n for n in names)
    assert not any("rtcorrelation" in n for n in names)


def test_check_output_errors_only_when_files_exist(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    settings = settings_for(root, output_dir=tmp_path / "out")
    outputs.check_output(settings)  # nothing there
    target = tmp_path / "out" / "sub-07" / "ses-nsd10" / "func"
    target.mkdir(parents=True)
    (target / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json").write_text(
        "{}"
    )
    with pytest.raises(FileExistsError, match="existing_results"):
        outputs.check_output(settings)
    outputs.check_output(
        settings_for(root, output_dir=tmp_path / "out", existing_results="overwrite")
    )


def test_check_output_ignores_other_tasks(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    target = tmp_path / "out" / "sub-07" / "ses-nsd10" / "func"
    target.mkdir(parents=True)
    (target / "sub-07_ses-nsd10_task-other_desc-boldtailor_metadata.json").write_text(
        "{}"
    )
    assert outputs.check_output(settings_for(root, output_dir=tmp_path / "out")) is None


@pytest.fixture
def selected(four_runs, settings_for):
    root, _ = four_runs
    settings = settings_for(root)
    runs, model, brain = _session(settings)
    library = HrfLibrary.from_parameters(
        [
            [3, 10, 0.5, 0.5, 2, 0, 36],
            [4, 12, 1, 1.5, 5, 1, 36],
            [6, 16, 1.5, 2.5, 8, 2, 36],
        ]
    )
    blocks = inputs.make_blocks(runs, block_size=2, max_grayordinates=4)
    selections = analysis.select_hrfs(runs, root, blocks, library, task_model=model)
    return settings, runs, model, brain, library, blocks, selections


def _metadata(selected, **extra):
    settings, runs, model, _, library, _, selections = selected
    options = dict(
        beta_models={},
        activation=None,
        selections=selections,
        skipped=(),
        report=None,
    )
    options.update(extra)
    return outputs.metadata(runs, library, settings, model, **options)


def test_metadata_records_the_task_model_fingerprint(selected):
    model = selected[2]
    assert _metadata(selected)["task_model_fingerprint"] == model.fingerprint


def test_metadata_records_peak_hrf_normalization(selected):
    assert _metadata(selected)["hrf_normalization"] == "peak_one_event_response"


def test_metadata_records_settings_runs_task_model_skips_and_report(selected):
    settings, runs, model, *_ = selected
    report = "sub-07_ses-nsd10_task-nsdcore_report.html"
    published = _metadata(
        selected, skipped=[("reliability", "too few runs")], report=report
    )
    assert published["settings"] == settings.to_dict()
    assert published["runs"] == [r.label for r in runs]
    assert published["task_model"] == model.to_dict()
    assert published["skipped"] == [("reliability", "too few runs")]
    assert published["report"] == report
    assert "uncentered" in published["task_model_description"]
    assert "value zero" in published["task_model_description"]
    json.dumps(published, allow_nan=False)


def test_metadata_prose_follows_the_task_model(four_runs, small_library, settings_for):
    root, _ = four_runs
    settings = settings_for(root, modulators=(Modulator("trial_type"),))
    runs, model, _ = _session(settings)
    published = outputs.metadata(
        runs,
        small_library,
        settings,
        model,
        beta_models={},
        activation=None,
        selections=None,
        skipped=(),
        report=None,
    )
    assert published["regressors"] == ["task", "trial_type"]
    assert "trial_type" in published
    assert "response_time" not in published
    assert "response_time" not in published["task_model_description"]
    assert "missing_response_time" not in published
    assert "rt_check" not in published
    assert "RT" not in published["hrf_selection"]


def test_metadata_rt_check_follows_the_rt_selection_switch(selected, settings_for):
    settings, runs, model, _, library, _, selections = selected
    assert "RT enters HRF selection" in _metadata(selected)["rt_check"]
    off = settings_for(settings.bids_dir, hrf_selection_rt=False)
    published = outputs.metadata(
        runs,
        library,
        off,
        model,
        beta_models={},
        activation=None,
        selections=selections,
        skipped=(),
        report=None,
    )
    assert "never used to select HRFs" in published["rt_check"]


def test_metadata_pools_hrf_bound_flags_over_blocks(selected):
    selections = selected[-1]
    assert len(selections) > 1
    rows = pd.DataFrame(_metadata(selected)["hrf_boundary_summary"])
    for scope in ("all", "odd", "even"):
        picked = [
            bundle[scope] if scope == "all" else bundle[scope].training_selection
            for bundle in selections.values()
        ]
        ids = np.concatenate([p.hrf_indices for p in picked])
        flags = np.concatenate([p.parameter_bound_flags for p in picked])[ids > 0]
        table = rows.loc[rows.scope == scope]
        assert (table.n_custom == (ids > 0).sum()).all()
        for row in table.itertuples():
            p = PARAMETER_NAMES.index(row.parameter)
            e = ("low", "high").index(row.edge)
            expected = flags[:, p, e].mean() if len(flags) else np.nan
            np.testing.assert_allclose(row.fraction_flagged, expected)


def test_hrf_artifacts_write_splits_only_when_requested(selected):
    settings, _, _, brain, library, _, selections = selected
    stem, space = settings.stem, settings.space_entity
    full = {a.path for a in outputs.hrf_artifacts(settings, brain, selections, library)}
    for descriptor in SPLIT_DESCRIPTORS:
        assert any(f"_desc-{descriptor}_" in n for n in full)
    alone = {
        a.path
        for a in outputs.hrf_artifacts(
            settings, brain, selections, library, include_splits=False
        )
    }
    assert f"{stem}_{space}_desc-HRFAll_stat-selection.dscalar.nii" in alone
    assert f"{stem}_{space}_desc-HRFAll_stat-hrfparameters.dscalar.nii" in alone
    assert f"{stem}_desc-HRF_library.tsv" in alone
    assert f"{stem}_desc-HRF_provenance.json" in alone
    for descriptor in SPLIT_DESCRIPTORS:
        assert not any(f"_desc-{descriptor}_" in n for n in alone)


def test_save_workflow_publishes_every_stage_and_the_report(
    selected, settings_for, tmp_path
):
    _, runs, model, _, library, blocks, selections = selected
    root = selected[0].bids_dir
    settings = settings_for(root, ridge_mode="off", output_dir=tmp_path / "out")
    glm = inputs.glm_model(runs, model)
    glms = {
        "CanonicalGLM": analysis.fit_glms(runs, root, blocks, glm),
        "OptimizedGLM": analysis.fit_glms(
            runs, root, blocks, glm, selections=selections
        ),
    }
    beta_models = beta_series.fit_beta_models(
        runs, root, blocks, settings, library, selections, model
    )
    paths = outputs.save_workflow(
        settings,
        runs,
        model,
        library,
        selections,
        glms,
        beta_models,
        figures={},
        activation=None,
        skipped=[("reliability", "skipped for the test")],
        report_html=b"<html></html>",
        include_hrf_splits=False,
    )
    out = settings.output_dir
    stem = settings.stem
    names = {p.relative_to(out).as_posix() for p in paths}
    assert "dataset_description.json" in names
    assert "sub-07_ses-nsd10_task-nsdcore_report.html" in names
    assert f"{stem}_desc-boldtailor_metadata.json" in names
    assert f"{stem}_desc-boldtailor_runs.tsv" in names
    assert (
        f"{stem}_space-fsLR_den-91k_desc-GLMComparison_stat-deltarsquared.dscalar.nii"
        in names
    )
    assert (
        f"{stem}_space-fsLR_den-91k_desc-OptimizedTrialOLS_stat-rsquared.dscalar.nii"
        in names
    )
    assert any(n.endswith("_desc-boldtailor_events.tsv") for n in names)
    assert not any("notebook" in n or "HRFOdd" in n for n in names)
    description = json.loads((out / "dataset_description.json").read_text())
    assert description["Name"] == "boldtailor"
    metadata = json.loads((out / f"{stem}_desc-boldtailor_metadata.json").read_text())
    assert metadata["report"] == "sub-07_ses-nsd10_task-nsdcore_report.html"
    assert metadata["skipped"] == [["reliability", "skipped for the test"]]
    effects = nib.load(
        out / f"{stem}_space-fsLR_den-91k_desc-CanonicalGLM_stat-effects.dscalar.nii"
    )
    assert effects.header.get_axis(0).name.tolist() == list(model.regressor_names)
    with pytest.raises(FileExistsError, match="existing_results"):
        outputs.check_output(settings)


def test_save_workflow_keeps_another_sessions_dataset_description(
    four_runs, small_library, settings_for, tmp_path
):
    root, _ = four_runs
    settings = settings_for(root, output_dir=tmp_path / "out")
    runs, model, _ = _session(settings)
    existing = settings.output_dir / "dataset_description.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"Name": "earlier session"}')
    paths = outputs.save_workflow(
        settings,
        runs,
        model,
        small_library,
        None,
        {},
        {},
        figures={},
        activation=None,
        skipped=(),
        report_html=None,
    )
    assert existing not in paths
    assert json.loads(existing.read_text()) == {"Name": "earlier session"}
    assert (settings.output_dir / f"{settings.stem}_desc-boldtailor_runs.tsv") in paths


def test_hrf_outputs_accept_all_only_selection_bundles(selected):
    settings, _, _, brain, library, _, selections = selected
    all_only = {k: {"all": bundle["all"]} for k, bundle in selections.items()}
    written = outputs.hrf_artifacts(
        settings, brain, all_only, library, include_splits=False
    )
    names = {a.path for a in written}
    stem, space = settings.stem, settings.space_entity
    assert f"{stem}_{space}_desc-HRFAll_stat-selection.dscalar.nii" in names
    full = outputs.hrf_artifacts(
        settings, brain, selections, library, include_splits=False
    )
    for one, two in zip(
        sorted(written, key=lambda a: a.path), sorted(full, key=lambda a: a.path)
    ):
        assert one.path == two.path
        if one.path.endswith(".dscalar.nii"):
            assert one.payload == two.payload
    rows = pd.DataFrame(outputs.hrf_boundary_summary(all_only))
    assert set(rows.scope) == {"all"}
    published = _metadata(selected, selections=all_only)
    assert {r["scope"] for r in published["hrf_boundary_summary"]} == {"all"}


def _save_inputs_only(settings, figures=None, report_html=None):
    runs, model, _ = _session(settings)
    library = HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])
    return outputs.save_workflow(
        settings,
        runs,
        model,
        library,
        None,
        {},
        {},
        figures=figures or {},
        activation=None,
        skipped=(),
        report_html=report_html,
    )


def _figure():
    figure = Figure()
    figure.subplots().plot([0, 1])
    return figure


def test_overwrite_removes_only_stale_files_boldtailor_listed(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    out = tmp_path / "out"
    first = _save_inputs_only(
        settings_for(root, output_dir=out), figures={"Old": _figure()}
    )
    func = out / "sub-07" / "ses-nsd10" / "func"
    stale = func / "sub-07_ses-nsd10_task-nsdcore_desc-Old_plot.png"
    assert stale in first
    other_session = out / "sub-07" / "ses-nsd11" / "func"
    other_session.mkdir(parents=True)
    kept = [
        func / "sub-07_ses-nsd10_task-nsdcore_desc-Mine_stat-x.dscalar.nii",
        func / "sub-07_ses-nsd10_task-other_desc-boldtailor_metadata.json",
        func / "notes.txt",
        out / "sub-07_ses-nsd10_task-nsdcore_notes.txt",
        other_session / "sub-07_ses-nsd11_task-nsdcore_desc-Old_stat-x.dscalar.nii",
    ]
    for path in kept:
        path.write_text("x")
    second = _save_inputs_only(
        settings_for(root, output_dir=out, existing_results="overwrite")
    )
    assert set(first) - {stale} <= set(second)
    assert not stale.exists()
    assert all(path.exists() for path in kept)
    assert all(path.exists() for path in second)


def test_overwrite_without_an_earlier_artifact_list_deletes_nothing(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    out = tmp_path / "out"
    func = out / "sub-07" / "ses-nsd10" / "func"
    func.mkdir(parents=True)
    foreign = (
        func / "sub-07_ses-nsd10_task-nsdcore_run-01_desc-confounds_timeseries.tsv"
    )
    foreign.write_text("x")
    _save_inputs_only(settings_for(root, output_dir=out, existing_results="overwrite"))
    assert foreign.read_text() == "x"


def test_overwrite_ignores_listed_paths_outside_the_output_dir(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    out = tmp_path / "out"
    settings = settings_for(root, output_dir=out)
    _save_inputs_only(settings)
    metadata_path = out / f"{settings.stem}_desc-boldtailor_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    victim = tmp_path / "victim.txt"
    victim.write_text("x")
    bold = next((root / "derivatives").rglob("*run-01*_bold.dtseries.nii"))
    metadata["artifacts"] += ["../victim.txt", str(victim), str(bold)]
    metadata_path.write_text(json.dumps(metadata))
    _save_inputs_only(settings_for(root, output_dir=out, existing_results="overwrite"))
    assert victim.exists() and bold.exists()


def test_metadata_lists_this_runs_published_artifacts(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    settings = settings_for(root, output_dir=tmp_path / "out")
    paths = _save_inputs_only(
        settings, figures={"Example": _figure()}, report_html=b"<html></html>"
    )
    out = settings.output_dir
    metadata = json.loads(
        (out / f"{settings.stem}_desc-boldtailor_metadata.json").read_text()
    )
    published = {p.relative_to(out).as_posix() for p in paths}
    assert set(metadata["artifacts"]) == published - {"dataset_description.json"}


def test_check_output_names_the_existing_files(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    settings = settings_for(root, output_dir=tmp_path / "out")
    func = tmp_path / "out" / "sub-07" / "ses-nsd10" / "func"
    func.mkdir(parents=True)
    (func / "sub-07_ses-nsd10_task-nsdcore_desc-Mine_plot.png").write_text("x")
    with pytest.raises(FileExistsError) as raised:
        outputs.check_output(settings)
    message = str(raised.value)
    assert "existing files" in message and "existing_results" in message
    assert "sub-07_ses-nsd10_task-nsdcore_desc-Mine_plot.png" in message
    assert "boldtailor outputs" not in message


def test_hrf_boundary_table_is_the_pooled_summary_as_a_frame(selected):
    selections = selected[-1]
    table = outputs.hrf_boundary_table(selections)
    expected = pd.DataFrame(outputs.hrf_boundary_summary(selections))
    pd.testing.assert_frame_equal(table, expected)
    assert outputs.hrf_boundary_table(None) is None


def test_workflow_artifacts_are_the_published_set_less_report_and_description(
    selected, settings_for, tmp_path
):
    _, runs, model, _, library, _, selections = selected
    settings = settings_for(selected[0].bids_dir, output_dir=tmp_path / "out")
    figure = Figure()
    figure.subplots().plot([0, 1])
    options = dict(activation=None, skipped=(), include_hrf_splits=False)
    report = "sub-07_ses-nsd10_task-nsdcore_report.html"
    listed = outputs.workflow_artifacts(
        settings,
        runs,
        model,
        library,
        selections,
        {},
        {},
        figures={"Example": figure},
        report=report,
        **options,
    )
    assert figure.get_axes(), "listing artifacts must leave figures intact"
    paths = outputs.save_workflow(
        settings,
        runs,
        model,
        library,
        selections,
        {},
        {},
        figures={"Example": figure},
        report_html=b"<html></html>",
        **options,
    )
    published = {p.relative_to(settings.output_dir).as_posix() for p in paths}
    assert published - {a.path for a in listed} == {report, "dataset_description.json"}
    assert {a.path for a in listed} <= published
    assert f"{settings.stem}_desc-Example_plot.png" in published
    assert not any("HRFOdd" in a.path for a in listed)


def test_metadata_records_undefined_bound_fractions_as_null(four_runs, settings_for):
    root, _ = four_runs
    settings = settings_for(root, hrf_library="canonical")
    runs, model, _ = _session(settings)
    library = settings.build_library()
    blocks = inputs.make_blocks(runs, block_size=2, max_grayordinates=4)
    selections = analysis.select_hrfs(
        runs, root, blocks, library, task_model=model, splits=False
    )
    published = outputs.metadata(
        runs,
        library,
        settings,
        model,
        beta_models={},
        activation=None,
        selections=selections,
        skipped=(),
        report=None,
    )
    rows = json.loads(json.dumps(published, allow_nan=False))["hrf_boundary_summary"]
    assert rows and all(r["n_custom"] == 0 for r in rows)
    assert all(r["fraction_flagged"] is None for r in rows)


def test_two_sessions_publish_into_one_output_dir_under_error(
    four_runs, settings_for, tmp_path
):
    from tests.workflow.synthetic_bids import copy_task

    root, prep = four_runs
    copy_task(root, prep, "other")
    out = tmp_path / "out"
    first = _save_inputs_only(settings_for(root, output_dir=out))
    second = _save_inputs_only(settings_for(root, task="other", output_dir=out))
    assert out / "dataset_description.json" in first
    assert out / "dataset_description.json" not in second
    assert any("task-other" in p.name for p in second)


def test_publication_waits_long_for_the_writer_lock(
    four_runs, settings_for, tmp_path, monkeypatch
):
    seen = {}
    original = outputs.publish_artifact_set

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(outputs, "publish_artifact_set", spy)
    root, _ = four_runs
    _save_inputs_only(settings_for(root, output_dir=tmp_path / "out"))
    assert seen["lock_timeout"] >= 3600
    assert seen["keep_existing"] == ("dataset_description.json",)


def test_check_output_sees_an_existing_root_report(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    settings = settings_for(root, output_dir=tmp_path / "out")
    report = tmp_path / "out" / "sub-07_ses-nsd10_task-nsdcore_report.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html></html>")
    with pytest.raises(FileExistsError, match="report.html"):
        outputs.check_output(settings)


def test_check_output_message_stays_short_for_many_files(
    four_runs, settings_for, tmp_path
):
    root, _ = four_runs
    func = tmp_path / "out" / "sub-07" / "ses-nsd10" / "func"
    func.mkdir(parents=True)
    for i in range(8):
        (func / f"sub-07_ses-nsd10_task-nsdcore_desc-M{i}_plot.png").write_text("x")
    with pytest.raises(FileExistsError) as raised:
        outputs.check_output(settings_for(root, output_dir=tmp_path / "out"))
    message = str(raised.value)
    assert message.count("_plot.png") == 5 and "8 files" in message
