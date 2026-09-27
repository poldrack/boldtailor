"""The tutorial must execute real CIFTI analyses with matched GLM predictors."""

import importlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401
from boldtailor.hrf_library import HrfLibrary

NOTEBOOK = Path(__file__).with_name("nsd_workflow.ipynb")


def workflow(module="workflow_inputs"):
    try:
        return importlib.import_module(f"examples.NSD.{module}")
    except ModuleNotFoundError as error:
        pytest.fail(f"The NSD notebook workflow is not implemented: {error}")


@pytest.fixture
def four_runs(dataset):
    root, prep, *_ = dataset
    for directory in (root, prep):
        func = directory / "sub-07/ses-nsd10/func"
        for source in list(func.glob("*run-0[12]*")):
            target = source.with_name(
                source.name.replace("run-01", "run-03").replace("run-02", "run-04")
            )
            target.write_bytes(source.read_bytes())
    for number, dropped in enumerate((1, 2, 3, 1), 1):
        path = next(prep.rglob(f"*run-{number:02d}*confounds_timeseries.tsv"))
        table = pd.read_csv(path, sep="\t")
        for i in range(dropped):
            table[f"non_steady_state_outlier{i:02d}"] = (
                np.arange(len(table)) == i
            ).astype(int)
        table.to_csv(path, sep="\t", index=False)
    return root, prep


@pytest.fixture
def small_library():
    return HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])


def test_glm_events_preserve_timing_and_center_two_joint_modulators(events):
    original = events.copy(deep=True)
    result = workflow().glm_events(events)
    amplitudes = {
        "task": np.ones(len(events)),
        "response_time": events.response_time - events.response_time.mean(),
        "trial_type": events.trial_type - events.trial_type.mean(),
    }
    assert set(result.trial_type) == set(amplitudes)
    assert len(result) == 3 * len(events)
    for name, expected in amplitudes.items():
        rows = result.loc[result.trial_type == name]
        np.testing.assert_allclose(rows.modulation, expected)
        np.testing.assert_allclose(
            rows[["onset", "duration"]], events[["onset", "duration"]]
        )
    pd.testing.assert_frame_equal(events, original)


@pytest.mark.parametrize("column,value", [("trial_type", 2), ("response_time", np.nan)])
def test_invalid_glm_covariates_fail_explicitly(events, column, value):
    events.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        workflow().glm_events(events)


def test_trimming_keeps_acquisition_times_and_matches_confounds(four_runs):
    root, prep = four_runs
    runs = workflow().load_session(root, prep)
    data = workflow().load_block(runs, root, [0, 2], glm=True)
    assert [len(t) for t in data.frame_times] == [95, 94, 93, 95]
    for run, y, dropped in zip(runs, data.signals, (1, 2, 3, 1), strict=True):
        np.testing.assert_allclose(
            run.frame_times, 0.775 + np.arange(dropped, 96) * 1.6
        )
        np.testing.assert_allclose(y, np.asarray(run.image.dataobj)[dropped:, [0, 2]])
        assert not any(c.startswith("non_steady") for c in run.confounds)
        assert run.events.onset.iloc[0] == 8.0
    assert len({tuple(r.confounds.columns) for r in runs}) == 1
    assert sum(c.startswith("a_comp_cor") for c in runs[0].confounds) == 6
    assert data.provenance.metadata_fingerprint is not None
    assert (
        data.provenance.sources[2].signal.annotations["retained_frame_indices"][0] == 3
    )


def test_interior_nonsteady_flag_is_rejected(four_runs):
    root, prep = four_runs
    path = next(prep.rglob("*run-01*confounds_timeseries.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.loc[12, "non_steady_state_outlier00"] = 1
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="leading|contiguous"):
        workflow().load_session(root, prep)


def independent_ols(runs, indices, library, ids):
    effects, residuals, nuisance_residuals, totals = [], [], [], []
    for run in runs:
        y = np.asarray(run.image.dataobj)[run.retained_frames][:, indices]
        nuisance = np.column_stack([run.confounds, np.ones(len(y))])
        per_feature, errors = [], []
        for col, hrf_id in enumerate(ids):
            candidate = library.candidates[hrf_id]
            e = run.events
            amplitudes = (
                np.ones(len(e)),
                e.response_time - e.response_time.mean(),
                e.trial_type - e.trial_type.mean(),
            )
            columns = [
                compute_regressor(
                    np.vstack([e.onset, e.duration, a]),
                    "spm" if hrf_id == 0 else candidate.kernel,
                    run.frame_times,
                )[0][:, 0]
                for a in amplitudes
            ]
            x = np.column_stack([*columns, nuisance])
            b = np.linalg.lstsq(x, y[:, col], rcond=None)[0]
            per_feature.append(b[:3])
            errors.append(np.sum((y[:, col] - x @ b) ** 2))
        effects.append(np.array(per_feature).T)
        residuals.append(errors)
        nuisance_residuals.append(
            np.sum(
                (y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]) ** 2,
                axis=0,
            )
        )
        totals.append(np.sum((y - y.mean(axis=0)) ** 2, axis=0))
    full = 1 - np.sum(residuals, axis=0) / np.sum(totals, axis=0)
    nuisance = 1 - np.sum(nuisance_residuals, axis=0) / np.sum(totals, axis=0)
    return np.mean(effects, axis=0), np.stack([full, nuisance, full - nuisance])


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_both_glms_match_independent_ols_and_keep_spatial_order(
    four_runs, small_library, n_jobs
):
    root, prep = four_runs
    runs = workflow().load_session(root, prep)
    inputs, analysis = workflow(), workflow("workflow_analysis")
    blocks = inputs.make_blocks(runs, block_size=2)
    assert [list(b) for b in blocks] == [
        [0, 1],
        [2],
    ]  # Constant vertex retained as NaN in maps.
    model = inputs.glm_model(runs)
    selected = analysis.select_hrfs(runs, root, blocks, small_library, n_jobs=n_jobs)
    for selection in (None, selected):
        result = analysis.fit_glms(
            runs, root, blocks, model, selections=selection, n_jobs=n_jobs
        )
        ids = (
            np.zeros(3, dtype=int)
            if selection is None
            else analysis.selection_maps(selected, 4)["all"][0, :3].astype(int)
        )
        effects, scores = independent_ols(runs, [0, 1, 2], small_library, ids)
        np.testing.assert_allclose(result["effects"][:, :3], effects, atol=1e-8)
        np.testing.assert_allclose(result["r2"][:, :3], scores, atol=1e-10)
        assert np.isnan(result["r2"][:, 3]).all()
        assert np.isnan(result["effects"][:, 3]).all()
        assert result["designs"] and result["provenance"]


def test_notebook_executes_full_workflow_and_exports_reusable_artifacts(
    four_runs, tmp_path
):
    assert NOTEBOOK.is_file(), "The full NSD workflow notebook has not been created"
    root, prep = four_runs
    output = tmp_path / "output"
    config = dict(
        bids_root=str(root),
        fmriprep_root=str(prep),
        output_root=str(output),
        block_size=2,
        n_jobs=1,
        hrf_parameters=[[3, 10, 0.5, 0.5, 2, 0, 36]],
        ridge_alpha=0.1,
    )
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(f"NSD_CONFIG = {config!r}"))
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parents[2])}},
    )
    executed = client.execute()
    nbformat.write(executed, tmp_path / "executed.ipynb")
    files = list(output.rglob("*"))
    scalars = [p for p in files if p.name.endswith(".dscalar.nii")]
    assert scalars and all("desc-notebook" in p.name for p in scalars)
    by_name = {p.name: p for p in scalars}

    def read(suffix):
        return nib.load(
            next(p for n, p in by_name.items() if n.endswith(suffix))
        ).get_fdata()

    canonical = read("desc-notebookCanonicalGLM_stat-rsquared.dscalar.nii")
    optimized = read("desc-notebookOptimizedGLM_stat-rsquared.dscalar.nii")
    comparison = read("desc-notebookGLMComparison_stat-deltarsquared.dscalar.nii")
    np.testing.assert_allclose(comparison[0], optimized[0] - canonical[0], atol=1e-7)
    np.testing.assert_allclose(canonical[2], canonical[0] - canonical[1], atol=1e-7)
    for label in ("All", "Odd", "Even"):
        assert any(
            f"desc-notebookHRF{label}_stat-hrfparameters" in p.name for p in scalars
        )
    assert len([p for p in scalars if "_betas." in p.name]) == 16
    assert len([p for p in files if p.name.endswith("_trials.tsv")]) == 4
    assert any(p.name.endswith("_library.tsv") for p in files)
    assert any(p.name.endswith("_designs.npz") for p in files)
    for path in scalars:
        image = nib.load(path)
        assert image.shape[1] == 4
        assert np.isnan(image.get_fdata()[:, 3]).all()
    metadata = json.loads(
        next(p for p in files if p.name.endswith("_metadata.json")).read_text()
    )
    assert metadata["regressors"] == ["task", "response_time", "trial_type"]
    assert metadata["retained_scans"] == [95, 94, 93, 95]
    assert metadata["library_candidates"] == 2
    assert metadata["noise_model"] == "ols"
    assert "independent" in metadata["glm_comparison"].lower()
    before = {p: p.read_bytes() for p in scalars}
    with pytest.raises(FileExistsError):
        workflow("workflow_outputs").check_output(output, "sub-07", "ses-nsd10")
    assert all(p.read_bytes() == value for p, value in before.items())


def test_rerunning_beta_cell_uses_current_settings(four_runs, small_library):
    assert NOTEBOOK.is_file(), "The full NSD workflow notebook has not been created"
    root, prep = four_runs
    inputs, analysis = workflow(), workflow("workflow_analysis")
    runs = inputs.load_session(root, prep)
    blocks = inputs.make_blocks(runs, block_size=4)
    selections = analysis.select_hrfs(runs, root, blocks, small_library)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(
        c
        for c in notebook.cells
        if c.cell_type == "code" and "beta_models =" in c.source
    )
    context = dict(
        runs=runs,
        root=root,
        blocks=blocks,
        selections=selections,
        fit_beta_series=analysis.fit_beta_series,
        workers=1,
        settings={"ridge_alpha": 0.1},
        display=lambda value: None,
    )
    exec(cell.source, context)
    original = context["beta_models"]["OptimizedTrialRidge"]["betas"][0].copy()
    context["settings"]["ridge_alpha"] = 0.4
    exec(cell.source, context)
    refit = context["beta_models"]["OptimizedTrialRidge"]
    assert refit["ridge_alpha"] == 0.4
    assert not np.allclose(original[:, :3], refit["betas"][0][:, :3])
