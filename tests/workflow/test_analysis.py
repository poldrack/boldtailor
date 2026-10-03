"""Blockwise GLM, HRF selection and beta-series fits on a detected task model."""

import logging

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor.workflow import analysis, inputs
from tests.oracles import scaled_condition


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
                e.response_time.fillna(0.0),  # Raw RT; missing becomes zero.
                e.trial_type,  # trial_type is left uncentered.
            )
            if e.response_time.isna().any():
                amplitudes += (e.response_time.isna().astype(float),)
            columns = [
                compute_regressor(
                    scaled_condition(
                        e.onset, e.duration, a, candidate.kernel, run.frame_times
                    ),
                    candidate.kernel,
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
@pytest.mark.parametrize("missing_rt", [False, True])
def test_both_glms_match_independent_ols_and_keep_spatial_order(
    four_runs, small_library, settings_for, n_jobs, caplog, capfd, missing_rt
):
    root, _ = four_runs
    if missing_rt:
        path = next(root.rglob("*run-01*events.tsv"))
        events = pd.read_csv(path, sep="\t")
        events.loc[1, "response_time"] = np.nan
        events.to_csv(path, sep="\t", index=False)
    runs = inputs.load_session(settings_for(root))
    task_model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=2)
    assert [list(b) for b in blocks] == [
        [0, 1],
        [2],
    ]  # Constant vertex retained as NaN in maps.
    model = inputs.glm_model(runs, task_model)
    selected = analysis.select_hrfs(
        runs, root, blocks, small_library, task_model=task_model, n_jobs=n_jobs
    )
    for selection in (None, selected):
        capfd.readouterr()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="boldtailor.workflow"):
            result = analysis.fit_glms(
                runs, root, blocks, model, selections=selection, n_jobs=n_jobs
            )
        captured = capfd.readouterr()
        assert "modulation" not in captured.out
        assert "make_first_level_design_matrix" not in captured.out
        assert any("GLM:" in m for m in caplog.messages)
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
        for (run_index, _), design in result["designs"].items():
            assert ("missing_response_time" in design) == (
                missing_rt and run_index == 0
            )


def test_glm_model_and_selection_share_the_detected_task_model(
    four_runs, small_library, settings_for
):
    from boldtailor.design import expand_events, task_columns

    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model_task = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=2, max_grayordinates=4)
    model = inputs.glm_model(runs, model_task)
    assert model.task_model == model_task
    selections = analysis.select_hrfs(
        runs, root, blocks, small_library, task_model=model_task
    )
    for bundle in selections.values():
        assert bundle["all"].task_model == model_task
    fitted = analysis.fit_glms(runs, root, blocks, model, selections=selections)
    assert fitted["designs"]
    for (run, cid), design in fitted["designs"].items():
        expected = task_columns(
            expand_events(runs[run].events, model_task, run),
            runs[run].frame_times,
            small_library.candidates[cid].kernel,
        )
        np.testing.assert_array_equal(
            design.iloc[:, : expected.shape[1]].to_numpy(), expected.to_numpy()
        )


def test_select_hrfs_without_rt_still_feeds_the_full_glm(
    four_runs, small_library, settings_for
):
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model_task = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=2, max_grayordinates=4)
    narrow = inputs.selection_task_model(model_task, False)
    selections = analysis.select_hrfs(
        runs, root, blocks, small_library, task_model=narrow
    )
    for bundle in selections.values():
        assert bundle["all"].task_model == narrow
        assert bundle["odd"].training_selection.task_model == narrow
        assert bundle["even"].training_selection.task_model == narrow
    fitted = analysis.fit_glms(
        runs,
        root,
        blocks,
        inputs.glm_model(runs, model_task),
        selections=selections,
    )
    assert all(
        "response_time" in design.columns for design in fitted["designs"].values()
    )
    assigned = np.isfinite(fitted["effects"][0, :4])
    assert assigned.any()
    assert np.isfinite(fitted["effects"][1, :4][assigned]).all()


def test_beta_series_without_response_time_has_no_rt_correlations(
    four_runs, settings_for
):
    root, _ = four_runs
    for path in (root / "sub-07" / "ses-nsd10" / "func").glob("*_events.tsv"):
        pd.read_csv(path, sep="\t").drop(columns="response_time").to_csv(
            path, sep="\t", index=False
        )
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=2)
    result = analysis.fit_beta_series(runs, root, blocks, task_model=model)
    assert result["rt"] is None and len(result["betas"]) == 4


def test_select_hrfs_without_splits_selects_on_all_runs_only(
    dataset, four_runs, small_library, settings_for
):
    root, *_ = dataset
    runs = inputs.load_session(settings_for(root), hrf_only=True)
    model_task = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=2, max_grayordinates=4)
    alone = analysis.select_hrfs(
        runs, root, blocks, small_library, task_model=model_task, splits=False
    )
    assert alone and all(set(bundle) == {"all"} for bundle in alone.values())
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    full = analysis.select_hrfs(
        runs, root, blocks, small_library, task_model=model_task
    )
    alone = analysis.select_hrfs(
        runs, root, blocks, small_library, task_model=model_task, splits=False
    )
    for key, bundle in alone.items():
        assert set(bundle) == {"all"}
        np.testing.assert_array_equal(
            bundle["all"].hrf_indices, full[key]["all"].hrf_indices
        )
