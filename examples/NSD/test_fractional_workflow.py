"""The NSD adapter preserves featurewise decisions through fitting and export."""

from dataclasses import replace
import json

import nibabel as nib
import numpy as np
import pytest

from boldtailor.fractional_ridge import (
    score_fraction_candidates,
    select_ridge_fractions,
)
from boldtailor.hrf_selection import select_hrfs
from boldtailor.trial_encoding import evaluate_trial_encoding
from boldtailor.publication import publish_artifact_set
from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
from examples.NSD.ridge_workflow import fit_cv_beta_series, trial_predictors
from examples.NSD.ridge_outputs import ridge_artifacts, tuning_table, tuning_figure
from examples.NSD.workflow_outputs import _beta_artifacts
from examples.NSD.workflow_inputs import NSD_TASK_MODEL, load_session, load_block


def fit(runs, root, library=None, **kwargs):
    return fit_cv_beta_series(
        runs, root, library=library, fractions=[0.3, 0.7, 1], **kwargs
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_fraction_workflow_matches_whole_array_and_exports(
    six_run_dataset, cv_library, tmp_path, optimized
):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    library = cv_library if optimized else None
    result = fit(runs, root, library, block_size=1)
    for scope, indices in [
        ("odd", [0, 2, 4]),
        ("even", [1, 3, 5]),
        ("all", list(range(6))),
    ]:
        subset = [runs[r] for r in indices]
        scores = score_fraction_candidates(
            load_block(subset, root, np.arange(4)),
            trial_predictors(subset),
            library=library,
            fractions=[0.3, 0.7, 1],
        )
        expected = select_ridge_fractions(scores.cv_r2, scores.grid)
        actual = result["tuning"][scope]["selection"]
        np.testing.assert_allclose(actual.ridge_fraction, expected.ridge_fraction)
        np.testing.assert_allclose(actual.selected_r2, expected.selected_r2, atol=1e-10)
    data = load_block(runs, root, np.arange(4))
    # Outer scores use OLS targets with HRFs learned only on outer training runs.
    train, test = [0, 2, 4], [1, 3, 5]
    chosen = result["tuning"]["odd"]["selection"].ridge_fraction
    outer_selection = (
        None
        if library is None
        else select_hrfs(
            load_block([runs[r] for r in train], root, np.arange(4)),
            library=library,
            task_model=NSD_TASK_MODEL,
        )
    )
    fitter = fit_single_trials if outer_selection is None else fit_selected_hrfs
    kwargs = {} if outer_selection is None else {"hrf_selection": outer_selection}
    shrunk = fitter(data, ridge_fraction=chosen, **kwargs)
    ols = fitter(
        data, ridge_fraction=np.where(np.isfinite(chosen), 1.0, np.nan), **kwargs
    )
    mixed = [ols.run_betas[r] if r in test else shrunk.run_betas[r] for r in range(6)]
    outer = evaluate_trial_encoding(
        mixed, trial_predictors(runs), train_runs=train, test_runs=test
    )
    actual_outer = result["evaluation"]["odd_to_even"]
    for target_beta, r in zip(actual_outer["targets"], test):
        np.testing.assert_allclose(target_beta, ols.run_betas[r], atol=2e-6)
    np.testing.assert_allclose(actual_outer["run_sse"], outer.run_sse, atol=1e-8)
    np.testing.assert_allclose(actual_outer["run_sst"], outer.run_sst, atol=1e-8)
    for actual_beta, r in zip(actual_outer["betas"], test):
        np.testing.assert_allclose(actual_beta, shrunk.run_betas[r], atol=1e-8)
    fractions = result["tuning"]["all"]["selection"].ridge_fraction
    selection = (
        None
        if library is None
        else select_hrfs(data, library=library, task_model=NSD_TASK_MODEL)
    )
    expected = (
        fit_single_trials(data, ridge_fraction=fractions)
        if selection is None
        else fit_selected_hrfs(data, hrf_selection=selection, ridge_fraction=fractions)
    )
    for beta, reference in zip(result["final"]["betas"], expected.run_betas):
        np.testing.assert_allclose(beta, reference, atol=2e-6)
    np.testing.assert_allclose(
        result["final"]["run_ridge_alphas"], expected.run_ridge_alphas, atol=1e-9
    )
    assert result["final"]["ridge_alpha"] is None
    assert np.isnan(fractions[-1])
    assert np.isnan(result["final"]["run_ridge_alphas"][:, -1]).all()
    decision = result["tuning"]["all"]["provenance"].analysis_fingerprint
    assert decision
    assert (
        result["final"]["provenance"][0]["record"]["activities"][-1][
            "tuning_analysis_fingerprint"
        ]
        == decision
    )
    mode = "Optimized" if optimized else "Canonical"
    stem = "sub-07/ses-nsd10/func/sub-07_ses-nsd10_task-nsdcore"
    brain = runs[0].image.header.get_axis(1)
    artifacts = ridge_artifacts(stem, brain, runs, {mode: result}, cv_library)
    artifacts += _beta_artifacts(
        stem, brain, {f"{mode}TrialFractionalCV": result["final"]}, runs
    )
    paths = publish_artifact_set(tmp_path / "outputs", artifacts)
    for scope in ("Odd", "Even", "All"):
        path = next(
            p
            for p in paths
            if f"{mode}FractionalCV{scope}_stat-ridgefraction." in p.name
        )
        image = nib.load(path)
        assert image.header.get_axis(1) == brain
        np.testing.assert_allclose(
            image.get_fdata()[0],
            result["tuning"][scope.lower()]["selection"].ridge_fraction,
        )
    for scope in ("odd_to_even", "even_to_odd"):
        outer = result["evaluation"][scope]
        descriptor = (
            mode + "FractionalCV" + "".join(w.title() for w in scope.split("_"))
        )
        for label, targets in zip(outer["test_run_labels"], outer["targets"]):
            run = next(r for r in runs if r.label == label)
            path = next(
                p
                for p in paths
                if run.inputs.stem in p.name
                and descriptor in p.name
                and p.name.endswith("_targets.dscalar.nii")
            )
            np.testing.assert_allclose(nib.load(path).get_fdata(), targets, atol=1e-6)
    alpha_path = next(
        p for p in paths if "TrialFractionalCV_stat-ridgealpha." in p.name
    )
    np.testing.assert_allclose(
        nib.load(alpha_path).get_fdata(), expected.run_ridge_alphas, atol=1e-7
    )
    metadata = json.loads(
        next(
            p for p in paths if f"{mode}FractionalCVAll_metadata.json" in p.name
        ).read_text()
    )
    assert metadata["selection_rule"] == "maximum_encoding_r2_per_grayordinate"
    assert metadata["validation_target"] == "fixed_ols_betas"
    assert metadata["percentile_role"] == "descriptive_only"
    table = tuning_table({mode: result})
    assert table.groupby("scope").selected_grayordinates.sum().tolist() == [3, 3, 3]
    assert "alpha" not in table
    assert tuning_figure({mode: result}).axes[0].get_xlabel() == "Ridge fraction"


def test_fraction_blocks_workers_and_missing_rt_preserve_results(
    six_run_dataset, cv_library
):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    runs = [
        replace(
            r,
            events=r.events.assign(
                response_time=r.events.response_time.mask(r.events.index == 0)
            ),
        )
        for r in runs
    ]
    serial = fit(runs, root, cv_library, block_size=1)
    parallel = fit(runs, root, cv_library, block_size=3, n_jobs=2)
    for scope in ("odd", "even", "all"):
        np.testing.assert_allclose(
            serial["tuning"][scope]["selection"].ridge_fraction,
            parallel["tuning"][scope]["selection"].ridge_fraction,
        )
    for a, b in zip(serial["final"]["betas"], parallel["final"]["betas"]):
        assert a.shape == (6, 4)
        np.testing.assert_allclose(a, b, atol=1e-6)
    assert all(not mask[0] for mask in serial["tuning"]["all"]["scores"].trial_masks)


@pytest.mark.parametrize("parity,scope", [(0, "odd_to_even"), (1, "even_to_odd")])
def test_outer_fraction_choices_and_training_coefficients_are_isolated(
    six_run_dataset, cv_library, parity, scope
):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    initial = fit(runs, root, cv_library, block_size=2)
    changed = []
    for run in runs:
        if run.number % 2 == parity:
            y = np.random.default_rng(run.number).normal(size=run.image.shape)
            changed.append(
                replace(
                    run,
                    image=nib.Cifti2Image(y, header=run.image.header),
                    events=run.events.assign(
                        response_time=run.events.response_time + 100
                    ),
                )
            )
        else:
            changed.append(run)
    altered = fit(changed, root, cv_library, block_size=2)
    for key in ("ridge_fraction", "hrf_indices", "coefficients", "predictor_means"):
        np.testing.assert_allclose(
            initial["evaluation"][scope][key],
            altered["evaluation"][scope][key],
            atol=1e-10,
        )
    assert not np.allclose(
        initial["evaluation"][scope]["encoding_r2"],
        altered["evaluation"][scope]["encoding_r2"],
        equal_nan=True,
    )


def test_fraction_and_alpha_grids_are_mutually_exclusive(six_run_dataset):
    root, prep = six_run_dataset
    with pytest.raises(ValueError):
        fit(load_session(root, prep), root, alphas=[0, 1])
