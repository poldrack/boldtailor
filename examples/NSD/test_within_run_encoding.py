"""Outer encoding and published losses preserve the selected scientific mode."""

import json

import nibabel as nib
import numpy as np
import pytest

from boldtailor.publication import publish_artifact_set
from boldtailor.single_trial import fit_single_trials
from boldtailor.workflow.beta_series import fit_cv_beta_series, trial_predictors
from boldtailor.workflow.outputs import ridge_artifacts
from boldtailor.workflow.inputs import load_session, load_block


@pytest.mark.parametrize("mode", ["within_run", "absolute"])
@pytest.mark.parametrize("fractional", [False, True])
def test_outer_oracle_blocks_and_export(
    six_run_dataset, cv_library, tmp_path, mode, fractional
):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    grid = dict(fractions=[1, 0.5]) if fractional else dict(alphas=[0, 0.1])
    options = dict(library=None, encoding_mode=mode, **grid)
    result = fit_cv_beta_series(runs, root, block_size=2, **options)
    blocked = fit_cv_beta_series(runs, root, block_size=3, **options)
    assert result["provenance"]["encoding_mode"] == mode
    decision = result["tuning"]["all"]["provenance"].to_dict()["activities"][-1]
    assert decision["encoding_mode"] == mode
    assert (
        result["final"]["provenance"][0]["record"]["activities"][-1]["encoding_mode"]
        == mode
    )
    data = load_block(runs, root, np.arange(4))
    predictors = trial_predictors(runs)
    for scope, train, test in [
        ("odd_to_even", [0, 2, 4], [1, 3, 5]),
        ("even_to_odd", [1, 3, 5], [0, 2, 4]),
    ]:
        outer = result["evaluation"][scope]
        assert outer["encoding_mode"] == mode
        for key in [
            "coefficients",
            "train_run_intercepts",
            "scoring_offsets",
            "encoding_r2",
            "train_run_predictor_means",
        ]:
            np.testing.assert_allclose(
                outer[key], blocked["evaluation"][scope][key], atol=1e-9
            )
        fit_options = (
            {"ridge_fraction": outer["ridge_fraction"]}
            if fractional
            else {"ridge_alpha": outer["ridge_alpha"]}
        )
        fitted = fit_single_trials(data, **fit_options)
        # Fractional CV now scores fixed OLS targets; alpha CV keeps fitted targets.
        target_fit = fit_single_trials(data) if fractional else fitted
        xs = [predictors[r].to_numpy() for r in train]
        ys = [fitted.run_betas[r][:, :3] for r in train]
        ids = np.repeat(np.arange(len(train)), [len(x) for x in xs])
        intercepts = (
            np.eye(len(train))[ids] if mode == "within_run" else np.ones((len(ids), 1))
        )
        oracle = np.linalg.lstsq(
            np.column_stack([intercepts, np.vstack(xs)]), np.vstack(ys), rcond=None
        )[0]
        np.testing.assert_allclose(
            outer["coefficients"][1:, :3], oracle[-2:], atol=1e-8
        )
        expected_intercepts = (
            oracle[:-2]
            if mode == "within_run"
            else np.repeat(oracle[:1], len(train), axis=0)
        )
        np.testing.assert_allclose(
            outer["train_run_intercepts"][:, :3], expected_intercepts, atol=1e-8
        )
        for j, r in enumerate(test):
            raw = (
                np.vstack(ys).mean(0)
                + (predictors[r].to_numpy() - np.vstack(xs).mean(0)) @ oracle[-2:]
            )
            np.testing.assert_allclose(outer["predictions"][j][:, :3], raw, atol=1e-6)
            target = target_fit.run_betas[r][:, :3]
            np.testing.assert_allclose(outer["targets"][j][:, :3], target, atol=1e-6)
            residual = target - raw
            offset = residual.mean(0) if mode == "within_run" else np.zeros(3)
            np.testing.assert_allclose(
                outer["scoring_offsets"][j, :3], offset, atol=1e-8
            )
            np.testing.assert_allclose(
                outer["run_sse"][j, :3],
                np.sum((residual - offset) ** 2, axis=0),
                atol=1e-8,
            )
    paths = publish_artifact_set(
        tmp_path / "exports",
        ridge_artifacts(
            "sub-07/ses-nsd10/func/test",
            runs[0].image.header.get_axis(1),
            runs,
            {"Canonical": result},
            cv_library,
        ),
    )
    kind = "FractionalCV" if fractional else "RidgeCV"
    base = f"Canonical{kind}OddToEven"
    metadata = json.loads(
        next(p for p in paths if base + "_metadata.json" in p.name).read_text()
    )
    assert metadata["encoding_mode"] == mode
    assert (
        metadata["prediction_reference"]
        == "pooled_training_beta_mean_at_pooled_predictor_mean"
    )
    with np.load(next(p for p in paths if base + "_loss.npz" in p.name)) as saved:
        outer = result["evaluation"]["odd_to_even"]
        np.testing.assert_allclose(
            saved["train_run_intercepts"], outer["train_run_intercepts"]
        )
        np.testing.assert_allclose(saved["scoring_offsets"], outer["scoring_offsets"])
        assert saved["train_run_labels"].tolist() == outer["train_run_labels"]
        assert saved["test_run_labels"].tolist() == outer["test_run_labels"]
        for j, label in enumerate(outer["test_run_labels"]):
            run = next(r for r in runs if r.label == label)
            prefix = run.inputs.stem + "_space-fsLR_den-91k_desc-notebook" + base
            observed = nib.load(
                next(p for p in paths if p.name == prefix + "_targets.dscalar.nii")
            ).get_fdata()
            predicted = nib.load(
                next(p for p in paths if p.name == prefix + "_predictions.dscalar.nii")
            ).get_fdata()
            residual = observed - predicted - saved["scoring_offsets"][j]
            np.testing.assert_allclose(
                np.sum(residual**2, axis=0), saved["sse"][j], rtol=1e-5, atol=1e-6
            )
