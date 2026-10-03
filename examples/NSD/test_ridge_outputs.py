"""Published ridge diagnostics identify the objective and retain CIFTI order."""

import importlib
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.publication import publish_artifact_set
from examples.NSD.ridge_workflow import fit_cv_beta_series
from examples.NSD.workflow_inputs import load_session


def outputs():
    try:
        return importlib.import_module("examples.NSD.ridge_outputs")
    except ImportError as error:
        pytest.fail(f"Missing ridge output support: {error}")


def test_ridge_artifacts_match_numeric_results(six_run_dataset, cv_library, tmp_path):
    writer = outputs()
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    result = fit_cv_beta_series(
        runs, root, library=None, alphas=[0.0, 0.1], block_size=2
    )
    stem = "sub-07/ses-nsd10/func/sub-07_ses-nsd10_task-nsdcore"
    artifacts = writer.ridge_artifacts(
        stem, runs[0].image.header.get_axis(1), runs, {"Canonical": result}, cv_library
    )
    output = tmp_path / "published"
    sentinel = output / "old_fixed_ridge.txt"
    output.mkdir()
    sentinel.write_text("retain prior fixed-alpha outputs")
    paths = publish_artifact_set(output, artifacts)
    for scope in ("Odd", "Even", "All"):
        selected = result["tuning"][scope.lower()]
        base = f"desc-notebookCanonicalRidgeCV{scope}"
        path = next(p for p in paths if f"{base}_stat-encodingcvr2." in p.name)
        image = nib.load(path)
        assert image.header.get_axis(1) == runs[0].image.header.get_axis(1)
        np.testing.assert_allclose(
            image.get_fdata(), selected["scores"].cv_r2, rtol=1e-6, atol=1e-6
        )
        scores = pd.read_csv(
            next(p for p in paths if f"{base}_scores.tsv" in p.name), sep="\t"
        )
        np.testing.assert_allclose(
            scores.objective, selected["selection"].objective_scores
        )
        assert (
            scores.loc[scores.selected, "alpha"].item()
            == selected["selection"].ridge_alpha
        )
        with np.load(next(p for p in paths if f"{base}_folds.npz" in p.name)) as arrays:
            np.testing.assert_allclose(arrays["sse"], selected["scores"].fold_sse)
            np.testing.assert_allclose(arrays["sst"], selected["scores"].fold_sst)
        record = json.loads(
            next(p for p in paths if f"{base}_provenance.json" in p.name).read_text()
        )
        assert (
            record["analysis_fingerprint"]
            == selected["provenance"].analysis_fingerprint
        )
        assert (
            record["activities"][-1]["selected_alpha"]
            == selected["selection"].ridge_alpha
        )
    for split in ("OddToEven", "EvenToOdd"):
        base = f"desc-notebookCanonicalRidgeCV{split}"
        metadata = json.loads(
            next(p for p in paths if f"{base}_metadata.json" in p.name).read_text()
        )
        assert not set(metadata["train_run_labels"]) & set(metadata["test_run_labels"])
        assert metadata["validation_target"] == "selected_penalty_regularized_betas"
        assert metadata["predictor_names"] == ["task", "trial_type", "response_time"]
    predictors = pd.read_csv(
        next(p for p in paths if p.name.endswith("RidgeCV_predictors.tsv")), sep="\t"
    )
    assert len(predictors) == 36 and predictors.trial_id.is_unique
    assert len([p for p in paths if p.name.endswith("_predictions.dscalar.nii")]) == 6
    assert len([p for p in paths if p.name.endswith("_targets.dscalar.nii")]) == 6
    assert any(p.name.endswith("_tuning.png") for p in paths)
    assert sentinel.read_text() == "retain prior fixed-alpha outputs"
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(FileExistsError):
        publish_artifact_set(output, artifacts)
    assert all(p.read_bytes() == original for p, original in before.items())
