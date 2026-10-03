"""Published outer losses reproduce the selected encoding mode's scores."""

import json

import nibabel as nib
import numpy as np
import pytest

from boldtailor.publication import publish_artifact_set
from boldtailor.workflow.analysis import select_hrfs
from boldtailor.workflow.beta_series import fit_beta_models
from boldtailor.workflow.inputs import detect_task_model, load_session, make_blocks
from boldtailor.workflow.outputs import beta_model_artifacts
from examples.NSD.nsd_settings import nsd_settings


def _fitted_models(root, prep, library, mode, fractional):
    grid = dict(ridge_fractions=[1, 0.5]) if fractional else dict(ridge_alphas=[0, 0.1])
    settings = nsd_settings(
        dict(bids_root=root, fmriprep_root=prep, subject="sub-07"),
        session="ses-nsd10",
        ridge_mode="fractional_cv" if fractional else "cv",
        encoding_mode=mode,
        block_size=2,
        n_jobs=1,
        **grid,
    )
    runs = load_session(settings)
    model = detect_task_model([r.events for r in runs])
    blocks = make_blocks(runs, block_size=2)
    selections = select_hrfs(runs, root, blocks, library, task_model=model)
    models = fit_beta_models(runs, root, blocks, settings, library, selections, model)
    return settings, runs, models


@pytest.mark.parametrize("mode", ["within_run", "absolute"])
@pytest.mark.parametrize("fractional", [False, True])
def test_outer_losses_and_maps_are_published_consistently(
    six_run_dataset, cv_library, tmp_path, mode, fractional
):
    root, prep = six_run_dataset
    settings, runs, models = _fitted_models(root, prep, cv_library, mode, fractional)
    kind = "FractionalCV" if fractional else "RidgeCV"
    model = models["CanonicalTrial" + kind]
    brain = runs[0].image.header.get_axis(1)
    paths = publish_artifact_set(
        tmp_path / "exports",
        beta_model_artifacts(settings, brain, runs, model, cv_library),
    )
    base = f"Canonical{kind}OddToEven"
    metadata = json.loads(
        next(p for p in paths if base + "_metadata.json" in p.name).read_text()
    )
    assert metadata["encoding_mode"] == mode
    assert (
        metadata["prediction_reference"]
        == "pooled_training_beta_mean_at_pooled_predictor_mean"
    )
    outer = model.evaluation["odd_to_even"]
    with np.load(next(p for p in paths if base + "_loss.npz" in p.name)) as saved:
        np.testing.assert_allclose(
            saved["train_run_intercepts"], outer["train_run_intercepts"]
        )
        np.testing.assert_allclose(saved["scoring_offsets"], outer["scoring_offsets"])
        assert saved["train_run_labels"].tolist() == outer["train_run_labels"]
        assert saved["test_run_labels"].tolist() == outer["test_run_labels"]
        for j, label in enumerate(outer["test_run_labels"]):
            run = next(r for r in runs if r.label == label)
            prefix = f"{run.inputs.stem}_{settings.space_entity}_desc-{base}"
            observed, predicted = (
                nib.load(
                    next(p for p in paths if p.name == f"{prefix}_{suffix}.dscalar.nii")
                ).get_fdata()
                for suffix in ("targets", "predictions")
            )
            residual = observed - predicted - saved["scoring_offsets"][j]
            np.testing.assert_allclose(
                np.sum(residual**2, axis=0), saved["sse"][j], rtol=1e-5, atol=1e-6
            )
