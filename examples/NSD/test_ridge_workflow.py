"""CIFTI ridge CV reduces across geometry, with independent outer evaluation."""

from dataclasses import replace
import importlib

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary
from boldtailor.ridge_selection import score_ridge_candidates, select_ridge_penalty
from boldtailor.hrf_selection import select_hrf
from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
from boldtailor.trial_encoding import evaluate_trial_encoding
from examples.NSD.nsd_hrf import spatial_signature
from examples.NSD.workflow_inputs import NSD_TASK_MODEL, load_session, load_block
from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401


def workflow():
    try:
        return importlib.import_module("examples.NSD.ridge_workflow")
    except ImportError as error:
        pytest.fail(f"Missing NSD ridge CV workflow: {error}")


@pytest.fixture
def six_run_dataset(dataset):
    root, prep, *_ = dataset
    for directory in (root, prep):
        func = directory / "sub-07/ses-nsd10/func"
        sources = list(func.glob("*run-01*"))
        for number in (3, 4, 5, 6):
            for source in sources:
                target = source.with_name(
                    source.name.replace("run-01", f"run-{number:02d}")
                )
                target.write_bytes(source.read_bytes())
    rng = np.random.default_rng(75)
    for number in range(1, 7):
        path = next(prep.rglob(f"*run-{number:02d}*dtseries.nii"))
        image = nib.load(path)
        y = image.get_fdata()
        y[:, :3] += rng.normal(0, 0.3, y[:, :3].shape)
        nib.save(nib.Cifti2Image(y, header=image.header), path)
        event_path = next((root / "sub-07").rglob(f"*run-{number:02d}*events.tsv"))
        table = pd.read_csv(event_path, sep="\t")
        table.response_time += 0.03 * number
        table["stimulus_id"] = np.arange(len(table)) + number * 100
        table.to_csv(event_path, sep="\t", index=False)
    return root, prep


@pytest.fixture
def cv_library():
    return HrfLibrary.from_parameters([[4, 12, 0.8, 1, 5, 0, 36]])


def options(library, **extra):
    return dict(library=library, alphas=[0.0, 0.1, 1.0], block_size=2, **extra)


def assert_candidate_equal(one, two):
    assert one["selection"].ridge_alpha == two["selection"].ridge_alpha
    np.testing.assert_array_equal(
        one["selection"].scoring_mask, two["selection"].scoring_mask
    )
    np.testing.assert_allclose(one["scores"].cv_r2, two["scores"].cv_r2, atol=1e-10)
    np.testing.assert_array_equal(
        one["scores"].fold_hrf_indices, two["scores"].fold_hrf_indices
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_workflow_matches_whole_array_reference(six_run_dataset, cv_library, optimized):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    library = cv_library if optimized else None
    result = workflow().fit_cv_beta_series(runs, root, **options(library))
    for scope, indices in (
        ("odd", [0, 2, 4]),
        ("even", [1, 3, 5]),
        ("all", list(range(6))),
    ):
        training = [runs[i] for i in indices]
        data = load_block(training, root, np.arange(4))
        predictors = [r.events[["trial_type", "response_time"]] for r in training]
        expected = score_ridge_candidates(
            data, predictors, library=library, alphas=[0.0, 0.1, 1.0]
        )
        np.testing.assert_allclose(
            result["tuning"][scope]["scores"].cv_r2, expected.cv_r2, atol=1e-10
        )
        assert (
            result["tuning"][scope]["selection"].ridge_alpha
            == select_ridge_penalty(expected.cv_r2, expected.grid).ridge_alpha
        )
    data = load_block(runs, root, np.arange(4))
    final_alpha = result["tuning"]["all"]["selection"].ridge_alpha
    signature = spatial_signature(runs[0].image.header.get_axis(1), np.arange(4))
    if library is None:
        expected_fit = fit_single_trials(data, ridge_alpha=final_alpha)
    else:
        selection = select_hrf(
            data,
            library=library,
            feature_signature=signature,
            task_model=NSD_TASK_MODEL,
        )
        expected_fit = fit_selected_hrfs(
            data,
            selection=selection,
            ridge_alpha=final_alpha,
            feature_signature=signature,
        )
    for actual, expected in zip(result["final"]["betas"], expected_fit.run_betas):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        assert np.isnan(actual[:, -1]).all()
    for name, train, test in (
        ("odd_to_even", [0, 2, 4], [1, 3, 5]),
        ("even_to_odd", [1, 3, 5], [0, 2, 4]),
    ):
        outer = result["evaluation"][name]
        assert outer["train_run_labels"] == [runs[i].label for i in train]
        assert outer["test_run_labels"] == [runs[i].label for i in test]
        alpha = result["tuning"]["odd" if train[0] == 0 else "even"][
            "selection"
        ].ridge_alpha
        if library is None:
            fitted = fit_single_trials(data, ridge_alpha=alpha)
        else:
            train_data = load_block([runs[i] for i in train], root, np.arange(4))
            selected = select_hrf(
                train_data,
                library=library,
                feature_signature=signature,
                task_model=NSD_TASK_MODEL,
            )
            fitted = fit_selected_hrfs(
                data, selection=selected, ridge_alpha=alpha, feature_signature=signature
            )
        reference = evaluate_trial_encoding(
            fitted.run_betas,
            [r.events[["trial_type", "response_time"]] for r in runs],
            train_runs=train,
            test_runs=test,
        )
        for saved, r in zip(outer["betas"], test):
            np.testing.assert_allclose(saved, fitted.run_betas[r], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(outer["encoding_r2"], reference.r2, atol=1e-6)
        for saved, predicted in zip(outer["predictions"], reference.predictions):
            np.testing.assert_allclose(saved, predicted, atol=1e-6)


def test_block_size_and_parallelism_preserve_global_choice(six_run_dataset, cv_library):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    serial = workflow().fit_cv_beta_series(runs, root, **options(cv_library))
    parallel = workflow().fit_cv_beta_series(
        runs, root, library=cv_library, alphas=[0.0, 0.1, 1.0], block_size=3, n_jobs=2
    )
    for scope in ("odd", "even", "all"):
        assert_candidate_equal(serial["tuning"][scope], parallel["tuning"][scope])
    for a, b in zip(serial["final"]["betas"], parallel["final"]["betas"]):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "changed_parity,scope,evaluation",
    [(0, "odd", "odd_to_even"), (1, "even", "even_to_odd")],
)
def test_outer_data_cannot_change_training_choices(
    six_run_dataset, cv_library, changed_parity, scope, evaluation
):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    original = workflow().fit_cv_beta_series(runs, root, **options(cv_library))
    changed = []
    for run in runs:
        if run.number % 2 == changed_parity:
            y = np.random.default_rng(run.number).normal(size=run.image.shape)
            image = nib.Cifti2Image(y, header=run.image.header)
            event = run.events.assign(response_time=np.arange(len(run.events)) + 10.0)
            changed.append(replace(run, image=image, events=event))
        else:
            changed.append(run)
    altered = workflow().fit_cv_beta_series(changed, root, **options(cv_library))
    assert_candidate_equal(original["tuning"][scope], altered["tuning"][scope])
    for key in ("coefficients", "predictor_means", "hrf_indices"):
        np.testing.assert_allclose(
            original["evaluation"][evaluation][key],
            altered["evaluation"][evaluation][key],
            atol=1e-10,
        )
    assert not np.allclose(
        original["evaluation"][evaluation]["encoding_r2"][:3],
        altered["evaluation"][evaluation]["encoding_r2"][:3],
    )


def test_missing_rt_keeps_beta_rows_and_requested_axis(six_run_dataset):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    altered = []
    for run in runs:
        e = run.events.copy()
        e.loc[0, "response_time"] = np.nan
        e.loc[1, "response_time"] = -1.0
        altered.append(replace(run, events=e))
    result = workflow().fit_cv_beta_series(
        altered, root, **options(None, max_grayordinates=2)
    )
    assert all(b.shape == (6, 4) for b in result["final"]["betas"])
    assert all(np.isnan(b[:, 2:]).all() for b in result["final"]["betas"])
    assert all(
        mask.tolist() == [False, False, True, True, True, True]
        for mask in result["tuning"]["all"]["scores"].trial_masks
    )


@pytest.mark.parametrize(
    "case", ["fewruns", "type", "axis", "block", "maximum", "jobs"]
)
def test_workflow_preflight_rejects_invalid_inputs(six_run_dataset, cv_library, case):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    kwargs = options(cv_library)
    if case == "fewruns":
        runs = runs[:4]
    elif case == "type":
        runs[0] = replace(runs[0], events=runs[0].events.assign(trial_type=3))
    elif case == "axis":
        brain = nib.cifti2.BrainModelAxis.from_surface(
            np.array([0, 1, 3, 6]), 8, name="CortexLeft"
        )
        header = nib.cifti2.Cifti2Header.from_axes(
            [runs[0].image.header.get_axis(0), brain]
        )
        runs[0] = replace(
            runs[0], image=nib.Cifti2Image(runs[0].image.get_fdata(), header=header)
        )
    elif case == "block":
        kwargs["block_size"] = 0
    elif case == "maximum":
        kwargs["max_grayordinates"] = -1
    else:
        kwargs["n_jobs"] = True
    with pytest.raises(ValueError):
        workflow().fit_cv_beta_series(runs, root, **kwargs)


def test_final_provenance_identifies_the_tuning_decision(six_run_dataset):
    root, prep = six_run_dataset
    runs = load_session(root, prep)

    def fit(run_set, **overrides):
        kwargs = dict(library=None, alphas=[1.0], percentile=90, block_size=2)
        kwargs.update(overrides)
        return workflow().fit_cv_beta_series(run_set, root, **kwargs)

    original = fit(runs)
    repeated = fit(runs)
    changed_predictors = [
        replace(r, events=r.events.assign(response_time=r.events.response_time + 1))
        for r in runs
    ]
    alternatives = [
        fit(runs, percentile=80),
        fit(runs, alphas=[0.0, 1.0]),
        fit(changed_predictors),
        fit(runs, max_grayordinates=2),
    ]
    records = []
    for result in (original, repeated, *alternatives):
        tuned = result["tuning"]["all"]
        assert "provenance" in tuned, "The global decision needs identified provenance"
        decision = tuned["provenance"]
        assert decision.analysis_fingerprint
        activity = decision.to_dict()["activities"][-1]
        assert activity["percentile"] == tuned["selection"].percentile
        assert activity["selected_alpha"] == tuned["selection"].ridge_alpha
        assert activity["scoring_mask_fingerprint"]
        assert activity["validation_target"] == "candidate_regularized_betas"
        final_record = result["final"]["provenance"][0]["record"]
        link = final_record["activities"][-1]
        assert link["name"] == "encoding_guided_ridge_refit"
        assert link["tuning_analysis_fingerprint"] == decision.analysis_fingerprint
        assert link["tuning_execution_id"] == decision.execution_id
        records.append(
            (decision.analysis_fingerprint, final_record["analysis_fingerprint"])
        )
    assert records[0] == records[1], "Execution UUIDs must not change analysis identity"
    assert all(record != records[0] for record in records[2:])
    # Percentile-only changes still select alpha 1; their final provenance differs.
    assert (
        original["final"]["ridge_alpha"] == alternatives[0]["final"]["ridge_alpha"] == 1
    )


def test_rt_switch_reaches_every_ridge_selection(six_run_dataset, cv_library):
    from examples.NSD.workflow_inputs import selection_task_model

    root, prep = six_run_dataset
    runs = load_session(root, prep)
    narrow = selection_task_model(False)
    result = workflow().fit_cv_beta_series(
        runs, root, **options(cv_library), task_model=narrow
    )
    finals = result["final"]["hrf_selections"]
    assert finals and all(b["all"].task_model == narrow for b in finals.values())
    for evaluation in result["evaluation"].values():
        assert evaluation["selection_task_model"] == narrow.to_dict()
