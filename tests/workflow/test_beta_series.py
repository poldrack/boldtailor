"""CV beta series, trial predictors, tuning provenance and BetaModel assembly."""

from dataclasses import replace
import logging

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor._software import software_environment
from boldtailor.cifti import spatial_signature
from boldtailor.data import from_arrays
from boldtailor.fractional_ridge import (
    score_fraction_candidates,
    select_ridge_fractions,
)
from boldtailor.hrf_selection import select_hrfs
from boldtailor.model import Modulator, TaskModel
from boldtailor.ridge_selection import score_ridge_candidates, select_ridge_penalty
from boldtailor.single_trial import fit_selected_hrfs, fit_single_trials
from boldtailor.trial_encoding import evaluate_trial_encoding
from boldtailor.workflow import analysis, beta_series, inputs
from tests.oracles import glm_run_sources

ORDER = ["response_time", "trial_type[1]"]


def encoded(events):
    """The hand-built encoding predictors: RT and the trial_type[1] indicator."""
    table = events[["response_time", "trial_type"]].astype(float)
    return table.rename(columns={"trial_type": "trial_type[1]"})


@pytest.fixture
def session(six_run_dataset, settings_for):
    root, _ = six_run_dataset
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    return root, list(runs), model


def options(library, **extra):
    return dict(library=library, alphas=[0.0, 0.1, 1.0], block_size=2, **extra)


def fit(session, library=None, **kwargs):
    root, runs, model = session
    return beta_series.fit_cv_beta_series(
        runs, root, task_model=model, library=library, **kwargs
    )


def fit_alpha(session, library, **extra):
    return fit(session, **options(library, **extra))


def fit_fractions(session, library=None, **kwargs):
    return fit(session, library, fractions=[0.3, 0.7, 1], **kwargs)


def assert_candidate_equal(one, two):
    assert one["selection"].ridge_alpha == two["selection"].ridge_alpha
    np.testing.assert_array_equal(
        one["selection"].scoring_mask, two["selection"].scoring_mask
    )
    np.testing.assert_allclose(one["scores"].cv_r2, two["scores"].cv_r2, atol=1e-10)
    np.testing.assert_array_equal(
        one["scores"].fold_hrf_indices, two["scores"].fold_hrf_indices
    )


def altered_runs(runs, parity, rt_shift):
    changed = []
    for run in runs:
        if run.number % 2 != parity:
            changed.append(run)
            continue
        y = np.random.default_rng(run.number).normal(size=run.image.shape)
        image = nib.Cifti2Image(y, header=run.image.header)
        events = run.events.assign(response_time=run.events.response_time + rt_shift)
        changed.append(replace(run, image=image, events=events))
    return changed


# --- trial_predictors -------------------------------------------------------


def test_trial_predictors_select_modulator_columns_in_model_order(session):
    _, runs, model = session
    tables = beta_series.trial_predictors(runs, model)
    assert all(list(t.columns) == ORDER for t in tables)
    assert all(len(t) == len(r.events) for t, r in zip(tables, runs))


def test_trial_predictors_follow_the_task_model(session):
    _, runs, model = session
    narrow = inputs.selection_task_model(model, False)
    tables = beta_series.trial_predictors(runs, narrow)
    assert all(list(t.columns) == ["trial_type[1]"] for t in tables)


def test_trial_predictors_make_nonpositive_or_missing_rt_nan(session):
    _, runs, model = session
    events = runs[0].events.copy()
    events.loc[0, "response_time"] = -1.0
    events.loc[1, "response_time"] = 0.0
    events.loc[2, "response_time"] = np.nan
    table = beta_series.trial_predictors([replace(runs[0], events=events)], model)[0]
    assert table.response_time.iloc[:3].isna().all()
    assert table.response_time.iloc[3:].notna().all()


def test_trial_predictors_leave_other_columns_untouched():
    events = pd.DataFrame(dict(effort=[-1.0, 0.0, 2.0]))
    run = type("Run", (), dict(label="run-01", events=events))()
    model = TaskModel((Modulator("effort"),))
    table = beta_series.trial_predictors([run], model)[0]
    assert table.effort.tolist() == [-1.0, 0.0, 2.0]


def test_trial_predictors_reject_missing_column(session):
    _, runs, model = session
    broken = replace(runs[0], events=runs[0].events.drop(columns="response_time"))
    with pytest.raises(ValueError, match="response_time"):
        beta_series.trial_predictors([broken], model)


def test_trial_predictors_reject_nonbinary_trial_type(session):
    _, runs, model = session
    broken = replace(runs[0], events=runs[0].events.assign(trial_type=3))
    with pytest.raises(ValueError, match="trial_type"):
        beta_series.trial_predictors([broken], model)


# --- shared-alpha CV --------------------------------------------------------


def test_trial_predictors_emit_indicator_columns_with_missing_as_nan(session):
    _, runs, _ = session
    mod = Modulator(
        "trial_type", kind="categorical", levels=("0", "1"), missing="indicator"
    )
    first = runs[0].events.copy()
    first.loc[first.index[0], "trial_type"] = np.nan
    runs = [replace(runs[0], events=first), *runs[1:]]
    tables = beta_series.trial_predictors(runs, TaskModel((mod,)))
    assert list(tables[0].columns) == ["trial_type[1]"]
    assert np.isnan(tables[0].iloc[0, 0])
    assert set(tables[0].iloc[1:, 0]) <= {0.0, 1.0}


@pytest.mark.parametrize("optimized", [False, True])
def test_workflow_matches_whole_array_reference(session, cv_library, optimized):
    root, runs, model = session
    library = cv_library if optimized else None
    result = fit_alpha(session, library)
    for scope, indices in (
        ("odd", [0, 2, 4]),
        ("even", [1, 3, 5]),
        ("all", list(range(6))),
    ):
        training = [runs[i] for i in indices]
        data = inputs.load_block(training, root, np.arange(4), model)
        predictors = [encoded(r.events) for r in training]
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
    data = inputs.load_block(runs, root, np.arange(4), model)
    final_alpha = result["tuning"]["all"]["selection"].ridge_alpha
    signature = spatial_signature(runs[0].image.header.get_axis(1), np.arange(4))
    if library is None:
        expected_fit = fit_single_trials(data, ridge_alpha=final_alpha)
    else:
        selection = select_hrfs(
            data, library=library, feature_signature=signature, task_model=model
        )
        expected_fit = fit_selected_hrfs(
            data,
            hrf_selection=selection,
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
            train_data = inputs.load_block(
                [runs[i] for i in train], root, np.arange(4), model
            )
            selected = select_hrfs(
                train_data,
                library=library,
                feature_signature=signature,
                task_model=model,
            )
            fitted = fit_selected_hrfs(
                data,
                hrf_selection=selected,
                ridge_alpha=alpha,
                feature_signature=signature,
            )
        reference = evaluate_trial_encoding(
            fitted.run_betas,
            [encoded(r.events) for r in runs],
            train_runs=train,
            test_runs=test,
        )
        for saved, r in zip(outer["betas"], test):
            np.testing.assert_allclose(saved, fitted.run_betas[r], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(outer["encoding_r2"], reference.r2, atol=1e-6)
        for saved, predicted in zip(outer["predictions"], reference.predictions):
            np.testing.assert_allclose(saved, predicted, atol=1e-6)


def test_block_size_and_parallelism_preserve_global_choice(session, cv_library):
    serial = fit_alpha(session, cv_library)
    parallel = fit(session, cv_library, alphas=[0.0, 0.1, 1.0], block_size=3, n_jobs=2)
    for scope in ("odd", "even", "all"):
        assert_candidate_equal(serial["tuning"][scope], parallel["tuning"][scope])
    for a, b in zip(serial["final"]["betas"], parallel["final"]["betas"]):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "parity,scope,evaluation", [(0, "odd", "odd_to_even"), (1, "even", "even_to_odd")]
)
def test_outer_data_cannot_change_training_choices(
    session, cv_library, parity, scope, evaluation
):
    root, runs, model = session
    original = fit_alpha(session, cv_library)
    changed = altered_runs(runs, parity, 10.0)
    altered = fit_alpha((root, changed, model), cv_library)
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


def test_missing_rt_keeps_beta_rows_and_requested_axis(session):
    root, runs, model = session
    altered = []
    for run in runs:
        e = run.events.copy()
        e.loc[0, "response_time"] = np.nan
        e.loc[1, "response_time"] = -1.0
        altered.append(replace(run, events=e))
    result = fit_alpha((root, altered, model), None, max_grayordinates=2)
    assert all(b.shape == (6, 4) for b in result["final"]["betas"])
    assert all(np.isnan(b[:, 2:]).all() for b in result["final"]["betas"])
    assert all(
        mask.tolist() == [False, False, True, True, True, True]
        for mask in result["tuning"]["all"]["scores"].trial_masks
    )


@pytest.mark.parametrize(
    "case", ["fewruns", "type", "axis", "block", "maximum", "jobs"]
)
def test_workflow_preflight_rejects_invalid_inputs(session, cv_library, case):
    root, runs, model = session
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
        image = nib.Cifti2Image(runs[0].image.get_fdata(), header=header)
        runs[0] = replace(runs[0], image=image)
    elif case == "block":
        kwargs["block_size"] = 0
    elif case == "maximum":
        kwargs["max_grayordinates"] = -1
    else:
        kwargs["n_jobs"] = True
    with pytest.raises(ValueError):
        beta_series.fit_cv_beta_series(runs, root, task_model=model, **kwargs)


def test_final_provenance_identifies_the_tuning_decision(session):
    root, runs, model = session

    def run_fit(run_set, **overrides):
        kwargs = dict(library=None, alphas=[1.0], percentile=90, block_size=2)
        kwargs.update(overrides)
        return fit((root, run_set, model), **kwargs)

    original = run_fit(runs)
    repeated = run_fit(runs)
    changed_predictors = [
        replace(r, events=r.events.assign(response_time=r.events.response_time + 1))
        for r in runs
    ]
    alternatives = [
        run_fit(runs, percentile=80),
        run_fit(runs, alphas=[0.0, 1.0]),
        run_fit(changed_predictors),
        run_fit(runs, max_grayordinates=2),
    ]
    records = []
    for result in (original, repeated, *alternatives):
        tuned = result["tuning"]["all"]
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
    assert (
        original["final"]["ridge_alpha"] == alternatives[0]["final"]["ridge_alpha"] == 1
    )


def test_rt_switch_reaches_every_ridge_selection(session, cv_library):
    root, runs, model = session
    narrow = inputs.selection_task_model(model, False)
    result = beta_series.fit_cv_beta_series(
        runs, root, **options(cv_library), task_model=narrow
    )
    finals = result["final"]["hrf_selections"]
    assert finals and all(b["all"].task_model == narrow for b in finals.values())
    for evaluation in result["evaluation"].values():
        assert evaluation["selection_task_model"] == narrow.to_dict()


def test_selection_model_leaves_predictors_and_fits_on_the_full_model(
    session, cv_library
):
    root, runs, model = session
    narrow = inputs.selection_task_model(model, False)
    result = beta_series.fit_cv_beta_series(
        runs, root, **options(cv_library), task_model=model, selection_model=narrow
    )
    assert all(list(p.columns) == ORDER for p in result["predictors"])
    assert result["final"]["rt"] is not None
    finals = result["final"]["hrf_selections"]
    assert all(b["all"].task_model == narrow for b in finals.values())


def test_coefficient_rows_are_named_after_the_predictors(session):
    outer = fit_alpha(session, None)["evaluation"]["odd_to_even"]
    assert outer["coefficient_names"] == ["task", *ORDER]
    assert outer["coefficients"].shape[0] == 3


# --- fractional CV ----------------------------------------------------------


@pytest.mark.parametrize("optimized", [False, True])
def test_fraction_workflow_matches_whole_array(session, cv_library, optimized):
    root, runs, model = session
    library = cv_library if optimized else None
    result = fit_fractions(session, library, block_size=1)
    for scope, indices in [
        ("odd", [0, 2, 4]),
        ("even", [1, 3, 5]),
        ("all", list(range(6))),
    ]:
        subset = [runs[r] for r in indices]
        scores = score_fraction_candidates(
            inputs.load_block(subset, root, np.arange(4), model),
            beta_series.trial_predictors(subset, model),
            library=library,
            fractions=[0.3, 0.7, 1],
        )
        expected = select_ridge_fractions(scores.cv_r2, scores.grid)
        actual = result["tuning"][scope]["selection"]
        np.testing.assert_allclose(actual.ridge_fraction, expected.ridge_fraction)
        np.testing.assert_allclose(actual.selected_r2, expected.selected_r2, atol=1e-10)
    data = inputs.load_block(runs, root, np.arange(4), model)
    train, test = [0, 2, 4], [1, 3, 5]
    chosen = result["tuning"]["odd"]["selection"].ridge_fraction
    outer_selection = (
        None
        if library is None
        else select_hrfs(
            inputs.load_block([runs[r] for r in train], root, np.arange(4), model),
            library=library,
            task_model=model,
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
        mixed,
        beta_series.trial_predictors(runs, model),
        train_runs=train,
        test_runs=test,
    )
    actual_outer = result["evaluation"]["odd_to_even"]
    for target_beta, r in zip(actual_outer["targets"], test):
        np.testing.assert_allclose(target_beta, ols.run_betas[r], atol=2e-6)
    np.testing.assert_allclose(actual_outer["run_sse"], outer.run_sse, atol=1e-8)
    np.testing.assert_allclose(actual_outer["run_sst"], outer.run_sst, atol=1e-8)
    for actual_beta, r in zip(actual_outer["betas"], test):
        np.testing.assert_allclose(actual_beta, shrunk.run_betas[r], atol=1e-8)
    assert_fractional_final(result, runs, data, library, model)


def assert_fractional_final(result, runs, data, library, model):
    fractions = result["tuning"]["all"]["selection"].ridge_fraction
    if library is None:
        expected = fit_single_trials(data, ridge_fraction=fractions)
    else:
        selection = select_hrfs(data, library=library, task_model=model)
        expected = fit_selected_hrfs(
            data, hrf_selection=selection, ridge_fraction=fractions
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
    link = result["final"]["provenance"][0]["record"]["activities"][-1]
    assert decision and link["tuning_analysis_fingerprint"] == decision


def test_fraction_blocks_workers_and_missing_rt_preserve_results(session, cv_library):
    root, runs, model = session
    runs = [
        replace(
            r,
            events=r.events.assign(
                response_time=r.events.response_time.mask(r.events.index == 0)
            ),
        )
        for r in runs
    ]
    pair = (root, runs, model)
    serial = fit_fractions(pair, cv_library, block_size=1)
    parallel = fit_fractions(pair, cv_library, block_size=3, n_jobs=2)
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
    session, cv_library, parity, scope
):
    root, runs, model = session
    initial = fit_fractions(session, cv_library, block_size=2)
    altered = fit_fractions(
        (root, altered_runs(runs, parity, 100.0), model), cv_library, block_size=2
    )
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


def test_fraction_and_alpha_grids_are_mutually_exclusive(session):
    with pytest.raises(ValueError):
        fit_fractions(session, alphas=[0, 1])


# --- encoding modes ---------------------------------------------------------


@pytest.mark.parametrize("mode", ["within_run", "absolute"])
@pytest.mark.parametrize("fractional", [False, True])
def test_outer_oracle_and_blocks(session, mode, fractional):
    root, runs, model = session
    grid = dict(fractions=[1, 0.5]) if fractional else dict(alphas=[0, 0.1])
    result = fit(session, encoding_mode=mode, block_size=2, **grid)
    blocked = fit(session, encoding_mode=mode, block_size=3, **grid)
    assert result["provenance"]["encoding_mode"] == mode
    decision = result["tuning"]["all"]["provenance"].to_dict()["activities"][-1]
    assert decision["encoding_mode"] == mode
    final = result["final"]["provenance"][0]["record"]["activities"][-1]
    assert final["encoding_mode"] == mode
    data = inputs.load_block(runs, root, np.arange(4), model)
    predictors = beta_series.trial_predictors(runs, model)
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
        assert_outer_oracle(outer, data, predictors, train, test, mode, fractional)


def assert_outer_oracle(outer, data, predictors, train, test, mode, fractional):
    fit_options = (
        {"ridge_fraction": outer["ridge_fraction"]}
        if fractional
        else {"ridge_alpha": outer["ridge_alpha"]}
    )
    fitted = fit_single_trials(data, **fit_options)
    # Fractional CV scores fixed OLS targets; alpha CV keeps fitted targets.
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
    np.testing.assert_allclose(outer["coefficients"][1:, :3], oracle[-2:], atol=1e-8)
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
        np.testing.assert_allclose(outer["scoring_offsets"][j, :3], offset, atol=1e-8)
        np.testing.assert_allclose(
            outer["run_sse"][j, :3],
            np.sum((residual - offset) ** 2, axis=0),
            atol=1e-8,
        )


def test_workflow_rejects_unknown_encoding_mode(session):
    with pytest.raises(ValueError, match="encoding_mode"):
        fit(session, alphas=[0], encoding_mode="bad")


# --- tuning provenance ------------------------------------------------------


def _problem():
    rng = np.random.default_rng(91)
    signals, events, predictors, times = [], [], [], []
    for run in range(3):
        t = 0.75 + 1.5 * np.arange(70)
        p = pd.DataFrame(
            dict(response_time=rng.uniform(0.3, 1.8, 7), trial_type=np.arange(7) % 2)
        )
        events.append(p.assign(onset=8 + 12.0 * np.arange(7) + run, duration=1.5))
        predictors.append(p)
        times.append(t)
        signals.append(rng.normal(100, 1, (len(t), 3)))
    sources = [glm_run_sources(run) for run in range(3)]
    data = from_arrays(signals, events, frame_times=times, sources=sources)
    return data, predictors


def _tuning(fractional):
    data, predictors = _problem()
    if fractional:
        scores = score_fraction_candidates(data, predictors, fractions=[0.5, 1.0])
        return beta_series.tuning_provenance(scores, select_ridge_fractions(scores))
    scores = score_ridge_candidates(data, predictors, alphas=[0.0, 1.0])
    return beta_series.tuning_provenance(scores, select_ridge_penalty(scores))


@pytest.mark.parametrize("fractional", [False, True])
def test_tuning_identity_does_not_depend_on_software(monkeypatch, fractional):
    first = _tuning(fractional)
    assert first.analysis_fingerprint is not None
    changed = {k: f"changed-{v}" for k, v in software_environment().items()}
    for module in ("_fit_lifecycle", "data", "prepared"):
        monkeypatch.setattr(
            f"boldtailor.{module}.software_environment", lambda: changed
        )
    second = _tuning(fractional)
    software = [a.get("software") for a in second.to_dict()["activities"]]
    assert changed in software
    assert first.analysis_fingerprint == second.analysis_fingerprint


# --- BetaModel assembly -----------------------------------------------------

ESTIMATOR = {
    "off": None,
    "fixed": "Ridge",
    "cv": "RidgeCV",
    "fractional_cv": "FractionalCV",
}


@pytest.mark.parametrize("mode", ["off", "fixed", "cv", "fractional_cv"])
def test_fit_beta_models_yields_one_model_per_hrf_and_estimator(
    six_run_dataset, cv_library, settings_for, mode
):
    root, _ = six_run_dataset
    settings = settings_for(
        root,
        ridge_mode=mode,
        ridge_fractions=(0.4, 1.0),
        ridge_alphas=(0.0, 0.1),
        block_size=4,
    )
    runs = inputs.load_session(settings)
    model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=4)
    selections = analysis.select_hrfs(runs, root, blocks, cv_library, task_model=model)
    models = beta_series.fit_beta_models(
        runs, root, blocks, settings, cv_library, selections, model
    )
    estimator = ESTIMATOR[mode]
    expected = {"CanonicalTrialOLS", "OptimizedTrialOLS"} | (
        {f"CanonicalTrial{estimator}", f"OptimizedTrial{estimator}"}
        if estimator
        else set()
    )
    assert set(models) == expected
    for name, item in models.items():
        assert item.name == name and item.hrf in ("canonical", "optimized")
        assert len(item.fit["betas"]) == 6
        tuned = item.estimator in ("RidgeCV", "FractionalCV")
        assert (item.tuning is not None) is tuned
        assert (item.evaluation is not None) is tuned
        assert (item.cv_provenance is not None) is tuned
        assert (item.predictors is not None) is tuned
        assert item.fractional == (item.estimator == "FractionalCV")


def test_cv_models_keep_rt_in_fits_and_predictors_when_selection_excludes_it(
    six_run_dataset, cv_library, settings_for
):
    root, _ = six_run_dataset
    settings = settings_for(
        root,
        ridge_mode="cv",
        ridge_alphas=(0.0, 0.1),
        block_size=4,
        hrf_selection_rt=False,
    )
    runs = inputs.load_session(settings)
    model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=4)
    models = beta_series.fit_beta_models(
        runs, root, blocks, settings, cv_library, None, model
    )
    item = models["OptimizedTrialRidgeCV"]
    assert all(list(p.columns) == ORDER for p in item.predictors)
    assert item.fit["rt"] is not None


def test_fixed_ridge_model_uses_the_setting_alpha(
    six_run_dataset, cv_library, settings_for
):
    root, _ = six_run_dataset
    settings = settings_for(root, ridge_mode="fixed", ridge_alpha=0.5, block_size=4)
    runs = inputs.load_session(settings)
    model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=4)
    models = beta_series.fit_beta_models(
        runs, root, blocks, settings, None, None, model
    )
    assert models["CanonicalTrialRidge"].fit["ridge_alpha"] == 0.5
    assert models["CanonicalTrialOLS"].fit["ridge_alpha"] == 0.0


def test_ridge_cv_progress_goes_to_the_log_not_stdout(session, capsys, caplog):
    caplog.set_level(logging.INFO, logger="boldtailor")
    fit_alpha(session, None)
    assert capsys.readouterr().out == ""
    assert any("Ridge CV" in r.getMessage() for r in caplog.records)
