"""Independent candidate-target oracles and strict inner-fold isolation."""

import importlib

import numpy as np
import pytest

from boldtailor._single_trial_design import compile_trial_run
from boldtailor.data import from_arrays
from boldtailor.hrf_selection import select_hrf
from tests.oracles import subset_runs as subset


def score(*args, **kwargs):
    module = importlib.import_module("boldtailor.ridge_selection")
    if not hasattr(module, "score_ridge_candidates"):
        pytest.fail("Missing run-wise encoding ridge CV")
    return module.score_ridge_candidates(*args, **kwargs)


def augmented_beta(data, r, ids, library, alpha):
    y = data.signals[r]
    beta = np.full((len(data.events[r]), data.n_features), np.nan)
    for cid in np.unique(ids[ids >= 0]):
        features = np.flatnonzero((ids == cid) & (np.ptp(y, axis=0) > 0))
        x, n, _ = compile_trial_run(
            data.events[r],
            data.frame_times[r],
            data.confounds[r],
            f"run-{r}",
            hrf="spm" if library is None else library.candidates[cid],
        )
        x, n = x.to_numpy(), n.to_numpy()
        xr = x - n @ np.linalg.lstsq(n, x, rcond=None)[0]
        scale = np.linalg.norm(xr, axis=0)
        matrix = np.vstack(
            [
                np.column_stack([x, n]),
                np.column_stack(
                    [
                        np.diag(np.sqrt(alpha) * scale),
                        np.zeros((x.shape[1], n.shape[1])),
                    ]
                ),
            ]
        )
        target = np.vstack([y[:, features], np.zeros((x.shape[1], len(features)))])
        beta[:, features] = np.linalg.lstsq(matrix, target, rcond=None)[0][: x.shape[1]]
    return beta


def reference_scores(
    data, predictors, library, alphas, *, ols_target=False, encoding_mode="within_run"
):
    masks = [np.isfinite(p).all(axis=1).to_numpy() for p in predictors]
    losses, totals, assignments = [], [], []
    for test in range(data.n_runs):
        train = [r for r in range(data.n_runs) if r != test]
        ids = (
            np.zeros(data.n_features, dtype=int)
            if library is None
            else select_hrf(subset(data, train), library=library).hrf_indices
        )
        assignments.append(ids)
        fold_loss, fold_total = [], []
        for alpha in alphas:
            betas = [
                augmented_beta(data, r, ids, library, alpha) for r in range(data.n_runs)
            ]
            x = np.concatenate([predictors[r].to_numpy()[masks[r]] for r in train])
            y = np.concatenate([betas[r][masks[r]] for r in train])
            valid = np.isfinite(y).all(axis=0)
            intercepts = np.ones((len(x), 1))
            if encoding_mode == "within_run":
                ids_run = np.repeat(
                    np.arange(len(train)), [masks[r].sum() for r in train]
                )
                intercepts = np.eye(len(train))[ids_run]
            coefficients = np.linalg.lstsq(
                np.column_stack([intercepts, x]), y[:, valid], rcond=None
            )[0]
            xtest = predictors[test].to_numpy()[masks[test]]
            predicted = xtest @ coefficients[-x.shape[1] :]
            if encoding_mode == "absolute":
                predicted += coefficients[0]
            target_beta = (
                augmented_beta(data, test, ids, library, 0.0)
                if ols_target
                else betas[test]
            )
            target = target_beta[masks[test]][:, valid]
            loss, total = np.full((2, data.n_features), np.nan)
            residual = target - predicted
            if encoding_mode == "within_run":
                residual -= residual.mean(0)
            loss[valid] = np.sum(residual**2, axis=0)
            total[valid] = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
            fold_loss.append(loss)
            fold_total.append(total)
        losses.append(fold_loss)
        totals.append(fold_total)
    losses, totals = np.array(losses), np.array(totals)
    return losses, totals, np.array(assignments)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("encoding_mode", ["within_run", "absolute"])
def test_cv_matches_independent_augmented_ols_and_encoding(
    ridge_problem, optimized, encoding_mode
):
    data, predictors, library = ridge_problem
    library = library if optimized else None
    result = score(
        data,
        predictors,
        alphas=[1.0, 0.0, 0.1],
        library=library,
        feature_signature="fixture",
        encoding_mode=encoding_mode,
    )
    expected_sse, expected_sst, assignments = reference_scores(
        data, predictors, library, [0.0, 0.1, 1.0], encoding_mode=encoding_mode
    )
    np.testing.assert_allclose(result.fold_sse, expected_sse, atol=1e-9)
    np.testing.assert_allclose(result.fold_sst, expected_sst, atol=1e-9)
    np.testing.assert_allclose(
        result.cv_r2, 1 - expected_sse.sum(axis=0) / expected_sst.sum(axis=0), atol=1e-9
    )
    np.testing.assert_array_equal(result.fold_hrf_indices, assignments)
    assert tuple(result.alphas) == (0.0, 0.1, 1.0)
    assert len(result.trial_masks[0]) == 8 and result.trial_masks[0].sum() == 7
    assert np.isnan(result.cv_r2[:, -1]).all()
    for array in (
        result.cv_r2,
        result.fold_sse,
        result.fold_hrf_indices,
        result.trial_masks[0],
    ):
        assert type(array) is np.ndarray
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


def test_validation_targets_use_candidate_penalty(ridge_problem):
    data, predictors, _ = ridge_problem
    result = score(data, predictors, alphas=[0.0, 1.0])
    loss, total, _ = reference_scores(
        data, predictors, None, [0.0, 1.0], ols_target=True
    )
    wrong = 1 - loss.sum(axis=0) / total.sum(axis=0)
    np.testing.assert_allclose(result.cv_r2[0], wrong[0], atol=1e-9)
    assert not np.allclose(result.cv_r2[1, :4], wrong[1, :4])


def test_inner_validation_cannot_train_hrf_or_encoding(ridge_problem, monkeypatch):
    data, predictors, library = ridge_problem
    score(data, predictors, alphas=[0.0], library=library)
    module = importlib.import_module("boldtailor._ridge_cv")
    original_select, original_encoding = (
        module.select_hrf,
        module.evaluate_trial_encoding,
    )
    seen_selections, seen_encodings = [], []

    def observe_selection(*args, **kwargs):
        result = original_select(*args, **kwargs)
        seen_selections.append(result)
        return result

    def observe_encoding(*args, **kwargs):
        result = original_encoding(*args, **kwargs)
        seen_encodings.append(result)
        return result

    monkeypatch.setattr(module, "select_hrf", observe_selection)
    monkeypatch.setattr(module, "evaluate_trial_encoding", observe_encoding)
    options = dict(
        alphas=[0.0, 0.2],
        library=library,
        run_labels=[f"run-{i}" for i in range(6)],
        feature_signature="fixture",
    )
    first = score(data, predictors, **options)
    original_first = seen_encodings[0]
    assert len(seen_selections) == 6
    assert all(f"run-{r}" not in s.run_labels for r, s in enumerate(seen_selections))
    signals = [y.copy() for y in data.signals]
    signals[0][:, :4] = np.random.default_rng(6).normal(size=signals[0][:, :4].shape)
    predictors = [p.copy() for p in predictors]
    predictors[0].loc[:, "response_time"] = 99.0
    changed = score(subset(data, range(6), signals=signals), predictors, **options)
    np.testing.assert_array_equal(
        first.fold_hrf_indices[0], changed.fold_hrf_indices[0]
    )
    np.testing.assert_array_equal(
        original_first.coefficients, seen_encodings[12].coefficients
    )
    np.testing.assert_array_equal(
        original_first.predictor_means, seen_encodings[12].predictor_means
    )
    np.testing.assert_array_equal(
        original_first.train_run_intercepts, seen_encodings[12].train_run_intercepts
    )
    np.testing.assert_array_equal(
        original_first.train_run_predictor_means,
        seen_encodings[12].train_run_predictor_means,
    )
    assert not np.allclose(first.fold_sse[0, :, :4], changed.fold_sse[0, :, :4])


def test_behavior_is_in_scoring_provenance(ridge_problem):
    data, predictors, _ = ridge_problem
    first = score(data, predictors, alphas=[0.0, 0.1])
    predictors[0] = predictors[0].assign(response_time=1.0)
    second = score(data, predictors, alphas=[0.0, 0.1])
    one = first.provenance.to_dict()["activities"][-1]
    two = second.provenance.to_dict()["activities"][-1]
    assert one["predictor_fingerprint"] != two["predictor_fingerprint"]
    assert one["validation_target"] == "candidate_regularized_betas"
    assert one["score"] == "pooled_within_run_centered_trial_encoding_r2"


@pytest.mark.parametrize("optimized,count", [(True, 2), (False, 1)])
def test_too_few_runs_rejected(ridge_problem, optimized, count):
    data, predictors, library = ridge_problem
    with pytest.raises(ValueError, match="runs"):
        score(
            subset(data, range(count)),
            predictors[:count],
            alphas=[0.0],
            library=library if optimized else None,
        )


def test_validation_design_failure_names_fold_and_run(ridge_problem):
    data, predictors, library = ridge_problem
    events = list(data.events)
    events[0].loc[1, "onset"] = events[0].loc[0, "onset"]
    with pytest.raises(ValueError, match="validation.*run-01|run-01.*validation"):
        score(
            subset(data, range(6), events=events),
            predictors,
            alphas=[0.0],
            library=library,
        )


@pytest.mark.parametrize("fractional", [False, True])
def test_modes_identify_objective_and_default(ridge_problem, fractional):
    from boldtailor.fractional_ridge import score_fraction_candidates

    data, predictors, _ = ridge_problem
    from boldtailor.provenance import RunSources, SourceRef

    def source(role, run):
        return SourceRef(
            role=role,
            uri=f"run-{run}/{role}.tsv",
            media_type="text/tab-separated-values",
            byte_size=128,
            modified_at="2026-09-28T00:00:00Z",
        )

    sources = [
        RunSources(
            signal=source("signal", r),
            events=source("events", r),
            confounds=source("confounds", r),
        )
        for r in range(data.n_runs)
    ]
    data = from_arrays(
        data.signals,
        data.events,
        frame_times=data.frame_times,
        confounds=data.confounds,
        sources=sources,
    )
    function = score_fraction_candidates if fractional else score
    options = {"fractions": [1, 0.5]} if fractional else {"alphas": [0, 1]}
    default = function(data, predictors, **options)
    within = function(data, predictors, encoding_mode="within_run", **options)
    absolute = function(data, predictors, encoding_mode="absolute", **options)
    np.testing.assert_array_equal(default.cv_r2, within.cv_r2)
    assert not np.allclose(within.cv_r2, absolute.cv_r2, equal_nan=True)
    assert (
        within.provenance.analysis_fingerprint
        != absolute.provenance.analysis_fingerprint
    )
    for mode, result in [("within_run", within), ("absolute", absolute)]:
        activity = result.provenance.to_dict()["activities"][-1]
        assert activity["encoding_mode"] == mode
        assert activity["encoding_objective_version"] == 2
        assert activity["encoding_model"] == (
            "ols_with_run_specific_intercepts"
            if mode == "within_run"
            else "ols_with_shared_intercept"
        )
        fold = activity["folds"][0]
        np.testing.assert_allclose(
            fold["train_run_predictor_means"],
            [p[np.isfinite(p).all(axis=1)].mean().to_numpy() for p in predictors[1:]],
        )


@pytest.mark.parametrize("fractional", [False, True])
@pytest.mark.parametrize("case", ["mode", "rank"])
def test_encoding_preflight_precedes_hrf_fitting(
    ridge_problem, monkeypatch, fractional, case
):
    from boldtailor.fractional_ridge import score_fraction_candidates
    import boldtailor._ridge_cv as cv

    data, predictors, library = ridge_problem

    def forbidden(*args, **kwargs):
        pytest.fail("invalid encoding design reached HRF fitting")

    monkeypatch.setattr(cv, "_fold_selection", forbidden)
    mode = "invalid" if case == "mode" else "within_run"
    if case == "rank":
        predictors = [
            p.assign(response_time=float(r)) for r, p in enumerate(predictors)
        ]
    function = score_fraction_candidates if fractional else score
    options = {"fractions": [1]} if fractional else {"alphas": [0]}
    with pytest.raises(ValueError, match="encoding_mode|rank"):
        function(data, predictors, library=library, encoding_mode=mode, **options)


@pytest.mark.parametrize("optimized", [False, True])
def test_prepared_run_betas_preserve_feature_order_and_candidate_independence(
    ridge_problem, optimized
):
    from boldtailor import _ridge_cv as cv

    data, _, library = ridge_problem
    prepared = cv.prepare_runs(data, library) if optimized else None
    ids = np.array([2, 0, 1, -1, 0]) if optimized else np.zeros(data.n_features, int)
    path = cv.prepare_run_beta_path(data, 0, prepared, ids, "run-01")
    for alpha in [0.0, 0.1, 2.0, 2.0, 0.1, 0.0]:
        expected = augmented_beta(data, 0, ids, library if optimized else None, alpha)
        np.testing.assert_allclose(path.betas_at(alpha), expected, atol=1e-9)
