"""Fraction selection is featurewise and targets use matched shrinkage."""

import importlib

import numpy as np
import pytest

from boldtailor._single_trial_design import compile_trial_run
from boldtailor.hrf_selection import select_hrf
from test_fractional_ridge import oracle
from test_ridge_cv import ridge_problem, subset  # noqa: F401


def module():
    try:
        return importlib.import_module("boldtailor.fractional_ridge")
    except ImportError as error:
        pytest.fail(f"Fractional CV API is missing: {error}")


def test_selector_chooses_different_fractions_and_preserves_undefined_locations():
    scores = np.array(
        [
            [0.7, -0.5, 0.8, np.nan, 0.5],
            [0.5, -0.2, 0.8 - 2e-13, np.nan, np.nan],
            [0.2, -0.9, 0.3, np.nan, 0.9],
        ]
    )
    selected = module().select_ridge_fractions(scores, [0.4, 1, 0.7])
    np.testing.assert_allclose(selected.ridge_fraction, [0.4, 1, 1, np.nan, np.nan])
    np.testing.assert_allclose(selected.selected_r2[:3], [0.7, -0.2, 0.8 - 2e-13])
    assert selected.fractions == (1.0, 0.7, 0.4)
    np.testing.assert_array_equal(selected.fraction_indices, [2, 0, 0, -1, -1])
    assert not selected.ridge_fraction.flags.writeable
    assert not selected.scoring_mask.flags.writeable
    undefined = module().select_ridge_fractions(np.full((2, 3), np.nan), [0.5, 1])
    assert np.isnan(undefined.ridge_fraction).all()
    masked = module().select_ridge_fractions(
        scores, [0.4, 1, 0.7], feature_mask=np.array([False, True, True, True, True])
    )
    assert np.isnan(masked.ridge_fraction[0])


@pytest.mark.parametrize("case", ["shape", "grid", "mask"])
def test_selector_rejects_misaligned_inputs(case):
    scores, fractions, mask = np.ones((2, 3)), [0.5, 1], None
    if case == "shape":
        scores = scores.T
    elif case == "grid":
        fractions = [0, 1]
    else:
        mask = [1, 1, 1]
    with pytest.raises(ValueError):
        module().select_ridge_fractions(scores, fractions, feature_mask=mask)


def reference(data, predictors, library, fractions):
    masks = [np.isfinite(p.to_numpy()).all(1) for p in predictors]
    losses, totals, assignments = [], [], []
    for test in range(data.n_runs):
        train = [r for r in range(data.n_runs) if r != test]
        ids = (
            np.zeros(data.n_features, int)
            if library is None
            else select_hrf(subset(data, train), library=library).hrf_indices
        )
        assignments.append(ids)
        fold_loss, fold_total = [], []
        for fraction in fractions:
            betas = []
            for r in range(data.n_runs):
                beta = np.full((len(data.events[r]), data.n_features), np.nan)
                for v, cid in enumerate(ids):
                    if cid < 0 or np.ptp(data.signals[r][:, v]) == 0:
                        continue
                    hrf = "spm" if library is None else library.candidates[cid]
                    x, n, _ = compile_trial_run(
                        data.events[r],
                        data.frame_times[r],
                        data.confounds[r],
                        f"run-{r}",
                        hrf=hrf,
                    )
                    beta[:, v] = oracle(
                        np.asarray(x), np.asarray(n), data.signals[r][:, v], fraction
                    )[0]
                betas.append(beta)
            xtrain = np.vstack([predictors[r].to_numpy()[masks[r]] for r in train])
            mean = xtrain.mean(0)
            x = np.column_stack([np.ones(len(xtrain)), xtrain - mean])
            y = np.vstack([betas[r][masks[r]] for r in train])
            valid = np.isfinite(y).all(0)
            coef = np.full((x.shape[1], data.n_features), np.nan)
            coef[:, valid] = np.linalg.lstsq(x, y[:, valid], rcond=None)[0]
            test_x = predictors[test].to_numpy()[masks[test]]
            predicted = np.column_stack([np.ones(len(test_x)), test_x - mean]) @ coef
            target = betas[test][masks[test]]
            fold_loss.append(np.sum((target - predicted) ** 2, 0))
            fold_total.append(np.sum((target - target.mean(0)) ** 2, 0))
        losses.append(fold_loss)
        totals.append(fold_total)
    losses, totals = np.array(losses), np.array(totals)
    return 1 - losses.sum(0) / totals.sum(0), losses, totals, assignments


@pytest.mark.parametrize("optimized", [False, True])
def test_fraction_cv_matches_nested_same_fraction_oracle(ridge_problem, optimized):
    data, predictors, library = ridge_problem
    library = library if optimized else None
    result = module().score_fraction_candidates(
        data, predictors, fractions=[0.3, 1, 0.7], library=library
    )
    expected = reference(data, predictors, library, [1, 0.7, 0.3])
    for actual, wanted in zip(
        (result.cv_r2, result.fold_sse, result.fold_sst, result.fold_hrf_indices),
        expected,
    ):
        np.testing.assert_allclose(actual, wanted, rtol=2e-7, atol=1e-8)
    assert result.fractions == (1.0, 0.7, 0.3)
    assert not result.cv_r2.flags.writeable
    assert not result.trial_masks[0][2]
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["validation_target"] == "candidate_fraction_regularized_betas"
    assert activity["fractions"] == [1.0, 0.7, 0.3]
    assert "alphas" not in activity


def test_inner_validation_data_does_not_change_hrf_or_training_transform(ridge_problem):
    data, predictors, library = ridge_problem
    original = module().score_fraction_candidates(
        data, predictors, fractions=[1, 0.5], library=library
    )
    changed_y = list(data.signals)
    changed_y[0] = np.random.default_rng(81).normal(size=data.signals[0].shape)
    changed_p = [p.copy() for p in predictors]
    changed_p[0]["response_time"] += 100
    changed = module().score_fraction_candidates(
        subset(data, range(data.n_runs), signals=changed_y),
        changed_p,
        fractions=[1, 0.5],
        library=library,
    )
    np.testing.assert_array_equal(
        original.fold_hrf_indices[0], changed.fold_hrf_indices[0]
    )
    a = original.provenance.to_dict()["activities"][-1]["folds"][0]
    b = changed.provenance.to_dict()["activities"][-1]["folds"][0]
    assert a["predictor_means"] == b["predictor_means"]
    assert not np.allclose(original.fold_sse[0], changed.fold_sse[0], equal_nan=True)
