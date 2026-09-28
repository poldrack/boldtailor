"""Held-out predictions use training coefficients and preserve event alignment."""

import importlib

import numpy as np
import pandas as pd
import pytest


def evaluate(*args, **kwargs):
    try:
        function = importlib.import_module(
            "boldtailor.trial_encoding"
        ).evaluate_trial_encoding
    except (ImportError, AttributeError) as error:
        pytest.fail(f"Missing trial encoding: {error}")
    return function(*args, **kwargs)


@pytest.fixture
def encoding_runs():
    rng = np.random.default_rng(32)
    predictors, betas = [], []
    for r, n in enumerate((9, 7, 6, 11)):
        x = pd.DataFrame(
            {"response_time": rng.uniform(0.3, 2, n), "trial_type": np.arange(n) % 2}
        )
        design = np.column_stack([np.ones(n), x])
        y = design @ np.array([[3.0, -1.0], [0.8, 2.0], [-0.5, 1.0]])
        y += rng.normal(0, 0.2, y.shape) + (2 if r == 3 else 0)
        predictors.append(x)
        betas.append(y)
    return betas, predictors


def test_no_validation_intercept_refit():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    predictors = [pd.DataFrame({"response_time": x}) for _ in range(3)]
    betas = [(2 + 3 * x)[:, None], (2 + 3 * x)[:, None], (12 + 3 * x)[:, None]]
    result = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2], encoding_mode="absolute")
    np.testing.assert_allclose(result.predictions[0][:, 0], 2 + 3 * x)
    np.testing.assert_allclose(result.r2, 1 - 400 / 45)


def test_predictions_and_pooled_loss_match_independent_ols(encoding_runs):
    betas, predictors = encoding_runs
    predictors[0].loc[2, "response_time"] = np.nan
    predictors[2].loc[1, "response_time"] = np.inf
    original = [x.copy(deep=True) for x in predictors]
    result = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3], encoding_mode="absolute")
    masks = [np.isfinite(x).all(axis=1).to_numpy() for x in predictors]
    xtrain = np.concatenate([predictors[r].to_numpy()[masks[r]] for r in (0, 1)])
    ytrain = np.concatenate([betas[r][masks[r]] for r in (0, 1)])
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(len(xtrain)), xtrain]), ytrain, rcond=None
    )[0]
    losses, totals = [], []
    for j, r in enumerate((2, 3)):
        x = predictors[r].to_numpy()[masks[r]]
        predicted = np.column_stack([np.ones(len(x)), x]) @ coefficients
        np.testing.assert_allclose(result.predictions[j][masks[r]], predicted)
        assert np.isnan(result.predictions[j][~masks[r]]).all()
        y = betas[r][masks[r]]
        losses.append(np.sum((y - predicted) ** 2, axis=0))
        totals.append(np.sum((y - y.mean(axis=0)) ** 2, axis=0))
    np.testing.assert_allclose(result.run_sse, losses)
    np.testing.assert_allclose(result.run_sst, totals)
    np.testing.assert_allclose(
        result.r2, 1 - np.sum(losses, axis=0) / np.sum(totals, axis=0)
    )
    assert not np.allclose(result.r2, np.mean(1 - np.array(losses) / totals, axis=0))
    np.testing.assert_allclose(result.predictor_means, xtrain.mean(axis=0))
    np.testing.assert_allclose(result.coefficients[1:], coefficients[1:])
    assert result.predictor_names == ("task", "response_time", "trial_type")
    for actual, expected in zip(predictors, original):
        pd.testing.assert_frame_equal(actual, expected)
    for actual, expected in zip(result.trial_masks, masks):
        np.testing.assert_array_equal(actual, expected)
    for array in [
        result.coefficients,
        result.predictor_means,
        result.r2,
        result.run_sse,
        result.run_sst,
        *result.predictions,
        *result.trial_masks,
    ]:
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_validation_covariates_do_not_change_training_transform(encoding_runs):
    betas, predictors = encoding_runs
    first = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])
    predictors[2] = predictors[2].assign(response_time=99.0, trial_type=0)
    second = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.predictor_means, second.predictor_means)
    assert not np.allclose(first.predictions[0], second.predictions[0])


def test_constant_and_incomplete_features_are_undefined(encoding_runs):
    betas, predictors = encoding_runs
    betas = [np.column_stack([y, np.ones(len(y))]) for y in betas]
    betas[0][1, 1] = np.nan
    result = evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])
    assert np.isfinite(result.r2[0])
    assert np.isnan(result.r2[1:]).all()
    assert np.isnan(result.predictions[0][:, 1]).all()


@pytest.mark.parametrize(
    "train,test",
    [
        ([], [2]),
        ([0, 0], [2]),
        ([0, 1], [1]),
        ([0], [4]),
        ([True], [2]),
        ([0, 1], []),
        ([0, 1], [2, 2]),
        ([0.5], [2]),
    ],
)
def test_bad_splits_rejected(encoding_runs, train, test):
    with pytest.raises(ValueError, match="run|split"):
        evaluate(*encoding_runs, train_runs=train, test_runs=test)


@pytest.mark.parametrize(
    "case",
    [
        "rows",
        "columns",
        "duplicate",
        "nonnumeric",
        "rank",
        "fewtrain",
        "fewtest",
        "run_count",
    ],
)
def test_invalid_predictors_rejected(encoding_runs, case):
    betas, predictors = encoding_runs
    if case == "rows":
        predictors[0] = predictors[0].iloc[:-1]
    elif case == "columns":
        predictors[2] = predictors[2][["trial_type", "response_time"]]
    elif case == "duplicate":
        predictors = [x.set_axis(["same", "same"], axis=1) for x in predictors]
    elif case == "nonnumeric":
        predictors[0] = predictors[0].astype(str)
    elif case == "rank":
        predictors = [x.assign(trial_type=1) for x in predictors]
    elif case == "fewtrain":
        for r in (0, 1):
            predictors[r].loc[1:, "response_time"] = np.nan
    elif case == "fewtest":
        predictors[2].loc[1:, "response_time"] = np.nan
    else:
        predictors.pop()
    with pytest.raises(ValueError):
        evaluate(betas, predictors, train_runs=[0, 1], test_runs=[2, 3])


@pytest.fixture
def offset_runs():
    rng = np.random.default_rng(731)
    predictors, betas = [], []
    for r, n in enumerate((8, 11, 6, 9)):
        x = rng.normal(size=(n, 2)) + r * np.array([3, -2])
        y = x @ np.array([[3., -2.], [.5, 4.]]) + [20*r, -7*r]
        y += rng.normal(0, .1, y.shape)
        predictors.append(pd.DataFrame(x, columns=['a', 'b']))
        betas.append(y)
    predictors[0].iloc[1, 0] = np.nan
    predictors[2].iloc[2, 1] = np.inf
    return betas, predictors


@pytest.mark.parametrize('mode', ['within_run', 'absolute'])
def test_run_intercepts_and_scores_match_dummy_ols(offset_runs, mode):
    betas, predictors = offset_runs
    train, test = [1, 0], [3, 2]
    masks = [np.isfinite(p).all(axis=1).to_numpy() for p in predictors]
    xs = [predictors[r].to_numpy()[masks[r]] for r in train]
    ys = [betas[r][masks[r]] for r in train]
    x, y = np.vstack(xs), np.vstack(ys)
    ids = np.repeat(np.arange(len(train)), [len(a) for a in xs])
    intercept_design = np.eye(len(train))[ids] if mode == 'within_run' else np.ones((len(x), 1))
    coef = np.linalg.lstsq(np.column_stack([intercept_design, x]), y, rcond=None)[0]
    slopes = coef[-2:]
    intercepts = coef[:-2] if mode == 'within_run' else np.repeat(coef[:1], 2, axis=0)
    result = evaluate(betas, predictors, train_runs=train, test_runs=test, encoding_mode=mode)
    assert result.encoding_mode == mode
    np.testing.assert_allclose(result.coefficients[1:], slopes)
    np.testing.assert_allclose(result.coefficients[0], y.mean(0))
    np.testing.assert_allclose(result.train_run_intercepts, intercepts)
    np.testing.assert_allclose(result.train_run_predictor_means, [a.mean(0) for a in xs])
    losses, totals = [], []
    for j, r in enumerate(test):
        observed = betas[r][masks[r]]
        prediction = y.mean(0) + (predictors[r].to_numpy()[masks[r]] - x.mean(0)) @ slopes
        residual = observed - prediction
        offset = residual.mean(0) if mode == 'within_run' else np.zeros(2)
        np.testing.assert_allclose(result.predictions[j][masks[r]], prediction)
        assert np.isnan(result.predictions[j][~masks[r]]).all()
        np.testing.assert_allclose(result.scoring_offsets[j], offset)
        losses.append(np.sum((residual-offset)**2, axis=0))
        totals.append(np.sum((observed-observed.mean(0))**2, axis=0))
    np.testing.assert_allclose(result.run_sse, losses)
    np.testing.assert_allclose(result.run_sst, totals)
    np.testing.assert_allclose(result.r2, 1-np.sum(losses,0)/np.sum(totals,0))
    assert not np.allclose(result.r2, np.mean(1-np.array(losses)/totals, axis=0))
    for array in (result.train_run_intercepts, result.train_run_predictor_means, result.scoring_offsets):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize('test_slope,expected', [(3, 1), (4, 15/16), (.5, -24)])
def test_default_centers_offsets_but_retains_slope_error(test_slope, expected):
    x = np.array([-1., 0., 1., 2.])
    predictors = [pd.DataFrame({'x': x}) for _ in range(3)]
    betas = [(2+3*x)[:,None], (22+3*x)[:,None], (102+test_slope*x)[:,None]]
    result = evaluate(betas, predictors, train_runs=[0,1], test_runs=[2])
    np.testing.assert_allclose(result.coefficients[1], [3])
    np.testing.assert_allclose(result.predictions[0][:,0], 12+3*x)
    np.testing.assert_allclose(result.r2, [expected])
    np.testing.assert_allclose(result.train_run_intercepts[:,0], [2,22])


def test_run_shifts_leave_slopes_and_scores_invariant(offset_runs):
    betas, predictors = offset_runs
    options = dict(train_runs=[0,1], test_runs=[2,3])
    before = evaluate(betas, predictors, **options)
    shifts = np.array([[100,-20], [-70,80], [11,-13], [7,9]])
    after = evaluate([b+s for b,s in zip(betas, shifts)], predictors, **options)
    np.testing.assert_allclose(after.coefficients[1:], before.coefficients[1:])
    np.testing.assert_allclose(after.train_run_intercepts-before.train_run_intercepts, shifts[:2])
    np.testing.assert_allclose(after.run_sse, before.run_sse, atol=1e-10)
    np.testing.assert_allclose(after.r2, before.r2)
    test_changed = evaluate([*betas[:2], betas[2]+shifts[2], betas[3]+shifts[3]], predictors, **options)
    np.testing.assert_array_equal(test_changed.coefficients, before.coefficients)
    np.testing.assert_array_equal(test_changed.train_run_intercepts, before.train_run_intercepts)
    for a,b in zip(test_changed.predictions, before.predictions):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(test_changed.scoring_offsets-before.scoring_offsets, shifts[2:])
    shifted_x = [p+np.array([5.,-8.]) for p in predictors]
    translated = evaluate([b+np.array([9.,12.]) for b in betas], shifted_x, **options)
    np.testing.assert_allclose(translated.train_run_intercepts, before.train_run_intercepts + [9,12] - np.array([5,-8]) @ before.coefficients[1:])


@pytest.mark.parametrize('mode', ['within_run', 'absolute'])
def test_feature_invalidity_and_zero_sst_are_preserved(offset_runs, mode):
    betas, predictors = offset_runs
    betas = [np.column_stack([b, np.ones(len(b))]) for b in betas]
    betas[0][0,1] = np.nan
    betas[2][0,0] = np.nan
    result = evaluate(betas, predictors, train_runs=[0,1], test_runs=[2,3], encoding_mode=mode)
    assert np.isnan(result.r2).all()
    assert np.isnan(result.train_run_intercepts[:,1]).all()
    assert np.isnan(result.scoring_offsets[0,:2]).all()
    assert np.isfinite(result.run_sse[1,0])
    assert result.run_sst[0,2] == 0


@pytest.mark.parametrize('case', ['empty', 'df', 'between', 'masked_rank'])
def test_within_run_design_rejects_unidentifiable_slopes(case):
    x = [np.arange(5., dtype=float)+r*10 for r in range(3)]
    if case == 'between':
        x[:2] = [np.ones(5)*r for r in range(2)]
    elif case == 'empty':
        x[0][:] = np.nan
    elif case == 'df':
        x[:2] = [np.array([0.]), np.array([1.,2.])]
    else:
        x[:2] = [np.array([0.,0.,np.nan]), np.array([1.,1.,np.nan])]
    predictors = [pd.DataFrame({'x': v}) for v in x]
    betas = [np.arange(len(v),dtype=float)[:,None] for v in x]
    with pytest.raises(ValueError, match='complete|rank|degrees'):
        evaluate(betas, predictors, train_runs=[0,1], test_runs=[2])


@pytest.mark.parametrize('sizes', [(5,), (1,5)])
def test_single_training_run_or_single_row_run_can_fit(sizes):
    predictors = [pd.DataFrame({'x': np.arange(n,dtype=float)}) for n in (*sizes,4)]
    betas = [(10*r+3*p.x.to_numpy())[:,None] for r,p in enumerate(predictors)]
    result = evaluate(betas,predictors,train_runs=list(range(len(sizes))),test_runs=[len(sizes)])
    np.testing.assert_allclose(result.r2,[1])
    np.testing.assert_allclose(result.train_run_intercepts[:,0],10*np.arange(len(sizes)),atol=1e-12)


@pytest.mark.parametrize('mode', ['centered', '', None, [], 1])
def test_unknown_encoding_mode_rejected(encoding_runs, mode):
    with pytest.raises(ValueError, match='encoding_mode'):
        evaluate(*encoding_runs, train_runs=[0,1], test_runs=[2], encoding_mode=mode)
