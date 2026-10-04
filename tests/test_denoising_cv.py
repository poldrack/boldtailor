"""Leave-one-run-out PC-count selection against an independent stacked oracle.

The oracle fits each fold's shared task coefficients by one stacked
least-squares problem: task columns shared across training runs next to a
block-diagonal matrix of run nuisances [confounds, 1, indicators, PCs]. It
scores the held-out run against a target projected off [confounds, 1,
indicators] only. Fold HRFs come from the public ``select_hrfs`` on training
runs and pool PCs from the (separately tested) pool module; no CV helper
under test is used to derive an expected value.
"""

import re

import numpy as np
import pytest
from scipy.linalg import block_diag

from boldtailor._denoising_cv import (
    choose_count,
    prepare_fold,
    score_count,
    select_component_count,
    validate_counts,
    validate_tolerance,
)
from boldtailor._denoising_pool import (
    analysis_components,
    pool_statistic,
    run_components,
)
from boldtailor._hrf_design import MIN_ONSET, OVERSAMPLING, hrf_model
from boldtailor._task_design import run_task_columns
from boldtailor.data import from_arrays
from boldtailor.hrf_selection import select_hrfs, subset_runs
from boldtailor.model import Modulator, TaskModel
from tests.denoising_fixtures import _trial_responses, make_denoising_fixture

COUNTS = (0, 1, 2, 4)
THRESHOLD = 0.0  # the plan default, valid with the indicator-consistent statistic
# The fixture's 12 noise features share one latent series, so their scores
# move together: in the fold training on runs 0, 1, 3 they sit at or above 0
# even without indicators (median +0.001 under the RT model). Under the
# missing-RT model the whole group lands at +0.003 to +0.05 there, which
# empties the pool. That is one correlated draw, not indicator bias (see the
# independent-noise test below). Only this variant's oracle tests need the
# higher, still noise/task-separating (task >= 0.3) threshold.
VARIANT_THRESHOLDS = dict(missing_rt=0.1)
MISSING_RT = TaskModel((Modulator("response_time", missing="indicator"),))
CATEGORICAL = TaskModel(
    (
        Modulator("response_time"),
        Modulator("cond", kind="categorical", levels=("a", "b"), missing="indicator"),
    )
)


# ---- data variants -----------------------------------------------------------


@pytest.fixture(scope="module")
def fixture():
    return make_denoising_fixture()


def rebuild(data, *, signals=None, events=None, confounds=None):
    return from_arrays(
        list(data.signals if signals is None else signals),
        list(data.events if events is None else events),
        frame_times=list(data.frame_times),
        confounds=list(data.confounds if confounds is None else confounds),
    )


def missing_rt_events(data):
    """Two missing response times in runs 0-2; run 3 has none (no indicator)."""
    events = [e.copy() for e in data.events]
    for run in (0, 1, 2):
        events[run].loc[[1, 4], "response_time"] = np.nan
    return events


def categorical_events(data):
    rng = np.random.default_rng(7)
    events = []
    for run, frame in enumerate(data.events):
        cond = np.where(np.arange(len(frame)) % 2 == 0, "a", "b").astype(object)
        cond = rng.permutation(cond)
        if run in (0, 1):
            cond[2] = np.nan
        events.append(frame.assign(cond=cond))
    return events


def collinear_confounds(data):
    """A scaled copy of motion and a column collinear with the intercept."""
    return [
        c.assign(motion_copy=2.0 * c.motion_x - 0.5 * c.drift, offset=3.0)
        for c in data.confounds
    ]


def variant(fixture, name):
    data = fixture.data
    if name == "rt":
        return data, fixture.task_model
    if name == "missing_rt":
        return rebuild(data, events=missing_rt_events(data)), MISSING_RT
    if name == "categorical":
        return rebuild(data, events=categorical_events(data)), CATEGORICAL
    return rebuild(data, confounds=collinear_confounds(data)), fixture.task_model


VARIANTS = ("rt", "missing_rt", "categorical", "collinear")


# ---- independent oracle ------------------------------------------------------


def design_parts(data, run, task_model, candidate):
    columns = run_task_columns(
        data.events[run],
        task_model,
        data.frame_times[run],
        hrf_model(candidate),
        run=run,
        min_onset=MIN_ONSET,
        oversampling=OVERSAMPLING,
    )
    x = columns[list(task_model.regressor_names)].to_numpy()
    present = [c for c in task_model.profiled_names if c in columns]
    return x, columns[present].to_numpy()


def baseline(data, run):
    confounds = data.confounds[run].to_numpy(dtype=float)
    return np.column_stack([confounds, np.ones(len(confounds))])


def residual(nuisance, y):
    return y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]


def oracle_coefficients(data, train, comps, features, candidate, count, task_model):
    xs, nuisances, ys = [], [], []
    for run, comp in zip(train, comps):
        x, profiled = design_parts(data, run, task_model, candidate)
        pcs = np.zeros((len(x), 0)) if comp is None else comp.components[:, :count]
        xs.append(x)
        nuisances.append(np.column_stack([baseline(data, run), profiled, pcs]))
        ys.append(data.signals[run][:, features])
    stacked = np.column_stack([np.vstack(xs), block_diag(*nuisances)])
    solution = np.linalg.lstsq(stacked, np.vstack(ys), rcond=None)[0]
    return solution[: x.shape[1]]


def oracle_heldout(data, run, features, candidate, beta, task_model):
    x, profiled = design_parts(data, run, task_model, candidate)
    nuisance = np.column_stack([baseline(data, run), profiled])
    observed = residual(nuisance, data.signals[run][:, features])
    predicted = residual(nuisance, x) @ beta
    energy = np.sum(observed**2, axis=0)
    return 1 - np.sum((observed - predicted) ** 2, axis=0) / energy, energy


def oracle_statistic(data, selection, task_model, library):
    """LORO CV R² of each winning HRF; denominator ||M_(q,qp) y||² per run."""
    ids, runs = selection.hrf_indices, range(data.n_runs)
    statistic = np.full(data.n_features, np.nan)
    for hrf in np.unique(ids[ids >= 0]):
        features, candidate = np.flatnonzero(ids == hrf), library.candidates[hrf]
        loss, total = 0.0, 0.0
        for held in runs:
            train = [r for r in runs if r != held]
            beta = oracle_coefficients(
                data, train, [None] * len(train), features, candidate, 0, task_model
            )
            r2, energy = oracle_heldout(
                data, held, features, candidate, beta, task_model
            )
            loss, total = loss + (1 - r2) * energy, total + energy
        statistic[features] = 1 - loss / total
    return statistic


def oracle_training(data, validation, library, task_model, brain_mask, threshold):
    train = [r for r in range(data.n_runs) if r != validation]
    subset = subset_runs(data, train)
    selection = select_hrfs(subset, library=library, task_model=task_model)
    scores = oracle_statistic(subset, selection, task_model, library)
    defined = brain_mask & np.isfinite(scores) & (selection.hrf_indices >= 0)
    pool, scoring = defined & (scores <= threshold), defined & (scores > threshold)
    comps = analysis_components(subset, pool)
    return train, selection, pool, scoring, comps, scores


def oracle_fold(
    data,
    validation,
    library,
    task_model,
    brain_mask,
    counts,
    zero=(),
    threshold=THRESHOLD,
):
    """Per count: feature R², coefficients, target energies, fold mean."""
    train, selection, pool, scoring, comps, _ = oracle_training(
        data, validation, library, task_model, brain_mask, threshold
    )
    n, k = data.n_features, len(task_model.regressor_names)
    results = {}
    for count in counts:
        r2, energy = np.full(n, np.nan), np.full(n, np.nan)
        beta = np.full((k, n), np.nan)
        for hrf in np.unique(selection.hrf_indices[scoring]):
            features = np.flatnonzero(scoring & (selection.hrf_indices == hrf))
            candidate = library.candidates[hrf]
            b = oracle_coefficients(
                data, train, comps, features, candidate, count, task_model
            )
            beta[:, features] = b
            r2[features], energy[features] = oracle_heldout(
                data, validation, features, candidate, b, task_model
            )
        r2[list(zero)] = np.nan
        results[count] = dict(r2=r2, beta=beta, energy=energy, mean=np.nanmean(r2))
    results.update(selection=selection, pool=pool, scoring=scoring, comps=comps)
    return results


def conservative_choice(counts, scores, tolerance):
    best = max(s for s in scores if np.isfinite(s))
    return min(c for c, s in zip(counts, scores) if s >= best - tolerance)


def fold(data, validation, fixture, task_model, threshold=THRESHOLD, mask=None):
    return prepare_fold(
        data,
        validation,
        brain_mask=fixture.brain_mask if mask is None else mask,
        task_model=task_model,
        library=fixture.library,
        threshold=threshold,
    )


# ---- validation of counts and tolerance --------------------------------------


def test_counts_are_sorted_deduplicated_integers_including_zero():
    assert validate_counts((4, 0, np.int64(2), 2, 1)) == (0, 1, 2, 4)
    assert validate_counts([0]) == (0,)


@pytest.mark.parametrize(
    "counts", [(), (1, 2), (0, -1), (0, True), (0, 1.0), (0, "2"), (False, 1), 3]
)
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError, match="counts"):
        validate_counts(counts)


def test_tolerance_must_be_finite_and_nonnegative():
    assert validate_tolerance(0) == 0.0
    assert validate_tolerance(0.001) == 0.001
    for value in (-1e-9, np.nan, np.inf, True, "0.1", None):
        with pytest.raises(ValueError, match="score_tolerance"):
            validate_tolerance(value)


# ---- conservative count choice ------------------------------------------------


def test_choice_is_smallest_count_within_tolerance_of_the_best():
    counts = (0, 1, 2, 4)
    scores = (0.10, 0.20, 0.2005, 0.19)
    assert choose_count(counts, scores, 0.001) == 1
    assert choose_count(counts, scores, 0.0) == 2
    assert choose_count(counts, (0.3, 0.3, 0.3, 0.3), 0.0) == 0
    assert choose_count(counts, (0.1, 0.2, 0.2, 0.2), 0.0) == 1


def test_unavailable_counts_cannot_win():
    counts = (0, 1, 2, 4)
    assert choose_count(counts, (0.10, 0.20, np.nan, 0.19), 0.0) == 1
    assert choose_count(counts, (0.10, np.nan, np.nan, np.nan), 0.0) == 0
    assert choose_count((4, 0, 2), (0.5, 0.1, 0.499), 0.002) == 2


def test_choice_requires_a_scored_zero_count_and_aligned_scores():
    with pytest.raises(ValueError, match="zero"):
        choose_count((0, 1), (np.nan, 0.2), 0.0)
    with pytest.raises(ValueError, match="zero"):
        choose_count((1, 2), (0.1, 0.2), 0.0)
    with pytest.raises(ValueError, match="one score per count"):
        choose_count((0, 1), (0.1,), 0.0)


# ---- fold preparation uses training runs only -------------------------------


def test_fold_selection_masks_and_pcs_come_from_training_runs(fixture):
    data, model = fixture.data, fixture.task_model
    result = fold(data, 1, fixture, model)
    train, selection, pool, scoring, comps, statistic = oracle_training(
        data, 1, fixture.library, model, fixture.brain_mask, THRESHOLD
    )
    assert result.validation_run == 1 and result.training_runs == tuple(train)
    np.testing.assert_allclose(result.pool_statistic, statistic, atol=1e-10)
    np.testing.assert_array_equal(result.selection.hrf_indices, selection.hrf_indices)
    np.testing.assert_array_equal(result.selection.cv_r2, selection.cv_r2)
    np.testing.assert_array_equal(result.masks.pool, pool)
    np.testing.assert_array_equal(result.masks.scoring, scoring)
    assert len(result.components) == len(train)
    for got, expected in zip(result.components, comps):
        np.testing.assert_array_equal(got.components, expected.components)
    assert pool.sum() >= 10 and scoring.sum() >= 6


def test_fold_arrays_are_read_only(fixture):
    result = fold(fixture.data, 0, fixture, fixture.task_model)
    score = score_count(result, 1)
    assert not result.pool_statistic.flags.writeable
    for array in (result.scored, result.zero_target, score.feature_r2):
        assert not array.flags.writeable
    for array in (score.coefficients, score.target_energy):
        assert not array.flags.writeable


# ---- numerical agreement with the stacked oracle -----------------------------


@pytest.fixture(scope="module")
def variant_folds(fixture):
    """Production and oracle results for every fold of every variant."""
    out = {}
    for name in VARIANTS:
        data, model = variant(fixture, name)
        threshold = VARIANT_THRESHOLDS.get(name, THRESHOLD)
        for v in range(data.n_runs):
            expected = oracle_fold(
                data,
                v,
                fixture.library,
                model,
                fixture.brain_mask,
                COUNTS,
                threshold=threshold,
            )
            result = fold(data, v, fixture, model, threshold=threshold)
            out[name, v] = (result, expected, model)
    return out


def test_missing_and_categorical_variants_exercise_indicators(fixture):
    data, _ = variant(fixture, "missing_rt")
    names = [
        run_task_columns(
            data.events[r],
            MISSING_RT,
            data.frame_times[r],
            "spm",
            run=r,
            min_onset=MIN_ONSET,
            oversampling=OVERSAMPLING,
        ).columns
        for r in range(4)
    ]
    assert ["missing_response_time" in n for n in names] == [True] * 3 + [False]
    assert CATEGORICAL.regressor_names == ("task", "response_time", "cond[b]")
    assert CATEGORICAL.profiled_names == ("missing_cond",)


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("count", COUNTS)
def test_scores_match_stacked_oracle(variant_folds, name, count):
    for v in range(4):
        result, expected, model = variant_folds[name, v]
        np.testing.assert_array_equal(result.masks.scoring, expected["scoring"])
        score = score_count(result, count)
        want = expected[count]
        assert score.count == count and score.reason == ""
        assert score.coefficients.shape == (len(model.regressor_names), 26)
        np.testing.assert_allclose(
            score.coefficients, want["beta"], rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(score.target_energy, want["energy"], rtol=1e-9)
        np.testing.assert_allclose(score.feature_r2, want["r2"], rtol=0, atol=1e-9)
        assert score.mean_r2 == pytest.approx(want["mean"], abs=1e-10)


@pytest.mark.parametrize("name", VARIANTS)
def test_target_denominator_and_scored_features_are_fixed_across_counts(
    variant_folds, name
):
    for v in range(4):
        result = variant_folds[name, v][0]
        np.testing.assert_array_equal(result.scored, result.masks.scoring)
        assert not result.zero_target.any()
        scores = [score_count(result, c) for c in COUNTS]
        for score in scores[1:]:
            np.testing.assert_array_equal(score.target_energy, scores[0].target_energy)
            np.testing.assert_array_equal(
                np.isfinite(score.feature_r2), np.isfinite(scores[0].feature_r2)
            )
        np.testing.assert_array_equal(np.isfinite(scores[0].feature_r2), result.scored)


def test_coefficients_change_with_count_so_the_comparison_is_live(variant_folds):
    result = variant_folds["rt", 0][0]
    zero, one = score_count(result, 0), score_count(result, 1)
    scored = result.scored
    assert np.max(np.abs(zero.coefficients - one.coefficients)[:, scored]) > 1e-3


def test_component_count_selection_aggregates_equal_weight_folds(fixture):
    data, model = fixture.data, fixture.task_model
    chosen = select_component_count(
        data,
        brain_mask=fixture.brain_mask,
        task_model=model,
        library=fixture.library,
        counts=(4, 0, 2, 1),
        threshold=THRESHOLD,
        tolerance=0.001,
    )
    folds = [
        oracle_fold(data, v, fixture.library, model, fixture.brain_mask, COUNTS)
        for v in range(4)
    ]
    expected = [np.mean([f[c]["mean"] for f in folds]) for c in COUNTS]
    assert chosen.counts == COUNTS
    table = chosen.scores
    assert list(table["count"]) == list(COUNTS)
    assert table["eligible"].all() and (table["reason"] == "").all()
    np.testing.assert_allclose(table["mean_r2"], expected, rtol=0, atol=1e-10)
    assert chosen.n_components == conservative_choice(COUNTS, expected, 0.001)
    per_fold = chosen.fold_scores
    assert len(per_fold) == 16 and len(chosen.folds) == 4
    assert list(per_fold["validation_run"]) == [v for v in range(4) for _ in COUNTS]
    for v, f in enumerate(folds):
        rows = per_fold[per_fold["validation_run"] == v]
        np.testing.assert_allclose(
            rows["mean_r2"], [f[c]["mean"] for c in COUNTS], atol=1e-10
        )
        assert (rows["n_scored"] == f["scoring"].sum()).all()
        assert (rows["n_zero_target"] == 0).all()


def test_selection_requires_three_runs(fixture):
    data = subset_runs(fixture.data, [0, 1])
    with pytest.raises(ValueError, match="three runs"):
        select_component_count(
            data,
            brain_mask=fixture.brain_mask,
            task_model=fixture.task_model,
            library=fixture.library,
            counts=COUNTS,
            threshold=THRESHOLD,
            tolerance=0.001,
        )


# ---- validation targets -------------------------------------------------------


def heldout_replaced(data, run, features, values):
    signals = [s.copy() for s in data.signals]
    signals[run][:, features] = values[:, None]
    return rebuild(data, signals=signals)


def test_zero_energy_targets_are_excluded_identically_for_every_count(fixture):
    model, run = fixture.task_model, 3
    in_nuisance = 50.0 + 4.0 * fixture.data.confounds[run]["drift"].to_numpy()
    zero = [0, 7]
    data = heldout_replaced(fixture.data, run, zero, in_nuisance)
    result = fold(data, run, fixture, model)
    expected = oracle_fold(
        data, run, fixture.library, model, fixture.brain_mask, COUNTS, zero=zero
    )
    assert result.masks.scoring[zero].all()
    np.testing.assert_array_equal(np.flatnonzero(result.zero_target), zero)
    np.testing.assert_array_equal(
        result.scored, result.masks.scoring & ~result.zero_target
    )
    for count in COUNTS:
        score = score_count(result, count)
        assert np.isnan(score.feature_r2[zero]).all()
        assert score.mean_r2 == pytest.approx(expected[count]["mean"], abs=1e-10)


def test_no_nonzero_validation_target_raises(fixture):
    model, run = fixture.task_model, 3
    in_nuisance = 50.0 + 4.0 * fixture.data.confounds[run]["cosine"].to_numpy()
    data = heldout_replaced(fixture.data, run, np.arange(26), in_nuisance)
    with pytest.raises(ValueError, match="no supported task signal"):
        fold(data, run, fixture, model)


def test_empty_scoring_mask_raises(fixture):
    with pytest.raises(ValueError, match="no supported task signal"):
        fold(fixture.data, 0, fixture, fixture.task_model, threshold=10.0)


# ---- eligibility ------------------------------------------------------------------


def test_empty_pool_leaves_only_zero(fixture):
    result = fold(fixture.data, 1, fixture, fixture.task_model, threshold=-10.0)
    assert result.masks.pool_size == 0
    assert score_count(result, 0).reason == ""
    for count in (1, 2):
        score = score_count(result, count)
        assert "empty noise pool" in score.reason
        assert np.isnan(score.mean_r2) and np.isnan(score.feature_r2).all()
        assert np.isnan(score.coefficients).all()
    chosen = select_component_count(
        fixture.data,
        brain_mask=fixture.brain_mask,
        task_model=fixture.task_model,
        library=fixture.library,
        counts=(0, 1, 2),
        threshold=-10.0,
        tolerance=0.0,
    )
    assert chosen.n_components == 0
    assert list(chosen.scores["eligible"]) == [True, False, False]
    assert chosen.scores["reason"].str.contains("empty noise pool")[1:].all()


def test_count_beyond_pool_rank_is_unavailable_not_capped(fixture):
    result = fold(fixture.data, 0, fixture, fixture.task_model)
    ranks = [c.rank for c in result.components]
    too_many = max(ranks) + 1
    score = score_count(result, too_many)
    assert "exceeds pool PCA rank" in score.reason
    assert np.isnan(score.mean_r2)


def short_run_data(fixture, extra=40, length=30):
    """Run 0 truncated to `length` scans plus `extra` latent-driven noise features."""
    rng = np.random.default_rng(11)
    data = fixture.data
    signals, events, times, confounds = [], [], [], []
    for run in range(data.n_runs):
        n = length if run == 0 else len(data.signals[run])
        latent = fixture.latent[run][:n]
        noise = np.outer(latent, rng.uniform(0.6, 1.4, extra))
        noise += rng.normal(scale=0.3, size=noise.shape) + 100.0
        signals.append(np.column_stack([data.signals[run][:n], noise]))
        frame_times = data.frame_times[run][:n]
        frame = data.events[run]
        events.append(frame[frame.onset < frame_times[-1] - 12.0])
        times.append(frame_times)
        confounds.append(data.confounds[run].iloc[:n].reset_index(drop=True))
    mask = np.concatenate([fixture.brain_mask, np.ones(extra, bool)])
    return from_arrays(signals, events, frame_times=times, confounds=confounds), mask


def test_count_without_positive_residual_df_is_unavailable_in_every_fold(fixture):
    data, mask = short_run_data(fixture)
    model = fixture.task_model
    dof_limit = 30 - 4 - len(model.regressor_names)  # scans - baseline - task
    result = fold(data, 3, fixture, model, mask=mask)
    assert 0 in result.training_runs and result.components[0].rank >= dof_limit
    assert all(c.unavailable_reason(dof_limit) == "" for c in result.components)
    assert score_count(result, dof_limit - 1).reason == ""
    blocked = score_count(result, dof_limit)
    assert "degrees of freedom" in blocked.reason
    assert np.isnan(blocked.mean_r2) and np.isnan(blocked.coefficients).all()
    chosen = select_component_count(
        data,
        brain_mask=mask,
        task_model=model,
        library=fixture.library,
        counts=(0, dof_limit),
        threshold=THRESHOLD,
        tolerance=0.0,
    )
    assert list(chosen.scores["eligible"]) == [True, False]
    assert np.isnan(chosen.scores["mean_r2"].iloc[1])
    rows = chosen.fold_scores[chosen.fold_scores["count"] == dof_limit]
    assert list(rows["eligible"]) == [True, False, False, False]
    assert chosen.n_components == 0


# ---- leakage: held-out signals never affect training quantities ---------------


def test_heldout_perturbation_changes_scores_but_not_training(fixture):
    model, run = fixture.task_model, 1
    rng = np.random.default_rng(3)
    signals = [s.copy() for s in fixture.data.signals]
    signals[run] += rng.normal(scale=5.0, size=signals[run].shape)
    signals[run][:, :8] *= -2.0
    perturbed = rebuild(fixture.data, signals=signals)
    a = fold(fixture.data, run, fixture, model)
    b = fold(perturbed, run, fixture, model)
    np.testing.assert_array_equal(a.selection.hrf_indices, b.selection.hrf_indices)
    np.testing.assert_array_equal(a.selection.cv_r2, b.selection.cv_r2)
    np.testing.assert_array_equal(a.masks.pool, b.masks.pool)
    np.testing.assert_array_equal(a.masks.scoring, b.masks.scoring)
    for x, y in zip(a.components, b.components):
        np.testing.assert_array_equal(x.components, y.components)
    for count in COUNTS:
        sa, sb = score_count(a, count), score_count(b, count)
        np.testing.assert_array_equal(sa.coefficients, sb.coefficients)
        assert sa.mean_r2 != pytest.approx(sb.mean_r2, abs=1e-3)


# ---- adversarial PCs aligned with unmodeled task variance ---------------------


def echo_data(fixture, scale=1.0):
    """Six pool features carrying the task features' RT-free trial variation."""
    rng = np.random.default_rng(13)
    data, signals = fixture.data, []
    for run in range(data.n_runs):
        times = data.frame_times[run]
        trials = _trial_responses(
            data.events[run], times, fixture.library.candidates[0]
        )
        variation = fixture.trial_amplitudes[run] - 3.0
        echo = scale * trials @ variation
        echo += np.outer(fixture.latent[run], rng.uniform(0.6, 1.4, 6))
        echo += rng.normal(scale=0.3, size=echo.shape) + 80.0
        signals.append(np.column_stack([data.signals[run], echo]))
    mask = np.concatenate([fixture.brain_mask, np.ones(6, bool)])
    return rebuild(data, signals=signals), mask


def naive_projected_r2(data, run, result, count):
    """Wrong objective: also project held-out PCs out of the target."""
    comp = run_components(data.signals[run], data.confounds[run], result.masks.pool)
    score = score_count(result, count)
    r2 = []
    for feature in np.flatnonzero(result.scored):
        hrf = result.selection.hrf_indices[feature]
        x, profiled = design_parts(
            data,
            run,
            result.selection.task_model,
            result.selection.library.candidates[hrf],
        )
        nuisance = np.column_stack([baseline(data, run), profiled, comp.prefix(count)])
        observed = residual(nuisance, data.signals[run][:, feature])
        predicted = residual(nuisance, x) @ score.coefficients[:, feature]
        r2.append(1 - np.sum((observed - predicted) ** 2) / np.sum(observed**2))
    return float(np.mean(r2))


def test_task_aligned_pcs_cannot_gain_by_removing_target_variance(fixture):
    data, mask = echo_data(fixture)
    model, run = fixture.task_model, 0
    result = fold(data, run, fixture, model, mask=mask)
    echo = np.arange(26, 32)
    assert result.masks.pool[echo].all()
    zero = score_count(result, 0)
    for count in (2, 4):
        score = score_count(result, count)
        naive_gain = naive_projected_r2(data, run, result, count) - zero.mean_r2
        assert naive_gain > 0.1
        assert score.mean_r2 - zero.mean_r2 < 0.25 * naive_gain
        np.testing.assert_array_equal(score.target_energy, zero.target_energy)
    expected = oracle_fold(data, run, fixture.library, model, mask, (0, 2, 4))
    for count in (0, 2, 4):
        assert score_count(result, count).mean_r2 == pytest.approx(
            expected[count]["mean"], abs=1e-10
        )


# ---- indicator-consistent pool statistic (ruling R4) --------------------------


def test_pool_statistic_equals_select_hrfs_cv_r2_without_indicators(fixture):
    for model in (fixture.task_model, MISSING_RT):  # fixture has no missing RTs
        selection = select_hrfs(fixture.data, library=fixture.library, task_model=model)
        statistic = pool_statistic(fixture.data, selection)
        assert not statistic.flags.writeable
        np.testing.assert_allclose(statistic, selection.cv_r2, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", ["missing_rt", "categorical"])
def test_pool_statistic_matches_stacked_loro_oracle_with_indicators(fixture, name):
    data, model = variant(fixture, name)
    selection = select_hrfs(data, library=fixture.library, task_model=model)
    statistic = pool_statistic(data, selection)
    expected = oracle_statistic(data, selection, model, fixture.library)
    np.testing.assert_allclose(statistic, expected, rtol=0, atol=1e-10)
    defined = selection.hrf_indices >= 0
    assert np.max(np.abs(statistic - selection.cv_r2)[defined]) > 1e-3


def test_pool_statistic_requires_matching_runs_and_features(fixture):
    selection = select_hrfs(
        fixture.data, library=fixture.library, task_model=fixture.task_model
    )
    with pytest.raises(ValueError, match="runs"):
        pool_statistic(subset_runs(fixture.data, [0, 1]), selection)
    with pytest.raises(ValueError, match="HrfSelectionResult"):
        pool_statistic(fixture.data, selection.cv_r2)


def with_white_noise(data, n=60, seed=17):
    """Append independent white-noise features (with confound loadings)."""
    rng = np.random.default_rng(seed)
    signals = []
    for signal, frame in zip(data.signals, data.confounds):
        confounds = frame.to_numpy()
        noise = rng.normal(size=(len(signal), n)) + 100.0
        noise += confounds @ rng.normal(scale=0.8, size=(confounds.shape[1], n))
        signals.append(np.column_stack([signal, noise]))
    return rebuild(data, signals=signals)


@pytest.mark.parametrize("name", ["missing_rt", "categorical"])
def test_indicator_models_keep_pure_noise_in_the_pool_at_threshold_zero(fixture, name):
    data, model = variant(fixture, name)
    data = with_white_noise(data)
    white = np.arange(26, 86)
    task = np.r_[fixture.groups["task"], fixture.groups["rt"]]
    mask = np.r_[fixture.brain_mask, np.ones(60, bool)]
    selection = select_hrfs(data, library=fixture.library, task_model=model)
    statistic = pool_statistic(data, selection)
    assert np.median(statistic[white]) <= 0.0
    assert np.median(selection.cv_r2[white]) > np.median(statistic[white])
    for v in range(data.n_runs):
        result = fold(data, v, fixture, model, threshold=0.0, mask=mask)
        assert np.median(result.pool_statistic[white]) <= 0.0
        assert result.masks.scoring[task].all()
        assert score_count(result, 4).reason == ""


@pytest.mark.filterwarnings("ignore:Matrix is singular:UserWarning")
def test_invalid_heldout_design_for_a_frozen_hrf_raises(fixture):
    events = [e.copy() for e in fixture.data.events]
    events[3]["response_time"] = 0.9  # task and RT columns become collinear
    data = rebuild(fixture.data, events=events)
    # Requirement change (Task 3 review): messages name runs by label.
    with pytest.raises(
        ValueError, match="held-out run 'run-04': HRF .* task design is invalid"
    ):
        fold(data, 3, fixture, fixture.task_model)


CV_LABELS = ("a1", "b2", "c3", "d4")


def test_fold_messages_and_training_selection_use_run_labels(fixture):
    result = prepare_fold(
        fixture.data,
        1,
        brain_mask=fixture.brain_mask,
        task_model=fixture.task_model,
        library=fixture.library,
        threshold=THRESHOLD,
        run_labels=CV_LABELS,
    )
    assert result.selection.run_labels == ("a1", "c3", "d4")
    assert score_count(result, 50).reason.startswith("training run 'a1': ")
    chosen = select_component_count(
        fixture.data,
        brain_mask=fixture.brain_mask,
        task_model=fixture.task_model,
        library=fixture.library,
        counts=(0, 50),
        threshold=THRESHOLD,
        tolerance=0.001,
        run_labels=CV_LABELS,
    )
    per_fold = chosen.fold_scores
    assert list(per_fold["validation_label"]) == [x for x in CV_LABELS for _ in (0, 50)]
    reason = chosen.scores.set_index("count").loc[50, "reason"]
    assert "fold holding out run 'b2': training run 'a1': " in reason
    assert not re.search(r"\brun \d", reason)


@pytest.mark.parametrize("broken", [0, 2, 3])
def test_invalid_run_design_names_only_that_run(fixture, broken):
    events = list(fixture.data.events)
    events[broken] = events[broken].drop(columns="response_time")
    data = rebuild(fixture.data, events=events)
    with pytest.raises(ValueError) as error:
        select_component_count(
            data,
            brain_mask=fixture.brain_mask,
            task_model=fixture.task_model,
            library=fixture.library,
            counts=(0, 1),
            threshold=THRESHOLD,
            tolerance=0.001,
            run_labels=CV_LABELS,
        )
    message = str(error.value)
    assert f"run '{CV_LABELS[broken]}'" in message
    others = [label for i, label in enumerate(CV_LABELS) if i != broken]
    assert not any(label in message for label in others)
    assert not re.search(r"\brun \d", message)
