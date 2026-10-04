"""Leave-one-run-out PC-count scoring against an independent stacked oracle.

HRFs, the noise pool, and its run-wise PCs are full-data inputs, frozen for
every fold and count (GLMsingle design). For each fold the oracle fits shared
task coefficients by one stacked least-squares problem: task columns shared
across the training runs next to a block-diagonal matrix of run nuisances
[confounds, 1, indicators, PCs]. It scores the held-out run against a target
projected off [confounds, 1, indicators] only, pools each feature's SSE and
SST across folds, and takes the median over scored features. Frozen inputs
come from the public ``select_hrfs`` and the (separately tested) pool module;
no CV helper under test derives an expected value.
"""

import re

import numpy as np
import pytest
from scipy.linalg import block_diag

from boldtailor._denoising_cv import (
    choose_count,
    prepare_scoring,
    score_count,
    select_component_count,
    validate_counts,
    validate_pcstop,
)
from boldtailor._denoising_pool import (
    analysis_components,
    onoff_r2,
    pool_masks,
    run_components,
)
from boldtailor._hrf_design import MIN_ONSET, OVERSAMPLING, hrf_model
from boldtailor._task_design import run_task_columns
from boldtailor.data import from_arrays
from boldtailor.hrf_selection import select_hrfs, subset_runs
from boldtailor.model import Modulator, TaskModel
from tests.denoising_fixtures import _trial_responses, make_denoising_fixture

COUNTS = (0, 1, 2, 4)
MISSING_RT = TaskModel((Modulator("response_time", missing="indicator"),))
CATEGORICAL = TaskModel(
    (
        Modulator("response_time"),
        Modulator("cond", kind="categorical", levels=("a", "b"), missing="indicator"),
    )
)
CV_LABELS = ("a1", "b2", "c3", "d4")


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


def frozen_inputs(data, model, library, threshold="auto"):
    """Full-data HRFs, masks, and pool PCs, as select_denoising builds them."""
    selection = select_hrfs(data, library=library, task_model=model)
    statistic = onoff_r2(data, library)
    masks = pool_masks(statistic, threshold, hrf_indices=selection.hrf_indices)
    return selection, masks, analysis_components(data, masks.pool)


def setup_for(data, model, library, inputs, run_labels=None):
    selection, masks, comps = inputs
    return prepare_scoring(
        data,
        hrf_indices=selection.hrf_indices,
        components=comps,
        scoring=masks.scoring,
        task_model=model,
        library=library,
        run_labels=run_labels,
    )


def select_count(data, model, library, inputs, **options):
    selection, masks, comps = inputs
    settings = dict(counts=COUNTS, pcstop=1.05)
    settings.update(options)
    return select_component_count(
        data,
        hrf_indices=selection.hrf_indices,
        components=comps,
        scoring=masks.scoring,
        task_model=model,
        library=library,
        **settings,
    )


@pytest.fixture(scope="module")
def base(fixture):
    """Frozen inputs and scoring setup for the unmodified fixture."""
    data, model = fixture.data, fixture.task_model
    inputs = frozen_inputs(data, model, fixture.library)
    return inputs, setup_for(data, model, fixture.library, inputs)


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
    for run in train:
        x, profiled = design_parts(data, run, task_model, candidate)
        pcs = comps[run].components[:, :count]
        xs.append(x)
        nuisances.append(np.column_stack([baseline(data, run), profiled, pcs]))
        ys.append(data.signals[run][:, features])
    stacked = np.column_stack([np.vstack(xs), block_diag(*nuisances)])
    solution = np.linalg.lstsq(stacked, np.vstack(ys), rcond=None)[0]
    return solution[: x.shape[1]]


def oracle_heldout(data, run, features, candidate, beta, task_model):
    """Held-out SSE and SST against the fixed [confounds, 1, indicators] target."""
    x, profiled = design_parts(data, run, task_model, candidate)
    nuisance = np.column_stack([baseline(data, run), profiled])
    observed = residual(nuisance, data.signals[run][:, features])
    predicted = residual(nuisance, x) @ beta
    return np.sum((observed - predicted) ** 2, axis=0), np.sum(observed**2, axis=0)


def oracle_count(data, inputs, library, task_model, count, scored):
    """Per-fold coefficients, SSE, SST; pooled feature R²; median performance."""
    selection, _, comps = inputs
    n_runs, n, k = data.n_runs, data.n_features, len(task_model.regressor_names)
    beta = np.full((n_runs, k, n), np.nan)
    sse, sst = np.full((n_runs, n), np.nan), np.full((n_runs, n), np.nan)
    ids = selection.hrf_indices
    for held in range(n_runs):
        train = [r for r in range(n_runs) if r != held]
        for hrf in np.unique(ids[scored]):
            features = np.flatnonzero(scored & (ids == hrf))
            candidate = library.candidates[hrf]
            b = oracle_coefficients(
                data, train, comps, features, candidate, count, task_model
            )
            beta[held][:, features] = b
            sse[held, features], sst[held, features] = oracle_heldout(
                data, held, features, candidate, b, task_model
            )
    r2 = np.full(n, np.nan)
    r2[scored] = 1 - sse[:, scored].sum(axis=0) / sst[:, scored].sum(axis=0)
    return dict(beta=beta, sse=sse, sst=sst, r2=r2, perf=np.median(r2[scored]))


def glmsingle_choice(counts, perf, pcstop):
    """GLMsingle select_noise_regressors walk over the available counts."""
    pairs = [(c, p) for c, p in zip(counts, perf) if np.isfinite(p)]
    curve = [(c, p - pairs[0][1]) for c, p in pairs]
    top = max(v for _, v in curve)
    if top <= 0:
        return 0
    best, chosen = -np.inf, None
    for c, v in curve:
        if v > best:
            best, chosen = v, c
        if best * pcstop >= top:
            return chosen


# ---- validation of counts and pcstop -----------------------------------------


def test_counts_are_sorted_deduplicated_integers_including_zero():
    assert validate_counts((4, 0, np.int64(2), 2, 1)) == (0, 1, 2, 4)
    assert validate_counts([0]) == (0,)


@pytest.mark.parametrize(
    "counts", [(), (1, 2), (0, -1), (0, True), (0, 1.0), (0, "2"), (False, 1), 3]
)
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError, match="counts"):
        validate_counts(counts)


def test_pcstop_must_be_finite_and_at_least_one():
    assert validate_pcstop(1) == 1.0 and validate_pcstop(1.05) == 1.05
    for value in (0.99, 0, -1, np.nan, np.inf, True, "1.05", None):
        with pytest.raises(ValueError, match="pcstop"):
            validate_pcstop(value)


# ---- GLMsingle pcstop stopping rule ---------------------------------------------


def test_pcstop_follows_glmsingle_semantics():
    counts = (0, 1, 2, 4)
    perf = (0.10, 0.20, 0.205, 0.19)  # curve (0, .10, .105, .09)
    assert choose_count(counts, perf, 1.05) == 1  # .10 * 1.05 >= .105
    assert choose_count(counts, perf, 1.0) == 2
    assert choose_count(counts, (0.3, 0.3, 0.3, 0.3), 1.05) == 0  # all equal
    assert choose_count(counts, (0.3, 0.1, 0.2, 0.25), 1.05) == 0  # negative curve
    assert choose_count(counts, (0.1, 0.2, 0.2, 0.2), 1.0) == 1  # ties keep fewer


def test_pcstop_walks_past_counts_far_below_the_maximum():
    counts = (0, 1, 2, 3, 4)
    perf = (0.0, 0.05, 0.02, 0.098, 0.1)
    assert choose_count(counts, perf, 1.05) == 3  # .098 * 1.05 >= .1
    assert choose_count(counts, perf, 1.0) == 4
    assert choose_count(counts, perf, 3.0) == 1  # .05 * 3 >= .1


def test_unavailable_counts_are_skipped():
    counts = (0, 1, 2, 4)
    assert choose_count(counts, (0.10, np.nan, 0.205, 0.19), 1.05) == 2
    assert choose_count(counts, (0.10, np.nan, np.nan, np.nan), 1.05) == 0
    assert choose_count((4, 0, 2), (0.5, 0.1, 0.49), 1.05) == 2  # sorted by count


def test_choice_requires_a_scored_zero_count_and_aligned_scores():
    with pytest.raises(ValueError, match="zero"):
        choose_count((0, 1), (np.nan, 0.2), 1.05)
    with pytest.raises(ValueError, match="zero"):
        choose_count((1, 2), (0.1, 0.2), 1.05)
    with pytest.raises(ValueError, match="one score per count"):
        choose_count((0, 1), (0.1,), 1.05)


# ---- numerical agreement with the stacked oracle -----------------------------


@pytest.fixture(scope="module")
def variant_setups(fixture):
    out = {}
    for name in VARIANTS:
        data, model = variant(fixture, name)
        inputs = frozen_inputs(data, model, fixture.library)
        out[name] = (
            data,
            model,
            inputs,
            setup_for(data, model, fixture.library, inputs),
        )
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
def test_scores_match_stacked_oracle(fixture, variant_setups, name, count):
    data, model, inputs, setup = variant_setups[name]
    want = oracle_count(data, inputs, fixture.library, model, count, setup.scored)
    score = score_count(setup, count)
    assert score.count == count and score.reason == ""
    assert score.coefficients.shape == (4, len(model.regressor_names), 26)
    scored = setup.scored
    np.testing.assert_allclose(
        score.coefficients[:, :, scored],
        want["beta"][:, :, scored],
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(score.fold_sst, want["sst"], rtol=1e-9)
    np.testing.assert_allclose(score.fold_sse, want["sse"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(score.feature_r2, want["r2"], rtol=0, atol=1e-9)
    assert score.perf == pytest.approx(want["perf"], abs=1e-10)


def test_scoring_features_are_the_task_features(fixture, base):
    inputs, setup = base
    groups = fixture.groups
    expected = np.r_[groups["task"], groups["rt"], groups["outside_task"]]
    np.testing.assert_array_equal(np.flatnonzero(setup.scored), expected)
    np.testing.assert_array_equal(setup.scoring, inputs[1].scoring)


@pytest.mark.parametrize("name", VARIANTS)
def test_target_denominator_and_scored_features_are_fixed_across_counts(
    variant_setups, name
):
    setup = variant_setups[name][3]
    np.testing.assert_array_equal(setup.scored, setup.scoring)
    assert not setup.zero_target.any()
    scores = [score_count(setup, c) for c in COUNTS]
    for score in scores[1:]:
        np.testing.assert_array_equal(score.fold_sst, scores[0].fold_sst)
        np.testing.assert_array_equal(
            np.isfinite(score.feature_r2), np.isfinite(scores[0].feature_r2)
        )
    np.testing.assert_array_equal(np.isfinite(scores[0].feature_r2), setup.scored)
    np.testing.assert_array_equal(scores[0].fold_sst, setup.target_energy)


def test_coefficients_change_with_count_so_the_comparison_is_live(base):
    setup = base[1]
    zero, one = score_count(setup, 0), score_count(setup, 1)
    scored = setup.scored
    difference = np.abs(zero.coefficients - one.coefficients)[:, :, scored]
    assert np.max(difference) > 1e-3


def test_setup_and_score_arrays_are_read_only(base):
    setup = base[1]
    score = score_count(setup, 1)
    for array in (setup.scored, setup.scoring, setup.zero_target, setup.target_energy):
        assert not array.flags.writeable
    for array in (score.feature_r2, score.coefficients, score.fold_sse, score.fold_sst):
        assert not array.flags.writeable


def test_scoring_reuses_cached_task_columns(fixture, base, monkeypatch):
    # Run designs (and their blocks) are cached, so a repeated setup must
    # not convolve the task design again for any run or HRF.
    from boldtailor._hrf_cv import RunDesign

    data, model = fixture.data, fixture.task_model
    want = setup_for(data, model, fixture.library, base[0])
    calls = []
    original = RunDesign.task_design

    def counted(self, candidate_id):
        calls.append(candidate_id)
        return original(self, candidate_id)

    monkeypatch.setattr(RunDesign, "task_design", counted)
    again = setup_for(data, model, fixture.library, base[0])
    assert calls == []
    for group, repeat in zip(want.groups, again.groups, strict=True):
        for terms, other in zip(group.runs, repeat.runs, strict=True):
            np.testing.assert_array_equal(terms.raw, other.raw)
            np.testing.assert_array_equal(terms.x, other.x)


def test_component_count_selection_uses_median_pooled_r2_and_pcstop(fixture, base):
    inputs, setup = base
    data, model = fixture.data, fixture.task_model
    chosen = select_count(data, model, fixture.library, inputs, counts=(4, 0, 2, 1))
    expected = [
        oracle_count(data, inputs, fixture.library, model, c, setup.scored)
        for c in COUNTS
    ]
    perf = [e["perf"] for e in expected]
    assert chosen.counts == COUNTS
    table = chosen.scores
    assert list(table["count"]) == list(COUNTS)
    assert table["eligible"].all() and (table["reason"] == "").all()
    np.testing.assert_allclose(table["perf"], perf, rtol=0, atol=1e-10)
    np.testing.assert_allclose(table["curve"], np.subtract(perf, perf[0]), atol=1e-10)
    assert chosen.n_components == glmsingle_choice(COUNTS, perf, 1.05)
    per_fold = chosen.fold_scores
    assert len(per_fold) == 16
    assert list(per_fold["validation_run"]) == [v for v in range(4) for _ in COUNTS]
    scored = setup.scored
    for v in range(4):
        rows = per_fold[per_fold["validation_run"] == v]
        fold_r2 = [1 - e["sse"][v, scored] / e["sst"][v, scored] for e in expected]
        np.testing.assert_allclose(
            rows["median_r2"], [np.median(r) for r in fold_r2], atol=1e-10
        )
        assert (rows["n_scored"] == scored.sum()).all()


def test_selection_finds_a_positive_count_on_the_fixture(fixture, base):
    chosen = select_count(fixture.data, fixture.task_model, fixture.library, base[0])
    assert chosen.n_components >= 1


def test_selection_requires_three_runs(fixture, base):
    data = subset_runs(fixture.data, [0, 1])
    selection, masks, comps = base[0]
    with pytest.raises(ValueError, match="three runs"):
        select_count(
            data, fixture.task_model, fixture.library, (selection, masks, comps[:2])
        )


@pytest.mark.parametrize("pcstop", [0.5, np.nan, True])
def test_selection_rejects_invalid_pcstop(fixture, base, pcstop):
    with pytest.raises(ValueError, match="pcstop"):
        select_count(
            fixture.data, fixture.task_model, fixture.library, base[0], pcstop=pcstop
        )


# ---- validation targets -------------------------------------------------------


def heldout_replaced(data, run, features, values):
    signals = [s.copy() for s in data.signals]
    signals[run][:, features] = values[:, None]
    return rebuild(data, signals=signals)


def test_zero_energy_targets_are_excluded_for_every_fold_and_count(fixture, base):
    inputs = base[0]
    model, run = fixture.task_model, 3
    in_nuisance = 50.0 + 4.0 * fixture.data.confounds[run]["drift"].to_numpy()
    zero = [0, 7]
    data = heldout_replaced(fixture.data, run, zero, in_nuisance)
    setup = setup_for(data, model, fixture.library, inputs)
    assert setup.scoring[zero].all()
    assert np.argwhere(setup.zero_target).tolist() == [[run, 0], [run, 7]]
    np.testing.assert_array_equal(
        setup.scored, setup.scoring & ~setup.zero_target.any(0)
    )
    for count in COUNTS:
        score = score_count(setup, count)
        assert np.isnan(score.feature_r2[zero]).all()
        want = oracle_count(data, inputs, fixture.library, model, count, setup.scored)
        assert score.perf == pytest.approx(want["perf"], abs=1e-10)


def test_no_nonzero_validation_target_raises(fixture, base):
    run = 3
    in_nuisance = 50.0 + 4.0 * fixture.data.confounds[run]["cosine"].to_numpy()
    data = heldout_replaced(fixture.data, run, np.arange(26), in_nuisance)
    with pytest.raises(ValueError, match="no supported task signal"):
        setup_for(data, fixture.task_model, fixture.library, base[0])


def test_empty_scoring_mask_raises(fixture, base):
    selection, masks, comps = base[0]
    empty = pool_masks(np.zeros(26), 0.0, hrf_indices=-np.ones(26, dtype=int))
    with pytest.raises(ValueError, match="no supported task signal"):
        select_count(
            fixture.data, fixture.task_model, fixture.library, (selection, empty, comps)
        )


# ---- eligibility ------------------------------------------------------------------


def test_empty_pool_leaves_only_zero(fixture, base):
    selection, masks, _ = base[0]
    data, model = fixture.data, fixture.task_model
    empty = analysis_components(data, np.zeros(26, dtype=bool))
    inputs = (selection, masks, empty)
    setup = setup_for(data, model, fixture.library, inputs)
    assert score_count(setup, 0).reason == ""
    for count in (1, 2):
        score = score_count(setup, count)
        assert "empty noise pool" in score.reason
        assert np.isnan(score.perf) and np.isnan(score.feature_r2).all()
        assert np.isnan(score.coefficients).all()
    chosen = select_count(data, model, fixture.library, inputs, counts=(0, 1, 2))
    assert chosen.n_components == 0
    assert list(chosen.scores["eligible"]) == [True, False, False]
    assert chosen.scores["reason"].str.contains("empty noise pool")[1:].all()


def test_count_beyond_pool_rank_is_unavailable_not_capped(base):
    inputs, setup = base
    too_many = max(c.rank for c in inputs[2]) + 1
    score = score_count(setup, too_many)
    assert "exceeds pool PCA rank" in score.reason
    assert np.isnan(score.perf)


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
    return from_arrays(signals, events, frame_times=times, confounds=confounds)


def test_count_without_positive_residual_df_is_unavailable_in_training_folds(fixture):
    data, model = short_run_data(fixture), fixture.task_model
    inputs = frozen_inputs(data, model, fixture.library)
    dof_limit = 30 - 4 - len(model.regressor_names)  # scans - baseline - task
    comps = inputs[2]
    assert comps[0].rank >= dof_limit
    assert all(c.unavailable_reason(dof_limit) == "" for c in comps)
    setup = setup_for(data, model, fixture.library, inputs)
    assert score_count(setup, dof_limit - 1).reason == ""
    blocked = score_count(setup, dof_limit)
    assert "degrees of freedom" in blocked.reason
    assert np.isnan(blocked.perf) and np.isnan(blocked.coefficients[1:]).all()
    chosen = select_count(data, model, fixture.library, inputs, counts=(0, dof_limit))
    assert list(chosen.scores["eligible"]) == [True, False]
    assert np.isnan(chosen.scores["perf"].iloc[1])
    rows = chosen.fold_scores[chosen.fold_scores["count"] == dof_limit]
    assert list(rows["eligible"]) == [True, False, False, False]
    assert chosen.n_components == 0


# ---- leakage: the held-out run never fits its own fold --------------------------
#
# HRFs, the pool, and the pool PCs are full-data quantities by design (GLMsingle
# selects HRFs and the pool once on all runs). What a fold must never use is the
# held-out run's BOLD to fit the task coefficients, or the held-out run's PCs in
# the target or the fit.


def test_heldout_bold_never_enters_the_fold_coefficients(fixture, base):
    inputs, setup = base
    run = 1
    rng = np.random.default_rng(3)
    signals = [s.copy() for s in fixture.data.signals]
    signals[run] += rng.normal(scale=5.0, size=signals[run].shape)
    signals[run][:, :8] *= -2.0
    perturbed = rebuild(fixture.data, signals=signals)
    other = setup_for(perturbed, fixture.task_model, fixture.library, inputs)
    for count in COUNTS:
        a, b = score_count(setup, count), score_count(other, count)
        np.testing.assert_array_equal(a.coefficients[run], b.coefficients[run])
        assert not np.allclose(a.fold_sse[run], b.fold_sse[run], equal_nan=True)
        trained_on_run = [v for v in range(4) if v != run]
        assert not np.allclose(
            a.coefficients[trained_on_run][:, :, setup.scored],
            b.coefficients[trained_on_run][:, :, setup.scored],
        )


def test_heldout_pcs_never_enter_their_own_fold(fixture, base):
    inputs, setup = base
    selection, masks, comps = inputs
    run = 2
    rng = np.random.default_rng(4)
    scrambled = list(comps)
    signal = rng.normal(size=fixture.data.signals[run].shape)
    scrambled[run] = run_components(signal, fixture.data.confounds[run], masks.pool)
    other = setup_for(
        fixture.data,
        fixture.task_model,
        fixture.library,
        (selection, masks, tuple(scrambled)),
    )
    for count in (1, 2, 4):
        a, b = score_count(setup, count), score_count(other, count)
        np.testing.assert_array_equal(a.coefficients[run], b.coefficients[run])
        np.testing.assert_array_equal(a.fold_sse[run], b.fold_sse[run])
        np.testing.assert_array_equal(a.fold_sst, b.fold_sst)
        assert not np.allclose(a.fold_sse[0], b.fold_sse[0], equal_nan=True)


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
    return rebuild(data, signals=signals)


def fold_mean_r2(score, setup, run):
    scored = setup.scored
    return float(np.mean(1 - score.fold_sse[run, scored] / score.fold_sst[run, scored]))


def naive_projected_r2(data, run, setup, inputs, count):
    """Wrong objective: also project the held-out run's PCs out of the target."""
    selection, masks, comps = inputs
    score = score_count(setup, count)
    r2 = []
    for feature in np.flatnonzero(setup.scored):
        hrf = selection.hrf_indices[feature]
        x, profiled = design_parts(
            data, run, selection.task_model, selection.library.candidates[hrf]
        )
        pcs = comps[run].prefix(count)
        nuisance = np.column_stack([baseline(data, run), profiled, pcs])
        observed = residual(nuisance, data.signals[run][:, feature])
        predicted = residual(nuisance, x) @ score.coefficients[run][:, feature]
        r2.append(1 - np.sum((observed - predicted) ** 2) / np.sum(observed**2))
    return float(np.mean(r2))


def test_task_aligned_pcs_cannot_gain_by_removing_target_variance(fixture):
    data, model, run = echo_data(fixture), fixture.task_model, 0
    inputs = frozen_inputs(data, model, fixture.library)
    echo = np.arange(26, 32)
    assert inputs[1].pool[echo].all()
    setup = setup_for(data, model, fixture.library, inputs)
    zero = score_count(setup, 0)
    for count in (2, 4):
        score = score_count(setup, count)
        honest = fold_mean_r2(score, setup, run) - fold_mean_r2(zero, setup, run)
        naive = naive_projected_r2(data, run, setup, inputs, count)
        naive_gain = naive - fold_mean_r2(zero, setup, run)
        assert naive_gain > 0.1
        assert honest < 0.25 * naive_gain
        np.testing.assert_array_equal(score.fold_sst, zero.fold_sst)


# ---- design errors and run labels ---------------------------------------------


@pytest.mark.filterwarnings("ignore:Matrix is singular:UserWarning")
def test_invalid_design_for_a_frozen_hrf_raises_naming_the_run(fixture, base):
    events = [e.copy() for e in fixture.data.events]
    events[3]["response_time"] = 0.9  # task and RT columns become collinear
    data = rebuild(fixture.data, events=events)
    # Requirement change (Task 6): every run is both a training and a held-out
    # run of the frozen design, so the message names the run, not its role.
    with pytest.raises(ValueError, match="run 'run-04': HRF .* task design is invalid"):
        setup_for(data, fixture.task_model, fixture.library, base[0])


def test_fold_messages_and_tables_use_run_labels(fixture, base):
    chosen = select_count(
        fixture.data,
        fixture.task_model,
        fixture.library,
        base[0],
        counts=(0, 50),
        run_labels=CV_LABELS,
    )
    per_fold = chosen.fold_scores
    assert list(per_fold["validation_label"]) == [x for x in CV_LABELS for _ in (0, 50)]
    reason = chosen.scores.set_index("count").loc[50, "reason"]
    assert "fold holding out run 'b2': training run 'a1': " in reason
    assert not re.search(r"\brun \d", reason)
    assert chosen.setup.run_labels == CV_LABELS


@pytest.mark.parametrize("broken", [0, 2, 3])
def test_invalid_run_design_names_only_that_run(fixture, base, broken):
    events = list(fixture.data.events)
    events[broken] = events[broken].drop(columns="response_time")
    data = rebuild(fixture.data, events=events)
    with pytest.raises(ValueError) as error:
        select_count(
            data,
            fixture.task_model,
            fixture.library,
            base[0],
            counts=(0, 1),
            run_labels=CV_LABELS,
        )
    message = str(error.value)
    assert f"run '{CV_LABELS[broken]}'" in message
    others = [label for i, label in enumerate(CV_LABELS) if i != broken]
    assert not any(label in message for label in others)
    assert not re.search(r"\brun \d", message)
