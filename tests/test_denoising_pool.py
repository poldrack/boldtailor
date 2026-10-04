"""Noise-pool masks from training HRF-selection scores and normalized pool PCA."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import orth

from boldtailor._denoising_pool import (
    PoolMasks,
    analysis_components,
    pool_masks,
    run_components,
)
from boldtailor.hrf_selection import select_hrfs, subset_runs
from boldtailor.model import TaskModel
from tests.denoising_fixtures import GROUP_SIZES, make_denoising_fixture

N_FEATURES = sum(GROUP_SIZES.values())


@pytest.fixture(scope="module")
def fixture():
    return make_denoising_fixture()


@pytest.fixture(scope="module")
def selection(fixture):
    return select_hrfs(
        fixture.data, library=fixture.library, task_model=fixture.task_model
    )


def with_scores(selection, scores, indices=None):
    indices = selection.hrf_indices if indices is None else indices
    return replace(selection, cv_r2=np.asarray(scores, float), hrf_indices=indices)


def oracle_normalized_pool(y, confounds):
    """Independent lstsq projection off [confounds, 1]; drop zeros; unit norm."""
    nuisance = np.column_stack([confounds, np.ones(len(y))])
    residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    norms = np.linalg.norm(residual, axis=0)
    keep = norms > 1e-9 * np.maximum(np.linalg.norm(y, axis=0), 1)
    return residual[:, keep] / norms[keep]


def projector(basis):
    q = orth(basis)
    return q @ q.T


# ---- pool and scoring masks -------------------------------------------------


def test_threshold_boundary_goes_to_pool(selection):
    n = len(selection.cv_r2)
    scores = np.linspace(-0.2, 0.2, n)
    scores[3] = 0.05
    masks = pool_masks(with_scores(selection, scores), np.ones(n, bool), 0.05)
    assert masks.pool[3] and not masks.scoring[3]
    np.testing.assert_array_equal(masks.pool, scores <= 0.05)
    np.testing.assert_array_equal(masks.scoring, scores > 0.05)
    assert masks.pool_size == int(np.sum(scores <= 0.05))
    assert masks.scoring_size == n - masks.pool_size


def test_nonfinite_scores_and_undefined_hrfs_are_in_neither_mask(selection):
    n = len(selection.cv_r2)
    scores = np.full(n, -0.5)
    scores[:3] = [np.nan, -np.inf, np.inf]
    scores[3:5] = 0.9
    indices = np.zeros(n, dtype=int)
    indices[5] = -1
    indices[3] = -1
    masks = pool_masks(with_scores(selection, scores, indices), np.ones(n, bool), 0)
    excluded = [0, 1, 2, 3, 5]
    assert not masks.pool[excluded].any() and not masks.scoring[excluded].any()
    assert masks.scoring[4] and masks.pool[6:].all()


def test_brain_mask_restricts_both_masks_and_masks_are_disjoint(fixture, selection):
    masks = pool_masks(selection, fixture.brain_mask, 0.0)
    assert not (masks.pool & masks.scoring).any()
    assert not (masks.pool | masks.scoring)[~fixture.brain_mask].any()


def test_masks_are_owned_and_readonly(fixture, selection):
    brain = fixture.brain_mask.copy()
    masks = pool_masks(selection, brain, 0.0)
    before = masks.pool.copy()
    brain[:] = False
    np.testing.assert_array_equal(masks.pool, before)
    assert not masks.pool.flags.writeable and not masks.scoring.flags.writeable
    assert masks.pool.dtype == bool and masks.threshold == 0.0


@pytest.mark.parametrize(
    "mask",
    [
        np.ones(3, bool),
        np.ones((N_FEATURES, 1), bool),
        np.ones(N_FEATURES, int),
        np.ones(N_FEATURES, float),
        [True] * (N_FEATURES - 1),
    ],
)
def test_brain_mask_must_be_boolean_vector_in_feature_order(selection, mask):
    with pytest.raises(ValueError, match="brain_mask"):
        pool_masks(selection, mask, 0.0)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf, "0", None, True])
def test_threshold_must_be_finite_real(fixture, selection, threshold):
    with pytest.raises(ValueError, match="pool_r2_threshold"):
        pool_masks(selection, fixture.brain_mask, threshold)


def test_selection_must_be_an_hrf_selection_result(fixture):
    with pytest.raises(ValueError, match="HrfSelectionResult"):
        pool_masks(np.zeros(N_FEATURES), fixture.brain_mask, 0.0)


def test_empty_pool_is_reported_not_substituted(fixture, selection):
    masks = pool_masks(selection, fixture.brain_mask, -1.0)
    assert masks.pool_size == 0 and not masks.pool.any()
    comps = analysis_components(fixture.data, masks.pool)
    for run, comp in zip(fixture.data.signals, comps):
        assert comp.rank == 0 and comp.components.shape == (len(run), 0)
        assert comp.unavailable_reason(0) == ""
        assert "empty noise pool" in comp.unavailable_reason(1)


def test_masks_must_be_disjoint():
    with pytest.raises(ValueError, match="disjoint"):
        PoolMasks(pool=[True, False], scoring=[True, True], threshold=0.0)


# ---- task-model time-series scores define the pool -------------------------


def test_strong_task_features_with_rt_free_trial_betas_are_not_pooled(
    fixture, selection
):
    """Trial amplitudes are orthogonal to RT, so a beta-encoding score fails."""
    groups = fixture.groups
    for amps, rt in zip(fixture.trial_amplitudes, fixture.response_times):
        centered = rt - rt.mean()
        np.testing.assert_allclose(centered @ (amps - amps.mean(axis=0)), 0, atol=1e-9)
    masks = pool_masks(selection, fixture.brain_mask, 0.0)
    assert masks.scoring[groups["task"]].all()
    assert not masks.pool[groups["task"]].any()
    assert masks.scoring[groups["rt"]].all()


def test_pool_is_the_nontask_in_brain_features(fixture, selection):
    groups = fixture.groups
    masks = pool_masks(selection, fixture.brain_mask, 0.0)
    np.testing.assert_array_equal(np.flatnonzero(masks.pool), groups["noise"])
    np.testing.assert_array_equal(
        np.flatnonzero(masks.scoring), np.r_[groups["task"], groups["rt"]]
    )


def test_constant_features_are_in_neither_mask(fixture, selection):
    masks = pool_masks(selection, fixture.brain_mask, 0.0)
    constant = fixture.groups["constant"]
    assert not masks.pool[constant].any() and not masks.scoring[constant].any()


def test_masks_follow_training_selection_scores(fixture):
    training = subset_runs(fixture.data, [0, 1, 2])
    train_selection = select_hrfs(
        training, library=fixture.library, task_model=fixture.task_model
    )
    masks = pool_masks(train_selection, fixture.brain_mask, 0.0)
    scores = train_selection.cv_r2
    expected = fixture.brain_mask & np.isfinite(scores) & (scores <= 0.0)
    np.testing.assert_array_equal(masks.pool, expected)


def test_plain_task_model_also_keeps_task_features_out_of_pool(fixture):
    plain = select_hrfs(fixture.data, library=fixture.library, task_model=TaskModel())
    masks = pool_masks(plain, fixture.brain_mask, 0.0)
    assert not masks.pool[fixture.groups["task"]].any()


# ---- normalized pool PCA: rank-two oracle ------------------------------------


@pytest.fixture
def rank_two():
    rng = np.random.default_rng(7)
    length = 40
    t = np.arange(length)
    confounds = pd.DataFrame(
        dict(motion=rng.normal(size=length), drift=(t - t.mean()) / length)
    )
    latent = rng.normal(size=(length, 2))
    mixing = rng.normal(size=(2, 6)) * np.array([1, 50, 0.01, 3, 7, 0.2])
    contamination = confounds.to_numpy() @ rng.normal(size=(2, 6)) + 40
    pool_part = latent @ mixing + contamination
    in_nuisance = confounds.to_numpy() @ [2.0, -1.0] + 5.0
    other = rng.normal(size=(length, 3))
    y = np.column_stack(
        [pool_part[:, :3], np.full(length, 9.0), in_nuisance, pool_part[:, 3:], other]
    )
    pool = np.r_[np.ones(8, bool), np.zeros(3, bool)]
    return y, confounds, pool


def test_rank_two_oracle_span_orthonormal_and_nuisance_orthogonal(rank_two):
    y, confounds, pool = rank_two
    result = run_components(y, confounds, pool)
    normalized = oracle_normalized_pool(y[:, pool], confounds.to_numpy())
    assert normalized.shape[1] == 6  # constant and in-nuisance columns discarded
    assert result.rank == 2 and result.retained_columns == 6
    assert result.pool_size == 8
    c = result.components
    assert c.shape == (len(y), 2)
    np.testing.assert_allclose(c.T @ c, np.eye(2), atol=1e-12)
    nuisance = np.column_stack([confounds, np.ones(len(y))])
    np.testing.assert_allclose(nuisance.T @ c, 0, atol=1e-10)
    np.testing.assert_allclose(projector(c), projector(normalized), atol=1e-10)


def test_singular_values_are_those_of_the_normalized_projected_pool(rank_two):
    y, confounds, pool = rank_two
    result = run_components(y, confounds, pool)
    normalized = oracle_normalized_pool(y[:, pool], confounds.to_numpy())
    expected = np.linalg.svd(normalized, compute_uv=False)
    np.testing.assert_allclose(result.singular_values[:2], expected[:2], rtol=1e-10)
    assert np.all(np.diff(result.singular_values) <= 0)
    u = np.linalg.svd(normalized, full_matrices=False)[0]
    for k in range(2):
        assert abs(abs(u[:, k] @ result.components[:, k]) - 1) < 1e-10


def test_sign_convention_largest_absolute_entry_positive(rank_two):
    y, confounds, pool = rank_two
    c = run_components(y, confounds, pool).components
    flipped = run_components(-y, confounds, pool).components
    for k in range(c.shape[1]):
        assert c[np.argmax(np.abs(c[:, k])), k] > 0
    np.testing.assert_allclose(flipped, c, atol=1e-12)


def test_column_scaling_does_not_change_components(rank_two):
    y, confounds, pool = rank_two
    scaled = y * np.r_[1e3, 1e-2, 5, 1, 1, 0.3, 80, 2, 1, 1, 1]
    a = run_components(y, confounds, pool)
    b = run_components(scaled, confounds, pool)
    np.testing.assert_allclose(a.components, b.components, atol=1e-10)
    np.testing.assert_allclose(a.singular_values, b.singular_values, atol=1e-10)


def test_components_ignore_nonpool_features(rank_two):
    y, confounds, pool = rank_two
    changed = y.copy()
    changed[:, ~pool] *= -17
    a = run_components(y, confounds, pool).components
    np.testing.assert_array_equal(
        a, run_components(changed, confounds, pool).components
    )


def test_outputs_are_owned_and_readonly(rank_two):
    y, confounds, pool = rank_two
    y, pool = y.copy(), pool.copy()
    result = run_components(y, confounds, pool)
    before = result.components.copy()
    y[:] = 0
    pool[:] = False
    np.testing.assert_array_equal(result.components, before)
    assert not result.components.flags.writeable
    assert not result.singular_values.flags.writeable


def test_duplicate_nuisance_columns_do_not_create_components(rank_two):
    y, confounds, pool = rank_two
    duplicated = confounds.assign(
        motion_copy=confounds.motion, double=2 * confounds.drift
    )
    a = run_components(y, confounds, pool)
    b = run_components(y, duplicated, pool)
    assert b.rank == 2
    np.testing.assert_allclose(
        projector(a.components), projector(b.components), atol=1e-10
    )


def test_all_pool_columns_in_nuisance_span_give_rank_zero(rank_two):
    y, confounds, _ = rank_two
    pool = np.zeros(y.shape[1], bool)
    pool[[3, 4]] = True  # constant and pure-nuisance columns
    result = run_components(y, confounds, pool)
    assert result.rank == 0 and result.retained_columns == 0
    assert result.components.shape == (len(y), 0)
    assert "no signal outside" in result.unavailable_reason(1)


def test_short_run_rank_is_capped_by_residual_dimension():
    rng = np.random.default_rng(3)
    length = 6
    confounds = pd.DataFrame(dict(motion=rng.normal(size=length)))
    y = rng.normal(size=(length, 20))
    result = run_components(y, confounds, np.ones(20, bool))
    assert result.rank == length - 2
    c = result.components
    np.testing.assert_allclose(c.T @ c, np.eye(length - 2), atol=1e-12)
    assert result.unavailable_reason(length - 2) == ""
    assert "exceeds" in result.unavailable_reason(length - 1)


def test_reserved_constant_confound_is_rejected(rank_two):
    y, confounds, pool = rank_two
    with pytest.raises(ValueError, match="constant"):
        run_components(y, confounds.assign(constant=1.0), pool)


@pytest.mark.parametrize("pool", [np.ones(3, bool), np.ones(11, int)])
def test_pool_must_be_boolean_feature_mask(rank_two, pool):
    y, confounds, _ = rank_two
    with pytest.raises(ValueError, match="pool"):
        run_components(y, confounds, pool)


# ---- tied singular-value blocks --------------------------------------------


@pytest.fixture
def tied():
    """Columns u1, u2, u3, u3 orthogonal to nuisance: s = [sqrt2, 1, 1]."""
    rng = np.random.default_rng(11)
    length = 30
    confounds = pd.DataFrame(dict(motion=rng.normal(size=length)))
    nuisance = np.column_stack([confounds, np.ones(length)])
    raw = rng.normal(size=(length, 3))
    raw -= nuisance @ np.linalg.lstsq(nuisance, raw, rcond=None)[0]
    u = np.linalg.qr(raw)[0]
    y = np.column_stack([u[:, 2], u[:, 0], u[:, 1], u[:, 0]]) * [4, 1, 9, 0.5]
    return y, confounds, np.ones(4, bool)


def test_tied_block_cutoff_is_unavailable(tied):
    y, confounds, pool = tied
    result = run_components(y, confounds, pool)
    assert result.rank == 3
    np.testing.assert_allclose(result.singular_values[:3], [np.sqrt(2), 1, 1])
    assert not result.splits_tied_block(1)
    assert result.splits_tied_block(2)
    assert not result.splits_tied_block(3)
    assert result.unavailable_reason(1) == ""
    assert "tied" in result.unavailable_reason(2)
    assert result.unavailable_reason(3) == ""


def test_prefix_returns_leading_components_and_rejects_unavailable(tied):
    y, confounds, pool = tied
    result = run_components(y, confounds, pool)
    assert result.prefix(0).shape == (len(y), 0)
    np.testing.assert_array_equal(result.prefix(1), result.components[:, :1])
    assert not result.prefix(1).flags.writeable
    with pytest.raises(ValueError, match="tied"):
        result.prefix(2)
    with pytest.raises(ValueError, match="exceeds"):
        result.prefix(4)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="count"):
            result.prefix(bad)


def test_available_cutoffs_define_a_unique_subspace(tied):
    y, confounds, pool = tied
    a = run_components(y, confounds, pool)
    b = run_components(y[:, [3, 2, 1, 0]], confounds, pool)
    for count in (1, 3):
        np.testing.assert_allclose(
            projector(a.prefix(count)), projector(b.prefix(count)), atol=1e-10
        )


# ---- run-wise components on the synthetic analysis ---------------------------


def test_analysis_components_are_per_run_and_recover_latent(fixture, selection):
    masks = pool_masks(selection, fixture.brain_mask, 0.0)
    comps = analysis_components(fixture.data, masks.pool)
    assert len(comps) == fixture.data.n_runs
    for signal, frame, comp, latent in zip(
        fixture.data.signals, fixture.data.confounds, comps, fixture.latent
    ):
        assert comp.components.shape[0] == len(signal)
        assert comp.pool_size == masks.pool_size
        nuisance = np.column_stack([frame, np.ones(len(signal))])
        target = latent - nuisance @ np.linalg.lstsq(nuisance, latent, rcond=None)[0]
        first = comp.prefix(1)[:, 0]
        assert abs(np.corrcoef(first, target)[0, 1]) > 0.95
        expected = run_components(signal, frame, masks.pool)
        np.testing.assert_array_equal(comp.components, expected.components)
