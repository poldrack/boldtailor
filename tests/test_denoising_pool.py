"""GLMsingle ON-OFF R² pool statistic, pool/scoring masks, and pool PCA.

The ON-OFF oracle is one stacked least-squares fit: a single canonical-HRF
task column (every trial, amplitude 1) shared by all runs next to a
block-diagonal matrix of each run's [confounds, 1]. Its denominator is each
run's energy after projecting off that run's [confounds, 1] alone.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag, orth

from boldtailor._denoising_pool import (
    FALLBACK_SIZE,
    PoolMasks,
    analysis_components,
    onoff_r2,
    pool_masks,
    run_components,
)
from boldtailor._hrf_design import MIN_ONSET, OVERSAMPLING, hrf_model
from boldtailor._mixture_threshold import mixture_threshold
from boldtailor._task_design import run_task_columns
from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import subset_runs
from boldtailor.model import TaskModel
from tests.denoising_fixtures import GROUP_SIZES, make_denoising_fixture

N_FEATURES = sum(GROUP_SIZES.values())
EPS = np.finfo(float).eps


@pytest.fixture(scope="module")
def fixture():
    return make_denoising_fixture()


@pytest.fixture(scope="module")
def statistic(fixture):
    return onoff_r2(fixture.data, fixture.library)


def rebuild(data, *, signals=None, events=None, confounds=None):
    return from_arrays(
        list(data.signals if signals is None else signals),
        list(data.events if events is None else events),
        frame_times=list(data.frame_times),
        confounds=list(data.confounds if confounds is None else confounds),
    )


def canonical_regressor(data, run, library):
    columns = run_task_columns(
        data.events[run],
        TaskModel(),
        data.frame_times[run],
        hrf_model(library.candidates[0]),
        run=run,
        min_onset=MIN_ONSET,
        oversampling=OVERSAMPLING,
    )
    return columns[["task"]].to_numpy()


def run_nuisance(data, run):
    confounds = data.confounds[run].to_numpy(dtype=float)
    return np.column_stack([confounds, np.ones(len(confounds))])


def oracle_onoff(data, library):
    """Stacked lstsq: shared task column, block-diagonal run nuisance."""
    nuisance = block_diag(*[run_nuisance(data, r) for r in range(data.n_runs)])
    task = np.vstack(
        [canonical_regressor(data, r, library) for r in range(data.n_runs)]
    )
    y = np.vstack(data.signals)
    full = np.column_stack([task, nuisance])
    sse = np.sum((y - full @ np.linalg.lstsq(full, y, rcond=None)[0]) ** 2, axis=0)
    sst = np.sum((y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]) ** 2, 0)
    # Numerically zero residual energy (constant features) is undefined.
    zero = np.sqrt(sst) <= np.linalg.norm(y, axis=0) * len(y) * EPS
    r2 = np.full(y.shape[1], np.nan)
    r2[~zero] = 1 - sse[~zero] / sst[~zero]
    return r2


def oracle_normalized_pool(y, confounds):
    """Independent lstsq projection off [confounds, 1]; drop zeros; unit norm."""
    nuisance = np.column_stack([confounds, np.ones(len(y))])
    residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    norms = np.linalg.norm(residual, axis=0)
    keep = norms > 1e-9 * np.maximum(np.linalg.norm(y, axis=0), 1)
    return residual[:, keep] / norms[keep]


def projector(basis, rcond=None):
    q = orth(basis, rcond=rcond)
    return q @ q.T


def defined(n=N_FEATURES):
    return np.zeros(n, dtype=int)


# ---- GLMsingle ON-OFF R² ---------------------------------------------------------


def test_onoff_r2_matches_the_stacked_oracle(fixture, statistic):
    expected = oracle_onoff(fixture.data, fixture.library)
    np.testing.assert_array_equal(np.isnan(statistic), np.isnan(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(statistic[finite], expected[finite], rtol=0, atol=1e-10)
    assert not statistic.flags.writeable and statistic.shape == (N_FEATURES,)


def test_onoff_r2_is_nan_for_zero_residual_energy(fixture, statistic):
    assert np.isnan(statistic[fixture.groups["constant"]]).all()
    others = np.setdiff1d(np.arange(N_FEATURES), fixture.groups["constant"])
    assert np.isfinite(statistic[others]).all()


def test_onoff_r2_separates_task_from_noise_features(fixture, statistic):
    groups = fixture.groups
    task = np.r_[groups["task"], groups["rt"], groups["outside_task"]]
    noise = np.r_[groups["noise"], groups["outside_noise"]]
    assert statistic[task].min() > 0.2 and statistic[noise].max() < 0.05


def test_onoff_r2_ignores_modulators_and_uses_canonical_hrf(fixture, statistic):
    """Amplitude-1 trials only: RT values and non-canonical candidates are unused."""
    events = [
        e.assign(response_time=e.response_time[::-1].to_numpy())
        for e in fixture.data.events
    ]
    changed = rebuild(fixture.data, events=events)
    library = HrfLibrary.from_parameters([[8, 20, 2, 3, 10, 3, 36]])
    np.testing.assert_array_equal(onoff_r2(changed, fixture.library), statistic)
    np.testing.assert_allclose(
        onoff_r2(fixture.data, library), statistic, rtol=0, atol=1e-12, equal_nan=True
    )


def test_onoff_r2_handles_categorical_events(fixture, statistic):
    events = [
        e.assign(cond=np.where(np.arange(len(e)) % 2 == 0, "a", "b"))
        for e in fixture.data.events
    ]
    changed = rebuild(fixture.data, events=events)
    np.testing.assert_array_equal(onoff_r2(changed, fixture.library), statistic)


def test_onoff_r2_uses_every_run_with_its_own_nuisance(fixture):
    subset = subset_runs(fixture.data, [0, 2, 3])
    expected = oracle_onoff(subset, fixture.library)
    got = onoff_r2(subset, fixture.library)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10, equal_nan=True)


def test_onoff_design_errors_name_the_run_label(fixture):
    events = list(fixture.data.events)
    events[2] = events[2].copy()
    events[2].loc[0, "onset"] = fixture.data.frame_times[2][-1] + 10.0
    data = rebuild(fixture.data, events=events)
    with pytest.raises(ValueError, match="run 'sesC'"):
        onoff_r2(data, fixture.library, run_labels=("sesA", "sesB", "sesC", "sesD"))


# ---- pool and scoring masks (GLMsingle badR2 / pcR2cutoff) ---------------------


def test_pool_is_below_and_scoring_above_the_threshold(statistic):
    values = np.linspace(-0.2, 0.2, N_FEATURES)
    values[3] = 0.05
    masks = pool_masks(values, 0.05, hrf_indices=defined())
    assert not masks.pool[3] and not masks.scoring[3]  # exactly at the threshold
    np.testing.assert_array_equal(masks.pool, values < 0.05)
    np.testing.assert_array_equal(masks.scoring, values > 0.05)
    assert masks.pool_size == int(np.sum(values < 0.05))
    assert masks.scoring_size == int(np.sum(values > 0.05))
    assert masks.threshold == 0.05 and masks.mixture is None and not masks.fallback


def test_nonfinite_values_are_in_neither_mask():
    values = np.full(N_FEATURES, -0.5)
    values[:3] = [np.nan, -np.inf, np.inf]
    values[3:5] = 0.9
    masks = pool_masks(values, 0.0, hrf_indices=defined())
    assert not (masks.pool | masks.scoring)[:3].any()
    assert masks.scoring[3:5].all() and masks.pool[5:].all()


def test_undefined_hrfs_are_excluded_from_scoring_only():
    values = np.full(N_FEATURES, -0.5)
    values[:4] = 0.9
    indices = defined()
    indices[[0, 5]] = -1
    masks = pool_masks(values, 0.0, hrf_indices=indices)
    assert not masks.scoring[0] and masks.scoring[1:4].all()
    assert masks.pool[5]  # the pool does not depend on HRF assignments


def test_every_feature_is_a_candidate_without_any_mask_parameter():
    import inspect

    parameters = inspect.signature(pool_masks).parameters
    assert not any("mask" in name for name in parameters)
    values = np.where(np.arange(N_FEATURES) % 2 == 0, -0.5, 0.5)
    masks = pool_masks(values, 0.0, hrf_indices=defined())
    np.testing.assert_array_equal(masks.pool | masks.scoring, np.ones(N_FEATURES, bool))


def test_masks_are_owned_and_readonly():
    values = np.linspace(-1, 1, N_FEATURES)
    masks = pool_masks(values, 0.0, hrf_indices=defined())
    before = masks.pool.copy()
    values[:] = 5.0
    np.testing.assert_array_equal(masks.pool, before)
    assert not masks.pool.flags.writeable and not masks.scoring.flags.writeable
    assert masks.pool.dtype == bool


@pytest.mark.parametrize(
    "threshold", [np.nan, np.inf, -np.inf, "0", "Auto", "", None, True]
)
def test_threshold_must_be_auto_or_finite_real(threshold):
    with pytest.raises(ValueError, match="pool_r2_threshold"):
        pool_masks(np.zeros(N_FEATURES), threshold, hrf_indices=defined())


@pytest.mark.parametrize(
    "values", [np.zeros(3), np.zeros((N_FEATURES, 1)), ["a"] * N_FEATURES]
)
def test_statistic_must_be_a_real_vector_in_feature_order(values):
    with pytest.raises(ValueError, match="statistic"):
        pool_masks(values, 0.0, hrf_indices=defined())


def test_masks_must_be_disjoint_unless_the_fallback_was_used():
    with pytest.raises(ValueError, match="disjoint"):
        PoolMasks(pool=[True, False], scoring=[True, True], threshold=0.0)
    masks = PoolMasks(
        pool=[True, False], scoring=[True, True], threshold=0.0, fallback=True
    )
    assert masks.fallback


# ---- automatic (GLMsingle findtailthreshold) threshold ----------------------------


def bimodal_statistic(n=N_FEATURES, seed=5):
    """Two thirds of features near 0 (SD 0.01), the rest near 0.5 (SD 0.1)."""
    rng = np.random.default_rng(seed)
    low = n * 2 // 3
    return np.concatenate([rng.normal(0, 0.01, low), rng.normal(0.5, 0.1, n - low)])


def test_auto_threshold_is_the_mixture_threshold_of_all_finite_values():
    values = bimodal_statistic()
    values[1] = np.nan
    indices = defined()
    indices[2] = -1  # undefined HRFs still inform the threshold
    masks = pool_masks(values, "auto", hrf_indices=indices)
    expected = mixture_threshold(values[np.isfinite(values)])
    assert masks.threshold == expected.threshold and masks.mixture == expected
    finite = np.isfinite(values)
    np.testing.assert_array_equal(masks.pool, finite & (values < expected.threshold))
    np.testing.assert_array_equal(
        masks.scoring, finite & (indices >= 0) & (values > expected.threshold)
    )
    assert 0 < masks.pool_size and 0 < masks.scoring_size


def test_auto_threshold_on_the_fixture_pools_the_noise_features(fixture, statistic):
    masks = pool_masks(statistic, "auto", hrf_indices=defined())
    groups = fixture.groups
    np.testing.assert_array_equal(
        np.flatnonzero(masks.pool), np.r_[groups["noise"], groups["outside_noise"]]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(masks.scoring),
        np.r_[groups["task"], groups["rt"], groups["outside_task"]],
    )


def test_degenerate_auto_threshold_names_the_context_and_suggests_a_fixed_one():
    with pytest.raises(ValueError, match="pool_r2_threshold") as error:
        pool_masks(
            np.full(N_FEATURES, 0.1),
            "auto",
            hrf_indices=defined(),
            context="the noise pool (runs 'sesA', 'sesB')",
        )
    message = str(error.value)
    assert "the noise pool (runs 'sesA', 'sesB')" in message and "distinct" in message
    assert "fixed pool_r2_threshold" in message


# ---- best-100 fallback (GLMsingle) ------------------------------------------------


def test_fallback_scores_the_100_highest_defined_features():
    rng = np.random.default_rng(2)
    n = 150
    values = rng.permutation(np.linspace(-0.5, 0.0, n))
    values[7] = np.nan
    indices = np.zeros(n, dtype=int)
    best = np.argsort(-np.nan_to_num(values, nan=-np.inf))
    indices[best[0]] = -1  # the very best has no HRF, so it is skipped
    masks = pool_masks(values, 0.5, hrf_indices=indices)
    assert masks.fallback and FALLBACK_SIZE == 100
    expected = np.zeros(n, dtype=bool)
    expected[best[1:101]] = True
    np.testing.assert_array_equal(masks.scoring, expected)
    np.testing.assert_array_equal(masks.pool, np.isfinite(values))


def test_fallback_with_fewer_candidates_scores_all_of_them():
    values = np.linspace(-0.5, 0.0, N_FEATURES)
    values[0] = np.nan
    masks = pool_masks(values, 1.0, hrf_indices=defined())
    assert masks.fallback
    np.testing.assert_array_equal(masks.scoring, np.isfinite(values))


def test_no_fallback_when_some_feature_passes():
    values = np.linspace(-0.5, 0.0, N_FEATURES)
    values[4] = 2.0
    masks = pool_masks(values, 1.0, hrf_indices=defined())
    assert not masks.fallback and np.flatnonzero(masks.scoring).tolist() == [4]


def test_empty_pool_is_reported_not_substituted(fixture, statistic):
    masks = pool_masks(statistic, -1.0, hrf_indices=defined())
    assert masks.pool_size == 0 and not masks.pool.any()
    comps = analysis_components(fixture.data, masks.pool)
    for run, comp in zip(fixture.data.signals, comps):
        assert comp.rank == 0 and comp.components.shape == (len(run), 0)
        assert comp.unavailable_reason(0) == ""
        assert "empty noise pool" in comp.unavailable_reason(1)


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
    oracle = projector(normalized, rcond=1e-9)  # roundoff of tiny columns
    assert np.linalg.matrix_rank(oracle) == 2
    np.testing.assert_allclose(projector(c), oracle, atol=1e-10)


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


def test_analysis_components_are_per_run_and_recover_latent(fixture, statistic):
    masks = pool_masks(statistic, "auto", hrf_indices=defined())
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
