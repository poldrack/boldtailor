"""F-test significance gate on the pcstop count, against independent oracles.

The gate is a Boldtailor addition (not part of GLMsingle). For each scoring
feature the oracle fits two nested OLS models in-sample on all runs by
separate ``np.linalg.lstsq`` calls on stacked designs: task columns shared
across runs next to block-diagonal run nuisances [confounds, 1, indicators]
(reduced) and [confounds, 1, indicators, first k PCs] (full). Degrees of
freedom come from ``np.linalg.matrix_rank``, p-values from
``scipy.stats.f.sf``, and the decision from ``scipy.stats.binomtest``.
"""

from dataclasses import FrozenInstanceError
from hashlib import sha256
import inspect

import numpy as np
import pytest
from scipy import stats
from scipy.linalg import block_diag

from boldtailor._denoising_cv import prepare_scoring, select_component_count
from boldtailor._denoising_gate import apply_gate, validate_gate
from boldtailor._denoising_pool import RunComponents, run_components
from boldtailor.denoising import select_denoising
from boldtailor.denoising_results import SignificanceGate
from tests.denoising_fixtures import make_denoising_fixture
from tests.test_denoising_cv import (
    MISSING_RT,
    VARIANTS,
    baseline,
    design_parts,
    frozen_inputs,
    heldout_replaced,
    rebuild,
    setup_for,
    variant,
)

COUNTS = (0, 1, 2, 4)
ALPHA = 0.05


@pytest.fixture(scope="module")
def fixture():
    return make_denoising_fixture()


@pytest.fixture(scope="module")
def base(fixture):
    data, model = fixture.data, fixture.task_model
    inputs = frozen_inputs(data, model, fixture.library)
    return inputs, setup_for(data, model, fixture.library, inputs)


@pytest.fixture(scope="module")
def pcstop_count(fixture, base):
    inputs, _ = base
    selection, masks, comps = inputs
    return select_component_count(
        fixture.data,
        hrf_indices=selection.hrf_indices,
        components=comps,
        scoring=masks.scoring,
        task_model=fixture.task_model,
        library=fixture.library,
        counts=COUNTS,
        pcstop=1.05,
    ).n_components


def gate(setup, count, **options):
    settings = dict(enabled=True, alpha=ALPHA, binomial_alpha=ALPHA)
    settings.update(options)
    return apply_gate(setup, count, **settings)


# ---- independent oracle --------------------------------------------------------


def stacked(data, comps, features, candidate, count, model):
    """Shared task columns, block-diagonal nuisances, and stacked BOLD."""
    xs, reduced, full, ys = [], [], [], []
    for run in range(data.n_runs):
        x, profiled = design_parts(data, run, model, candidate)
        nuisance = np.column_stack([baseline(data, run), profiled])
        xs.append(x)
        reduced.append(nuisance)
        full.append(np.column_stack([nuisance, comps[run].components[:, :count]]))
        ys.append(data.signals[run][:, features])
    task = np.vstack(xs)
    return (
        np.column_stack([task, block_diag(*reduced)]),
        np.column_stack([task, block_diag(*full)]),
        np.vstack(ys),
    )


def sse(design, y):
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return np.sum((y - design @ beta) ** 2, axis=0)


def oracle_f(data, comps, features, candidate, count, model):
    reduced, full, y = stacked(data, comps, features, candidate, count, model)
    rank_r, rank_f = np.linalg.matrix_rank(reduced), np.linalg.matrix_rank(full)
    df1, df2 = rank_f - rank_r, len(y) - rank_f
    sse_r, sse_f = sse(reduced, y), sse(full, y)
    f = ((sse_r - sse_f) / df1) / (sse_f / df2)
    return f, stats.f.sf(f, df1, df2), df1, df2


def oracle_gate(data, inputs, library, model, count, scoring, indices=None):
    selection, _, comps = inputs
    ids = selection.hrf_indices if indices is None else indices
    n = data.n_features
    out = {key: np.full(n, np.nan) for key in ("f", "p", "df1", "df2")}
    for hrf in np.unique(ids[scoring]):
        features = np.flatnonzero(scoring & (ids == hrf))
        values = oracle_f(data, comps, features, library.candidates[hrf], count, model)
        for key, value in zip(("f", "p", "df1", "df2"), values):
            out[key][features] = value
    return out


# ---- settings ------------------------------------------------------------------------


def test_gate_settings_are_validated():
    assert validate_gate(True, 0.05, 0.01) == (True, 0.05, 0.01)
    assert validate_gate(False, 0.5, 0.5) == (False, 0.5, 0.5)


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, -0.1, 1.5, np.nan, np.inf, True])
def test_gate_alphas_must_be_finite_and_strictly_between_zero_and_one(value):
    with pytest.raises(ValueError, match="gate_alpha"):
        validate_gate(True, value, 0.05)
    with pytest.raises(ValueError, match="gate_binomial_alpha"):
        validate_gate(True, 0.05, value)


@pytest.mark.parametrize("value", [1, 0, "yes", None, np.True_])
def test_significance_gate_must_be_a_bool(value):
    with pytest.raises(ValueError, match="significance_gate"):
        validate_gate(value, 0.05, 0.05)


def test_gate_alpha_strings_are_rejected():
    with pytest.raises(ValueError, match="gate_alpha"):
        validate_gate(True, "0.05", 0.05)


# ---- per-feature F-tests -------------------------------------------------------------


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


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("count", (1, 2, 4))
def test_f_statistics_match_the_stacked_oracle(fixture, variant_setups, name, count):
    data, model, inputs, setup = variant_setups[name]
    want = oracle_gate(data, inputs, fixture.library, model, count, setup.scoring)
    result = gate(setup, count)
    tested = setup.scoring
    np.testing.assert_array_equal(result.tested, tested)
    assert not result.excluded.any()
    np.testing.assert_array_equal(result.df1[tested], want["df1"][tested])
    np.testing.assert_array_equal(result.df2[tested], want["df2"][tested])
    np.testing.assert_allclose(
        result.f_statistic[tested], want["f"][tested], rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        result.p_value[tested], want["p"][tested], rtol=1e-6, atol=1e-12
    )
    for array in (result.f_statistic, result.p_value, result.df1, result.df2):
        assert np.isnan(array[~tested]).all()


def test_binomial_decision_matches_scipy(base, pcstop_count):
    _, setup = base
    result = gate(setup, pcstop_count)
    tested = result.tested
    m = int(np.sum(result.p_value[tested] < ALPHA))
    n = int(tested.sum())
    expected = stats.binomtest(m, n, ALPHA, alternative="greater").pvalue
    assert (result.m, result.n) == (m, n)
    assert result.binomial_p == pytest.approx(expected, rel=1e-12)
    keep = expected < ALPHA
    assert result.decision == ("kept" if keep else "rejected")
    assert result.n_components == (pcstop_count if keep else 0)


@pytest.mark.parametrize("binomial_alpha", [0.01, 0.2])
def test_gate_alpha_settings_change_the_test_as_specified(base, binomial_alpha):
    _, setup = base
    result = gate(setup, 2, alpha=0.01, binomial_alpha=binomial_alpha)
    m = int(np.sum(result.p_value[result.tested] < 0.01))
    expected = stats.binomtest(m, result.n, 0.01, alternative="greater").pvalue
    assert result.m == m and result.binomial_p == pytest.approx(expected, rel=1e-12)
    assert (result.decision == "kept") == (expected < binomial_alpha)
    assert result.alpha == 0.01 and result.binomial_alpha == binomial_alpha


def test_shared_noise_keeps_the_pcstop_count(base, pcstop_count):
    # The fixture's shared latent series loads on the task features.
    _, setup = base
    assert pcstop_count > 0
    result = gate(setup, pcstop_count)
    assert result.decision == "kept"
    assert result.n_components == result.pcstop_count == pcstop_count
    assert result.m > ALPHA * result.n


def white_noise_components(data, seed=11, n=40):
    """Run-wise PCs of independent white noise, unrelated to every feature."""
    rng = np.random.default_rng(seed)
    return tuple(
        run_components(rng.normal(size=(len(t), n)), frame, np.ones(n, bool))
        for t, frame in zip(data.frame_times, data.confounds)
    )


def test_independent_noise_components_are_rejected(fixture, base):
    inputs, _ = base
    selection, masks, _ = inputs
    setup = prepare_scoring(
        fixture.data,
        hrf_indices=selection.hrf_indices,
        components=white_noise_components(fixture.data),
        scoring=masks.scoring,
        task_model=fixture.task_model,
        library=fixture.library,
    )
    result = gate(setup, 2)
    assert result.decision == "rejected"
    assert result.n_components == 0 and result.pcstop_count == 2
    expected = stats.binomtest(result.m, result.n, ALPHA, alternative="greater")
    assert result.binomial_p == pytest.approx(expected.pvalue, rel=1e-12)
    assert result.binomial_p >= ALPHA


def test_zero_count_skips_the_gate(base):
    _, setup = base
    result = gate(setup, 0)
    assert result.decision == "skipped_zero_count"
    assert result.n_components == result.pcstop_count == 0
    assert (result.m, result.n) == (0, 0) and np.isnan(result.binomial_p)
    assert not result.tested.any() and np.isnan(result.f_statistic).all()


def test_disabled_gate_returns_the_pcstop_count(base, pcstop_count):
    _, setup = base
    result = gate(setup, pcstop_count, enabled=False, alpha=1e-300)
    assert result.decision == "disabled" and result.enabled is False
    assert result.n_components == result.pcstop_count == pcstop_count
    assert not result.tested.any() and np.isnan(result.binomial_p)


def test_no_significant_feature_rejects(base, pcstop_count):
    _, setup = base
    result = gate(setup, pcstop_count, alpha=1e-300)
    assert result.m == 0 and result.binomial_p == pytest.approx(1.0)
    assert result.decision == "rejected" and result.n_components == 0


# ---- rank-deficient features -----------------------------------------------------------


def task_aligned_components(fixture, hrf_id, extra_seed=3):
    """Per run: the run's projected HRF-``hrf_id`` task column plus a random PC."""
    rng = np.random.default_rng(extra_seed)
    data, library = fixture.data, fixture.library
    comps = []
    for run in range(data.n_runs):
        x, _ = design_parts(data, run, fixture.task_model, library.candidates[hrf_id])
        nuisance = baseline(data, run)
        columns = np.column_stack([x[:, 0], rng.normal(size=len(x))])
        columns -= nuisance @ np.linalg.lstsq(nuisance, columns, rcond=None)[0]
        q, r = np.linalg.qr(columns)
        q *= np.sign(np.diag(r))
        comps.append(
            RunComponents(
                components=q,
                singular_values=np.array([2.0, 1.0]),
                rank_tolerance=1e-12,
                pool_size=2,
                retained_columns=2,
            )
        )
    return tuple(comps)


@pytest.fixture(scope="module")
def two_hrf_setup(fixture, base):
    """Scoring features 22-23 use HRF 1; PC 1 is HRF 1's task column."""
    inputs, _ = base
    selection, masks, _ = inputs
    indices = np.array(selection.hrf_indices)
    indices[[22, 23]] = 1
    comps = task_aligned_components(fixture, 1)
    setup = prepare_scoring(
        fixture.data,
        hrf_indices=indices,
        components=comps,
        scoring=masks.scoring,
        task_model=fixture.task_model,
        library=fixture.library,
    )
    return indices, comps, setup


def test_rank_deficient_features_are_excluded_from_n(fixture, base, two_hrf_setup):
    inputs, _ = base
    _, masks, _ = inputs
    indices, comps, setup = two_hrf_setup
    result = gate(setup, 2)
    deficient = masks.scoring & (indices == 1)
    assert deficient.sum() == 2
    np.testing.assert_array_equal(result.excluded, deficient)
    np.testing.assert_array_equal(result.tested, masks.scoring & ~deficient)
    assert result.n == int((masks.scoring & ~deficient).sum())
    assert result.n_excluded == 2
    assert np.isnan(result.f_statistic[deficient]).all()
    assert np.isnan(result.p_value[deficient]).all()
    ((hrf, size, reason),) = result.exclusions
    assert (hrf, size) == (1, 2) and "task" in reason
    # The stacked full design is rank deficient for those features only.
    n_pcs = 2 * fixture.data.n_runs
    for hrf, features in ((0, masks.scoring & (indices == 0)), (1, deficient)):
        reduced, full, _ = stacked(
            fixture.data,
            comps,
            np.flatnonzero(features),
            fixture.library.candidates[hrf],
            2,
            fixture.task_model,
        )
        added = np.linalg.matrix_rank(full) - np.linalg.matrix_rank(reduced)
        assert (added < n_pcs) == (hrf == 1)
    want = oracle_gate(
        fixture.data,
        (None, None, comps),
        fixture.library,
        fixture.task_model,
        2,
        masks.scoring & ~deficient,
        indices,
    )
    tested = result.tested
    np.testing.assert_allclose(result.f_statistic[tested], want["f"][tested], rtol=1e-6)


def test_gate_tests_exactly_the_features_pcstop_scored(fixture, base):
    # Features 0 and 7 are in the baseline span in every run; feature 1 in
    # run 3 only. pcstop scores neither, so the gate must not test them.
    inputs, _ = base
    data = fixture.data
    for run in range(data.n_runs):
        drift = data.confounds[run]["drift"].to_numpy()
        features = [0, 7, 1] if run == 3 else [0, 7]
        data = heldout_replaced(data, run, features, 50.0 + 4.0 * drift)
    setup = setup_for(data, fixture.task_model, fixture.library, inputs)
    zero = np.zeros(data.n_features, bool)
    zero[[0, 1, 7]] = True
    assert setup.scoring[zero].all()
    np.testing.assert_array_equal(setup.scored, setup.scoring & ~zero)
    result = gate(setup, 2)
    np.testing.assert_array_equal(result.tested, setup.scored)
    np.testing.assert_array_equal(result.excluded, zero)
    assert result.n == int(setup.scored.sum())
    reasons = [(size, reason) for _, size, reason in result.exclusions]
    assert sum(size for size, _ in reasons) == 3
    assert all("zero target" in reason for _, reason in reasons)
    for array in (result.f_statistic, result.p_value, result.df1, result.df2):
        assert np.isnan(array[zero]).all()
    want = oracle_gate(
        data, inputs, fixture.library, fixture.task_model, 2, setup.scored
    )
    tested = result.tested
    np.testing.assert_allclose(
        result.f_statistic[tested], want["f"][tested], rtol=1e-6, atol=1e-9
    )


def missing_everywhere(data):
    """Missing response times in every run, so every run has an indicator."""
    events = [e.copy() for e in data.events]
    for frame in events:
        frame.loc[[1, 4], "response_time"] = np.nan
    return rebuild(data, events=events)


def indicator_components(data, candidate):
    """Per run: the indicator column off the baseline, as a single unit PC."""
    comps = []
    for run in range(data.n_runs):
        _, profiled = design_parts(data, run, MISSING_RT, candidate)
        nuisance = baseline(data, run)
        fitted = nuisance @ np.linalg.lstsq(nuisance, profiled, rcond=None)[0]
        column = profiled - fitted
        comps.append(
            RunComponents(
                components=column / np.linalg.norm(column),
                singular_values=np.array([1.0]),
                rank_tolerance=1e-12,
                pool_size=1,
                retained_columns=1,
            )
        )
    return tuple(comps)


def test_pcs_inside_the_indicator_span_leave_nothing_to_test(fixture, base):
    # df1 = 0 for every feature, so n = 0: rejected with a NaN binomial p.
    _, masks, _ = base[0]
    data = missing_everywhere(fixture.data)
    setup = prepare_scoring(
        data,
        hrf_indices=np.zeros(data.n_features, dtype=int),
        components=indicator_components(data, fixture.library.candidates[0]),
        scoring=masks.scoring,
        task_model=MISSING_RT,
        library=fixture.library,
    )
    result = gate(setup, 1)
    assert result.decision == "rejected"
    assert result.n_components == 0 and result.pcstop_count == 1
    assert (result.m, result.n) == (0, 0) and np.isnan(result.binomial_p)
    assert not result.tested.any()
    np.testing.assert_array_equal(result.excluded, masks.scoring)
    ((hrf, size, reason),) = result.exclusions
    assert (hrf, size) == (0, int(masks.scoring.sum()))
    assert "no columns" in reason


def test_internal_errors_in_the_fit_are_not_hidden_as_exclusions(
    base, pcstop_count, monkeypatch
):
    import boldtailor._denoising_gate as module

    def broken(*args):
        raise ValueError("internal failure")

    _, setup = base
    monkeypatch.setattr(module, "pooled_amplitude", broken)
    with pytest.raises(ValueError, match="internal failure"):
        gate(setup, max(pcstop_count, 1))


# ---- result object ---------------------------------------------------------------------


def test_gate_arrays_are_read_only_and_the_result_is_frozen(base, pcstop_count):
    _, setup = base
    result = gate(setup, pcstop_count)
    assert isinstance(result, SignificanceGate)
    for array in (
        result.f_statistic,
        result.p_value,
        result.df1,
        result.df2,
        result.tested,
        result.excluded,
    ):
        assert not array.flags.writeable
    with pytest.raises(FrozenInstanceError):
        result.n_components = 0


# ---- public selection ------------------------------------------------------------------


def run_selection(fixture, **options):
    settings = dict(
        task_model=fixture.task_model, library=fixture.library, counts=COUNTS
    )
    settings.update(options)
    return select_denoising(fixture.data, **settings)


@pytest.fixture(scope="module")
def result(fixture):
    return run_selection(fixture)


def test_gate_defaults_are_declared_in_the_signature():
    parameters = inspect.signature(select_denoising).parameters
    assert parameters["significance_gate"].default is True
    assert parameters["gate_alpha"].default == 0.05
    assert parameters["gate_binomial_alpha"].default == 0.05


@pytest.mark.parametrize(
    "options,match",
    [
        (dict(significance_gate=1), "significance_gate"),
        (dict(gate_alpha=0.0), "gate_alpha"),
        (dict(gate_alpha=1.0), "gate_alpha"),
        (dict(gate_alpha=np.nan), "gate_alpha"),
        (dict(gate_binomial_alpha=-0.5), "gate_binomial_alpha"),
        (dict(gate_binomial_alpha=np.inf), "gate_binomial_alpha"),
        (dict(gate_binomial_alpha=True), "gate_binomial_alpha"),
    ],
)
def test_select_denoising_validates_gate_settings(fixture, options, match):
    with pytest.raises(ValueError, match=match):
        run_selection(fixture, **options)


def test_result_records_pcstop_and_gated_counts(base, pcstop_count, result):
    _, setup = base
    want = gate(setup, pcstop_count)
    assert result.pcstop_count == pcstop_count
    assert result.n_components == want.n_components
    assert isinstance(result.significance_gate, SignificanceGate)
    recorded = result.significance_gate
    assert recorded.decision == want.decision
    assert (recorded.m, recorded.n) == (want.m, want.n)
    assert recorded.binomial_p == want.binomial_p
    np.testing.assert_array_equal(recorded.f_statistic, want.f_statistic)
    np.testing.assert_array_equal(recorded.p_value, want.p_value)
    assert recorded.alpha == ALPHA and recorded.binomial_alpha == ALPHA


def test_disabled_gate_reproduces_pcstop_only_selection(fixture, pcstop_count):
    disabled = run_selection(fixture, significance_gate=False)
    assert disabled.n_components == disabled.pcstop_count == pcstop_count
    assert disabled.significance_gate.decision == "disabled"
    assert [c.shape[1] for c in disabled.run_components] == [pcstop_count] * 4


def test_rejected_gate_returns_zero_components(fixture, pcstop_count):
    rejected = run_selection(fixture, gate_alpha=1e-300)
    assert rejected.pcstop_count == pcstop_count > 0
    assert rejected.n_components == 0
    assert rejected.significance_gate.decision == "rejected"
    assert rejected.component_names == ()
    for run, components in enumerate(rejected.run_components):
        assert components.shape == (len(fixture.data.frame_times[run]), 0)


def digest(array, dtype="<f8"):
    return sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def test_provenance_records_the_gate(result):
    activity = result.provenance.to_dict()["activities"][-1]
    recorded = result.significance_gate
    assert activity["count_rule"] == "glmsingle_pcstop"
    assert activity["n_components"] == result.n_components
    assert activity["significance_gate"] == dict(
        enabled=True,
        test="ols_nested_f_in_sample_all_runs",
        aggregate="one_sided_binomial_over_tested_features",
        gate_alpha=ALPHA,
        gate_binomial_alpha=ALPHA,
        pcstop_count=result.pcstop_count,
        n_components=result.n_components,
        decision=recorded.decision,
        m=recorded.m,
        n=recorded.n,
        n_excluded=recorded.n_excluded,
        binomial_p=recorded.binomial_p,
        f_statistic_fingerprint=digest(recorded.f_statistic),
        p_value_fingerprint=digest(recorded.p_value),
        exclusions=[],
    )


def test_result_rejects_an_inconsistent_gate(result):
    from dataclasses import replace

    with pytest.raises(ValueError, match="significance_gate"):
        replace(result, pcstop_count=result.pcstop_count + 1)
