import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import boldtailor.reml as reml
from boldtailor.reml import (
    RemlGroup,
    RemlSelection,
    fit_reml,
    select_random_effects_design,
)
from tests.reml_oracles import simulate

N_FEATURES = 4
FIT_FIELDS = ("ratio", "noise_variance", "effect_variance", "criterion", "at_upper")


def _groups(signals, fixed):
    return [RemlGroup(y, c) for y, c in zip(signals, fixed, strict=True)]


def _columns(signals, index):
    return [y[:, index] for y in signals]


def _decoys(rng, fixed, count, n_random=3):
    return [[rng.normal(size=(len(c), n_random)) for c in fixed] for _ in range(count)]


@pytest.fixture
def planted():
    """Signals simulated from candidate 2 of 5 at ratio 5."""
    rng = np.random.default_rng(3)
    signals, fixed, random = simulate(rng, ratio=5.0, n_features=N_FEATURES)
    candidates = _decoys(rng, fixed, 4)
    candidates.insert(2, random)
    return signals, fixed, candidates


def _assert_same_fit(fit, expected):
    for name in FIT_FIELDS:
        assert_array_equal(getattr(fit, name), getattr(expected, name))
    assert fit.status == expected.status
    assert_array_equal(fit.rho, expected.rho)
    assert fit.restricted_dof == expected.restricted_dof


def _orthogonal_signals(rng, fixed, candidates):
    """Signals with no component along any candidate beyond the fixed design."""
    signals = []
    for index, c in enumerate(fixed):
        span = np.column_stack([c] + [z[index] for z in candidates])
        noise = rng.normal(size=(len(c), 2))
        residual = noise - span @ np.linalg.lstsq(span, noise, rcond=None)[0]
        signals.append(residual + c @ rng.normal(size=(c.shape[1], 2)))
    return signals


@pytest.mark.integration
def test_planted_design_wins(planted):
    """Verify planted design wins."""
    signals, fixed, candidates = planted
    groups = _groups(signals, fixed)
    selection = select_random_effects_design(groups, candidates)
    assert isinstance(selection, RemlSelection)
    assert selection.n_candidates == 5
    assert selection.winner.dtype == np.int64
    assert_array_equal(selection.winner, np.full(N_FEATURES, 2))
    assert (selection.gap > 0).all()
    _assert_same_fit(selection.fit, fit_reml(groups, candidates[2]))


def test_results_are_read_only(planted):
    """Verify results are read only."""
    signals, fixed, candidates = planted
    selection = select_random_effects_design(_groups(signals, fixed), candidates)
    with pytest.raises(ValueError):
        selection.winner[0] = 0
    with pytest.raises(ValueError):
        selection.gap[0] = 0.0


def test_ties_go_to_earlier_candidate(planted):
    """Verify ties go to earlier candidate."""
    signals, fixed, candidates = planted
    design = candidates[2]
    selection = select_random_effects_design(_groups(signals, fixed), [design, design])
    assert_array_equal(selection.winner, np.zeros(N_FEATURES))
    assert_array_equal(selection.gap, np.zeros(N_FEATURES))


def test_no_supported_candidate_gives_sentinel():
    """Verify no supported candidate gives sentinel."""
    rng = np.random.default_rng(4)
    _, fixed, _ = simulate(rng)
    candidates = _decoys(rng, fixed, 3)
    signals = _orthogonal_signals(rng, fixed, candidates)
    groups = _groups(signals, fixed)
    selection = select_random_effects_design(groups, candidates)
    at_zero = fit_reml(groups, [None] * len(groups))
    assert_array_equal(selection.winner, [-1, -1])
    assert np.isnan(selection.gap).all()
    assert selection.fit.status == ("variance_zero", "variance_zero")
    assert_array_equal(selection.fit.ratio, 0.0)
    assert_array_equal(selection.fit.effect_variance, 0.0)
    assert_allclose(selection.fit.criterion, at_zero.criterion, rtol=1e-12)
    assert_allclose(selection.fit.noise_variance, at_zero.noise_variance, rtol=1e-12)


def test_no_residual_signal_feature_gives_sentinel(planted):
    """Verify no residual signal feature gives sentinel."""
    signals, fixed, candidates = planted
    signals = [np.column_stack([y[:, 0], np.zeros(len(y)), y[:, 1]]) for y in signals]
    groups = _groups(signals, fixed)
    selection = select_random_effects_design(groups, candidates)
    reference = select_random_effects_design(
        _groups(_columns(signals, [0, 2]), fixed), candidates
    )
    assert_array_equal(selection.winner, [2, -1, 2])
    assert np.isnan(selection.gap[1])
    assert selection.fit.status[1] == "no_residual_signal"
    assert np.isnan(selection.fit.noise_variance[1])
    assert np.isnan(selection.fit.criterion[1])
    assert np.isnan(selection.fit.ratio[1])
    assert_array_equal(selection.gap[[0, 2]], reference.gap)
    assert_array_equal(selection.fit.criterion[[0, 2]], reference.fit.criterion)


def test_single_ok_candidate_has_nan_gap(planted):
    """Verify single ok candidate has NaN gap."""
    signals, fixed, candidates = planted
    groups = _groups(signals, fixed)
    selection = select_random_effects_design(groups, [[None, None], candidates[2]])
    assert_array_equal(selection.winner, np.ones(N_FEATURES))
    assert np.isnan(selection.gap).all()
    _assert_same_fit(selection.fit, fit_reml(groups, candidates[2]))


class _OnePass:
    """Iterable of candidates that counts ``next`` calls and allows one pass."""

    def __init__(self, candidates):
        self.candidates, self.calls, self.passes = candidates, 0, 0

    def __iter__(self):
        self.passes += 1
        if self.passes > 1:
            raise AssertionError("candidates iterated twice")
        for candidate in self.candidates:
            self.calls += 1
            yield candidate


def test_candidates_consumed_once(planted):
    """Verify candidates consumed once."""
    signals, fixed, candidates = planted
    once = _OnePass(candidates)
    rho = np.array([0.0, 0.2, 0.2, 0.4])
    selection = select_random_effects_design(_groups(signals, fixed), once, rho=rho)
    assert once.calls == len(candidates)
    assert once.passes == 1
    assert selection.n_candidates == len(candidates)
    assert_array_equal(selection.fit.rho, rho)


def test_invalid_candidates_raise(planted):
    """Verify invalid candidates raise."""
    signals, fixed, candidates = planted
    groups = _groups(signals, fixed)
    with pytest.raises(ValueError, match="candidate"):
        select_random_effects_design(groups, [])
    with pytest.raises(ValueError, match="group"):
        select_random_effects_design(groups, [candidates[0], candidates[1][:1]])
    short = [candidates[1][0][:-1], candidates[1][1]]
    with pytest.raises(ValueError, match="scan"):
        select_random_effects_design(groups, [candidates[0], short])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_invalid_candidate_message_names_candidate_and_group(planted, value):
    """Verify invalid candidate message names candidate and group."""
    signals, fixed, candidates = planted
    groups = _groups(signals, fixed)
    corrupt = [z.copy() for z in candidates[3]]
    corrupt[1][2, 0] = value
    with pytest.raises(ValueError, match=r"candidate 3.*group 1|group 1.*candidate 3"):
        select_random_effects_design(groups, [*candidates[:3], corrupt])
    short = [candidates[3][0], candidates[3][1][:-1]]
    with pytest.raises(ValueError, match=r"candidate 2.*group 1|group 1.*candidate 2"):
        select_random_effects_design(groups, [*candidates[:2], short])


def test_candidate_may_be_any_iterable_of_group_designs(planted):
    """Verify candidate may be any iterable of group designs."""
    signals, fixed, candidates = planted
    groups = _groups(signals, fixed)
    expected = select_random_effects_design(groups, candidates)
    lazy = ((z for z in candidate) for candidate in candidates)
    selection = select_random_effects_design(groups, lazy)
    assert_array_equal(selection.winner, expected.winner)
    assert_array_equal(selection.gap, expected.gap)
    _assert_same_fit(selection.fit, expected.fit)
    fit = fit_reml(groups, iter(candidates[2]))
    _assert_same_fit(fit, fit_reml(groups, candidates[2]))


def test_candidates_are_fitted_as_they_are_drawn(planted, monkeypatch):
    """Verify candidates are fitted as they are drawn."""
    signals, fixed, candidates = planted
    drawn, at_fit = [0], []
    original = reml.minimize_ratio

    def recording(*args, **kwargs):
        at_fit.append(drawn[0])
        return original(*args, **kwargs)

    def stream():
        for candidate in candidates:
            drawn[0] += 1
            yield candidate

    monkeypatch.setattr(reml, "minimize_ratio", recording)
    select_random_effects_design(_groups(signals, fixed), stream())
    assert at_fit == list(range(len(candidates) + 1))


def test_per_feature_rho_equals_separate_selections(planted):
    """Verify per feature rho equals separate selections."""
    signals, fixed, candidates = planted
    rho = np.array([0.0, 0.3, 0.3, 0.6])
    selection = select_random_effects_design(
        _groups(signals, fixed), candidates, rho=rho
    )
    for value, index in ((0.0, [0]), (0.3, [1, 2]), (0.6, [3])):
        groups = _groups(_columns(signals, index), fixed)
        single = select_random_effects_design(groups, candidates, rho=value)
        assert_array_equal(selection.winner[index], single.winner)
        assert_array_equal(selection.gap[index], single.gap)
        for name in FIT_FIELDS:
            assert_array_equal(
                getattr(selection.fit, name)[index], getattr(single.fit, name)
            )
        assert [selection.fit.status[i] for i in index] == list(single.fit.status)


def test_fixed_projection_built_once_per_rho(planted, monkeypatch):
    """Verify fixed projection built once per rho."""
    signals, fixed, candidates = planted
    calls = []
    original = reml.project_fixed

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(reml, "project_fixed", counting)
    rho = np.array([0.0, 0.0, 0.5, 0.5])
    select_random_effects_design(_groups(signals, fixed), candidates[:4], rho=rho)
    assert len(calls) == 4


def _noisy_copies(rng, random, levels):
    """Candidates that perturb the generating design by increasing amounts."""
    return [[z + level * rng.normal(size=z.shape) for z in random] for level in levels]


@pytest.fixture
def graded():
    """Signals with several ok candidates per feature, plus one weak decoy."""
    rng = np.random.default_rng(9)
    signals, fixed, random = simulate(rng, ratio=3.0, n_features=6)
    candidates = _noisy_copies(rng, random, (0.3, 0.6, 1.0, 1.5))
    candidates.append(_decoys(rng, fixed, 1)[0])
    return _groups(signals, fixed), candidates


def _brute_force(groups, candidates):
    """Winner and gap from one fit_reml per candidate, over ok fits only."""
    fits = [fit_reml(groups, candidate) for candidate in candidates]
    criteria = np.array([f.criterion for f in fits])
    ok = np.array([[s == "ok" for s in f.status] for f in fits])
    scores = np.where(ok, criteria, np.inf)
    ordered = np.sort(scores, axis=0)
    winner = np.where(ok.any(axis=0), np.argmin(scores, axis=0), -1)
    gap = np.where(np.isfinite(ordered[1]), ordered[1] - ordered[0], np.nan)
    return winner, gap, ok


def test_winner_and_gap_match_per_candidate_fits_in_every_order(graded):
    """Verify winner and gap match per candidate fits in every order."""
    groups, candidates = graded
    winner, gap, ok = _brute_force(groups, candidates)
    assert (ok.sum(axis=0) >= 4).all()
    for order in itertools.permutations(range(len(candidates))):
        selection = select_random_effects_design(groups, [candidates[i] for i in order])
        mapped = np.asarray(order)[selection.winner]
        assert_array_equal(mapped, winner)
        assert_allclose(selection.gap, gap, rtol=1e-12, atol=0.0)


@pytest.fixture
def noise_free():
    """One feature exactly explained by a candidate, one ordinary feature."""
    rng = np.random.default_rng(4)
    signals, fixed, random = simulate(rng, scans=(60,), ratio=5.0, n_features=2)
    exact = fixed[0] @ rng.normal(size=4) + random[0] @ rng.normal(size=3)
    mixed = [np.column_stack([exact, signals[0][:, 1]])]
    return mixed, fixed, random, _decoys(rng, fixed, 1)[0]


@pytest.mark.parametrize("exact_first", [True, False])
def test_selection_reports_features_with_a_truncated_candidate(noise_free, exact_first):
    """A criterion truncated at the search edge is not a comparable optimum;
    the selection says which features had one, whoever won."""
    signals, fixed, exact, decoy = noise_free
    candidates = [exact, decoy] if exact_first else [decoy, exact]
    selection = select_random_effects_design(_groups(signals, fixed), candidates)
    assert_array_equal(selection.any_at_upper, [True, False])
    assert_array_equal(selection.any_unrefined, [True, False])
    assert selection.any_at_upper.dtype == bool
    with pytest.raises(ValueError):
        selection.any_at_upper[0] = False


def test_selection_carries_the_winning_fit_unrefined_flag(noise_free):
    """Verify selection carries the winning fit unrefined flag."""
    signals, fixed, exact, decoy = noise_free
    selection = select_random_effects_design(_groups(signals, fixed), [decoy, exact])
    assert_array_equal(selection.fit.unrefined, selection.fit.at_upper)
    assert selection.fit.unrefined[0] and not selection.fit.unrefined[1]


def test_untruncated_selection_has_no_flag(planted):
    """Verify untruncated selection has no flag."""
    signals, fixed, candidates = planted
    selection = select_random_effects_design(_groups(signals, fixed), candidates)
    assert not selection.any_at_upper.any()
    assert not selection.any_unrefined.any()


def test_losing_interior_fallback_makes_selection_provisional(monkeypatch):
    """Verify losing interior fallback makes selection provisional."""
    import boldtailor.reml as api
    from boldtailor._reml_search import minimize_ratio
    from tests.test_reml_adversarial import _diagonal_problem

    y, z, _ = _diagonal_problem(54, "wide")
    # Compare the same covariance searched at two resolutions. The refined
    # winner is interior, so only the losing candidate carries a qualification.
    resolutions = iter([3, 121])

    def search(terms):
        size = next(resolutions) if terms.singular2.size else 121
        return minimize_ratio(terms, grid_size=size)

    monkeypatch.setattr(api, "minimize_ratio", search)
    groups = [RemlGroup(y, np.empty((len(y), 0)))]
    selected = api.select_random_effects_design(groups, [[z], [z]])
    assert selected.winner[0] == 1
    assert not selected.fit.unrefined[0]
    assert not selected.any_at_upper[0]
    assert selected.any_unrefined[0]
