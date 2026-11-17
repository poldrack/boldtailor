"""Winner selection among HRF candidates with lazy structural eligibility."""

import numpy as np
import pytest

from boldtailor._hrf_cv import choose_eligible


class Run:
    """A run whose listed candidates are structurally ineligible."""

    def __init__(self, bad=()):
        self.bad, self.asked = dict(bad), []

    def eligible(self, cid):
        self.asked.append(cid)
        return (cid not in self.bad, self.bad.get(cid, ""))


SCORES = np.array(
    [
        [0.1, 0.5, np.nan, -np.inf],
        [0.4, 0.5, np.nan, -np.inf],
        [0.9, 0.2, np.nan, -np.inf],
    ]
)


def test_each_feature_takes_its_best_candidate_and_ties_keep_the_first():
    """The highest finite score wins; equal scores keep the lower candidate ID."""
    runs = [Run(), Run()]
    indices, scores, table = choose_eligible(SCORES, runs)
    np.testing.assert_array_equal(indices, [2, 0, -1, -1])
    np.testing.assert_array_equal(scores, SCORES)
    assert table.hrf_id.tolist() == [0, 1, 2]
    assert table.eligible.tolist() == [True, None, True]
    assert table.reason.tolist() == ["", "unchecked", ""]


def test_only_winners_and_the_canonical_candidate_are_checked():
    """Eligibility is evaluated lazily: candidate zero, then each winner once."""
    runs = [Run(), Run()]
    choose_eligible(SCORES, runs)
    assert sorted(runs[0].asked) == [0, 2] and sorted(runs[1].asked) == [0, 2]


def test_an_ineligible_winner_is_excluded_everywhere_and_the_choice_repeats():
    """A structural exclusion removes the candidate for every feature."""
    runs = [Run(), Run({2: "rank deficient"})]
    indices, scores, table = choose_eligible(SCORES, runs)
    np.testing.assert_array_equal(indices, [1, 0, -1, -1])
    assert np.isneginf(scores[2]).all()
    np.testing.assert_array_equal(scores[:2], SCORES[:2])
    assert table.eligible.tolist() == [True, True, False]
    assert table.reason.tolist() == ["", "", "run 1: rank deficient"]
    assert not np.isneginf(SCORES[2, :2]).any()  # the input is not modified


def test_reasons_from_several_runs_are_joined():
    """Every run that excludes a candidate contributes its reason."""
    runs = [Run({0: "no events"}), Run({0: "rank deficient"})]
    _, _, table = choose_eligible(SCORES, runs)
    assert table.reason[0] == "run 0: no events; run 1: rank deficient"
    assert table.eligible[0] is False or table.eligible[0] == False  # noqa: E712


def test_no_eligible_candidate_is_an_error():
    """With every candidate structurally excluded there is nothing to select."""
    bad = {cid: "empty" for cid in range(3)}
    with pytest.raises(ValueError, match="no eligible, estimable HRF candidate"):
        choose_eligible(SCORES, [Run(bad)])


def test_a_scored_feature_left_without_a_candidate_is_an_error():
    """A feature whose only finite scores belong to excluded candidates fails."""
    scores = np.array([[0.2, -np.inf], [0.1, 0.7]])
    with pytest.raises(ValueError, match="no eligible HRF with finite prediction"):
        choose_eligible(scores, [Run({1: "rank deficient"})])


def test_unscored_candidates_are_checked_when_no_winner_is_eligible():
    """If no scored feature has a winner, candidates are checked until one is eligible."""
    scores = np.full((3, 2), -np.inf)
    runs = [Run({0: "empty"})]
    indices, _, table = choose_eligible(scores, runs)
    np.testing.assert_array_equal(indices, [-1, -1])
    assert table.eligible.tolist() == [False, True, None]
