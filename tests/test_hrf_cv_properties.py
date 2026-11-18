"""Property-based tests for HRF cross-validation scoring and candidate selection."""

from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd

from boldtailor._hrf_cv import choose_eligible, loro_scores, signal_statistics
from boldtailor._run_design import prepare_runs
from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import TaskModel


def _hrf_problem(seed, n_runs=3, n_features=4):
    rng = np.random.default_rng(seed)
    times = np.arange(50) * 1.5
    signals, events, confounds, frame_times = [], [], [], []
    for _ in range(n_runs):
        ev = pd.DataFrame({"onset": [6.0, 20.0, 36.0, 50.0], "duration": [1.0] * 4})
        conf = pd.DataFrame({"motion": rng.normal(size=len(times))})
        y = rng.normal(size=(len(times), n_features))
        signals.append(y)
        events.append(ev)
        confounds.append(conf)
        frame_times.append(times)
    library = HrfLibrary.from_parameters(
        [[4, 12, 0.8, 1.0, 4, 0.5, 32], [6, 16, 1.2, 1.5, 6, 1.0, 32]]
    )
    data = from_arrays(signals, events, frame_times=frame_times, confounds=confounds)
    return data, library


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    scale=st.floats(
        min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)
def test_loro_scores_are_invariant_to_signal_scale_and_run_order(seed, scale):
    """Verify LORO HRF scores are invariant to positive signal scaling and run-order permutation."""
    data, library = _hrf_problem(seed)
    runs = prepare_runs(data, library, TaskModel())
    a, b, c, energy = signal_statistics(runs, data.signals, batch_size=2)
    base_scores = loro_scores(a, b, c, energy)

    a_s, b_s, c_s, e_s = signal_statistics(
        runs, tuple(s * scale for s in data.signals), batch_size=2
    )
    scaled_scores = loro_scores(a_s, b_s, c_s, e_s)
    np.testing.assert_allclose(scaled_scores, base_scores, rtol=1e-10, atol=1e-12)

    perm = np.random.default_rng(seed + 1).permutation(data.n_runs)
    perm_scores = loro_scores(a[perm], b[perm], c[perm], energy[perm])
    np.testing.assert_allclose(perm_scores, base_scores, rtol=1e-10, atol=1e-12)


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n_candidates=st.integers(2, 6),
    n_features=st.integers(2, 8),
)
def test_choose_eligible_is_feature_permutation_equivariant(
    seed, n_candidates, n_features
):
    """Verify choose_eligible selects each feature independently and breaks ties by lowest ID."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-0.5, 0.9, size=(n_candidates, n_features))
    scores[1:, 0] = scores[0, 0]  # exact tie on feature 0

    class _AlwaysEligible:
        @staticmethod
        def eligible(_cid):
            return True, ""

    runs = [_AlwaysEligible()]
    indices, _, _ = choose_eligible(scores, runs)
    assert indices[0] == 0

    perm = rng.permutation(n_features)
    perm_indices, _, _ = choose_eligible(scores[:, perm], runs)
    np.testing.assert_array_equal(perm_indices, indices[perm])
