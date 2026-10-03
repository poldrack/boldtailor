"""Method recovery under realistic noise: shrinkage, penalty, and HRF choice.

These tests validate the selection methods, not code paths. Truth comes only
from the simulation; production internals are used solely to build signals.
"""

import numpy as np
import pandas as pd

from boldtailor._hrf_design import convolve_events
from boldtailor._single_trial_design import compile_trial_run
from boldtailor.data import from_arrays
from boldtailor.fractional_ridge import score_fraction_candidates
from boldtailor.fractional_ridge import select_ridge_fractions
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import select_hrf
from boldtailor.ridge_selection import score_ridge_candidates
from boldtailor.ridge_selection import select_ridge_penalty
from boldtailor.single_trial import fit_single_trials

FRACTIONS = (1.0, 0.8, 0.6, 0.4, 0.2)


def _ar1(rng, n, rho, sd):
    e = rng.normal(0, sd, n)
    for t in range(1, n):
        e[t] += rho * e[t - 1]
    return e


def _trial_events(rng, n_trials, isi):
    rt = rng.uniform(0.4, 1.6, n_trials)
    kind = rng.integers(0, 2, n_trials)
    onsets = 6 + np.arange(n_trials) * isi
    return pd.DataFrame(
        dict(onset=onsets, duration=1.0, response_time=rt, trial_type=kind)
    )


def _true_betas(rng, events):
    rt, kind = events.response_time.to_numpy(), events.trial_type.to_numpy()
    noise = rng.normal(0, 0.3, len(events))
    return 2.0 + 1.0 * (rt - rt.mean()) - 0.8 * kind + noise


def _dense_trial_problem(rng, *, n_runs=6, n_trials=24, isi=3.0, tr=1.0, noise_sd=1.5):
    signals, events, times, confounds, predictors, truth = [], [], [], [], [], []
    for r in range(n_runs):
        t = np.arange(int(n_trials * isi / tr) + 30) * tr
        e = _trial_events(rng, n_trials, isi)
        n = pd.DataFrame(dict(drift=np.linspace(-1, 1, len(t))))
        x, _, _ = compile_trial_run(e, t, n, f"run-{r}")
        beta = _true_betas(rng, e)
        y = x.to_numpy() @ beta + 100 + _ar1(rng, len(t), 0.4, noise_sd)
        signals.append(np.column_stack([y, y + rng.normal(0, 0.01, len(t))]))
        events.append(e)
        times.append(t)
        confounds.append(n)
        predictors.append(e[["response_time", "trial_type"]])
        truth.append(beta)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    return data, predictors, truth


def _centered_rmse(fit, truth):
    err = [
        b[:, 0] - b[:, 0].mean() - (tb - tb.mean())
        for b, tb in zip(fit.run_betas, truth)
    ]
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def test_fractional_cv_prefers_shrinkage_when_trials_overlap_and_noise_is_high():
    rng = np.random.default_rng(11)
    data, predictors, truth = _dense_trial_problem(rng)
    scores = score_fraction_candidates(data, predictors, fractions=FRACTIONS)
    choice = select_ridge_fractions(scores.cv_r2, scores.grid)
    assert np.all(choice.ridge_fraction < 1.0)
    ols = fit_single_trials(data)
    ridge = fit_single_trials(data, ridge_fraction=choice.ridge_fraction)
    assert _centered_rmse(ridge, truth) < _centered_rmse(ols, truth)


def test_fractional_cv_selects_ols_when_noise_is_negligible():
    rng = np.random.default_rng(12)
    data, predictors, _ = _dense_trial_problem(rng, isi=8.0, noise_sd=0.01)
    scores = score_fraction_candidates(data, predictors, fractions=FRACTIONS)
    choice = select_ridge_fractions(scores.cv_r2, scores.grid)
    assert np.all(choice.ridge_fraction == 1.0)


def test_shared_alpha_cv_prefers_a_positive_penalty_under_high_noise():
    rng = np.random.default_rng(13)
    data, predictors, _ = _dense_trial_problem(rng)
    alphas = (0.0, 0.3, 1.0, 3.0, 10.0)
    scores = score_ridge_candidates(data, predictors, alphas=alphas)
    choice = select_ridge_penalty(scores.cv_r2, scores.grid, percentile=50.0)
    assert choice.ridge_alpha > 0.0


HRF_TIMES = 0.8 + 1.6 * np.arange(150)


def _hrf_run_events(rng, jitter):
    onsets = 10 + np.arange(16) * 14.0
    if jitter:
        onsets = onsets + rng.uniform(-1, 1, 16)
    return pd.DataFrame(dict(onset=onsets, duration=2.0))


def _hrf_selection(rng, library, kernels, motion, *, motion_weight, jitter):
    """Simulate 4 runs (TR 1.6, AR(1) rho 0.4 sd 1.0, amplitude 3.0) and select."""
    signals, events, times, confounds = [], [], [], []
    for r in range(4):
        e = _hrf_run_events(rng, jitter)
        n = pd.DataFrame(dict(motion=motion(np.arange(len(HRF_TIMES)), r)))
        cols = []
        for kernel in kernels:
            x = convolve_events(e.onset, e.duration, HRF_TIMES, kernel)
            noise = _ar1(rng, len(HRF_TIMES), 0.4, 1.0)
            cols.append(3.0 * x + motion_weight * n.motion.to_numpy() + 100 + noise)
        signals.append(np.column_stack(cols))
        events.append(e)
        times.append(HRF_TIMES)
        confounds.append(n)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    return select_hrf(data, library=library)


def test_hrf_selection_recovers_the_generating_kernel_under_ar1_noise():
    rng = np.random.default_rng(21)
    library = HrfLibrary.from_parameters(
        [
            [3, 10, 0.5, 0.5, 2, 0, 36],
            [4, 12, 0.8, 1.0, 4, 0.5, 36],
            [5, 14, 1.0, 1.5, 6, 1.0, 36],
            [6, 16, 1.5, 2.5, 8, 2, 36],
        ]
    )
    true_ids = np.array([0, 1, 2, 3, 2, 1, 0, 3])
    kernels = [library.candidates[int(i)] for i in true_ids]

    def motion(s, r):
        return np.sin(s / 10 + r)

    selection = _hrf_selection(
        rng, library, kernels, motion, motion_weight=0.5, jitter=True
    )
    assert np.mean(selection.hrf_indices == true_ids) >= 0.75
    assert np.nanmedian(selection.delta_cv_r2[true_ids != 0]) > 0.0


def test_hrf_selection_keeps_canonical_when_truth_is_canonical():
    rng = np.random.default_rng(22)
    library = HrfLibrary.from_parameters(
        [[4, 12, 0.8, 1.0, 4, 0.5, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )

    def motion(s, r):
        return np.cos(s / 9 + r)

    selection = _hrf_selection(
        rng, library, ["spm"] * 6, motion, motion_weight=0.0, jitter=False
    )
    assert np.mean(selection.hrf_indices == 0) >= 0.75
