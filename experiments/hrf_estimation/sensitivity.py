"""Reproducible simulations on real event/nuisance timing, without real BOLD."""

from dataclasses import replace
import numpy as np
from scipy.signal import lfilter

from boldtailor.hrf_library import HrfLibrary
from experiments.hrf_estimation.design import build_design


def parametric_library(library):
    candidates = tuple(replace(c, id=i) for i, c in enumerate(
        c for c in library.candidates if c.kind != "glmsingle"))
    return HrfLibrary(candidates, origin={"parent": library.fingerprint, "subset": "parametric"})


def ar1_noise(length, rho, rng):
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("AR coefficient must be finite and strictly within (-1, 1)")
    innovations = rng.normal(size=length) * np.sqrt(1 - rho**2)
    innovations[0] = rng.normal()
    return lfilter([1], [1, -rho], innovations)


def simulate_runs(templates, truth, mode, rho, snr, seed):
    if snr <= 0 or np.isnan(snr):
        raise ValueError("SNR must be positive (infinity means noiseless)")
    rng = np.random.default_rng(seed)
    return tuple(_simulate_run(r, truth, mode, rho, snr, rng) for r in templates)


def _simulate_run(template, truth, mode, rho, snr, rng):
    events = template.events.copy()
    valid = (events.missing_data == 0) & (events.response_time > 0)
    # Fixed condition correlation plus trial variation, independent of observed BOLD.
    events.loc[valid, "response_time"] = (
        .4 + .35 * events.loc[valid, "trial_type"] + rng.lognormal(-1, .35, valid.sum()))
    run = replace(template, events=events, feature_ids=template.feature_ids[:1].copy(),
                  brain_axis=None if template.brain_axis is None else template.brain_axis[:1])
    parts = build_design(run, truth, mode)
    rt_beta = 1.5 if mode == "amplitude" else .05
    signal = parts.task @ np.array([2., 3., rt_beta]) + parts.junk @ np.array([1., 1.5])
    # Nuisance coefficients vary across runs; normalization only sets simulation scale.
    c = parts.nuisance
    norms = np.sqrt(np.mean(c**2, axis=0))
    nuisance = (c / np.where(norms > 0, norms, 1)) @ rng.normal(0, .3, c.shape[1])
    noise = ar1_noise(len(signal), rho, rng) * np.std(signal) / snr
    return replace(run, signals=(20 + signal + nuisance + noise)[:, None])


def synthetic_templates(seed=20261107):
    """Synthetic NSD-like timing, explicitly not a reconstruction of NSD data."""
    import pandas as pd
    from experiments.hrf_estimation.inputs import RunData
    from experiments.hrf_estimation.timing import MOTION_COLUMNS, build_nuisance

    rng = np.random.default_rng(seed)
    runs = []
    times = np.arange(226) * 4 / 3
    for i in range(4):
        events = pd.DataFrame(dict(onset=np.arange(8, 260, 4.), duration=3.,
                                   trial_type=rng.integers(0, 2, 63),
                                   response_time=rng.uniform(.4, 1.8, 63),
                                   is_correct=1, missing_data=0))
        events.loc[[5, 18], "missing_data"] = 1
        events.loc[[9, 25, 42], "is_correct"] = 0
        motion = pd.DataFrame({c: ar1_noise(226, .8, rng) for c in MOTION_COLUMNS})
        runs.append(RunData(f"synthetic_run-{i+1:02}", np.zeros((226, 1)), events,
                            times, build_nuisance(motion, times), np.array([0])))
    return tuple(runs)
