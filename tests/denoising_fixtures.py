"""Seeded multi-run synthetic data with known task, noise-pool, and nuisance parts.

Every run has its own length, trial timing, response times, baseline motion and
drift confounds, and a shared non-task latent time series that loads on most
in-brain features. Ground truth is returned alongside the AnalysisData so tests
can check pool membership and PC recovery without reusing code under test.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boldtailor._hrf_design import convolve_events
from boldtailor.data import AnalysisData, from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import Modulator, TaskModel

TR = 1.5
RUN_LENGTHS = (72, 86, 79, 93)
LIBRARY_PARAMETERS = [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
RT_MODEL = TaskModel((Modulator("response_time"),))
# Feature groups in column order; counts are kept small so tests stay fast.
GROUP_SIZES = dict(task=6, rt=2, noise=12, constant=2, outside_task=2, outside_noise=2)


@dataclass(frozen=True)
class DenoisingFixture:
    data: AnalysisData
    library: HrfLibrary
    task_model: TaskModel
    brain_mask: np.ndarray
    groups: dict[str, np.ndarray]
    latent: tuple[np.ndarray, ...]
    trial_amplitudes: tuple[np.ndarray, ...]
    response_times: tuple[np.ndarray, ...]


def fixture_library():
    return HrfLibrary.from_parameters(LIBRARY_PARAMETERS)


def feature_groups():
    groups, start = {}, 0
    for name, size in GROUP_SIZES.items():
        groups[name] = np.arange(start, start + size)
        start += size
    return groups


def _events(rng, length):
    onsets = np.arange(6.0, TR * length - 24.0, 9.0)
    onsets = onsets + rng.uniform(0.0, 3.0, size=len(onsets))
    return pd.DataFrame(
        dict(
            onset=onsets,
            duration=1.0,
            response_time=rng.uniform(0.4, 1.4, size=len(onsets)),
        )
    )


def _confounds(rng, length):
    t = np.arange(length)
    motion = np.convolve(rng.normal(size=length + 4), np.ones(5) / 5, "valid")
    return pd.DataFrame(
        dict(
            motion_x=motion,
            drift=(t - t.mean()) / length,
            cosine=np.cos(np.pi * (t + 0.5) / length),
        )
    )


def _latent(rng, length):
    values = rng.normal(size=length)
    for k in range(1, length):
        values[k] += 0.6 * values[k - 1]
    return values / values.std()


def _trial_responses(events, times, candidate):
    return np.column_stack(
        [
            convolve_events([onset], [duration], times, candidate)
            for onset, duration in zip(events.onset, events.duration)
        ]
    )


def _rt_free_amplitudes(rng, rt, n):
    """Mean 3 plus unit-SD trial variation exactly orthogonal to [1, RT]."""
    basis = np.column_stack([np.ones_like(rt), rt])
    variation = rng.normal(size=(len(rt), n))
    variation -= basis @ np.linalg.lstsq(basis, variation, rcond=None)[0]
    return 3.0 + variation / variation.std(axis=0)


def _signals(rng, groups, trials, events, latent, confounds):
    length = len(latent)
    n = sum(GROUP_SIZES.values())
    y = rng.normal(scale=0.3, size=(length, n))
    loading = rng.uniform(0.6, 1.4, size=n)
    y += np.outer(latent, loading)
    y += confounds.to_numpy() @ rng.normal(scale=0.8, size=(confounds.shape[1], n))
    rt = events.response_time.to_numpy()
    amplitudes = _rt_free_amplitudes(rng, rt, len(groups["task"]))
    y[:, groups["task"]] += trials @ amplitudes
    y[:, groups["outside_task"]] += trials @ amplitudes[:, :2]
    rt_amplitude = 2.0 + 3.0 * (rt - 0.9)
    y[:, groups["rt"]] += (trials @ rt_amplitude)[:, None]
    y += rng.uniform(50.0, 150.0, size=n)
    y[:, groups["constant"]] = 100.0
    return y, amplitudes


def make_denoising_fixture(seed=20261003):
    """Four unequal runs; trial amplitudes of `task` features ignore RT."""
    rng = np.random.default_rng(seed)
    library, groups = fixture_library(), feature_groups()
    parts = dict(signals=[], events=[], confounds=[], latent=[], amps=[], rts=[])
    for length in RUN_LENGTHS:
        times = TR * np.arange(length)
        events, confounds = _events(rng, length), _confounds(rng, length)
        latent = _latent(rng, length)
        trials = _trial_responses(events, times, library.candidates[0])
        y, amplitudes = _signals(rng, groups, trials, events, latent, confounds)
        for key, value in zip(
            parts,
            (y, events, confounds, latent, amplitudes, events.response_time),
        ):
            parts[key].append(value)
    mask = np.ones(sum(GROUP_SIZES.values()), dtype=bool)
    mask[groups["outside_task"]] = mask[groups["outside_noise"]] = False
    data = from_arrays(
        parts["signals"], parts["events"], tr=TR, confounds=parts["confounds"]
    )
    return DenoisingFixture(
        data=data,
        library=library,
        task_model=RT_MODEL,
        brain_mask=mask,
        groups=groups,
        latent=tuple(parts["latent"]),
        trial_amplitudes=tuple(parts["amps"]),
        response_times=tuple(r.to_numpy() for r in parts["rts"]),
    )
