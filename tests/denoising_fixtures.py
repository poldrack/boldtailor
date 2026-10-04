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
    data = from_arrays(
        parts["signals"], parts["events"], tr=TR, confounds=parts["confounds"]
    )
    return DenoisingFixture(
        data=data,
        library=library,
        task_model=RT_MODEL,
        groups=groups,
        latent=tuple(parts["latent"]),
        trial_amplitudes=tuple(parts["amps"]),
        response_times=tuple(r.to_numpy() for r in parts["rts"]),
    )


# ---- validation datasets ---------------------------------------------------------
#
# Predeclared before the validation tests were first run; never tuned afterward.
# Five runs: the first four are given to selection, the fifth is the untouched
# outer run. All features share HRF candidate 0. `task` features have known
# amplitudes; `noise` features carry no task response. With `shared_noise`,
# two AR(1) latent series per run (not in the baseline confounds) load on every
# feature; without it, every feature has only independent white noise.

VALIDATION_SEEDS = dict(recovery=20261004, no_benefit=20261005)
VALIDATION_LENGTHS = (90, 96, 84, 102, 92)
VALIDATION_AMPLITUDES = np.linspace(1.0, 3.0, 10)
VALIDATION_NOISE_FEATURES = 30
VALIDATION_LATENTS = 2
VALIDATION_LOADING_RANGE = (1.0, 2.0)
VALIDATION_WHITE_SD = 0.5


@dataclass(frozen=True)
class ValidationDataset:
    training: AnalysisData
    outer: AnalysisData
    library: HrfLibrary
    task: np.ndarray
    amplitudes: np.ndarray
    outer_task_signal: np.ndarray


def _validation_confounds(length):
    t = np.arange(length)
    return pd.DataFrame(
        dict(drift=(t - t.mean()) / length, cosine=np.cos(np.pi * (t + 0.5) / length))
    )


def _validation_run(rng, length, library, shared_noise):
    times = TR * np.arange(length)
    events = _events(rng, length)[["onset", "duration"]].assign(trial_type="task")
    regressor = _trial_responses(events, times, library.candidates[0]).sum(axis=1)
    n_task = len(VALIDATION_AMPLITUDES)
    n = n_task + VALIDATION_NOISE_FEATURES
    task_signal = np.zeros((length, n))
    task_signal[:, :n_task] = np.outer(regressor, VALIDATION_AMPLITUDES)
    confounds = _validation_confounds(length)
    y = rng.normal(scale=VALIDATION_WHITE_SD, size=(length, n))
    y += confounds.to_numpy() @ rng.normal(scale=0.5, size=(2, n))
    if shared_noise:
        latents = np.column_stack(
            [_latent(rng, length) for _ in range(VALIDATION_LATENTS)]
        )
        loadings = rng.uniform(*VALIDATION_LOADING_RANGE, size=(VALIDATION_LATENTS, n))
        y += latents @ loadings
    y += task_signal + rng.uniform(50.0, 150.0, size=n)
    return y, events, times, confounds, task_signal


def _runs_data(runs):
    signals, events, times, confounds, _ = zip(*runs)
    return from_arrays(
        list(signals), list(events), frame_times=list(times), confounds=list(confounds)
    )


def make_validation_dataset(kind):
    """Selection runs, an untouched outer run, and the known task coefficients."""
    rng = np.random.default_rng(VALIDATION_SEEDS[kind])
    library = fixture_library()
    runs = [
        _validation_run(rng, length, library, kind == "recovery")
        for length in VALIDATION_LENGTHS
    ]
    n_task = len(VALIDATION_AMPLITUDES)
    return ValidationDataset(
        training=_runs_data(runs[:-1]),
        outer=_runs_data(runs[-1:]),
        library=library,
        task=np.arange(n_task),
        amplitudes=VALIDATION_AMPLITUDES.copy(),
        outer_task_signal=runs[-1][4],
    )
