import sys
from pathlib import Path

# Modules import siblings as experiments.nsd_replication.*; tests run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import (
    CANONICAL_PARAMETERS,
    HrfCandidate,
    spm_parameters,
    timing_parameters,
)
from boldtailor.workflow.inputs import detect_task_model

from experiments.nsd_replication.confounds import glmsingle_polynomials
from experiments.nsd_replication.inputs import Session

TR = 4 / 3
N_RUNS, N_VOLUMES, N_TRIALS = 6, 96, 24
N_IMAGES, N_FEATURES, N_RESPONSIVE = 40, 30, 20
N_FILLERS = N_RUNS * N_TRIALS - 3 * N_IMAGES  # 6 x 24 slots exceed 40 x 3
OVERSAMPLING = 16
# Response lobe peaks at 6.5 s; the canonical lobe (SPM delay 6) peaks at 5 s.
DELAYED = spm_parameters((6.5, *timing_parameters(CANONICAL_PARAMETERS)[1:]))


def _events(images, rng):
    return pd.DataFrame(
        {
            "onset": 8.0 + 4.0 * np.arange(N_TRIALS),
            "duration": 3.0,
            "trial_type": np.arange(N_TRIALS) % 2,
            "response_time": rng.uniform(0.5, 1.5, N_TRIALS),
            "73k_id": images,
        }
    )


def _responses(events):
    """Unit-amplitude delayed-HRF responses, frames x trials."""
    dt = TR / OVERSAMPLING
    kernel = HrfCandidate(1, "double_gamma", DELAYED).kernel(TR, OVERSAMPLING)
    grid = np.arange(N_VOLUMES * OVERSAMPLING) * dt
    boxes = np.stack(
        [(grid >= o) & (grid < o + d) for o, d in zip(events.onset, events.duration)],
        axis=1,
    ).astype(float)
    full = np.apply_along_axis(lambda b: np.convolve(b, kernel)[: len(grid)], 0, boxes)
    return full[::OVERSAMPLING] * dt


def _ar1(rng, rho=0.3):
    shocks = rng.standard_normal((N_VOLUMES, N_FEATURES))
    noise = np.empty_like(shocks)
    noise[0] = shocks[0]
    for t in range(1, N_VOLUMES):
        noise[t] = rho * noise[t - 1] + shocks[t]
    return noise


def _signal(events, effects, loadings, rng):
    amplitude = effects[events["73k_id"].to_numpy()] + 0.5 * rng.standard_normal(
        (N_TRIALS, N_RESPONSIVE)
    )
    y = 0.5 * rng.standard_normal((N_VOLUMES, 2)) @ loadings + _ar1(rng)
    y[:, :N_RESPONSIVE] += _responses(events) @ amplitude
    return (100.0 + y).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_session():
    rng = np.random.default_rng(0)
    shown = np.r_[np.repeat(np.arange(N_IMAGES), 3), N_IMAGES + np.arange(N_FILLERS)]
    images = rng.permutation(shown).reshape(N_RUNS, N_TRIALS)
    effects = 3.0 + 1.5 * rng.standard_normal((N_IMAGES + N_FILLERS, N_RESPONSIVE))
    loadings = rng.standard_normal((2, N_FEATURES))
    events = [_events(run, rng) for run in images]
    labels = tuple(f"run-{i + 1:02d}" for i in range(N_RUNS))
    return Session(
        subject="sub-synthetic",
        session="ses-synthetic",
        labels=labels,
        signals=tuple(_signal(e, effects, loadings, rng) for e in events),
        events=tuple(events),
        confounds=tuple(glmsingle_polynomials(N_VOLUMES, TR) for _ in events),
        frame_times=tuple(np.arange(N_VOLUMES) * TR for _ in events),
        brain=nib.cifti2.BrainModelAxis.from_surface(
            np.arange(N_FEATURES), N_FEATURES, "CIFTI_STRUCTURE_CORTEX_LEFT"
        ),
        task_model=detect_task_model(events, labels=list(labels)),
        source="ppdata",
    )
