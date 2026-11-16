"""Small package-only problems for HRF likelihood selection."""

import numpy as np
import pandas as pd
import pytest

from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import Modulator, TaskModel


@pytest.fixture
def reml_hrf_problem():
    """Ragged runs with duration variation, missing RT, and repeated images."""
    rng = np.random.default_rng(812)
    library = HrfLibrary.from_parameters([(4.0, 12.0, 1.0, 1.0, 6.0, 0.0, 32.0)])
    events = pd.DataFrame(
        {
            "onset": np.arange(8) * 10 + 0.3,
            "duration": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            "trial_type": ["face", "house"] * 4,
            "response_time": [0.4, 0.8, 0.5, np.nan, 0.7, 0.6, 0.9, 1.0],
            "image_id": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    data = from_arrays(
        [rng.normal(size=(n, 3)) for n in (100, 110, 105)],
        [events] * 3,
        tr=1.0,
        confounds=[
            pd.DataFrame({"motion": rng.normal(size=n)}) for n in (100, 110, 105)
        ],
    )
    model = TaskModel(
        (
            Modulator("response_time", missing="indicator"),
            Modulator("trial_type", kind="categorical", levels=("face", "house")),
        )
    )
    return data, library, model


@pytest.fixture
def planted_reml_hrfs(reml_hrf_problem):
    """Strong trial deviations with a fixed canonical mean and a dead feature."""
    from boldtailor._hrf_design import trial_regressors

    data, library, _ = reml_hrf_problem
    rng = np.random.default_rng(40)
    events = [e.assign(duration=np.maximum(e.duration, 0.5)) for e in data.events]
    signals = []
    for e, times, confounds in zip(
        events, data.frame_times, data.confounds, strict=True
    ):
        reference = trial_regressors(e, times, library.candidates[0])
        random = trial_regressors(e, times, library.candidates[1])
        y = 100 + 3 * reference.sum(axis=1, keepdims=True)
        y = (
            y
            + random @ rng.normal(scale=4, size=(len(e), 3))
            + rng.normal(scale=0.1, size=(len(times), 3))
        )
        y[:, -1] = 100
        signals.append(y)
    return (
        from_arrays(
            signals, events, frame_times=data.frame_times, confounds=data.confounds
        ),
        library,
    )


def small_reml_problem(seed=23, n_features=3, effect=3.0, mean_id=0, rho=0.2):
    """Generate independent trial effects with a known fixed mean and AR noise."""
    from boldtailor._hrf_design import trial_regressors
    from tests.reml_oracles import ar1_correlation

    rng = np.random.default_rng(seed)
    library = HrfLibrary.from_parameters([(3.5, 12, 1, 1, 6, 0, 32)])
    events = pd.DataFrame(
        {
            "onset": [1.2, 9.4, 18.1, 27.3],
            "duration": [0.5, 1, 0.5, 1],
            "trial_type": "task",
        }
    )
    times = np.arange(48.0)
    designs = [trial_regressors(events, times, h) for h in library.candidates]
    mean = designs[mean_id].sum(axis=1, keepdims=True)
    factor = np.linalg.cholesky(ar1_correlation(len(times), rho))
    signals = [
        100
        + 4 * mean
        + effect * designs[1] @ rng.normal(size=(4, n_features))
        + factor @ rng.normal(size=(48, n_features))
        for _ in range(2)
    ]
    data = from_arrays(signals, [events] * 2, frame_times=[times] * 2)
    return data, library, designs
