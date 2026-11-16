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
    library = HrfLibrary.from_parameters([(4., 12., 1., 1., 6., 0., 32.)])
    events = pd.DataFrame({
        "onset": np.arange(8) * 10 + 0.3,
        "duration": [0., 1., 2., 0., 1., 2., 1., 0.],
        "trial_type": ["face", "house"] * 4,
        "response_time": [0.4, 0.8, 0.5, np.nan, 0.7, 0.6, 0.9, 1.],
        "image_id": [1, 1, 2, 2, 3, 3, 4, 4],
    })
    data = from_arrays(
        [rng.normal(size=(n, 3)) for n in (100, 110, 105)],
        [events] * 3, tr=1.,
        confounds=[pd.DataFrame({"motion": rng.normal(size=n)}) for n in (100, 110, 105)],
    )
    model = TaskModel((Modulator("response_time", missing="indicator"),
                       Modulator("trial_type", kind="categorical", levels=("face", "house"))))
    return data, library, model
