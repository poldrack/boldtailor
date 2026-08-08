import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix

from boldtailor.data import from_arrays
from boldtailor.design import compile_designs
from boldtailor.model import ModelSpec


@pytest.fixture
def inputs():
    events = pd.DataFrame(
        {
            "onset": [0.0, 8.0, 16.0, 24.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["face", "house", "face", "house"],
        }
    )
    confounds = pd.DataFrame(
        {
            "trans_x": np.linspace(-1.0, 1.0, 20),
            "unused": np.linspace(1.0, -1.0, 20),
        }
    )
    return events, confounds


def test_compile_designs_matches_nilearn(inputs):
    events, confounds = inputs
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(
        contrasts={"face_gt_house": "face - house"},
        confounds=("trans_x",),
        drift_model=None,
        noise_model="ols",
    )

    actual = compile_designs(data, model)[0].matrix
    expected = make_first_level_design_matrix(
        np.arange(20) * 2.0,
        events=events,
        hrf_model="glover",
        drift_model=None,
        high_pass=0.01,
        drift_order=1,
        add_regs=confounds[["trans_x"]],
        min_onset=-24.0,
        oversampling=50,
    )

    assert_frame_equal(actual, expected)
    assert "unused" not in actual


def test_compile_designs_rejects_missing_selected_confound(inputs):
    events, confounds = inputs
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(contrasts={"face": "face"}, confounds=("missing",))

    with pytest.raises(ValueError, match="run 0.*missing confounds: missing"):
        compile_designs(data, model)


def test_compile_designs_rejects_nonfinite_selected_confound(inputs):
    events, confounds = inputs
    confounds.loc[3, "trans_x"] = np.nan
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(contrasts={"face": "face"}, confounds=("trans_x",))

    with pytest.raises(ValueError, match="run 0.*finite"):
        compile_designs(data, model)


def test_compile_designs_warns_and_records_early_event_exclusion(inputs):
    events, _ = inputs
    early = pd.DataFrame(
        {
            "onset": [-30.0],
            "duration": [1.0],
            "trial_type": ["training"],
        }
    )
    events = pd.concat([early, events], ignore_index=True)
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0)
    model = ModelSpec(
        contrasts={"face": "face"},
        drift_model=None,
        min_onset=-24.0,
    )

    with pytest.warns(
        UserWarning,
        match=r"run 0.*excluding 1 event.*-24",
    ):
        compiled = compile_designs(data, model)[0]

    assert compiled.excluded_event_count == 1
    assert compiled.min_onset_cutoff == -24.0
    assert "training" not in compiled.matrix.columns


@pytest.mark.parametrize(
    ("option", "value"),
    [("hrf_model", "bogus"), ("drift_model", "bogus")],
)
def test_compile_designs_contextualizes_invalid_nilearn_options(inputs, option, value):
    events, _ = inputs
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0)
    model = ModelSpec(contrasts={"face": "face"}, **{option: value})

    with pytest.raises(ValueError, match=r"run 0.*design compilation"):
        compile_designs(data, model)
