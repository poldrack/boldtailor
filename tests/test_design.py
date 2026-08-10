import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix

from boldtailor.data import from_arrays
from boldtailor.design import compile_designs, compile_nuisance_designs
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


def test_compile_nuisance_designs_excludes_events_and_retains_nuisance(inputs):
    events, _ = inputs
    confounds = pd.DataFrame(
        {
            "trans_x": np.linspace(-1.0, 1.0, 80),
            "unused": np.linspace(0.5, -0.5, 80),
        }
    )
    data = from_arrays(np.zeros((80, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(
        contrasts={"face_gt_house": "face - house"},
        confounds=("trans_x",),
        drift_model="cosine",
        high_pass=0.01,
        noise_model="ar1",
    )

    compiled = compile_nuisance_designs(data, model)[0]

    assert "face" not in compiled.matrix
    assert "house" not in compiled.matrix
    assert "trans_x" in compiled.matrix
    assert "unused" not in compiled.matrix
    assert "constant" in compiled.matrix
    assert any(column.startswith("drift_") for column in compiled.matrix)
    assert compiled.excluded_event_count == 0
    assert compiled.min_onset_cutoff == -24.0


def test_compile_nuisance_designs_matches_nilearn_without_events(inputs):
    events, _ = inputs
    confounds = pd.DataFrame(
        {
            "trans_x": np.linspace(-1.0, 1.0, 80),
            "unused": np.linspace(0.5, -0.5, 80),
        }
    )
    data = from_arrays(np.zeros((80, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(
        contrasts={"face_gt_house": "face - house"},
        confounds=("trans_x",),
        drift_model=None,
        high_pass=0.01,
        noise_model="ar1",
    )

    actual = compile_nuisance_designs(data, model)[0].matrix
    expected = make_first_level_design_matrix(
        data.frame_times[0],
        events=None,
        hrf_model=None,
        drift_model=None,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        add_regs=confounds[["trans_x"]],
        min_onset=model.min_onset,
        oversampling=model.oversampling,
    )

    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "compiler", [compile_designs, compile_nuisance_designs]
)
def test_compile_designs_rejects_missing_selected_confound(inputs, compiler):
    events, confounds = inputs
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(contrasts={"face": "face"}, confounds=("missing",))

    with pytest.raises(ValueError, match="run 0.*missing confounds: missing"):
        compiler(data, model)


@pytest.mark.parametrize(
    "compiler", [compile_designs, compile_nuisance_designs]
)
def test_compile_designs_rejects_nonfinite_selected_confound(inputs, compiler):
    events, confounds = inputs
    confounds.loc[3, "trans_x"] = np.nan
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(contrasts={"face": "face"}, confounds=("trans_x",))

    with pytest.raises(ValueError, match="run 0.*finite"):
        compiler(data, model)


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
