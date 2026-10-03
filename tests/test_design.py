import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix

from boldtailor.data import from_arrays
from boldtailor.design import compile_designs, compile_nuisance_designs
from boldtailor.model import ModelSpec, Modulator, TaskModel
from tests.oracles import peak_design_matrix


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
    expected = peak_design_matrix(
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


@pytest.mark.parametrize("compiler", [compile_designs, compile_nuisance_designs])
def test_compile_designs_rejects_missing_selected_confound(inputs, compiler):
    events, confounds = inputs
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(contrasts={"face": "face"}, confounds=("missing",))

    with pytest.raises(ValueError, match="run 0.*missing confounds: missing"):
        compiler(data, model)


@pytest.mark.parametrize("compiler", [compile_designs, compile_nuisance_designs])
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


def _task_model_data():
    times = [0.775 + 1.6 * np.arange(80), 0.775 + 1.6 * np.arange(85)]
    events = [
        pd.DataFrame(
            dict(
                onset=[8.0, 22.0, 38.0, 60.0, 90.0],
                duration=[3.0, 1.0, 2.0, 3.0, 1.5],
                trial_type=[0, 1, 0, 1, 1],
                response_time=[1.0, np.nan if r else 2.0, 3.0, 2.0, 4.0],
            )
        )
        for r in range(2)
    ]
    confounds = [pd.DataFrame(dict(motion=np.linspace(-1, 1, len(t)))) for t in times]
    signals = [
        np.random.default_rng(r).normal(size=(len(t), 3)) for r, t in enumerate(times)
    ]
    return from_arrays(signals, events, frame_times=times, confounds=confounds)


def test_compile_designs_with_task_model_uses_nilearn_task_columns_and_nuisance():
    from boldtailor._task_design import expand_events

    data = _task_model_data()
    task_model = TaskModel(
        (
            Modulator("response_time", missing="indicator"),
            Modulator("trial_type", center=False),
        )
    )
    model = ModelSpec(
        contrasts={"task": {"task": 1}},
        confounds=("motion",),
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
        task_model=task_model,
    )
    compiled = compile_designs(data, model)
    expected_names = [
        ["task", "response_time", "trial_type"],
        ["task", "response_time", "trial_type", "missing_response_time"],
    ]
    for run, design in enumerate(compiled):
        expected = peak_design_matrix(
            data.frame_times[run],
            events=expand_events(data.events[run], task_model, run),
            hrf_model="spm",
            drift_model="cosine",
            high_pass=0.01,
            add_regs=data.confounds[run][["motion"]],
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
        assert (
            list(design.matrix.columns[: len(expected_names[run])])
            == expected_names[run]
        )
        assert "constant" in design.matrix.columns
        assert any(c.startswith("drift") for c in design.matrix.columns)
        for name in expected_names[run]:
            np.testing.assert_allclose(design.matrix[name], expected[name], atol=1e-12)
        np.testing.assert_allclose(
            design.matrix["motion"], expected["motion"], atol=1e-12
        )


def test_compile_designs_with_task_model_reports_data_errors_with_run():
    data = _task_model_data()
    model = ModelSpec(
        contrasts={"task": {"task": 1}},
        hrf_model="spm",
        task_model=TaskModel((Modulator("response_time"),)),
    )
    with pytest.raises(
        ValueError,
        match=r"^run 1 design compilation failed: (?!run 1).*response_time",
    ):
        compile_designs(data, model)


def test_plain_string_hrf_keeps_confound_names_ending_in_kernel(inputs):
    events, confounds = inputs
    confounds = confounds.rename(columns={"trans_x": "face_kernel"})
    data = from_arrays(np.zeros((20, 2)), events, tr=2.0, confounds=confounds)
    model = ModelSpec(
        contrasts={"face_gt_house": "face - house"},
        confounds=("face_kernel",),
        drift_model=None,
        noise_model="ols",
    )
    columns = list(compile_designs(data, model)[0].matrix.columns)
    assert columns == ["face", "house", "face_kernel", "constant"]


@pytest.fixture
def access_counts(monkeypatch):
    """Count copies of per-run tables made through AnalysisData accessors."""
    from boldtailor.data import AnalysisData

    counts = {"events": 0, "confounds": 0}
    for name in counts:
        original = getattr(AnalysisData, name)

        def counted(self, _name=name, _original=original):
            counts[_name] += 1
            return _original.fget(self)

        monkeypatch.setattr(AnalysisData, name, property(counted))
    return counts


def _three_runs(inputs):
    events, _ = inputs
    confounds = pd.DataFrame({"trans_x": np.linspace(-1.0, 1.0, 80)})
    return from_arrays(
        [np.random.default_rng(r).normal(size=(80, 2)) for r in range(3)],
        [events] * 3,
        tr=2.0,
        confounds=[confounds] * 3,
    )


def test_per_run_loops_copy_run_tables_once(inputs, access_counts):
    from boldtailor import _ridge_cv as cv

    data = _three_runs(inputs)
    model = ModelSpec(contrasts={"face": "face"}, confounds=("trans_x",))
    calls = [
        lambda: compile_nuisance_designs(data, model),
        lambda: cv.subset_runs(data, [0, 2]),
        lambda: cv.prepare_run_beta_path(data, 1, None, np.zeros(2, int), "r"),
    ]
    for call in calls:
        access_counts.update(events=0, confounds=0)
        call()
        assert access_counts["events"] <= 1
        assert access_counts["confounds"] <= 1
