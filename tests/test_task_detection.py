"""Task-model detection from event tables, independent of the BIDS workflow."""

import subprocess
import sys

import pandas as pd
import pytest

from boldtailor.model import Modulator, TaskModel
from boldtailor.task_detection import (
    TRIAL_TYPE_NOTE,
    detect_task_model,
    task_model_notes,
)

TT = Modulator("trial_type", kind="categorical", levels=("0", "1"))


@pytest.fixture
def events():
    return pd.DataFrame(
        {
            "onset": [8.0, 22.0, 38.0, 60.0, 90.0, 112.0],
            "duration": [3.0] * 6,
            "trial_type": [0, 1, 0, 1, 1, 0],
            "response_time": [0.5, 1.0, 2.5, 1.5, 3.0, 0.5],
        }
    )


def test_detection_uses_rt_and_categorical_trial_type(events):
    model = detect_task_model([events, events])
    assert model == TaskModel((Modulator("response_time", missing="indicator"), TT))
    assert model.regressor_names == ("task", "response_time", "trial_type[1]")
    without_rt = detect_task_model([events.drop(columns="response_time"), events])
    assert without_rt == TaskModel((TT,))
    assert detect_task_model([events[["onset", "duration"]]]) == TaskModel()


def test_string_and_mixed_dtype_trial_types_become_one_level_set(events):
    words = events.assign(trial_type=["face", "house", "scrambled"] * 2)
    model = detect_task_model([words, words])
    assert model.regressor_names[-2:] == ("trial_type[house]", "trial_type[scrambled]")
    as_text = events.assign(trial_type=events.trial_type.astype(str))
    assert detect_task_model([events, as_text]).regressor_names[-1] == "trial_type[1]"


def test_single_level_trial_type_is_left_out_with_a_note(events):
    flat = events.assign(trial_type="face")
    assert detect_task_model([flat]).regressor_names == ("task", "response_time")
    assert task_model_notes([flat]) == [TRIAL_TYPE_NOTE]


def test_explicit_categorical_resolves_levels_and_checks_reference(events):
    wanted = (Modulator("trial_type", kind="categorical", reference="1"),)
    assert detect_task_model([events, events], wanted).regressor_names == (
        "task",
        "trial_type[0]",
    )
    bad = (Modulator("trial_type", kind="categorical", reference="7"),)
    with pytest.raises(ValueError, match="trial_type.*7"):
        detect_task_model([events], bad)


def test_explicit_modulators_must_exist_in_every_run(events):
    wanted = (Modulator("stimulus_id"),)
    with pytest.raises(ValueError, match="run 2.*stimulus_id"):
        detect_task_model([events.assign(stimulus_id=1), events], wanted)
    assert detect_task_model([events.assign(stimulus_id=1)] * 2, wanted) == TaskModel(
        wanted
    )


def test_explicit_modulator_errors_name_the_labelled_run(events):
    wanted = (Modulator("stimulus_id"),)
    with pytest.raises(ValueError, match="run-02.*stimulus_id"):
        detect_task_model(
            [events.assign(stimulus_id=1), events], wanted, labels=["run-01", "run-02"]
        )


def test_detection_takes_any_multilevel_trial_type_without_a_note(events):
    for values in (["face", "house"] * 3, [0, 1, 2, 0, 1, 2]):
        tables = [events, events.assign(trial_type=values)]
        assert detect_task_model(tables).modulators[-1].kind == "categorical"
        assert task_model_notes(tables) == []


def test_detection_keeps_binary_trial_type_written_as_text(events):
    text = events.assign(trial_type=events.trial_type.astype(str))
    assert TT in detect_task_model([text]).modulators
    assert task_model_notes([text]) == []


def test_notes_are_only_for_automatic_detection(events):
    strings = events.assign(trial_type=["face", "house"] * 3)
    assert task_model_notes([strings], modulators=()) == []
    assert task_model_notes([strings], modulators=(Modulator("x"),)) == []


def test_no_modulators_means_a_task_only_model(events):
    assert detect_task_model([events], ()) == TaskModel()


def test_core_detection_does_not_import_the_workflow():
    code = (
        "import sys, boldtailor.task_detection; "
        "sys.exit(any(m.startswith('boldtailor.workflow') for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
