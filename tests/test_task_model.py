"""A task model is a frozen, fingerprinted list of task regressors."""

import json

import pytest

from boldtailor.model import Modulator, TaskModel


def test_default_task_model_is_task_only():
    model = TaskModel()
    assert model.regressor_names == ("task",)
    assert model.profiled_names == ()
    assert model.to_dict() == {"regressors": ["task"], "modulators": []}
    assert model == TaskModel(())
    assert hash(model) == hash(TaskModel())


def test_nsd_task_model_names_and_dict():
    model = TaskModel(
        (
            Modulator("response_time", center=True, missing="indicator"),
            Modulator("trial_type", center=False),
        )
    )
    assert model.regressor_names == ("task", "response_time", "trial_type")
    assert model.profiled_names == ("missing_response_time",)
    assert model.to_dict() == {
        "regressors": ["task", "response_time", "trial_type"],
        "modulators": [
            {"column": "response_time", "center": True, "missing": "indicator"},
            {"column": "trial_type", "center": False, "missing": "error"},
        ],
    }


def test_fingerprint_is_sha256_of_canonical_dict_and_changes_with_settings():
    model = TaskModel((Modulator("response_time"),))
    assert len(model.fingerprint) == 64
    assert int(model.fingerprint, 16) >= 0
    json.dumps(model.to_dict(), sort_keys=True)  # must be JSON serializable
    assert model.fingerprint == TaskModel((Modulator("response_time"),)).fingerprint
    assert model.fingerprint != TaskModel().fingerprint
    assert (
        model.fingerprint
        != TaskModel((Modulator("response_time", center=False),)).fingerprint
    )
    assert (
        model.fingerprint
        != TaskModel((Modulator("response_time", missing="indicator"),)).fingerprint
    )


@pytest.mark.parametrize(
    "column", ["task", "constant", "onset", "duration", "missing_rt", "", None, 3]
)
def test_reserved_or_invalid_modulator_columns_rejected(column):
    with pytest.raises(ValueError, match="reserved|nonempty"):
        Modulator(column)


def test_trial_type_is_an_ordinary_modulator_column():
    assert Modulator("trial_type", center=False).column == "trial_type"


@pytest.mark.parametrize("missing", ["drop", "", None, True])
def test_invalid_missing_policy_rejected(missing):
    with pytest.raises(ValueError, match="missing"):
        Modulator("response_time", missing=missing)


@pytest.mark.parametrize("center", [1, "yes", None])
def test_center_must_be_boolean(center):
    with pytest.raises(ValueError, match="center"):
        Modulator("response_time", center=center)


def test_duplicate_or_non_modulator_entries_rejected():
    with pytest.raises(ValueError, match="unique"):
        TaskModel((Modulator("rt"), Modulator("rt", center=False)))
    with pytest.raises(ValueError, match="Modulator"):
        TaskModel(("rt",))


def test_task_model_is_immutable_and_owns_its_tuple():
    modulators = [Modulator("rt")]
    model = TaskModel(modulators)
    modulators.append(Modulator("other"))
    assert model.regressor_names == ("task", "rt")
    with pytest.raises(AttributeError):
        model.modulators = ()
