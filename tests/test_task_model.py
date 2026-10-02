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
    ("build", "message"),
    [
        (lambda: Modulator("task"), "reserved|nonempty"),
        (lambda: Modulator("constant"), "reserved|nonempty"),
        (lambda: Modulator("onset"), "reserved|nonempty"),
        (lambda: Modulator("duration"), "reserved|nonempty"),
        (lambda: Modulator("missing_rt"), "reserved|nonempty"),
        (lambda: Modulator(""), "reserved|nonempty"),
        (lambda: Modulator(None), "reserved|nonempty"),
        (lambda: Modulator(3), "reserved|nonempty"),
        (lambda: Modulator("response_time", missing="drop"), "missing"),
        (lambda: Modulator("response_time", missing=""), "missing"),
        (lambda: Modulator("response_time", missing=None), "missing"),
        (lambda: Modulator("response_time", missing=True), "missing"),
        (lambda: Modulator("response_time", center=1), "center"),
        (lambda: Modulator("response_time", center="yes"), "center"),
        (lambda: Modulator("response_time", center=None), "center"),
        (
            lambda: TaskModel((Modulator("rt"), Modulator("rt", center=False))),
            "unique",
        ),
        (lambda: TaskModel(("rt",)), "Modulator"),
    ],
)
def test_task_model_rejects_invalid_arguments(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_task_model_is_immutable_and_owns_its_tuple():
    modulators = [Modulator("rt")]
    model = TaskModel(modulators)
    modulators.append(Modulator("other"))
    assert model.regressor_names == ("task", "rt")
    with pytest.raises(AttributeError):
        model.modulators = ()


def test_subset_requires_identical_shared_modulators():
    rt = Modulator("response_time", center=True, missing="indicator")
    trial = Modulator("trial_type", center=False)
    full = TaskModel((rt, trial))
    assert TaskModel().is_subset_of(full)
    assert TaskModel((trial,)).is_subset_of(full)
    assert full.is_subset_of(full)
    assert not full.is_subset_of(TaskModel((trial,)))
    assert not TaskModel((Modulator("trial_type", center=True),)).is_subset_of(full)
    assert not TaskModel((Modulator("response_time"),)).is_subset_of(full)
    assert not TaskModel((Modulator("image"),)).is_subset_of(full)
    with pytest.raises(ValueError, match="TaskModel"):
        full.is_subset_of("nsd")
