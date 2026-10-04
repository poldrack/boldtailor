"""A task model is a frozen, fingerprinted list of task regressors."""

import json

import numpy as np
import pytest

from boldtailor.model import Modulator, TaskModel, level_name


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
            Modulator("response_time", missing="indicator"),
            Modulator("trial_type"),
        )
    )
    assert model.regressor_names == ("task", "response_time", "trial_type")
    assert model.profiled_names == ("missing_response_time",)
    assert model.to_dict() == {
        "regressors": ["task", "response_time", "trial_type"],
        "modulators": [
            {"column": "response_time", "missing": "indicator", "kind": "numeric"},
            {"column": "trial_type", "missing": "error", "kind": "numeric"},
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
        != TaskModel((Modulator("response_time", missing="indicator"),)).fingerprint
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Modulator("task"), "is reserved"),
        (lambda: Modulator("constant"), "is reserved"),
        (lambda: Modulator("onset"), "is reserved"),
        (lambda: Modulator("duration"), "is reserved"),
        (lambda: Modulator("missing_rt"), "is reserved"),
        (lambda: Modulator(""), "nonempty string"),
        (lambda: Modulator(None), "nonempty string"),
        (lambda: Modulator(3), "nonempty string"),
        (lambda: Modulator("response_time", missing="drop"), "missing policy"),
        (lambda: Modulator("response_time", missing=""), "missing policy"),
        (lambda: Modulator("response_time", missing=None), "missing policy"),
        (lambda: Modulator("response_time", missing=True), "missing policy"),
        (
            lambda: TaskModel((Modulator("rt"), Modulator("rt", missing="indicator"))),
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
    rt = Modulator("response_time", missing="indicator")
    trial = Modulator("trial_type")
    full = TaskModel((rt, trial))
    assert TaskModel().is_subset_of(full)
    assert TaskModel((trial,)).is_subset_of(full)
    assert full.is_subset_of(full)
    assert not full.is_subset_of(TaskModel((trial,)))
    assert not TaskModel((Modulator("trial_type", missing="indicator"),)).is_subset_of(
        full
    )
    assert not TaskModel((Modulator("response_time"),)).is_subset_of(full)
    assert not TaskModel((Modulator("image"),)).is_subset_of(full)
    with pytest.raises(ValueError, match="TaskModel"):
        full.is_subset_of("nsd")


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, "1"),
        (1.0, "1"),
        (np.int64(2), "2"),
        ("1", "1"),
        (" face ", "face"),
        (2.5, "2.5"),
        (None, None),
        (float("nan"), None),
        ("", None),
        ("n/a", None),
    ],
)
def test_level_name_canonicalises_values(value, expected):
    assert level_name(value) == expected


def test_level_name_rejects_other_types():
    with pytest.raises(ValueError, match="categorical value"):
        level_name([1])


def test_categorical_levels_are_sorted_and_reference_defaults_to_first():
    mod = Modulator(
        "trial_type", kind="categorical", levels=("scrambled", "face", "house")
    )
    assert mod.levels == ("face", "house", "scrambled")
    assert mod.reference == "face"
    assert mod.regressor_names == ("trial_type[house]", "trial_type[scrambled]")


def test_numeric_levels_sort_numerically_and_normalise():
    mod = Modulator("cond", kind="categorical", levels=(10, 2.0, "1"))
    assert mod.levels == ("1", "2", "10")
    assert mod.regressor_names == ("cond[2]", "cond[10]")


def test_explicit_reference_and_binary_naming():
    mod = Modulator("trial_type", kind="categorical", levels=(0, 1), reference=1)
    assert mod.reference == "1"
    assert mod.regressor_names == ("trial_type[0]",)
    assert Modulator(
        "trial_type", kind="categorical", levels=(1, 0)
    ).regressor_names == ("trial_type[1]",)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(kind="ordinal"), "kind"),
        (dict(levels=("a", "b")), "categorical"),
        (dict(reference="a"), "categorical"),
        (dict(kind="categorical", levels=("a",)), "at least two"),
        (dict(kind="categorical", levels=("a", "a ")), "distinct"),
        (dict(kind="categorical", levels=("a", None)), "nonmissing"),
        (dict(kind="categorical", levels=("a", "b"), reference="c"), "reference"),
    ],
)
def test_invalid_categorical_modulators(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Modulator("cond", **kwargs)


def test_unresolved_categorical_gets_levels_later():
    mod = Modulator("trial_type", kind="categorical", reference="house")
    assert not mod.resolved
    with pytest.raises(ValueError, match="no levels"):
        mod.regressor_names
    resolved = mod.with_levels({"face", "house"})
    assert resolved.resolved and resolved.reference == "house"
    assert resolved.regressor_names == ("trial_type[face]",)


def test_task_model_expands_and_rejects_unresolved_or_duplicate_names():
    cat = Modulator("trial_type", kind="categorical", levels=("a", "b", "c"))
    model = TaskModel((Modulator("response_time", missing="indicator"), cat))
    assert model.regressor_names == (
        "task",
        "response_time",
        "trial_type[b]",
        "trial_type[c]",
    )
    assert model.to_dict()["regressors"] == list(model.regressor_names)
    with pytest.raises(ValueError, match="levels"):
        TaskModel((Modulator("trial_type", kind="categorical"),))
    with pytest.raises(ValueError, match="unique"):
        TaskModel((cat, Modulator("trial_type[b]")))


def test_to_dict_and_fingerprint_record_kind_levels_and_reference():
    cat = Modulator(
        "trial_type", kind="categorical", levels=("a", "b"), missing="indicator"
    )
    assert cat.to_dict() == {
        "column": "trial_type",
        "missing": "indicator",
        "kind": "categorical",
        "levels": ["a", "b"],
        "reference": "a",
    }
    assert Modulator("rt").to_dict() == {
        "column": "rt",
        "missing": "error",
        "kind": "numeric",
    }
    other = Modulator(
        "trial_type",
        kind="categorical",
        levels=("a", "b"),
        reference="b",
        missing="indicator",
    )
    assert TaskModel((cat,)).fingerprint != TaskModel((other,)).fingerprint
    assert not TaskModel((cat,)).is_subset_of(TaskModel((other,)))
    assert Modulator(**cat.to_dict()) == cat
