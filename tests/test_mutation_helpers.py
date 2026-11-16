"""Mutating helpers name their destination and preserve source ownership."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from boldtailor import _arrays, reml, trial_encoding


@pytest.mark.parametrize("keep", [slice(None), np.array([True, False, True])])
def test_copy_fit_values_into_updates_only_selected_destination_columns(keep):
    """Verify copy fit values into updates only selected destination columns."""
    destination = {"ratio": np.full(6, -1.0), "criterion": np.full(6, -2.0)}
    values = {
        "ratio": np.array([10.0, 20.0, 30.0]),
        "criterion": np.array([1.0, 2.0, 3.0]),
    }
    before = {name: array.copy() for name, array in values.items()}
    columns = np.array([4, 0, 2])
    returned = reml._copy_fit_values_into(destination, columns, values, keep)
    assert returned is None
    wanted_ratio = (
        [20.0, -1.0, 30.0, -1.0, 10.0, -1.0]
        if isinstance(keep, slice)
        else [-1.0, -1.0, 30.0, -1.0, 10.0, -1.0]
    )
    wanted_criterion = (
        [2.0, -2.0, 3.0, -2.0, 1.0, -2.0]
        if isinstance(keep, slice)
        else [-2.0, -2.0, 3.0, -2.0, 1.0, -2.0]
    )
    np.testing.assert_array_equal(destination["ratio"], wanted_ratio)
    np.testing.assert_array_equal(destination["criterion"], wanted_criterion)
    for name in values:
        np.testing.assert_array_equal(values[name], before[name])


def test_copy_fit_values_into_defaults_to_all_values():
    """Verify copy fit values into defaults to all values."""
    destination = {"ratio": np.zeros(3)}
    reml._copy_fit_values_into(
        destination, np.array([2, 0]), {"ratio": np.array([7.0, 9.0])}
    )
    np.testing.assert_array_equal(destination["ratio"], [9.0, 0.0, 7.0])


def test_censor_response_time_in_place_changes_only_invalid_times():
    """Verify censor response time in place changes only invalid times."""
    table = pd.DataFrame(
        {"response_time": [1.0, 0.0, -1.0, np.inf], "other": [1, 2, 3, 4]}
    )
    assert trial_encoding._censor_response_time_in_place(table) is None
    np.testing.assert_allclose(
        table.response_time, [1.0, np.nan, np.nan, np.nan], equal_nan=True
    )
    assert table.other.tolist() == [1, 2, 3, 4]


def test_field_mutators_replace_fields_without_aliasing_source_arrays():
    """Verify field mutators replace fields without aliasing source arrays."""
    source = np.array([1.0, 2.0])
    instance = SimpleNamespace(array=source, arrays=[source], labels=["a", "b"])
    assert _arrays.copy_array_fields_in_place(instance, ["array"]) is None
    assert _arrays.copy_array_tuple_fields_in_place(instance, ["arrays"]) is None
    assert _arrays.freeze_sequence_fields_in_place(instance, ["labels"]) is None
    assert _arrays.set_fields_in_place(instance, label="owned") is None
    source[:] = 99
    np.testing.assert_array_equal(instance.array, [1.0, 2.0])
    np.testing.assert_array_equal(instance.arrays[0], [1.0, 2.0])
    assert not instance.array.flags.writeable
    assert not instance.arrays[0].flags.writeable
    assert isinstance(instance.arrays, tuple)
    assert instance.labels == ("a", "b")
    assert instance.label == "owned"
