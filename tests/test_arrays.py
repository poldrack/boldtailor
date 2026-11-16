"""Owned ordinary arrays protect against accidental writes."""

import numpy as np
import pytest

from boldtailor import _arrays


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
def test_readonly_array_owns_values_and_keeps_dtype(dtype):
    """Verify read-only array owns values and keeps dtype."""
    source = np.arange(24).reshape(4, 6).astype(dtype)[:, ::2]
    expected = source.copy()
    result = _arrays.readonly_array(source, dtype=dtype)
    assert type(result) is np.ndarray
    assert result.flags.owndata and result.flags.c_contiguous
    assert result.dtype == np.dtype(dtype)
    assert not np.shares_memory(result, source)
    assert not result.flags.writeable
    np.testing.assert_array_equal(result, expected)
    source[...] = 0
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.flat[0] = 0
    editable = result.copy()
    editable.flat[0] = 0
    assert editable.flags.writeable


def test_readonly_array_removes_subclass_behavior():
    """Verify read-only array removes subclass behavior."""

    class CallerArray(np.ndarray):
        pass

    source = np.arange(6.0).view(CallerArray)
    result = _arrays.readonly_array(source)
    assert type(result) is np.ndarray
    assert not np.shares_memory(result, source)


def test_sum_arrays_preserves_sequential_rounding_and_ownership():
    """Array totals use the original left-to-right sum and own their storage."""
    from boldtailor._arrays import sum_arrays

    arrays = [np.array([1e16]), np.array([1.0]), np.array([-1e16])]
    result = sum_arrays(arrays)
    np.testing.assert_array_equal(result, [0.0])
    assert not any(np.shares_memory(result, array) for array in arrays)


def test_sum_arrays_rejects_empty_input():
    """An empty reduction must not masquerade as a scalar-valued array."""
    from boldtailor._arrays import sum_arrays

    with pytest.raises(ValueError, match="at least one array"):
        sum_arrays([])
