"""Owned NumPy arrays with protection against accidental writes."""

import numpy as np


def readonly_array(values, *, dtype=np.float64):
    """Copy values into a plain array; callers should copy again before editing."""
    array = np.array(values, dtype=dtype, order="C", copy=True, subok=False)
    array.setflags(write=False)
    return array
