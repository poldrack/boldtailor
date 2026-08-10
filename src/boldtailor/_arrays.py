from __future__ import annotations

import numpy as np


class _ImmutableFloatArray(np.ndarray):
    def setflags(self, write=None, align=None, uic=None) -> None:
        if write:
            raise ValueError("cannot set WRITEABLE flag to True of this array")
        super().setflags(write=write, align=align, uic=uic)


def immutable_float_array(values: object) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64, order="C")
    immutable = np.ndarray.__new__(
        _ImmutableFloatArray,
        source.shape,
        dtype=np.float64,
        order="C",
    )
    immutable[...] = source
    np.ndarray.setflags(immutable, write=False)
    return immutable
