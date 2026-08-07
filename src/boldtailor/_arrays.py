from __future__ import annotations

import numpy as np


def immutable_float_array(values: object) -> np.ndarray:
    contiguous = np.array(values, dtype=float, copy=True, order="C")
    immutable = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)
