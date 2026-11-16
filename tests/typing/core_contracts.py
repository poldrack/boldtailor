"""Static contracts for dtype-preserving arrays and parallel worker results."""

from collections.abc import Iterator
from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from boldtailor._arrays import readonly_array, copy_array_fields_in_place
from boldtailor.parallel import map_blocks


def check_owned_arrays(instance: object) -> None:
    """Boolean and integer requests retain their dtypes rather than defaulting to float."""
    assert_type(readonly_array([1, 2]), NDArray[np.float64])
    assert_type(readonly_array([1, 2], dtype=np.int64), NDArray[np.int64])
    assert_type(readonly_array([True, False], dtype=bool), NDArray[np.bool_])
    copy_array_fields_in_place(instance, ("flags",), dtype=bool)
    copy_array_fields_in_place(instance, ("indices",), dtype=np.int64)


def _worker(block: int, offset: int) -> str:
    return str(block + offset)


def check_worker_results() -> None:
    """Parallel execution preserves the worker result type without adding None."""
    assert_type(map_blocks(_worker, [1, 2], args=(3,)), Iterator[tuple[int, str]])
