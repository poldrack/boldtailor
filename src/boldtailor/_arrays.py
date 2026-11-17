"""Owned NumPy arrays and tables with protection against accidental writes."""

from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from typing import Any, TypeVar, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike, NDArray

_Scalar = TypeVar("_Scalar", bound=np.generic)


@overload
def readonly_array(values: ArrayLike) -> NDArray[np.float64]: ...


@overload
def readonly_array(values: ArrayLike, *, dtype: type[bool]) -> NDArray[np.bool_]: ...


@overload
def readonly_array(values: ArrayLike, *, dtype: type[_Scalar]) -> NDArray[_Scalar]: ...


@overload
def readonly_array(values: ArrayLike, *, dtype: DTypeLike) -> NDArray[Any]: ...


def readonly_array(values: ArrayLike, *, dtype: DTypeLike = np.float64) -> NDArray[Any]:
    """Copy values into a plain array; callers should copy again before editing."""
    array = np.array(values, dtype=dtype, order="C", copy=True, subok=False)
    array.setflags(write=False)
    return array


def set_fields_in_place(instance, **values):
    """Set fields of a frozen dataclass from its own ``__post_init__``."""
    for name, value in values.items():
        object.__setattr__(instance, name, value)


def copy_array_fields_in_place(
    instance: object, names: Iterable[str], *, dtype: DTypeLike = np.float64
) -> None:
    """Replace each named field with a read-only owned array."""
    set_fields_in_place(
        instance,
        **{
            name: readonly_array(getattr(instance, name), dtype=dtype) for name in names
        },
    )


def copy_array_tuple_fields_in_place(
    instance: object, names: Iterable[str], *, dtype: DTypeLike = np.float64
) -> None:
    """Replace each named field with a tuple of read-only owned arrays."""
    set_fields_in_place(
        instance,
        **{
            name: tuple(readonly_array(a, dtype=dtype) for a in getattr(instance, name))
            for name in names
        },
    )


def freeze_sequence_fields_in_place(instance, names):
    """Replace each named sequence field with an immutable tuple."""
    set_fields_in_place(
        instance, **{name: tuple(getattr(instance, name)) for name in names}
    )


def sum_arrays(values: Iterable[NDArray[Any]]) -> NDArray[Any]:
    """Sum nonempty arrays in Python's existing left-to-right order."""
    arrays = iter(values)
    first = next(arrays, None)
    if first is None:
        raise ValueError("sum_arrays requires at least one array")
    return np.asarray(sum(chain((first,), arrays)))


def owned_table(frame: pd.DataFrame) -> pd.DataFrame:
    """A deep copy that also copies the payloads of object columns."""
    copied = frame.copy(deep=True)
    for column, dtype in copied.dtypes.items():
        if dtype != object:
            continue
        payloads = [deepcopy(value) for value in copied[column]]
        copied[column] = pd.Series(payloads, index=copied.index, dtype=object)
    return copied
