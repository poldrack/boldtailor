"""Owned NumPy arrays with protection against accidental writes."""

import numpy as np


def readonly_array(values, *, dtype=np.float64):
    """Copy values into a plain array; callers should copy again before editing."""
    array = np.array(values, dtype=dtype, order="C", copy=True, subok=False)
    array.setflags(write=False)
    return array


def set_fields_in_place(instance, **values):
    """Set fields of a frozen dataclass from its own ``__post_init__``."""
    for name, value in values.items():
        object.__setattr__(instance, name, value)


def copy_array_fields_in_place(instance, names, *, dtype=np.float64):
    """Replace each named field with a read-only owned array."""
    set_fields_in_place(
        instance,
        **{
            name: readonly_array(getattr(instance, name), dtype=dtype) for name in names
        },
    )


def copy_array_tuple_fields_in_place(instance, names, *, dtype=np.float64):
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
