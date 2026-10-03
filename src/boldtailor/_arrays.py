"""Owned NumPy arrays with protection against accidental writes."""

import numpy as np


def readonly_array(values, *, dtype=np.float64):
    """Copy values into a plain array; callers should copy again before editing."""
    array = np.array(values, dtype=dtype, order="C", copy=True, subok=False)
    array.setflags(write=False)
    return array


def rebind(instance, **values):
    """Set fields of a frozen dataclass from its own ``__post_init__``."""
    for name, value in values.items():
        object.__setattr__(instance, name, value)


def own_fields(instance, names, *, dtype=np.float64):
    """Replace each named field with a read-only owned array."""
    rebind(
        instance,
        **{
            name: readonly_array(getattr(instance, name), dtype=dtype) for name in names
        },
    )


def own_array_tuples(instance, names, *, dtype=np.float64):
    """Replace each named field with a tuple of read-only owned arrays."""
    rebind(
        instance,
        **{
            name: tuple(readonly_array(a, dtype=dtype) for a in getattr(instance, name))
            for name in names
        },
    )


def own_tuples(instance, names):
    """Replace each named sequence field with an immutable tuple."""
    rebind(instance, **{name: tuple(getattr(instance, name)) for name in names})
