"""Scalar type predicates that never treat booleans as numbers."""

from numbers import Integral, Real

import numpy as np


def is_boolean(value: object) -> bool:
    """True for Python and NumPy booleans."""
    return isinstance(value, (bool, np.bool_))


def is_integer(value: object) -> bool:
    """True for integral numbers (including NumPy integers) that are not booleans."""
    return isinstance(value, Integral) and not is_boolean(value)


def is_real(value: object) -> bool:
    """True for real numbers (including NumPy scalars) that are not booleans."""
    return isinstance(value, Real) and not is_boolean(value)
