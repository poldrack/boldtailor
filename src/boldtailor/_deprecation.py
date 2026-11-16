"""One-release aliases for renamed keywords and functions."""

import warnings
from typing import TypeVar, cast

_Value = TypeVar("_Value")


class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


UNSET = _Unset()


def deprecated(message: str, *, stacklevel: int = 3) -> None:
    """Emit a DeprecationWarning attributed to the caller's caller."""
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def renamed_keyword(
    owner: str,
    old: str,
    new: str,
    old_value: _Value | _Unset,
    new_value: _Value | _Unset,
    *,
    default: _Value | _Unset = UNSET,
) -> _Value:
    """Value for ``new``, accepting ``old`` for one release with a warning.

    Callers must use UNSET for both parameter defaults, preserving omission.
    Supplying both raises TypeError; ``default`` is applied only after resolution.
    """
    if old_value is not UNSET:
        if new_value is not UNSET:
            raise TypeError(f"{owner}() got both {old!r} and {new!r}; use {new!r}")
        deprecated(f"{owner}({old}=...) is deprecated; use {new}=...", stacklevel=4)
        # Identity checks exclude the singleton sentinel; preserve all other values.
        return cast(_Value, old_value)
    if new_value is not UNSET:
        return cast(_Value, new_value)
    if default is not UNSET:
        return cast(_Value, default)
    raise TypeError(f"{owner}() missing required keyword argument {new!r}")
