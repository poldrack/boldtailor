"""One-release aliases for renamed keywords and functions."""

import warnings


class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


UNSET = _Unset()


def deprecated(message: str, *, stacklevel: int = 3) -> None:
    """Emit a DeprecationWarning attributed to the caller's caller."""
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def renamed_keyword(owner, old, new, old_value, new_value, *, default=UNSET):
    """Value for ``new``, accepting ``old`` for one release with a warning.

    Callers must use UNSET for both parameter defaults, preserving omission.
    Supplying both raises TypeError; ``default`` is applied only after resolution.
    """
    if old_value is not UNSET:
        if new_value is not UNSET:
            raise TypeError(f"{owner}() got both {old!r} and {new!r}; use {new!r}")
        deprecated(f"{owner}({old}=...) is deprecated; use {new}=...", stacklevel=4)
        return old_value
    if new_value is not UNSET:
        return new_value
    if default is not UNSET:
        return default
    raise TypeError(f"{owner}() missing required keyword argument {new!r}")
