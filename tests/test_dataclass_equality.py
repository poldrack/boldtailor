"""Dataclasses that hold arrays or tables compare by identity, never field by field."""

import dataclasses
import importlib
import pkgutil

import numpy as np
import pytest

import boldtailor
from boldtailor.denoising import PoolMasks
from boldtailor.reml import RemlGroup

ARRAY_TYPES = ("ndarray", "NDArray", "DataFrame")


def _package_dataclasses():
    for info in pkgutil.walk_packages(boldtailor.__path__, "boldtailor."):
        module = importlib.import_module(info.name)
        for item in vars(module).values():
            if (
                isinstance(item, type)
                and dataclasses.is_dataclass(item)
                and item.__module__ == info.name
            ):
                yield item


def _holds_arrays(item):
    return any(
        name in str(field.type)
        for field in dataclasses.fields(item)
        for name in ARRAY_TYPES
    )


def test_array_holding_dataclasses_do_not_generate_field_equality():
    """A generated ``__eq__`` on array fields raises or misleads, so none is generated."""
    offenders = sorted(
        f"{item.__module__}.{item.__name__}"
        for item in _package_dataclasses()
        if _holds_arrays(item) and item.__dataclass_params__.eq
    )
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize(
    "build",
    [
        lambda: PoolMasks(
            pool=np.array([True, False]),
            scoring=np.array([False, True]),
            threshold=0.1,
            fallback=False,
        ),
        lambda: RemlGroup(np.ones((4, 2)), np.ones((4, 1))),
    ],
)
def test_equal_content_compares_by_identity_without_raising(build):
    """Two records with the same arrays are distinct objects; ``==`` never raises."""
    one, two = build(), build()
    assert one == one
    assert (one == two) is False
    assert len({one, two}) == 2
