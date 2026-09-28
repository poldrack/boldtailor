from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

import numpy as np

ContrastWeights = Mapping[str, float]
ContrastValue = str | ContrastWeights
HRFModel = str | Callable[..., np.ndarray] | None


@dataclass(frozen=True)
class ModelSpec:
    contrasts: Mapping[str, ContrastValue]
    confounds: Sequence[str] = ()
    hrf_model: HRFModel = "glover"
    drift_model: str | None = "cosine"
    high_pass: float = 0.01
    drift_order: int = 1
    oversampling: int = 50
    min_onset: float = -24.0
    noise_model: str = "ar1"
    _contrast_names: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = _prepare_contrasts(self.contrasts)
        selected_confounds = _prepare_confound_names(self.confounds)
        _validate_noise_model(self.noise_model)
        _validate_design_options(
            self.high_pass,
            self.drift_order,
            self.oversampling,
            self.min_onset,
        )
        object.__setattr__(self, "contrasts", MappingProxyType(normalized))
        object.__setattr__(self, "confounds", selected_confounds)
        object.__setattr__(self, "_contrast_names", tuple(normalized))

    @property
    def contrast_names(self) -> tuple[str, ...]:
        return self._contrast_names


def contrast_metadata(contrasts):
    return {
        name: (
            {"kind": "expression", "value": value}
            if isinstance(value, str)
            else {"kind": "weights", "weights": dict(value)}
        )
        for name, value in contrasts.items()
    }


def _prepare_contrasts(
    contrasts: Mapping[str, ContrastValue],
) -> dict[str, ContrastValue]:
    if not contrasts:
        raise ValueError("ModelSpec requires at least one contrast")
    normalized: dict[str, ContrastValue] = {}
    for name, value in contrasts.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("contrast names must be nonempty strings")
        normalized[name] = _prepare_contrast(value, name)
    return normalized


def _prepare_contrast(value: ContrastValue, name: str) -> ContrastValue:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"contrast {name!r} must not be empty")
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            f"contrast {name!r} must use a semantic expression or "
            "regressor-weight mapping"
        )
    return _prepare_weights(value, name)


def _prepare_weights(values: Mapping[str, float], name: str) -> ContrastWeights:
    if not values:
        raise ValueError(f"contrast {name!r} must contain at least one weight")
    prepared: dict[str, float] = {}
    for regressor, value in values.items():
        if not isinstance(regressor, str) or not regressor:
            raise ValueError(f"contrast {name!r} has an invalid regressor name")
        if _is_boolean(value):
            raise ValueError(f"contrast {name!r} weights must be numeric")
        try:
            weight = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"contrast {name!r} weights must be numeric") from error
        if not np.isfinite(weight):
            raise ValueError(f"contrast {name!r} weights must be finite")
        prepared[regressor] = weight
    if not any(weight != 0.0 for weight in prepared.values()):
        raise ValueError(f"contrast {name!r} must contain a nonzero weight")
    return MappingProxyType(prepared)


def _prepare_confound_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("confound names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("confound names must be unique")
    return names


def _validate_noise_model(value: str) -> None:
    if value not in {"ols", "ar1"}:
        raise ValueError("noise_model must be 'ols' or 'ar1' in Phase 1")


def _validate_design_options(
    high_pass: float,
    drift_order: int,
    oversampling: int,
    min_onset: float,
) -> None:
    if not _is_real_number(high_pass) or not np.isfinite(high_pass) or high_pass <= 0:
        raise ValueError("high_pass must be positive and finite")
    if not _is_integer(drift_order) or drift_order < 0:
        raise ValueError("drift_order must be a non-negative integer")
    if not _is_integer(oversampling) or oversampling < 1:
        raise ValueError("oversampling must be a positive integer")
    if not _is_real_number(min_onset) or not np.isfinite(min_onset):
        raise ValueError("min_onset must be finite")


def _is_boolean(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not _is_boolean(value)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not _is_boolean(value)
