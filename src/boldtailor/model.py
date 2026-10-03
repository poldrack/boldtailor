from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from boldtailor._fit_diagnostics import DIAGNOSTIC_NOISE_MODEL
from boldtailor._hrf_design import (  # HRF_NORMALIZATION: public re-export
    HRF_NORMALIZATION,
    MIN_ONSET,
    OVERSAMPLING,
)
from boldtailor._scalars import is_boolean, is_integer, is_real  # public

ContrastWeights = Mapping[str, float]
ContrastValue = str | ContrastWeights
HRFModel = str | Callable[..., np.ndarray] | None

_RESERVED_TASK_COLUMNS = frozenset({"task", "constant", "onset", "duration"})
_MISSING_POLICIES = ("error", "indicator")


@dataclass(frozen=True)
class Modulator:
    """One parametric task regressor derived from a raw events column."""

    column: str
    center: bool = True
    missing: str = "error"

    def __post_init__(self) -> None:
        _validate_modulator_column(self.column)
        if not isinstance(self.center, bool):
            raise ValueError("modulator center must be a boolean")
        if self.missing not in _MISSING_POLICIES:
            raise ValueError("modulator missing policy must be 'error' or 'indicator'")

    @property
    def indicator_name(self) -> str:
        return f"missing_{self.column}"

    def to_dict(self) -> dict[str, object]:
        return {"column": self.column, "center": self.center, "missing": self.missing}


@dataclass(frozen=True)
class TaskModel:
    """Task regressor plus modulators; the default is the task regressor alone."""

    modulators: tuple[Modulator, ...] = ()

    def __post_init__(self) -> None:
        modulators = tuple(self.modulators)
        if any(not isinstance(m, Modulator) for m in modulators):
            raise ValueError("task model modulators must be Modulator instances")
        columns = [m.column for m in modulators]
        if len(set(columns)) != len(columns):
            raise ValueError("modulator columns must be unique")
        object.__setattr__(self, "modulators", modulators)

    @property
    def regressor_names(self) -> tuple[str, ...]:
        return ("task", *(m.column for m in self.modulators))

    @property
    def profiled_names(self) -> tuple[str, ...]:
        return tuple(
            m.indicator_name for m in self.modulators if m.missing == "indicator"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "regressors": list(self.regressor_names),
            "modulators": [m.to_dict() for m in self.modulators],
        }

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def is_subset_of(self, other: "TaskModel") -> bool:
        """True when every modulator here appears unchanged in the other model."""
        if not isinstance(other, TaskModel):
            raise ValueError("is_subset_of expects a TaskModel")
        return set(self.modulators) <= set(other.modulators)


def _validate_modulator_column(column: object) -> None:
    if not isinstance(column, str) or not column:
        raise ValueError("modulator column must be a nonempty string")
    if column in _RESERVED_TASK_COLUMNS or column.startswith("missing_"):
        raise ValueError(f"modulator column {column!r} is reserved")


@dataclass(frozen=True)
class ModelSpec:
    contrasts: Mapping[str, ContrastValue]
    confounds: Sequence[str] = ()
    hrf_model: HRFModel = "glover"
    drift_model: str | None = "cosine"
    high_pass: float = 0.01
    drift_order: int = 1
    oversampling: int = OVERSAMPLING
    min_onset: float = MIN_ONSET
    noise_model: str = "ar1"
    task_model: TaskModel | None = None
    _contrast_names: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = _prepare_contrasts(self.contrasts)
        selected_confounds = _prepare_confound_names(self.confounds)
        _validate_noise_model(self.noise_model)
        _validate_task_model(self.task_model, self.hrf_model)
        _validate_design_options(
            self.high_pass,
            self.drift_order,
            self.oversampling,
            self.min_onset,
        )
        object.__setattr__(self, "contrasts", MappingProxyType(normalized))
        object.__setattr__(self, "confounds", selected_confounds)
        # Stored as builtin ints so model identities stay JSON-native.
        object.__setattr__(self, "drift_order", int(self.drift_order))
        object.__setattr__(self, "oversampling", int(self.oversampling))
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


def contrasts_from_metadata(metadata):
    """Invert :func:`contrast_metadata` back to expressions and weight maps."""
    return {
        name: (
            value["value"] if value["kind"] == "expression" else dict(value["weights"])
        )
        for name, value in metadata.items()
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
        if is_boolean(value):
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
        raise ValueError("noise_model must be 'ols' or 'ar1'")


def _validate_task_model(task_model: object, hrf_model: HRFModel) -> None:
    if task_model is None:
        return
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel or None")
    if hrf_model not in ("spm", "glover"):
        raise ValueError("task_model requires hrf_model 'spm' or 'glover'")


def _validate_design_options(
    high_pass: float,
    drift_order: int,
    oversampling: int,
    min_onset: float,
) -> None:
    if not is_real(high_pass) or not np.isfinite(high_pass) or high_pass <= 0:
        raise ValueError("high_pass must be positive and finite")
    if not is_integer(drift_order) or drift_order < 0:
        raise ValueError("drift_order must be a non-negative integer")
    if not is_integer(oversampling) or oversampling < 1:
        raise ValueError("oversampling must be a positive integer")
    if not is_real(min_onset) or not np.isfinite(min_onset):
        raise ValueError("min_onset must be finite")


@dataclass(frozen=True)
class ModelIdentity:
    activity: dict[str, object]
    fingerprint: dict[str, object] | None
    warnings: tuple[Mapping[str, object], ...]


def nuisance_model_settings(model: ModelSpec) -> dict[str, object]:
    return {
        "events": False,
        "confounds": list(model.confounds),
        "drift_model": model.drift_model,
        "high_pass": model.high_pass,
        "drift_order": model.drift_order,
        "noise_model": DIAGNOSTIC_NOISE_MODEL,
    }


def model_identity(model: ModelSpec, *, hrf_model: object = "keep") -> ModelIdentity:
    """Return the activity and fingerprint settings that identify a model.

    ``hrf_model`` overrides how the HRF is recorded, for example
    ``{"kind": "selected"}`` when per-feature HRFs replace ``model.hrf_model``.
    """
    value = (
        model.hrf_model
        if isinstance(hrf_model, str) and hrf_model == "keep"
        else hrf_model
    )
    serialized, callable_warnings = _serialize_hrf(value)
    activity = _identity_activity(model, serialized, _hrf_normalization(value))
    fingerprint = None if callable_warnings else activity.copy()
    return ModelIdentity(activity, fingerprint, callable_warnings)


def _identity_activity(
    model: ModelSpec, hrf_model: object, normalization: str | None
) -> dict[str, object]:
    activity = {
        "contrasts": contrast_metadata(model.contrasts),
        "confounds": list(model.confounds),
        "hrf_model": hrf_model,
        "drift_model": model.drift_model,
        "high_pass": model.high_pass,
        "drift_order": model.drift_order,
        "oversampling": model.oversampling,
        "min_onset": model.min_onset,
        "noise_model": model.noise_model,
    }
    if normalization is not None:
        activity["hrf_normalization"] = normalization
    if model.task_model is not None:
        activity["task_model"] = model.task_model.to_dict()
        activity["task_model_fingerprint"] = model.task_model.fingerprint
    return activity


def _hrf_normalization(value: object) -> str | None:
    if value in ("spm", "glover"):
        return HRF_NORMALIZATION
    return "nilearn_sum_one" if isinstance(value, str) else None


def _serialize_hrf(
    value: object,
) -> tuple[object, tuple[Mapping[str, object], ...]]:
    if not callable(value):
        return value, ()
    identity = _callable_identity(value)
    if identity is not None:
        return {"kind": "callable", **identity}, ()
    warning = {
        "code": "reproducibility",
        "message": "HRF callable is not importable; reproducibility is partial",
    }
    return {"kind": "callable", "reproducibility": "partial"}, (warning,)


def _callable_identity(value: object) -> dict[str, str] | None:
    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        return None
    if "<" in qualname or module_name not in sys.modules:
        return None
    resolved = sys.modules[module_name]
    try:
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except AttributeError:
        return None
    if resolved is not value:
        return None
    return {"module": module_name, "qualname": qualname}
