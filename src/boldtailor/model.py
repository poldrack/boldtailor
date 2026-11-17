from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from boldtailor._fit_diagnostics import DIAGNOSTIC_NOISE_MODEL
from boldtailor._hrf_design import HRF_NORMALIZATION
from boldtailor._constants import MIN_ONSET, OVERSAMPLING
from boldtailor._scalars import is_boolean, is_integer, is_real  # public

ContrastWeights = Mapping[str, float]
ContrastValue = str | ContrastWeights
HRFModel = str | Callable[..., np.ndarray] | None

_RESERVED_TASK_COLUMNS = frozenset({"task", "constant", "onset", "duration"})
_MISSING_POLICIES = ("error", "indicator")


_KINDS = ("numeric", "categorical")
_MISSING_TEXT = frozenset({"", "n/a", "nan"})


def level_name(value: object) -> str | None:
    """Canonical string for one categorical value, or None when it is missing.

    Numbers and strings that parse as numbers share one form (1, 1.0 and "1.0"
    all become "1"), so a level written differently across runs stays one
    level. NaN, None, "" and "n/a" are missing; infinities and booleans are
    rejected with a ValueError rather than silently becoming levels or gaps.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _MISSING_TEXT:
            return None
        number = _parse_number(text)
        return text if number is None else _number_name(number)
    if value is None:
        return None
    if is_real(value):
        return _number_name(float(value))
    raise ValueError(f"categorical value {value!r} must be a string or a number")


def _parse_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _number_name(number: float) -> str | None:
    if np.isnan(number):
        return None
    if np.isinf(number):
        raise ValueError(f"categorical value {number!r} must be finite")
    return str(int(number)) if number.is_integer() else repr(number)


def _sorted_levels(levels):
    """Numeric order when every level is a number, else lexical order."""
    if all(_parse_number(level) is not None for level in levels):
        return tuple(sorted(levels, key=float))
    return tuple(sorted(levels))


class Modulator:
    """One task regressor family derived from a raw events column.

    Numeric modulators enter uncentered, so the task regressor is the response
    at modulator value zero. Categorical modulators add one 0/1 indicator per
    non-reference level, so the task regressor is the reference-level response.
    """

    __slots__ = ("_column", "_missing", "_kind", "_levels", "_reference")

    def __init__(
        self,
        column: str,
        missing: str = "error",
        kind: str = "numeric",
        levels: Sequence[object] | None = None,
        reference: object = None,
    ) -> None:
        _validate_modulator_column(column)
        if missing not in _MISSING_POLICIES:
            raise ValueError("modulator missing policy must be 'error' or 'indicator'")
        if kind not in _KINDS:
            raise ValueError("modulator kind must be 'numeric' or 'categorical'")
        if kind == "numeric":
            if levels is not None or reference is not None:
                raise ValueError(
                    "levels and reference apply only to categorical modulators"
                )
            resolved_levels = None
            resolved_reference = None
        else:
            resolved_reference = _resolve_reference_name(column, reference)
            if levels is not None:
                resolved_levels, resolved_reference = _resolve_levels(
                    column, levels, resolved_reference
                )
            else:
                resolved_levels = None
        self._column = column
        self._missing = missing
        self._kind = kind
        self._levels = resolved_levels
        self._reference = resolved_reference

    @property
    def column(self) -> str:
        return self._column

    @property
    def missing(self) -> str:
        return self._missing

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def levels(self) -> tuple[str, ...] | None:
        return self._levels

    @property
    def reference(self) -> str | None:
        return self._reference

    def replace(self, **changes: object) -> Modulator:
        values: dict[str, object] = {
            "column": self._column,
            "missing": self._missing,
            "kind": self._kind,
            "levels": self._levels,
            "reference": self._reference,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unexpected Modulator fields: {sorted(unknown)}")
        values.update(changes)
        return Modulator(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Modulator):
            return NotImplemented
        return (
            self._column == other._column
            and self._missing == other._missing
            and self._kind == other._kind
            and self._levels == other._levels
            and self._reference == other._reference
        )

    def __hash__(self) -> int:
        return hash(
            (self._column, self._missing, self._kind, self._levels, self._reference)
        )

    def __repr__(self) -> str:
        return (
            f"Modulator(column={self._column!r}, missing={self._missing!r}, "
            f"kind={self._kind!r}, levels={self._levels!r}, "
            f"reference={self._reference!r})"
        )

    @property
    def resolved(self) -> bool:
        return self._kind == "numeric" or self._levels is not None

    def with_levels(self, values) -> Modulator:
        return self.replace(levels=tuple(values))

    @property
    def regressor_names(self) -> tuple[str, ...]:
        if self._kind == "numeric":
            return (self._column,)
        if self._levels is None:
            raise ValueError(
                f"categorical modulator {self._column!r} has no levels yet"
            )
        return tuple(
            f"{self._column}[{lv}]" for lv in self._levels if lv != self._reference
        )

    @property
    def indicator_name(self) -> str:
        return f"missing_{self._column}"

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "column": self._column,
            "missing": self._missing,
            "kind": self._kind,
        }
        if self._kind == "categorical":
            values["levels"] = None if self._levels is None else list(self._levels)
            values["reference"] = self._reference
        return values


def _resolve_reference_name(column: str, reference: object) -> str | None:
    if reference is None:
        return None
    name = level_name(reference)
    if name is None:
        raise ValueError(f"modulator {column!r} reference must be a nonmissing level")
    return name


def _resolve_levels(
    column: str, values: Sequence[object], reference: str | None
) -> tuple[tuple[str, ...], str]:
    """Validate and return normalized levels and reference."""
    names = [level_name(v) for v in values]
    if any(n is None for n in names):
        raise ValueError(f"modulator {column!r} levels must be nonmissing")
    if len(set(names)) != len(names):
        raise ValueError(f"modulator {column!r} levels must be distinct")
    if len(names) < 2:
        raise ValueError(f"categorical modulator {column!r} needs at least two levels")
    levels = _sorted_levels(names)
    if reference is not None and reference not in levels:
        raise ValueError(
            f"reference {reference!r} is not a level of {column!r}: {list(levels)}"
        )
    return levels, (reference or levels[0])


class TaskModel:
    """Task regressor plus modulators; the default is the task regressor alone."""

    __slots__ = ("_modulators",)

    def __init__(self, modulators: Sequence[Modulator] = ()) -> None:
        resolved = tuple(modulators)
        if any(not isinstance(m, Modulator) for m in resolved):
            raise ValueError("task model modulators must be Modulator instances")
        columns = [m.column for m in resolved]
        if len(set(columns)) != len(columns):
            raise ValueError("modulator columns must be unique")
        if any(not m.resolved for m in resolved):
            raise ValueError("categorical modulators need levels before fitting")
        names = [n for m in resolved for n in m.regressor_names]
        if len(set(names)) != len(names) or "task" in names:
            raise ValueError("task-model regressor names must be unique")
        self._modulators = resolved

    @property
    def modulators(self) -> tuple[Modulator, ...]:
        return self._modulators

    def replace(self, **changes: object) -> TaskModel:
        values: dict[str, object] = {"modulators": self._modulators}
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unexpected TaskModel fields: {sorted(unknown)}")
        values.update(changes)
        return TaskModel(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskModel):
            return NotImplemented
        return self._modulators == other._modulators

    def __hash__(self) -> int:
        return hash(self._modulators)

    def __repr__(self) -> str:
        return f"TaskModel(modulators={self._modulators!r})"

    @property
    def regressor_names(self) -> tuple[str, ...]:
        return ("task", *(n for m in self._modulators for n in m.regressor_names))

    @property
    def profiled_names(self) -> tuple[str, ...]:
        return tuple(
            m.indicator_name for m in self._modulators if m.missing == "indicator"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "regressors": list(self.regressor_names),
            "modulators": [m.to_dict() for m in self._modulators],
        }

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def is_subset_of(self, other: TaskModel) -> bool:
        """True when every modulator here appears unchanged in the other model."""
        if not isinstance(other, TaskModel):
            raise ValueError("is_subset_of expects a TaskModel")
        return set(self._modulators) <= set(other._modulators)


def _validate_modulator_column(column: object) -> None:
    if not isinstance(column, str) or not column:
        raise ValueError("modulator column must be a nonempty string")
    if column in _RESERVED_TASK_COLUMNS or column.startswith("missing_"):
        raise ValueError(f"modulator column {column!r} is reserved")


class ModelSpec:
    __slots__ = (
        "_contrasts",
        "_confounds",
        "_hrf_model",
        "_drift_model",
        "_high_pass",
        "_drift_order",
        "_oversampling",
        "_min_onset",
        "_noise_model",
        "_task_model",
        "_contrast_names",
    )

    def __init__(
        self,
        contrasts: Mapping[str, ContrastValue],
        confounds: Sequence[str] = (),
        hrf_model: HRFModel = "glover",
        drift_model: str | None = "cosine",
        high_pass: float = 0.01,
        drift_order: int = 1,
        oversampling: int = OVERSAMPLING,
        min_onset: float = MIN_ONSET,
        noise_model: str = "ar1",
        task_model: TaskModel | None = None,
    ) -> None:
        normalized = prepare_contrasts(contrasts)
        selected_confounds = _prepare_confound_names(confounds)
        validate_noise_model(noise_model)
        _validate_task_model(task_model, hrf_model)
        _validate_design_options(high_pass, drift_order, oversampling, min_onset)
        self._contrasts: Mapping[str, ContrastValue] = MappingProxyType(normalized)
        self._confounds = selected_confounds
        self._hrf_model = hrf_model
        self._drift_model = drift_model
        self._high_pass = high_pass
        # Stored as builtin ints so model identities stay JSON-native.
        self._drift_order = int(drift_order)
        self._oversampling = int(oversampling)
        self._min_onset = min_onset
        self._noise_model = noise_model
        self._task_model = task_model
        self._contrast_names = tuple(normalized)

    @property
    def contrasts(self) -> Mapping[str, ContrastValue]:
        return self._contrasts

    @property
    def confounds(self) -> tuple[str, ...]:
        return self._confounds

    @property
    def hrf_model(self) -> HRFModel:
        return self._hrf_model

    @property
    def drift_model(self) -> str | None:
        return self._drift_model

    @property
    def high_pass(self) -> float:
        return self._high_pass

    @property
    def drift_order(self) -> int:
        return self._drift_order

    @property
    def oversampling(self) -> int:
        return self._oversampling

    @property
    def min_onset(self) -> float:
        return self._min_onset

    @property
    def noise_model(self) -> str:
        return self._noise_model

    @property
    def task_model(self) -> TaskModel | None:
        return self._task_model

    @property
    def contrast_names(self) -> tuple[str, ...]:
        return self._contrast_names

    def replace(self, **changes: object) -> ModelSpec:
        values: dict[str, object] = {
            "contrasts": self._contrasts,
            "confounds": self._confounds,
            "hrf_model": self._hrf_model,
            "drift_model": self._drift_model,
            "high_pass": self._high_pass,
            "drift_order": self._drift_order,
            "oversampling": self._oversampling,
            "min_onset": self._min_onset,
            "noise_model": self._noise_model,
            "task_model": self._task_model,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unexpected ModelSpec fields: {sorted(unknown)}")
        values.update(changes)
        return ModelSpec(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelSpec):
            return NotImplemented
        return (
            self._contrasts == other._contrasts
            and self._confounds == other._confounds
            and self._hrf_model == other._hrf_model
            and self._drift_model == other._drift_model
            and self._high_pass == other._high_pass
            and self._drift_order == other._drift_order
            and self._oversampling == other._oversampling
            and self._min_onset == other._min_onset
            and self._noise_model == other._noise_model
            and self._task_model == other._task_model
        )

    def __repr__(self) -> str:
        return (
            f"ModelSpec(contrasts={self._contrasts!r}, confounds={self._confounds!r}, "
            f"hrf_model={self._hrf_model!r}, drift_model={self._drift_model!r}, "
            f"high_pass={self._high_pass!r}, drift_order={self._drift_order!r}, "
            f"oversampling={self._oversampling!r}, min_onset={self._min_onset!r}, "
            f"noise_model={self._noise_model!r}, task_model={self._task_model!r})"
        )


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


def prepare_contrasts(
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


def validate_noise_model(value: str) -> None:
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
    """Peak-one for named canonical shapes; Nilearn's sum-one for its
    canonical derivative bases; nothing for FIR and user-supplied kernels,
    which are used exactly as given."""
    if value in ("spm", "glover"):
        return HRF_NORMALIZATION
    if isinstance(value, str) and value != "fir":
        return "nilearn_sum_one"
    return None


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
