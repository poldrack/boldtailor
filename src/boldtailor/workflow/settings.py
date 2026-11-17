"""Resolved, validated settings for one workflow run; the CLI, stages, writer and metadata share it."""

import math
import re
from collections.abc import Mapping
from typing import Literal, overload

from pathlib import Path

from boldtailor._scalars import is_integer, is_real
from boldtailor.hrf_library import (
    HrfLibrary,
    default_hrf_library,
    expanded_hrf_library,
    sobol_hrf_library,
)
from boldtailor.model import Modulator
from boldtailor.hrf_options import RemlHrfOptions

SUPPORTED_SPACES = ("fsLR-91k",)
SPACE_ENTITIES = {"fsLR-91k": "space-fsLR_den-91k"}
RIDGE_MODES = ("fractional_cv", "cv", "fixed", "off")
HRF_LIBRARIES = ("default", "sobol", "expanded", "canonical")
STAGES = ("glms", "reliability", "betas", "summaries")
STAGE_REQUIRES = {"betas": "glms", "summaries": "betas"}
EXISTING_RESULTS = ("error", "overwrite")
ENCODING_MODES = ("within_run", "absolute")
_LABEL = re.compile(r"[A-Za-z0-9]+")


def resolve_fmriprep_dir(bids_dir):
    derivatives = Path(bids_dir) / "derivatives"
    candidates = sorted(p for p in derivatives.glob("fmriprep*") if p.is_dir())
    if not candidates:
        raise ValueError(
            f"no derivatives/fmriprep* directory under {bids_dir}; pass fmriprep_dir"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"several fMRIPrep directories under {derivatives} ({names}); "
            "pass fmriprep_dir"
        )
    return candidates[0]


_MODULATOR_SYNTAX = (
    "COLUMN[:OPTION[,OPTION...]] with OPTION indicator, categorical, or reference=LEVEL"
)


def check_modulator_expressible(mod):
    """Raise ValueError if ``--modulator`` syntax cannot spell ``mod``."""
    if ":" in mod.column:
        raise ValueError(f"modulator column {mod.column!r} contains ':'")
    reference = mod.reference
    if reference is not None and ("," in reference or ":" in reference):
        raise ValueError(
            f"modulator reference {reference!r} contains ',' or ':', which "
            "--modulator cannot express"
        )


def _modulator_options(text, options):
    seen, values = set(), {}
    for option in options:
        key, _, value = option.partition("=")
        if key in seen or key not in ("indicator", "categorical", "reference"):
            raise ValueError(f"modulator must be {_MODULATOR_SYNTAX}, not {text!r}")
        if (key == "reference") != bool(value):
            raise ValueError(f"modulator must be {_MODULATOR_SYNTAX}, not {text!r}")
        seen.add(key)
        values[key] = value
    if "reference" in values and "categorical" not in values:
        raise ValueError(f"modulator reference needs categorical, not {text!r}")
    return values


def parse_modulator(text):
    column, sep, rest = str(text).partition(":")
    if not column or ":" in rest or (sep and not rest):
        raise ValueError(f"modulator must be {_MODULATOR_SYNTAX}, not {text!r}")
    values = _modulator_options(text, rest.split(",") if rest else [])
    return Modulator(
        column,
        missing="indicator" if "indicator" in values else "error",
        kind="categorical" if "categorical" in values else "numeric",
        reference=values.get("reference"),
    )


def _check_unresolved(mod):
    if mod.kind == "categorical" and mod.levels is not None:
        raise ValueError(
            f"modulators: categorical modulator {mod.column!r} must not set "
            "levels; they come from the events"
        )


def _check_expressible(mod):
    try:
        check_modulator_expressible(mod)
    except ValueError as error:
        raise ValueError(f"modulators: {error}") from None


def _nested(path, other):
    """Whether two resolved paths are equal or one contains the other."""
    return path == other or path in other.parents or other in path.parents


def _check_label(value, name, prefix):
    text = str(value)
    body = text.removeprefix(prefix + "-") if prefix else text
    if (prefix and not text.startswith(prefix + "-")) or not _LABEL.fullmatch(body):
        wanted = f"{prefix}-<alphanumeric>" if prefix else "alphanumeric"
        raise ValueError(f"{name} must be {wanted}, not {text!r}")
    return text


def _check_choice(value, name, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return value


@overload
def _check_positive_int(value, name, *, optional: Literal[False] = False) -> int: ...


@overload
def _check_positive_int(value, name, *, optional: Literal[True]) -> int | None: ...


def _check_positive_int(value, name, *, optional=False):
    if optional and value is None:
        return None
    if not is_integer(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer, not {value!r}")
    return int(value)


def _check_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _fraction(value):
    return 0 < value <= 1


def _penalty(value):
    return math.isfinite(value) and value >= 0


def _check_grid(values, name, accept, wanted):
    """A non-empty grid of distinct real values, each accepted by ``accept``."""
    grid = tuple(values)
    valid = grid and all(is_real(v) for v in grid)
    grid = tuple(float(v) for v in grid) if valid else grid
    if not valid or len(set(grid)) != len(grid) or not all(map(accept, grid)):
        raise ValueError(
            f"{name} must be non-empty, distinct and {wanted}, not {list(values)}"
        )
    return grid


class WorkflowSettings:
    DEFAULTS: dict[str, object] = {
        "fmriprep_dir": None,
        "output_dir": None,
        "space": "fsLR-91k",
        "modulators": None,
        "hrf_method": "cv",
        "reml_rho": None,
        "reml_reference_hrf_id": None,
        "hrf_library": "default",
        "hrf_n_samples": 512,
        "hrf_seed": 0,
        "hrf_selection_rt": True,
        "ridge_mode": "fractional_cv",
        "ridge_fractions": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        "ridge_alphas": (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
        "ridge_percentile": 90.0,
        "ridge_alpha": 0.1,
        "encoding_mode": "within_run",
        "stages": frozenset(STAGES),
        "surface_maps": True,
        "surface_meshes": None,
        "n_jobs": 4,
        "block_size": 4096,
        "max_grayordinates": None,
        "existing_results": "error",
    }

    _FIELD_NAMES = (
        "bids_dir",
        "subject",
        "session",
        "task",
        "fmriprep_dir",
        "output_dir",
        "space",
        "modulators",
        "hrf_method",
        "reml_rho",
        "reml_reference_hrf_id",
        "hrf_library",
        "hrf_n_samples",
        "hrf_seed",
        "hrf_selection_rt",
        "ridge_mode",
        "ridge_fractions",
        "ridge_alphas",
        "ridge_percentile",
        "ridge_alpha",
        "encoding_mode",
        "stages",
        "surface_maps",
        "surface_meshes",
        "n_jobs",
        "block_size",
        "max_grayordinates",
        "existing_results",
    )

    __slots__ = tuple(f"_{name}" for name in _FIELD_NAMES)

    def __init__(
        self,
        *,
        bids_dir: Path,
        subject: str,
        session: str,
        task: str,
        fmriprep_dir: Path | None = None,
        output_dir: Path | None = None,
        space: str = "fsLR-91k",
        modulators: tuple[Modulator, ...] | None = None,
        hrf_method: str = "cv",
        reml_rho: float | None = None,
        reml_reference_hrf_id: int | None = None,
        hrf_library: str = "default",
        hrf_n_samples: int = 512,
        hrf_seed: int = 0,
        hrf_selection_rt: bool = True,
        ridge_mode: str = "fractional_cv",
        ridge_fractions: tuple[float, ...] = (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ),
        ridge_alphas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
        ridge_percentile: float = 90.0,
        ridge_alpha: float = 0.1,
        encoding_mode: str = "within_run",
        stages: frozenset[str] = frozenset(STAGES),
        surface_maps: bool = True,
        surface_meshes: Mapping[str, Path] | None = None,
        n_jobs: int = 4,
        block_size: int = 4096,
        max_grayordinates: int | None = None,
        existing_results: str = "error",
    ) -> None:
        self._bids_dir, self._fmriprep_dir, self._space = self._validate_paths(
            bids_dir, fmriprep_dir, space
        )
        self._subject = _check_label(subject, "subject", "sub")
        self._session = _check_label(session, "session", "ses")
        self._task = _check_label(task, "task", "")
        self._modulators = self._validate_modulators(modulators)
        self._hrf_library = _check_choice(hrf_library, "hrf_library", HRF_LIBRARIES)
        self._hrf_n_samples = self._validate_hrf_n_samples(hrf_n_samples)
        if not is_integer(hrf_seed) or hrf_seed < 0:
            raise ValueError("hrf_seed must be a nonnegative integer")
        self._hrf_seed = hrf_seed
        _check_bool(hrf_selection_rt, "hrf_selection_rt")
        self._hrf_selection_rt = hrf_selection_rt
        self._hrf_method, self._reml_rho, self._reml_reference_hrf_id = (
            self._validate_reml(hrf_method, reml_rho, reml_reference_hrf_id)
        )
        (
            self._ridge_mode,
            self._ridge_fractions,
            self._ridge_alphas,
            self._ridge_percentile,
            self._ridge_alpha,
            self._encoding_mode,
        ) = self._validate_ridge(
            ridge_mode,
            ridge_fractions,
            ridge_alphas,
            ridge_percentile,
            ridge_alpha,
            encoding_mode,
        )
        self._stages = self._validate_stages(stages)
        self._surface_maps, self._surface_meshes = self._validate_surfaces(
            surface_maps, surface_meshes
        )
        self._n_jobs = _check_positive_int(n_jobs, "n_jobs")
        self._block_size = _check_positive_int(block_size, "block_size")
        self._max_grayordinates = _check_positive_int(
            max_grayordinates, "max_grayordinates", optional=True
        )
        self._existing_results = _check_choice(
            existing_results, "existing_results", EXISTING_RESULTS
        )
        out = output_dir
        if out is None:
            out = self._bids_dir / "derivatives" / self.output_name()
        self._output_dir = Path(out).expanduser()
        self._check_output_overlap()

    @property
    def bids_dir(self) -> Path:
        return self._bids_dir

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def session(self) -> str:
        return self._session

    @property
    def task(self) -> str:
        return self._task

    @property
    def fmriprep_dir(self) -> Path:
        return self._fmriprep_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def space(self) -> str:
        return self._space

    @property
    def modulators(self) -> tuple[Modulator, ...] | None:
        return self._modulators

    @property
    def hrf_method(self) -> str:
        return self._hrf_method

    @property
    def reml_rho(self) -> float | None:
        return self._reml_rho

    @property
    def reml_reference_hrf_id(self) -> int | None:
        return self._reml_reference_hrf_id

    @property
    def hrf_library(self) -> str:
        return self._hrf_library

    @property
    def hrf_n_samples(self) -> int:
        return self._hrf_n_samples

    @property
    def hrf_seed(self) -> int:
        return self._hrf_seed

    @property
    def hrf_selection_rt(self) -> bool:
        return self._hrf_selection_rt

    @property
    def ridge_mode(self) -> str:
        return self._ridge_mode

    @property
    def ridge_fractions(self) -> tuple[float, ...]:
        return self._ridge_fractions

    @property
    def ridge_alphas(self) -> tuple[float, ...]:
        return self._ridge_alphas

    @property
    def ridge_percentile(self) -> float:
        return self._ridge_percentile

    @property
    def ridge_alpha(self) -> float:
        return self._ridge_alpha

    @property
    def encoding_mode(self) -> str:
        return self._encoding_mode

    @property
    def stages(self) -> frozenset[str]:
        return self._stages

    @property
    def surface_maps(self) -> bool:
        return self._surface_maps

    @property
    def surface_meshes(self) -> Mapping[str, Path] | None:
        return self._surface_meshes

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def max_grayordinates(self) -> int | None:
        return self._max_grayordinates

    @property
    def existing_results(self) -> str:
        return self._existing_results

    def replace(self, **changes: object) -> "WorkflowSettings":
        values = {name: getattr(self, name) for name in self._FIELD_NAMES}
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unexpected WorkflowSettings fields: {sorted(unknown)}")
        values.update(changes)
        return WorkflowSettings(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkflowSettings):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name) for name in self._FIELD_NAMES
        )

    def __repr__(self) -> str:
        fields_repr = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in self._FIELD_NAMES
        )
        return f"WorkflowSettings({fields_repr})"

    @staticmethod
    def _validate_paths(bids_dir, fmriprep_dir, space):
        root = Path(bids_dir).expanduser()
        if not root.is_dir():
            raise ValueError(f"bids_dir {root} is not a directory")
        prep = (
            Path(fmriprep_dir).expanduser()
            if fmriprep_dir is not None
            else resolve_fmriprep_dir(root)
        )
        return root, prep, _check_choice(space, "space", SUPPORTED_SPACES)

    @staticmethod
    def _validate_modulators(modulators):
        if modulators is None:
            return None
        mods = tuple(modulators)
        if any(not isinstance(m, Modulator) for m in mods):
            raise ValueError("modulators must be Modulator instances or None")
        for mod in mods:
            _check_unresolved(mod)
            _check_expressible(mod)
        return mods

    @staticmethod
    def _validate_hrf_n_samples(hrf_n_samples):
        n = _check_positive_int(hrf_n_samples, "hrf_n_samples")
        if n & (n - 1):
            raise ValueError("hrf_n_samples must be a power of two")
        return n

    def _validate_reml(self, hrf_method, reml_rho, reml_reference_hrf_id):
        _check_choice(hrf_method, "hrf_method", ("cv", "reml"))
        if hrf_method == "cv":
            if reml_rho is not None or reml_reference_hrf_id is not None:
                raise ValueError("REML settings require hrf_method='reml'")
            return hrf_method, None, None
        rho = 0.0 if reml_rho is None else reml_rho
        reference = 0 if reml_reference_hrf_id is None else reml_reference_hrf_id
        if not is_real(rho):
            raise ValueError("reml_rho must be a scalar correlation")
        options = RemlHrfOptions(rho=rho, reference_hrf_id=reference)
        if reference:
            options.validate(1, self.build_library())
        return hrf_method, float(rho), int(reference)

    def selection_options(self):
        """Resolved public HRF selection options used by every workflow stage."""
        options = None
        if self.hrf_method != "cv":
            assert self.reml_rho is not None
            assert self.reml_reference_hrf_id is not None
            options = RemlHrfOptions(
                rho=self.reml_rho, reference_hrf_id=self.reml_reference_hrf_id
            )
        return dict(method=self.hrf_method, reml_options=options)

    @staticmethod
    def _validate_ridge(
        ridge_mode,
        ridge_fractions,
        ridge_alphas,
        ridge_percentile,
        ridge_alpha,
        encoding_mode,
    ):
        _check_choice(ridge_mode, "ridge_mode", RIDGE_MODES)
        fractions = _check_grid(
            ridge_fractions, "ridge_fractions", _fraction, "in (0, 1]"
        )
        alphas = _check_grid(ridge_alphas, "ridge_alphas", _penalty, "finite and >= 0")
        if not is_real(ridge_percentile) or not 0 <= ridge_percentile <= 100:
            raise ValueError("ridge_percentile must lie in [0, 100]")
        fixed_bad = not is_real(ridge_alpha) or ridge_alpha <= 0
        if ridge_mode == "fixed" and fixed_bad:
            raise ValueError("ridge_alpha must be positive in fixed mode")
        _check_choice(encoding_mode, "encoding_mode", ENCODING_MODES)
        return (
            ridge_mode,
            fractions,
            alphas,
            ridge_percentile,
            ridge_alpha,
            encoding_mode,
        )

    @staticmethod
    def _validate_stages(stages):
        resolved = frozenset(stages)
        unknown = resolved - set(STAGES)
        if unknown:
            raise ValueError(
                f"stages must be drawn from {', '.join(STAGES)}; "
                f"unknown: {sorted(unknown)}"
            )
        for stage, needed in STAGE_REQUIRES.items():
            if stage in resolved and needed not in resolved:
                raise ValueError(f"stage {stage} requires stage {needed}")
        return resolved

    @staticmethod
    def _validate_surfaces(surface_maps, surface_meshes):
        _check_bool(surface_maps, "surface_maps")
        if surface_meshes is None:
            return surface_maps, None
        meshes = {k: Path(v).expanduser() for k, v in dict(surface_meshes).items()}
        if set(meshes) != {"left", "right"}:
            raise ValueError("surface_meshes must map exactly 'left' and 'right'")
        return surface_maps, meshes

    def _check_output_overlap(self):
        """Outputs may never share a directory tree with the inputs."""
        assert self.output_dir is not None and self.fmriprep_dir is not None
        out = self.output_dir.resolve()
        prep = self.fmriprep_dir.resolve()
        if out == self.bids_dir.resolve() or _nested(out, prep):
            raise ValueError(
                f"output_dir {self.output_dir} must not be bids_dir, nor be, "
                f"contain, or lie inside the fMRIPrep directory {prep}"
            )

    def output_name(self):
        library = self.hrf_library
        if library in ("default", "sobol"):
            library = f"{library}{self.hrf_n_samples}s{self.hrf_seed}"
        ridge = self.ridge_mode.replace("_", "")
        if self.ridge_mode == "fixed":
            ridge = f"fixed{self.ridge_alpha:g}"
        name = f"boldtailor_hrf-{library}_ridge-{ridge}"
        if self.hrf_method == "reml":
            name += (
                f"_method-reml_rho-{self.reml_rho!r}_ref-{self.reml_reference_hrf_id}"
            )
        return name

    @property
    def stem(self):
        return f"{self.subject}/{self.session}/func/{self.subject}_{self.session}_task-{self.task}"

    @property
    def space_entity(self):
        return SPACE_ENTITIES[self.space]

    def build_library(self):
        if self.hrf_library == "default":
            return default_hrf_library(self.hrf_n_samples, seed=self.hrf_seed)
        if self.hrf_library == "sobol":
            return sobol_hrf_library(self.hrf_n_samples, seed=self.hrf_seed)
        if self.hrf_library == "expanded":
            return expanded_hrf_library()
        return HrfLibrary.from_parameters(
            [], origin={"kind": "canonical_only", "n_candidates": 0}
        )

    def to_dict(self):
        return {
            name: _to_plain(name, getattr(self, name)) for name in self._FIELD_NAMES
        }

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if values.get("modulators") is not None:
            values["modulators"] = tuple(Modulator(**m) for m in values["modulators"])
        if "stages" in values:
            values["stages"] = frozenset(values["stages"])
        for name in ("ridge_fractions", "ridge_alphas"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


def _to_plain(name, value):
    if isinstance(value, Path):
        return str(value)
    if name == "modulators" and value is not None:
        return [m.to_dict() for m in value]
    if name == "stages":
        assert value is not None  # Settings resolves stages during construction.
        return [s for s in STAGES if s in value]
    if name == "surface_meshes" and value is not None:
        return {k: str(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value
