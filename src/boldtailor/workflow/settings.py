"""Resolved, validated settings for one workflow run; the CLI, stages, writer and metadata share it."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from boldtailor._scalars import is_integer, is_real
from boldtailor.hrf_library import (
    HrfLibrary,
    default_hrf_library,
    expanded_hrf_library,
    sobol_hrf_library,
)
from boldtailor.model import Modulator

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


def parse_modulator(text):
    parts = str(text).split(":")
    if not parts[0] or len(parts) > 2 or (len(parts) == 2 and parts[1] != "indicator"):
        raise ValueError(f"modulator must be COLUMN or COLUMN:indicator, not {text!r}")
    return Modulator(parts[0], missing="indicator" if len(parts) == 2 else "error")


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


@dataclass(frozen=True, kw_only=True)
class WorkflowSettings:
    bids_dir: Path
    subject: str
    session: str
    task: str
    fmriprep_dir: Path | None = None
    output_dir: Path | None = None
    space: str = "fsLR-91k"
    modulators: tuple[Modulator, ...] | None = None
    hrf_library: str = "default"
    hrf_n_samples: int = 512
    hrf_seed: int = 0
    hrf_selection_rt: bool = True
    ridge_mode: str = "fractional_cv"
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
    )
    ridge_alphas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    ridge_percentile: float = 90.0
    ridge_alpha: float = 0.1
    encoding_mode: str = "within_run"
    stages: frozenset[str] = field(default_factory=lambda: frozenset(STAGES))
    surface_maps: bool = True
    surface_meshes: Mapping[str, Path] | None = None
    n_jobs: int = 4
    block_size: int = 4096
    max_grayordinates: int | None = None
    existing_results: str = "error"

    def __post_init__(self):
        self._validate_paths()
        self._validate_labels()
        self._validate_modulators()
        self._validate_hrf()
        self._validate_ridge()
        self._validate_stages()
        self._validate_surfaces()
        self._validate_execution()
        self._resolve_output_dir()
        self._check_output_overlap()

    def _set(self, name, value):
        object.__setattr__(self, name, value)

    def _validate_paths(self):
        root = Path(self.bids_dir).expanduser()
        if not root.is_dir():
            raise ValueError(f"bids_dir {root} is not a directory")
        self._set("bids_dir", root)
        prep = self.fmriprep_dir
        self._set(
            "fmriprep_dir",
            Path(prep).expanduser() if prep is not None else resolve_fmriprep_dir(root),
        )
        self._set("space", _check_choice(self.space, "space", SUPPORTED_SPACES))

    def _validate_labels(self):
        self._set("subject", _check_label(self.subject, "subject", "sub"))
        self._set("session", _check_label(self.session, "session", "ses"))
        self._set("task", _check_label(self.task, "task", ""))

    def _validate_modulators(self):
        if self.modulators is None:
            return
        mods = tuple(self.modulators)
        if any(not isinstance(m, Modulator) for m in mods):
            raise ValueError("modulators must be Modulator instances or None")
        self._set("modulators", mods)

    def _validate_hrf(self):
        _check_choice(self.hrf_library, "hrf_library", HRF_LIBRARIES)
        n = _check_positive_int(self.hrf_n_samples, "hrf_n_samples")
        if n & (n - 1):
            raise ValueError("hrf_n_samples must be a power of two")
        if not is_integer(self.hrf_seed) or self.hrf_seed < 0:
            raise ValueError("hrf_seed must be a nonnegative integer")
        _check_bool(self.hrf_selection_rt, "hrf_selection_rt")

    def _validate_ridge(self):
        _check_choice(self.ridge_mode, "ridge_mode", RIDGE_MODES)
        self._set(
            "ridge_fractions",
            _check_grid(
                self.ridge_fractions, "ridge_fractions", _fraction, "in (0, 1]"
            ),
        )
        self._set(
            "ridge_alphas",
            _check_grid(self.ridge_alphas, "ridge_alphas", _penalty, "finite and >= 0"),
        )
        if not is_real(self.ridge_percentile) or not 0 <= self.ridge_percentile <= 100:
            raise ValueError("ridge_percentile must lie in [0, 100]")
        fixed_bad = not is_real(self.ridge_alpha) or self.ridge_alpha <= 0
        if self.ridge_mode == "fixed" and fixed_bad:
            raise ValueError("ridge_alpha must be positive in fixed mode")
        _check_choice(self.encoding_mode, "encoding_mode", ENCODING_MODES)

    def _validate_stages(self):
        stages = frozenset(self.stages)
        unknown = stages - set(STAGES)
        if unknown:
            raise ValueError(
                f"stages must be drawn from {', '.join(STAGES)}; "
                f"unknown: {sorted(unknown)}"
            )
        for stage, needed in STAGE_REQUIRES.items():
            if stage in stages and needed not in stages:
                raise ValueError(f"stage {stage} requires stage {needed}")
        self._set("stages", stages)

    def _validate_surfaces(self):
        _check_bool(self.surface_maps, "surface_maps")
        if self.surface_meshes is None:
            return
        meshes = {k: Path(v).expanduser() for k, v in dict(self.surface_meshes).items()}
        if set(meshes) != {"left", "right"}:
            raise ValueError("surface_meshes must map exactly 'left' and 'right'")
        self._set("surface_meshes", meshes)

    def _validate_execution(self):
        self._set("n_jobs", _check_positive_int(self.n_jobs, "n_jobs"))
        self._set("block_size", _check_positive_int(self.block_size, "block_size"))
        self._set(
            "max_grayordinates",
            _check_positive_int(
                self.max_grayordinates, "max_grayordinates", optional=True
            ),
        )
        _check_choice(self.existing_results, "existing_results", EXISTING_RESULTS)

    def _resolve_output_dir(self):
        out = self.output_dir
        if out is None:
            out = self.bids_dir / "derivatives" / self.output_name()
        self._set("output_dir", Path(out).expanduser())

    def _check_output_overlap(self):
        """Outputs may never share a directory tree with the inputs."""
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
        return f"boldtailor_hrf-{library}_ridge-{ridge}"

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
            item.name: _to_plain(item.name, getattr(self, item.name))
            for item in fields(self)
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
        return [s for s in STAGES if s in value]
    if name == "surface_meshes" and value is not None:
        return {k: str(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value
