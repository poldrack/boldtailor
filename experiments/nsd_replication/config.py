"""Experiment configuration loaded from a TOML file."""

from dataclasses import dataclass, fields, replace
import math
from pathlib import Path
import re
import tomllib


def _is_label(value, entity):
    return isinstance(value, str) and re.fullmatch(rf"{entity}-[A-Za-z0-9]+", value)


def _labels(name, values, entity):
    values = tuple(values)
    if (
        not values
        or len(set(values)) != len(values)
        or not all(_is_label(v, entity) for v in values)
    ):
        raise ValueError(f"{name} must be distinct {entity}-<label> values: {values}")
    return values


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    bids_dir: Path
    output_dir: Path
    freesurfer_dir: Path
    subjects: tuple[str, ...]
    sessions: tuple[str, ...]
    nsdpreproc_dir: Path | None = None
    released_dir: Path | None = None
    alignment_floor: float | None = None
    task: str = "nsdcore"
    image_column: str = "73k_id"
    n_jobs: int = 4
    block_size: int = 4096
    onset_offset: float = 0.0
    # every session in the TOML file; ``sessions`` may be a subset of them
    configured_sessions: tuple[str, ...] | None = None

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "subjects", _labels("subjects", self.subjects, "sub"))
        set_(self, "sessions", _labels("sessions", self.sessions, "ses"))
        set_(self, "configured_sessions", self._configured())
        for name in ("n_jobs", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.onset_offset):
            raise ValueError("onset_offset must be finite")
        if self.nsdpreproc_dir is None:
            set_(self, "nsdpreproc_dir", Path(self.bids_dir) / "derivatives/nsdpreproc")

    def _configured(self):
        if self.configured_sessions is None:
            return self.sessions
        configured = _labels("configured_sessions", self.configured_sessions, "ses")
        if not set(self.sessions) <= set(configured):
            raise ValueError(f"sessions must be among configured_sessions {configured}")
        return configured


def load_config(path) -> ExperimentConfig:
    values = tomllib.loads(Path(path).read_text())
    if "ppdata_dir" in values:
        raise ValueError(
            "ppdata_dir is no longer read; set nsdpreproc_dir to the BIDS "
            "derivatives/nsdpreproc directory (defaults to <bids_dir>/derivatives/nsdpreproc)"
        )
    known = {f.name for f in fields(ExperimentConfig)} - {"configured_sessions"}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {unknown}")
    paths = {k: Path(v) for k, v in values.items() if k.endswith("_dir")}
    return ExperimentConfig(**{**values, **paths})


def select_sessions(config, sessions):
    """``config`` narrowed to ``sessions`` (kept in config order); all if None."""
    if not sessions:
        return config
    _labels("sessions", sessions, "ses")
    unknown = [s for s in sessions if s not in config.configured_sessions]
    if unknown:
        raise ValueError(f"sessions {unknown} are not in the config")
    chosen = tuple(s for s in config.configured_sessions if s in sessions)
    return replace(config, sessions=chosen)


def session_summary(config):
    """One line naming the sessions in use, for logs and figure footers."""
    n, total = len(config.sessions), len(config.configured_sessions)
    return f"sessions: {', '.join(config.sessions)} ({n} of {total} in config)"
