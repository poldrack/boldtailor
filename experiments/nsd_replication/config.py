"""Experiment configuration loaded from a TOML file."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
import tomllib

from boldtailor.workflow.files import bids_label


def _labels(name, values, entity):
    values = tuple(values)
    if (
        not values
        or len(set(values)) != len(values)
        or not all(bids_label(v, entity) for v in values)
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
    ppdata_dir: Path | None = None
    released_dir: Path | None = None
    alignment_floor: float | None = None
    task: str = "nsdcore"
    image_column: str = "73k_id"
    n_jobs: int = 4
    block_size: int = 4096
    onset_offset: float = 0.0

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "subjects", _labels("subjects", self.subjects, "sub"))
        set_(self, "sessions", _labels("sessions", self.sessions, "ses"))
        for name in ("n_jobs", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.onset_offset):
            raise ValueError("onset_offset must be finite")


def load_config(path) -> ExperimentConfig:
    values = tomllib.loads(Path(path).read_text())
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {unknown}")
    paths = {k: Path(v) for k, v in values.items() if k.endswith("_dir")}
    return ExperimentConfig(**{**values, **paths})
