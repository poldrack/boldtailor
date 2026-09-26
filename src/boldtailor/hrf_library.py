"""Deterministic, sum-normalized HRFs with an exact Nilearn SPM anchor."""

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
import json
from numbers import Integral

import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from scipy.stats import gamma

from boldtailor._arrays import immutable_float_array

PARAMETER_NAMES = (
    "response_delay",
    "undershoot_delay",
    "response_dispersion",
    "undershoot_dispersion",
    "response_undershoot_ratio",
    "onset_delay",
    "duration",
)


def _parameters(values):
    try:
        values = tuple(float(v) for v in values)
    except (TypeError, ValueError) as error:
        raise ValueError("HRF parameters must be seven finite numbers") from error
    if len(values) != 7 or not np.isfinite(values).all():
        raise ValueError("HRF parameters must be seven finite numbers")
    if min(values[:5]) <= 0 or not 0 <= values[5] < values[6]:
        raise ValueError("HRF delays, dispersions, ratio and duration must be positive")
    return values


def _sampling(tr, oversampling):
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("TR must be positive and finite")
    if (
        isinstance(oversampling, (bool, np.bool_))
        or not isinstance(oversampling, Integral)
        or oversampling < 1
    ):
        raise ValueError("oversampling must be a positive integer")
    return float(tr) / int(oversampling)


@dataclass(frozen=True)
class HrfCandidate:
    """Stable library index and identified kernel, sampled at TR/oversampling."""

    id: int
    kind: str
    parameters: tuple[float, ...]

    def __post_init__(self):
        if self.kind not in ("spm", "double_gamma"):
            raise ValueError("unknown HRF kind")
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def kernel(self, tr, oversampling=50):
        dt = _sampling(tr, oversampling)
        if self.kind == "spm":
            return immutable_float_array(spm_hrf(tr, oversampling))
        a, b, c, d, ratio, onset, duration = self.parameters
        # TR inferred from frame differences carries roundoff; do not add an
        # extra tail sample when duration/dt is numerically an integer.
        ratio_samples = duration / dt
        nearest = round(ratio_samples)
        if abs(ratio_samples - nearest) <= 1e-12 * max(1, ratio_samples):
            ratio_samples = nearest
        times = np.arange(int(np.ceil(ratio_samples))) * dt - onset
        values = (
            gamma.pdf(times, a / c, scale=c) - gamma.pdf(times, b / d, scale=d) / ratio
        )
        total = values.sum()
        if (
            not np.isfinite(values).all()
            or abs(total) <= np.finfo(float).eps * np.abs(values).sum()
        ):
            raise ValueError("HRF kernel must be finite with nonzero sum")
        return immutable_float_array(values / total)


@dataclass(frozen=True)
class HrfLibrary:
    """Owned candidate order and 0.1-second curves; table access returns a copy."""

    candidates: tuple[HrfCandidate, ...]
    curves: np.ndarray = field(init=False, repr=False, compare=False)
    times: np.ndarray = field(init=False, repr=False, compare=False)
    fingerprint: str = field(init=False)

    def __post_init__(self):
        candidates = tuple(self.candidates)
        if (
            not candidates
            or candidates[0].kind != "spm"
            or any(c.id != i for i, c in enumerate(candidates))
        ):
            raise ValueError(
                "library requires canonical candidate 0 and ordered stable IDs"
            )
        kernels = [c.kernel(0.1, 1) for c in candidates]
        curves = np.zeros((len(kernels), max(map(len, kernels))))
        for row, values in zip(curves, kernels, strict=True):
            row[: len(values)] = values
        identity = [(c.id, c.kind, c.parameters) for c in candidates]
        digest = sha256(json.dumps(identity).encode() + curves.tobytes()).hexdigest()
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "curves", immutable_float_array(curves))
        object.__setattr__(
            self, "times", immutable_float_array(np.arange(curves.shape[1]) * 0.1)
        )
        object.__setattr__(self, "fingerprint", digest)

    @classmethod
    def from_parameters(cls, parameters):
        rows = [_parameters(row) for row in parameters]
        if len(rows) != len(set(rows)):
            raise ValueError("duplicate HRF parameter rows")
        canonical = HrfCandidate(0, "spm", (6.0, 16.0, 1.0, 1.0, 6.0, 0.0, 32.0))
        custom = tuple(
            HrfCandidate(i, "double_gamma", row)
            for i, row in enumerate(sorted(rows), 1)
        )
        return cls((canonical, *custom))

    @property
    def parameter_table(self):
        rows = []
        for candidate, curve in zip(self.candidates, self.curves, strict=True):
            rows.append(
                dict(
                    hrf_id=candidate.id,
                    kind=candidate.kind,
                    **dict(zip(PARAMETER_NAMES, candidate.parameters, strict=True)),
                    peak_time=float(self.times[np.argmax(curve)]),
                )
            )
        return pd.DataFrame(rows)


def expanded_hrf_library():
    """The notebook's 648 double-gamma kernels, plus the exact legacy anchor."""
    return HrfLibrary.from_parameters(
        product(
            (3, 4.5, 6),
            (10, 16),
            (0.5, 1, 1.5),
            (0.5, 1.5, 2.5),
            (2, 4, 6, 8),
            (0, 1, 2),
            (36,),
        )
    )
