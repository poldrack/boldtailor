"""Deterministic, peak-normalized HRFs; candidate 0 is Nilearn's SPM shape."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
import json
from types import MappingProxyType

import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from scipy.stats import gamma, qmc

from boldtailor._arrays import readonly_array
from boldtailor._constants import OVERSAMPLING
from boldtailor._scalars import is_integer

_INTEGRAL_RATIO_TOLERANCE = 1e-12

PARAMETER_NAMES = (
    "response_delay",
    "undershoot_delay",
    "response_dispersion",
    "undershoot_dispersion",
    "response_undershoot_ratio",
    "onset_delay",
    "duration",
)
CANONICAL_PARAMETERS = (6.0, 16.0, 1.0, 1.0, 6.0, 0.0, 32.0)
TIMING_NAMES = (
    "response_peak",
    "response_sd",
    "undershoot_peak",
    "undershoot_sd",
    "undershoot_depth",
    "onset",
    "duration",
)
TIMING_BOUNDS = MappingProxyType(
    {
        "response_peak": (3.5, 7.5),
        "response_sd": (1.0, 3.0),
        "undershoot_peak": (10.0, 18.0),
        "undershoot_sd": (2.5, 6.0),
        "undershoot_depth": (0.05, 0.5),
    }
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
    if not is_integer(oversampling) or oversampling < 1:
        raise ValueError("oversampling must be a positive integer")
    return float(tr) / int(oversampling)


def _peak_normalized(values):
    """Scale a sampled kernel to unit peak so betas are peak BOLD responses."""
    if not np.isfinite(values).all() or np.max(values) <= 0:
        raise ValueError("HRF kernel must be finite with a positive peak")
    return readonly_array(values / np.max(values))


@dataclass(frozen=True)
class HrfCandidate:
    """Stable library index and identified kernel, sampled at TR/oversampling.

    Every kernel is scaled to a peak of one, including canonical SPM.
    """

    id: int
    kind: str
    parameters: tuple[float, ...]

    def __post_init__(self):
        if self.kind not in ("spm", "double_gamma"):
            raise ValueError("unknown HRF kind")
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        if self.kind == "spm" and self.parameters != CANONICAL_PARAMETERS:
            raise ValueError("SPM candidate parameters must match the canonical kernel")

    def kernel(self, tr, oversampling=OVERSAMPLING):
        dt = _sampling(tr, oversampling)
        if self.kind == "spm":
            return _peak_normalized(spm_hrf(tr, oversampling))
        a, b, c, d, ratio, onset, duration = self.parameters
        # TR inferred from frame differences carries roundoff; do not add an
        # extra tail sample when duration/dt is numerically an integer.
        ratio_samples = duration / dt
        nearest = round(ratio_samples)
        if abs(ratio_samples - nearest) <= _INTEGRAL_RATIO_TOLERANCE * max(
            1, ratio_samples
        ):
            ratio_samples = nearest
        times = np.arange(int(np.ceil(ratio_samples))) * dt - onset
        values = (
            gamma.pdf(times, a / c, scale=c) - gamma.pdf(times, b / d, scale=d) / ratio
        )
        return _peak_normalized(values)


@dataclass(frozen=True)
class HrfLibrary:
    """Owned candidate order and 0.1-second curves; table access returns a copy."""

    candidates: tuple[HrfCandidate, ...]
    curves: np.ndarray = field(init=False, repr=False, compare=False)
    times: np.ndarray = field(init=False, repr=False, compare=False)
    fingerprint: str = field(init=False)
    origin: Mapping[str, object] = field(default_factory=dict, compare=False)

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
        object.__setattr__(self, "curves", readonly_array(curves))
        object.__setattr__(
            self, "times", readonly_array(np.arange(curves.shape[1]) * 0.1)
        )
        object.__setattr__(self, "fingerprint", digest)
        object.__setattr__(self, "origin", MappingProxyType(dict(self.origin)))

    @classmethod
    def from_parameters(cls, parameters, origin=None):
        rows = [_parameters(row) for row in parameters]
        if len(rows) != len(set(rows)):
            raise ValueError("duplicate HRF parameter rows")
        canonical = HrfCandidate(0, "spm", CANONICAL_PARAMETERS)
        custom = tuple(
            HrfCandidate(i, "double_gamma", row)
            for i, row in enumerate(sorted(rows), 1)
        )
        origin = (
            origin
            if origin is not None
            else {"kind": "explicit", "n_candidates": len(rows)}
        )
        return cls((canonical, *custom), origin=origin)

    @property
    def parameter_bounds(self):
        """Per-parameter ``low``/``high`` over the six sampled box parameters.

        Computed over custom candidates; canonical SPM is used only when there
        are none. A parameter shared by all candidates has zero width.
        """
        rows = self._custom_parameters()
        return pd.DataFrame(
            {"low": rows.min(axis=0), "high": rows.max(axis=0)},
            index=list(PARAMETER_NAMES[:6]),
        )

    def _custom_parameters(self):
        rows = [c.parameters[:6] for c in self.candidates if c.kind != "spm"]
        return np.array(rows or [CANONICAL_PARAMETERS[:6]], dtype=float)

    @property
    def informative_parameters(self) -> tuple[str, ...]:
        """Sampled parameters with at least three distinct custom values.

        Constant and two-level grid parameters put every pick at an edge, so
        they are excluded from the scalar ``at_parameter_bound`` diagnostic.
        """
        rows = self._custom_parameters()
        return tuple(
            name
            for name, column in zip(PARAMETER_NAMES[:6], rows.T, strict=True)
            if len(np.unique(column)) >= 3 and np.ptp(column) > 0
        )

    def candidate_bound_flags(self, margin=0.02):
        """Per-candidate ``(n_candidates, 6, 2)`` flags for ``[low, high]`` edges.

        True where a custom candidate's parameter lies within ``margin`` of the
        box width of that edge; canonical SPM rows are False.
        """
        bounds = self.parameter_bounds
        low, high = bounds["low"].to_numpy(), bounds["high"].to_numpy()
        tol = margin * (high - low)
        params = np.array([c.parameters[:6] for c in self.candidates], dtype=float)
        flags = np.stack([params - low <= tol, high - params <= tol], axis=-1)
        flags[[c.kind == "spm" for c in self.candidates]] = False
        return flags

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

    @property
    def timing_table(self):
        """Both parameterizations per candidate: SPM gamma values and lobe timing."""
        rows = []
        for candidate in self.candidates:
            rows.append(
                dict(
                    hrf_id=candidate.id,
                    kind=candidate.kind,
                    **dict(zip(PARAMETER_NAMES, candidate.parameters, strict=True)),
                    **dict(
                        zip(
                            TIMING_NAMES[:6],
                            timing_parameters(candidate.parameters)[:6],
                            strict=True,
                        )
                    ),
                )
            )
        return pd.DataFrame(rows)


def _lobe_timing(delay, dispersion, onset):
    """Mode and standard deviation of a gamma lobe with shape delay/dispersion."""
    return onset + delay - dispersion, float(np.sqrt(delay * dispersion))


def _lobe_parameters(peak, sd, onset):
    """Invert :func:`_lobe_timing`: delay - dispersion = peak - onset, product = sd**2."""
    mode = peak - onset
    dispersion = (-mode + np.sqrt(mode * mode + 4.0 * sd * sd)) / 2.0
    return mode + dispersion, dispersion


def _lobe_height(delay, dispersion):
    return gamma.pdf(delay - dispersion, delay / dispersion, scale=dispersion)


def timing_parameters(parameters):
    """SPM gamma parameters to lobe timing: peak, SD, undershoot peak/SD, depth, onset.

    Peak and SD are the mode and standard deviation of each gamma lobe; depth is
    the undershoot lobe's height relative to the response lobe's height. The
    response peak predicts the sampled kernel's maximum to about 0.1 s; the
    undershoot values describe the lobe, not the trough of the combined curve.
    """
    a, b, c, d, ratio, onset, duration = _parameters(parameters)
    response_peak, response_sd = _lobe_timing(a, c, onset)
    undershoot_peak, undershoot_sd = _lobe_timing(b, d, onset)
    depth = (_lobe_height(b, d) / ratio) / _lobe_height(a, c)
    return (
        response_peak,
        response_sd,
        undershoot_peak,
        undershoot_sd,
        depth,
        onset,
        duration,
    )


def spm_parameters(timing):
    """Lobe timing (see :func:`timing_parameters`) back to SPM gamma parameters."""
    values = tuple(float(v) for v in timing)
    if len(values) != 7 or not np.isfinite(values).all():
        raise ValueError("timing parameters must be seven finite numbers")
    (
        response_peak,
        response_sd,
        undershoot_peak,
        undershoot_sd,
        depth,
        onset,
        duration,
    ) = values
    if (
        min(response_sd, undershoot_sd, depth) <= 0
        or min(response_peak, undershoot_peak) <= onset
    ):
        raise ValueError(
            "timing SDs and depth must be positive and peaks must follow the onset"
        )
    a, c = _lobe_parameters(response_peak, response_sd, onset)
    b, d = _lobe_parameters(undershoot_peak, undershoot_sd, onset)
    ratio = (_lobe_height(b, d) / _lobe_height(a, c)) / depth
    return _parameters((a, b, c, d, ratio, onset, duration))


def expanded_hrf_library():
    """The original 648 double-gamma grid plus peak-scaled canonical SPM."""
    return HrfLibrary.from_parameters(
        product(
            (3, 4.5, 6),
            (10, 16),
            (0.5, 1, 1.5),
            (0.5, 1.5, 2.5),
            (2, 4, 6, 8),
            (0, 1, 2),
            (36,),
        ),
        origin={"kind": "expanded_grid"},
    )


def sobol_hrf_library(n_samples=512, *, seed=0):
    """Sample continuous HRF parameters, plus peak-scaled canonical SPM at ID zero.

    ``n_samples`` must be a positive power of two; ``seed`` must be a
    nonnegative integer. Scrambled Sobol points cover the six-dimensional
    parameter box of :func:`expanded_hrf_library`, with duration fixed at
    36 seconds. This balances parameter coverage, not waveform distances.
    Custom candidates are sorted by parameters, not Sobol sequence order.
    """
    points = _sobol_points(n_samples, seed, dimensions=6)
    parameters = qmc.scale(points, [3, 10, 0.5, 0.5, 2, 0], [6, 16, 1.5, 2.5, 8, 2])
    rows = np.column_stack([parameters, np.full(int(n_samples), 36.0)])
    origin = {"kind": "sobol", "n_samples": int(n_samples), "seed": int(seed)}
    return HrfLibrary.from_parameters(rows, origin={**origin, "duration": 36.0})


def _sobol_points(n_samples, seed, *, dimensions):
    if (
        not is_integer(n_samples)
        or n_samples < 1
        or int(n_samples) & (int(n_samples) - 1)
    ):
        raise ValueError("n_samples must be a positive integer power of two")
    if not is_integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    sampler = qmc.Sobol(d=dimensions, scramble=True, rng=int(seed))
    return sampler.random_base2(int(n_samples).bit_length() - 1)


def _timing_bounds(bounds):
    merged = dict(TIMING_BOUNDS)
    for name, values in dict(bounds or {}).items():
        if name not in merged:
            raise ValueError(f"bounds names must be timing parameters, not {name!r}")
        low, high = (float(v) for v in values)
        if not np.isfinite((low, high)).all() or low >= high:
            raise ValueError(f"bounds for {name!r} must be finite with low < high")
        merged[name] = (low, high)
    return merged


def timing_hrf_library(n_samples=512, *, seed=0, bounds=None, onset=0.0, duration=36.0):
    """Sample lobe timing with scrambled Sobol, plus peak-scaled canonical SPM at ID zero.

    The five sampled quantities are the response lobe's peak time and SD, the
    undershoot lobe's peak time and SD, and the undershoot depth relative to the
    response (see :func:`timing_parameters`); ``onset`` is fixed because onset
    and response delay trade off into the same peak time, which is what makes
    the gamma-parameter box oversample near-identical waveforms. ``bounds``
    overrides entries of :data:`TIMING_BOUNDS`. Candidates are stored with
    their SPM gamma parameters, so downstream modeling is unchanged.
    """
    limits = _timing_bounds(bounds)
    points = _sobol_points(n_samples, seed, dimensions=5)
    names = TIMING_NAMES[:5]
    sampled = qmc.scale(
        points, [limits[n][0] for n in names], [limits[n][1] for n in names]
    )
    rows = [spm_parameters((*row, float(onset), float(duration))) for row in sampled]
    origin = dict(
        kind="timing_sobol",
        n_samples=int(n_samples),
        seed=int(seed),
        onset=float(onset),
        duration=float(duration),
        bounds={name: list(limits[name]) for name in names},
    )
    return HrfLibrary.from_parameters(rows, origin=origin)
