"""Two-component Gaussian-mixture tail threshold (GLMsingle's findtailthreshold).

Follows GLMsingle's ``findtailthreshold``: a two-component
``sklearn.mixture.GaussianMixture`` (``tol=1e-10``, ``reg_covar=0``, 3
initialisations) is fit to the raw finite values of any 1-D statistic,
subsampled to at most ``MAX_VALUES`` values. Posteriors are evaluated on
500 evenly spaced points over GLMsingle's ``robustrange`` of the values,
widened to include both fitted means. The tail component is the one whose
posterior exceeds 0.5 at the right end of that grid, and the threshold is
the rightmost grid point where its posterior is at most 0.5.

Deviations, recorded in ``to_dict()``: ``random_state=0`` and a seeded
subsample make the result reproducible; convergence warnings are captured
and recorded (``converged``, ``n_iter``) instead of printed; failures raise
``ValueError`` rather than asserting.
"""

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

SETTINGS = dict(n_components=2, tol=1e-10, reg_covar=0.0, n_init=3, random_state=0)
GRID_POINTS = 500
MAX_VALUES = 1_000_000
SUBSAMPLE_SEED = 0
RULE = (
    "GLMsingle findtailthreshold: rightmost grid point where the posterior of "
    "the component dominating the grid's right end is <= 0.5"
)


@dataclass(frozen=True, kw_only=True)
class MixtureThreshold:
    """Threshold and fitted components, ordered by ascending mean.

    ``tail`` indexes (in that order) the component dominating the right end.
    """

    threshold: float
    means: tuple[float, float]
    sds: tuple[float, float]
    weights: tuple[float, float]
    tail: int
    n_values: int
    n_fitted: int
    converged: bool
    n_iter: int

    def to_dict(self) -> dict:
        values = asdict(self)
        for key in ("means", "sds", "weights"):
            values[key] = list(values[key])
        return dict(
            method="gaussian_mixture_tail_threshold",
            estimator="sklearn.mixture.GaussianMixture",
            sklearn_version=sklearn.__version__,
            rule=RULE,
            **SETTINGS,
            grid_points=GRID_POINTS,
            grid_range="robustrange widened to include both means",
            max_values=MAX_VALUES,
            subsample_seed=SUBSAMPLE_SEED,
            **values,
        )


def robust_range(values) -> tuple[float, float]:
    """GLMsingle ``robustrange``: data range with outliers trimmed recursively.

    Bounds are median +/- 5x the median-to-10th/90th percentile distance;
    values beyond them are dropped and the range recomputed. Otherwise each
    end is the absolute extreme if within 1.1x the median-to-0.1/99.9th
    percentile distance, else that percentile.
    """
    m = np.asarray(values, dtype=float)
    while True:
        p01, p10, p50, p90, p999 = np.percentile(m, [0.1, 10, 50, 90, 99.9])
        low, high = p50 - 5 * (p50 - p10), p50 + 5 * (p90 - p50)
        if p999 <= high and p01 >= low:
            top = m.max() if m.max() <= p50 + 1.1 * (p999 - p50) else p999
            bottom = m.min() if m.min() >= p50 - 1.1 * (p50 - p01) else p01
            return float(bottom), float(top)
        if p999 > high:
            m = m[~(m > high)]
        if p01 < low:
            m = m[~(m < low)]


def _finite_values(values) -> np.ndarray:
    message = "values must be a real vector"
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if array.ndim != 1:
        raise ValueError(message)
    finite = array[np.isfinite(array)]
    distinct = len(np.unique(finite))
    if distinct < 2:
        raise ValueError(
            "a two-component mixture needs at least 2 distinct finite values; "
            f"got {distinct}"
        )
    return finite


def _subsample(finite):
    if len(finite) <= MAX_VALUES:
        return finite
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    return finite[np.sort(rng.choice(len(finite), MAX_VALUES, replace=False))]


def _fit(fitted):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return GaussianMixture(**SETTINGS).fit(fitted[:, None])
    except Exception as error:  # e.g. singular covariance with reg_covar=0
        raise ValueError(f"the Gaussian mixture fit failed: {error}") from error


def _grid(finite, means):
    low, high = robust_range(finite)
    return np.linspace(min(low, means.min()), max(high, means.max()), GRID_POINTS)


def _threshold(model, grid):
    posterior = model.predict_proba(grid[:, None])
    tail = 0 if posterior[-1, 0] > 0.5 else 1
    below = np.flatnonzero(posterior[:, tail] <= 0.5)
    if not len(below):
        raise ValueError(
            "the tail component's posterior is never at or below 0.5 on the grid"
        )
    return float(grid[below[-1]]), tail


def _ordered(values, order):
    return tuple(float(v) for v in np.asarray(values).ravel()[order])


def mixture_threshold(values) -> MixtureThreshold:
    """GLMsingle-style tail threshold of the finite entries of ``values``.

    Raises ``ValueError`` for fewer than two distinct finite values, a
    failed fit, or a tail posterior that never reaches 0.5; no fallback.
    """
    finite = _finite_values(values)
    fitted = _subsample(finite)
    model = _fit(fitted)
    means = model.means_.ravel()
    threshold, tail = _threshold(model, _grid(finite, means))
    order = np.argsort(means)
    return MixtureThreshold(
        threshold=threshold,
        means=_ordered(means, order),
        sds=tuple(float(np.sqrt(v)) for v in _ordered(model.covariances_, order)),
        weights=_ordered(model.weights_, order),
        tail=int(np.flatnonzero(order == tail)[0]),
        n_values=len(finite),
        n_fitted=len(fitted),
        converged=bool(model.converged_),
        n_iter=int(model.n_iter_),
    )
