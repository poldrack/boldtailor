"""Two-component Gaussian-mixture tail threshold (GLMsingle's findtailthreshold).

A two-component ``sklearn.mixture.GaussianMixture`` is fit to the finite
values of any 1-D statistic. The threshold is the rightmost point, on an even
grid over the data range, where the lower-mean component's posterior falls
through 0.5 (GLMsingle's "50/50 posterior" rule). Values are z-scored before
fitting so the rule is shift- and scale-equivariant and scikit-learn's fixed
covariance regularization is negligible; reported means, standard
deviations, and the threshold are in the original units.
"""

from dataclasses import asdict, dataclass

import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture

SETTINGS = dict(n_components=2, random_state=0, n_init=3, tol=1e-10, max_iter=1000)
GRID_POINTS = 10001
RULE = "rightmost 50/50 posterior of the lower-mean component"


@dataclass(frozen=True, kw_only=True)
class MixtureThreshold:
    """Threshold and fitted components, ordered by ascending mean."""

    threshold: float
    means: tuple[float, float]
    sds: tuple[float, float]
    weights: tuple[float, float]
    n_values: int

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
            standardized=True,
            **values,
        )


def _finite_values(values) -> np.ndarray:
    message = "values must be a real vector"
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if array.ndim != 1:
        raise ValueError(message)
    finite = array[np.isfinite(array)]
    if len(np.unique(finite)) < 2:
        raise ValueError(
            "a two-component mixture needs at least 2 distinct finite values; "
            f"got {len(np.unique(finite))}"
        )
    return finite


def _crossing(model, grid) -> int:
    """Index of the last grid point whose lower posterior is >= 0.5 before < 0.5."""
    lower = int(np.argmin(model.means_.ravel()))
    posterior = model.predict_proba(grid[:, None])[:, lower]
    falls = np.flatnonzero((posterior[:-1] >= 0.5) & (posterior[1:] < 0.5))
    if not len(falls):
        raise ValueError(
            "the lower-mean component's posterior never falls through 0.5 "
            "within the data range"
        )
    return int(falls[-1])


def _tupled(values, order):
    return tuple(float(v) for v in np.asarray(values).ravel()[order])


def mixture_threshold(values) -> MixtureThreshold:
    """GLMsingle-style tail threshold of the finite entries of ``values``.

    Raises ``ValueError`` for fewer than two distinct finite values or when
    the posterior never crosses 0.5; there is no fallback threshold.
    """
    finite = _finite_values(values)
    center, scale = finite.mean(), finite.std()
    z = (finite - center) / scale
    model = GaussianMixture(**SETTINGS).fit(z[:, None])
    grid = np.linspace(z.min(), z.max(), GRID_POINTS)
    index = _crossing(model, grid)
    order = np.argsort(model.means_.ravel())
    return MixtureThreshold(
        threshold=float(center + scale * grid[index]),
        means=tuple(float(center + scale * m) for m in _tupled(model.means_, order)),
        sds=tuple(
            float(scale * np.sqrt(v)) for v in _tupled(model.covariances_, order)
        ),
        weights=_tupled(model.weights_, order),
        n_values=len(finite),
    )
