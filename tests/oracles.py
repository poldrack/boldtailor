"""Independent numerical references and run subsets for scientific tests."""

import numpy as np
from scipy.optimize import brentq

from boldtailor.data import from_arrays


def fractional_beta_oracle(x, n, y, fraction):
    """Solve augmented regression directly, not through the production SVD."""
    scale = np.ones(x.shape[1])
    matrix = np.column_stack([x / scale, n])

    def coefficients(alpha):
        penalty = np.column_stack(
            [np.sqrt(alpha) * np.eye(x.shape[1]), np.zeros((x.shape[1], n.shape[1]))]
        )
        return np.linalg.lstsq(
            np.vstack([matrix, penalty]), np.r_[y, np.zeros(x.shape[1])], rcond=None
        )[0]

    baseline = np.linalg.norm(coefficients(0)[: x.shape[1]])
    alpha = (
        0
        if fraction == 1
        else brentq(
            lambda a: np.linalg.norm(coefficients(a)[: x.shape[1]]) / baseline
            - fraction,
            0,
            1e7,
            xtol=1e-13,
        )
    )
    coef = coefficients(alpha)
    return coef[: x.shape[1]] / scale, coef[x.shape[1] :], alpha


def subset_runs(data, indices, *, signals=None, events=None):
    return from_arrays(
        [(data.signals if signals is None else signals)[i] for i in indices],
        [(data.events if events is None else events)[i] for i in indices],
        frame_times=[data.frame_times[i] for i in indices],
        confounds=[data.confounds[i] for i in indices],
        sources=[data.provenance.sources[i] for i in indices],
    )
