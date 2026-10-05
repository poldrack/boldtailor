"""GLMsingle's polynomial drift basis, used as boldtailor's confounds for ppdata."""

import math

import numpy as np
import pandas as pd


def polynomial_degree(n_volumes, tr):
    """GLMsingle default: alt_round(duration in minutes / 2), half away from zero."""
    return int(math.floor(n_volumes * tr / 60 / 2 + 0.5))


def glmsingle_polynomials(n_volumes, tr):
    degree = polynomial_degree(n_volumes, tr)
    x = np.linspace(-1, 1, n_volumes)
    basis, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, degree))
    return pd.DataFrame(basis, columns=[f"poly_{d}" for d in range(degree + 1)])
