"""Dense reference constructions for REML tests; imports nothing from boldtailor."""

import numpy as np


def ar1_correlation(n: int, rho: float) -> np.ndarray:
    """Dense AR(1) correlation matrix with entries rho ** |i - j|."""
    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return float(rho) ** lags
