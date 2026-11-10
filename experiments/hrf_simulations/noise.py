"""Stationary controls and empirical-spectrum noise surrogates."""

import numpy as np
from scipy.signal import lfilter


def ar1(length, rho, variance, rng):
    if not isinstance(length, (int, np.integer)) or length < 2:
        raise ValueError("length must be an integer >= 2")
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("rho must be finite and strictly inside (-1, 1)")
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("variance must be nonnegative and finite")
    innovations = rng.normal(size=length) * np.sqrt(variance * (1 - rho * rho))
    innovations[0] = rng.normal() * np.sqrt(variance)
    return lfilter([1.0], [1.0, -rho], innovations)


def phase_surrogate(residual, rng):
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 1 or len(residual) < 2 or not np.isfinite(residual).all():
        raise ValueError("residual must be a finite vector")
    spectrum = np.fft.rfft(residual - residual.mean())
    phases = rng.uniform(-np.pi, np.pi, len(spectrum))
    generated = abs(spectrum) * np.exp(1j * phases)
    generated[0] = 0
    if len(residual) % 2 == 0:
        generated[-1] = abs(spectrum[-1]) * rng.choice([-1.0, 1.0])
    return np.fft.irfft(generated, n=len(residual))


def diagnostics(values, max_lag=10):
    x = np.asarray(values, float)
    x = x - x.mean()
    energy = x @ x
    acf = [
        float(x[k:] @ x[:-k] / energy) if energy > 0 else np.nan
        for k in range(1, min(max_lag + 1, len(x)))
    ]
    return dict(
        variance=float(np.var(x)),
        acf=acf,
        periodogram=(abs(np.fft.rfft(x)) ** 2 / len(x)).tolist(),
    )
