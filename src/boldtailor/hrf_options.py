"""Owned settings for likelihood-based HRF selection."""

from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import readonly_array
from boldtailor._scalars import is_integer
from boldtailor.hrf_library import HrfLibrary


@dataclass(frozen=True)
class RemlHrfOptions:
    """Fixed noise correlation and reference HRF shared by all candidates."""

    rho: float | np.ndarray = 0.0
    reference_hrf_id: int = 0

    def __post_init__(self):
        raw = np.asarray(self.rho)
        if raw.dtype.kind not in "iuf" or raw.ndim > 1 or not raw.size:
            raise ValueError("rho must be a scalar or nonempty feature vector")
        values = np.asarray(raw, dtype=float)
        if not np.isfinite(values).all() or np.any(np.abs(values) >= 1):
            raise ValueError("rho must be finite with abs(rho) < 1")
        owned = float(values) if values.ndim == 0 else readonly_array(values)
        object.__setattr__(self, "rho", owned)
        if not is_integer(self.reference_hrf_id) or self.reference_hrf_id < 0:
            raise ValueError("reference_hrf_id must be a nonnegative integer")

    def validate(self, n_features: int, library: HrfLibrary) -> None:
        """Check settings against the data and candidate library."""
        if np.ndim(self.rho) and self.rho.shape != (n_features,):
            raise ValueError("rho must have one value per feature")
        if self.reference_hrf_id >= len(library.candidates):
            raise ValueError("reference_hrf_id is absent from the library")

    def subset(self, indices) -> "RemlHrfOptions":
        """Preserve supplied noise correlations when partitioning features."""
        rho = self.rho[indices] if np.ndim(self.rho) else self.rho
        return RemlHrfOptions(rho=rho, reference_hrf_id=self.reference_hrf_id)
