"""Owned settings for likelihood-based HRF selection."""

import numpy as np

from boldtailor._arrays import readonly_array
from boldtailor._scalars import is_integer
from boldtailor.hrf_library import HrfLibrary


class RemlHrfOptions:
    """Fixed noise correlation and reference HRF shared by all candidates."""

    __slots__ = ("_rho", "_reference_hrf_id")

    def __init__(
        self,
        rho: float | np.ndarray = 0.0,
        reference_hrf_id: int = 0,
    ) -> None:
        raw = np.asarray(rho)
        if raw.dtype.kind not in "iuf" or raw.ndim > 1 or not raw.size:
            raise ValueError("rho must be a scalar or nonempty feature vector")
        values = np.asarray(raw, dtype=float)
        if not np.isfinite(values).all() or np.any(np.abs(values) >= 1):
            raise ValueError("rho must be finite with abs(rho) < 1")
        if not is_integer(reference_hrf_id) or reference_hrf_id < 0:
            raise ValueError("reference_hrf_id must be a nonnegative integer")
        self._rho: float | np.ndarray = (
            float(values) if values.ndim == 0 else readonly_array(values)
        )
        self._reference_hrf_id: int = int(reference_hrf_id)

    @property
    def rho(self) -> float | np.ndarray:
        return self._rho

    @property
    def reference_hrf_id(self) -> int:
        return self._reference_hrf_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RemlHrfOptions):
            return NotImplemented
        if self._reference_hrf_id != other._reference_hrf_id:
            return False
        if isinstance(self._rho, np.ndarray) or isinstance(other._rho, np.ndarray):
            return (
                isinstance(self._rho, np.ndarray)
                and isinstance(other._rho, np.ndarray)
                and np.array_equal(self._rho, other._rho)
            )
        return self._rho == other._rho

    def __repr__(self) -> str:
        return (
            f"RemlHrfOptions(rho={self._rho!r}, "
            f"reference_hrf_id={self._reference_hrf_id!r})"
        )

    def validate(self, n_features: int, library: HrfLibrary) -> None:
        """Check settings against the data and candidate library."""
        if isinstance(self._rho, np.ndarray) and self._rho.shape != (n_features,):
            raise ValueError("rho must have one value per feature")
        if self._reference_hrf_id >= len(library.candidates):
            raise ValueError("reference_hrf_id is absent from the library")

    def subset(self, indices) -> "RemlHrfOptions":
        """Preserve supplied noise correlations when partitioning features."""
        rho = self._rho[indices] if isinstance(self._rho, np.ndarray) else self._rho
        return RemlHrfOptions(rho=rho, reference_hrf_id=self._reference_hrf_id)


def resolve_reml_options(method, options):
    """Reject ambiguous method settings instead of silently choosing another model."""
    if method not in ("cv", "reml"):
        raise ValueError("HRF method must be 'cv' or 'reml'")
    if method == "cv":
        if options is not None:
            raise ValueError("REML options require method='reml'")
        return None
    if options is None:
        return RemlHrfOptions()
    if not isinstance(options, RemlHrfOptions):
        raise ValueError("reml_options must be RemlHrfOptions")
    return options
