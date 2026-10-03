from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from nilearn.glm import compute_contrast
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.glm.first_level import run_glm

from boldtailor.model import ContrastValue
from boldtailor.results import contrast_result, mask_contrast


@dataclass(frozen=True)
class RunFit:
    contrasts: Mapping[str, object]
    r2: np.ndarray
    residual_sum: np.ndarray
    total_sum: np.ndarray


@dataclass(frozen=True)
class ConventionalFit:
    run_fits: tuple[RunFit, ...]
    contrasts: Mapping[str, object]
    aggregate_r2: np.ndarray


@dataclass(frozen=True)
class _GLMFit:
    labels: np.ndarray
    regression_results: dict
    residual_sum: np.ndarray
    total_sum: np.ndarray
    basis: np.ndarray | None


def fit_designs(
    signals: Sequence[np.ndarray],
    designs: Sequence[pd.DataFrame],
    contrasts: Mapping[str, ContrastValue],
    noise_model: str,
) -> ConventionalFit:
    resolved = _preflight_contrasts(designs, contrasts)
    _validate_designs(designs)
    run_fits = tuple(
        _fit_run(signal, design, run_contrasts, noise_model)
        for signal, design, run_contrasts in zip(
            signals,
            designs,
            resolved,
            strict=True,
        )
    )
    combined = _combine_contrasts(run_fits, tuple(contrasts))
    undefined = np.any([run.total_sum <= 0 for run in run_fits], axis=0)
    combined = {
        name: mask_contrast(value, undefined) for name, value in combined.items()
    }
    residual_sum = np.sum([run.residual_sum for run in run_fits], axis=0)
    total_sum = np.sum([run.total_sum for run in run_fits], axis=0)
    return ConventionalFit(run_fits, combined, _r2_from_sums(residual_sum, total_sum))


def fit_r2_designs(
    signals: Sequence[np.ndarray],
    designs: Sequence[pd.DataFrame],
    noise_model: str,
) -> np.ndarray:
    _validate_designs(designs)
    fits = tuple(
        _fit_glm(signal, design.to_numpy(), noise_model)
        for signal, design in zip(signals, designs, strict=True)
    )
    residual_sum = np.sum([fit.residual_sum for fit in fits], axis=0)
    total_sum = np.sum([fit.total_sum for fit in fits], axis=0)
    return _r2_from_sums(residual_sum, total_sum)


def _fit_run(
    signals: np.ndarray,
    design: pd.DataFrame,
    contrasts: Mapping[str, np.ndarray],
    noise_model: str,
) -> RunFit:
    matrix = design.to_numpy()
    glm_fit = _fit_glm(signals, matrix, noise_model)
    results = {
        name: _nilearn_t_contrast(
            glm_fit.labels,
            glm_fit.regression_results,
            _in_fitted_basis(vector, glm_fit.basis),
        )
        for name, vector in contrasts.items()
    }
    return RunFit(
        contrasts=results,
        r2=_r2_from_sums(glm_fit.residual_sum, glm_fit.total_sum),
        residual_sum=glm_fit.residual_sum,
        total_sum=glm_fit.total_sum,
    )


def _fit_glm(
    signals: np.ndarray,
    design: np.ndarray,
    noise_model: str,
) -> _GLMFit:
    basis = _full_rank_basis(design)
    fitted = design if basis is None else design @ basis
    labels, regression_results = run_glm(signals, fitted, noise_model=noise_model)
    prediction = _prediction(labels, regression_results, fitted, signals.shape)
    residual_sum, total_sum = _sums_of_squares(signals, prediction)
    return _GLMFit(labels, regression_results, residual_sum, total_sum, basis)


def _full_rank_basis(design: np.ndarray) -> np.ndarray | None:
    """Orthonormal row-space basis when the design is rank deficient, else None.

    Nilearn divides the dispersion by ``n - columns`` but reports
    ``n - rank`` degrees of freedom, so rank-deficient designs are fitted as
    ``design @ basis`` instead.
    """
    _, singular, vt = np.linalg.svd(design, full_matrices=False)
    tolerance = singular[0] * max(design.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    if rank == design.shape[1]:
        return None
    return vt[:rank].T


def _in_fitted_basis(vector: np.ndarray, basis: np.ndarray | None) -> np.ndarray:
    """Map an estimable contrast onto the columns actually fitted."""
    return vector if basis is None else basis.T @ vector


def _combine_contrasts(
    run_fits: tuple[RunFit, ...],
    names: tuple[str, ...],
) -> dict[str, object]:
    return {
        name: contrast_result(_fixed_effects([run.contrasts[name] for run in run_fits]))
        for name in names
    }


def _fixed_effects(contrasts: list[object]) -> object:
    combined = contrasts[0]
    for contrast in contrasts[1:]:
        combined = combined + contrast
    return (1.0 / len(contrasts)) * combined


def _preflight_contrasts(
    designs: Sequence[pd.DataFrame],
    contrasts: Mapping[str, ContrastValue],
) -> tuple[Mapping[str, np.ndarray], ...]:
    return tuple(
        _resolve_run_contrasts(design, contrasts, run)
        for run, design in enumerate(designs)
    )


def _resolve_run_contrasts(
    design: pd.DataFrame,
    contrasts: Mapping[str, ContrastValue],
    run: int,
) -> dict[str, np.ndarray]:
    matrix = design.to_numpy()
    return {
        name: _resolve_contrast(value, design.columns, matrix, name, run)
        for name, value in contrasts.items()
    }


def _resolve_contrast(
    value: ContrastValue,
    columns: pd.Index,
    design: np.ndarray,
    name: str,
    run: int,
) -> np.ndarray:
    vector = _contrast_vector(value, columns, name, run)
    if not np.any(vector):
        raise ValueError(f"run {run} contrast {name!r} resolves to all zeros")
    try:
        _validate_estimable(vector, design, name, run)
    except ValueError:
        _warn_if_rank_deficient(design, run)
        raise
    return vector


def _nilearn_t_contrast(
    labels: np.ndarray,
    regression_results: dict,
    vector: np.ndarray,
) -> object:
    # Constant features divide 0/0 here; fit_designs masks them to NaN afterwards.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^divide by zero encountered in divide$",
            category=RuntimeWarning,
            module=r"^nilearn\.glm\._utils$",
        )
        return compute_contrast(
            labels,
            regression_results,
            vector,
            stat_type="t",
        )


def _contrast_vector(
    value: ContrastValue,
    columns: pd.Index,
    name: str,
    run: int,
) -> np.ndarray:
    if isinstance(value, str):
        try:
            return expression_to_contrast_vector(value, columns)
        except (KeyError, NameError, SyntaxError, TypeError, ValueError) as error:
            raise ValueError(
                f"run {run} contrast {name!r} is invalid: {error}"
            ) from error
    return _weight_vector(value, columns, name, run)


def _weight_vector(
    weights: Mapping[str, float],
    columns: pd.Index,
    name: str,
    run: int,
) -> np.ndarray:
    vector = np.zeros(len(columns), dtype=float)
    positions = {column: index for index, column in enumerate(columns)}
    for regressor, weight in weights.items():
        if weight == 0.0:
            continue
        if regressor not in positions:
            raise ValueError(
                f"run {run} contrast {name!r} references missing "
                f"regressor {regressor!r}"
            )
        vector[positions[regressor]] = weight
    return vector


def _validate_estimable(
    vector: np.ndarray,
    design: np.ndarray,
    name: str,
    run: int,
) -> None:
    projection = vector @ np.linalg.pinv(design) @ design
    if not np.allclose(vector, projection, rtol=1e-7, atol=1e-9):
        raise ValueError(f"run {run} contrast {name!r} is not estimable")


def _warn_if_rank_deficient(design: np.ndarray, run: int) -> None:
    rank = np.linalg.matrix_rank(design)
    if rank < design.shape[1]:
        warnings.warn(
            f"run {run} design rank is {rank} for {design.shape[1]} columns",
            UserWarning,
            stacklevel=2,
        )


def _validate_designs(designs: Sequence[pd.DataFrame]) -> None:
    for run, design in enumerate(designs):
        matrix = design.to_numpy()
        _warn_if_rank_deficient(matrix, run)
        _validate_residual_dof(matrix, run)


def _validate_residual_dof(design: np.ndarray, run: int) -> None:
    residual_dof = design.shape[0] - np.linalg.matrix_rank(design)
    if residual_dof <= 0:
        raise ValueError(
            f"run {run} has residual degrees of freedom {residual_dof}; "
            "contrast inference requires a positive value"
        )


def _prediction(
    labels: np.ndarray,
    regression_results: dict,
    design: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    prediction = np.empty(shape, dtype=float)
    for label, result in regression_results.items():
        prediction[:, labels == label] = design @ result.theta
    return prediction


def _sums_of_squares(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual_sum = np.sum((observed - predicted) ** 2, axis=0)
    total_sum = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
    constant = np.ptp(observed, axis=0) == 0
    residual_sum[constant] = 0.0
    total_sum[constant] = 0.0
    return residual_sum, total_sum


def _r2_from_sums(
    residual_sum: np.ndarray,
    total_sum: np.ndarray,
) -> np.ndarray:
    ratio = np.full(residual_sum.shape, np.nan, dtype=float)
    np.divide(residual_sum, total_sum, out=ratio, where=total_sum > 0)
    return 1.0 - ratio
