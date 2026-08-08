from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from nilearn.glm import compute_contrast
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.glm.first_level import run_glm

from boldtailor.data import AnalysisData
from boldtailor.design import CompiledDesign, compile_designs
from boldtailor.model import ContrastValue, ModelSpec
from boldtailor.results import AnalysisResult, contrast_result, make_result


@dataclass(frozen=True)
class _RunFit:
    contrasts: dict[str, object]
    r2: np.ndarray
    residual_sum: np.ndarray
    total_sum: np.ndarray


def fit(data: AnalysisData, model: ModelSpec) -> AnalysisResult:
    compiled = compile_designs(data, model)
    run_fits = tuple(
        _fit_run(signals, design, model, run)
        for run, (signals, design) in enumerate(
            zip(data.signals, compiled, strict=True)
        )
    )
    combined = _combine_contrasts(run_fits, model.contrast_names)
    aggregate_r2 = _r2_from_sums(
        np.sum([run.residual_sum for run in run_fits], axis=0),
        np.sum([run.total_sum for run in run_fits], axis=0),
    )
    return make_result(
        combined,
        tuple(design.matrix for design in compiled),
        tuple(_design_provenance(design) for design in compiled),
        tuple(run.r2 for run in run_fits),
        aggregate_r2,
    )


def _fit_run(
    signals: np.ndarray,
    compiled: CompiledDesign,
    model: ModelSpec,
    run: int,
) -> _RunFit:
    design = compiled.matrix
    matrix = design.to_numpy()
    _warn_if_rank_deficient(matrix, run)
    _validate_residual_dof(matrix, run)
    labels, regression_results = run_glm(
        signals,
        matrix,
        noise_model=model.noise_model,
    )
    prediction = _prediction(labels, regression_results, matrix, signals.shape)
    residual_sum, total_sum = _sums_of_squares(signals, prediction)
    contrasts = {
        name: _compute_contrast(
            labels,
            regression_results,
            value,
            design.columns,
            matrix,
            name,
            run,
        )
        for name, value in model.contrasts.items()
    }
    return _RunFit(
        contrasts=contrasts,
        r2=_r2_from_sums(residual_sum, total_sum),
        residual_sum=residual_sum,
        total_sum=total_sum,
    )


def _combine_contrasts(
    run_fits: tuple[_RunFit, ...],
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


def _design_provenance(compiled: CompiledDesign) -> dict[str, int | float]:
    return {
        "excluded_event_count": compiled.excluded_event_count,
        "min_onset_cutoff": compiled.min_onset_cutoff,
    }


def _compute_contrast(
    labels: np.ndarray,
    regression_results: dict,
    value: ContrastValue,
    columns: pd.Index,
    design: np.ndarray,
    name: str,
    run: int,
) -> object:
    vector = _contrast_vector(value, columns, name, run)
    if not np.any(vector):
        raise ValueError(f"run {run} contrast {name!r} resolves to all zeros")
    _validate_estimable(vector, design, name, run)
    return _nilearn_t_contrast(labels, regression_results, vector)


def _nilearn_t_contrast(
    labels: np.ndarray,
    regression_results: dict,
    vector: np.ndarray,
) -> object:
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
    return residual_sum, total_sum


def _r2_from_sums(
    residual_sum: np.ndarray,
    total_sum: np.ndarray,
) -> np.ndarray:
    ratio = np.full(residual_sum.shape, np.nan, dtype=float)
    np.divide(residual_sum, total_sum, out=ratio, where=total_sum > 0)
    return 1.0 - ratio
