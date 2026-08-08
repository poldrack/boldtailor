from __future__ import annotations

from collections.abc import Mapping
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


def fit(data: AnalysisData, model: ModelSpec) -> AnalysisResult:
    if data.n_runs != 1:
        raise NotImplementedError("multi-run fitting is added in Task 6")
    compiled = compile_designs(data, model)[0]
    labels, regression_results = _fit_run(data.signals[0], compiled.matrix, model)
    contrasts = _compute_contrasts(
        labels,
        regression_results,
        compiled.matrix,
        model.contrasts,
    )
    prediction = _prediction(
        labels,
        regression_results,
        data.signals[0].shape,
    )
    run_r2 = _r2(data.signals[0], prediction)
    return make_result(
        contrasts,
        (compiled.matrix,),
        (_design_provenance(compiled),),
        (run_r2,),
        run_r2,
    )


def _fit_run(
    signals: np.ndarray,
    design: pd.DataFrame,
    model: ModelSpec,
) -> tuple[np.ndarray, dict]:
    matrix = design.to_numpy()
    _warn_if_rank_deficient(matrix, 0)
    return run_glm(signals, matrix, noise_model=model.noise_model)


def _compute_contrasts(
    labels: np.ndarray,
    regression_results: dict,
    design: pd.DataFrame,
    contrasts: Mapping[str, ContrastValue],
) -> dict:
    matrix = design.to_numpy()
    return {
        name: contrast_result(
            _compute_contrast(
                labels,
                regression_results,
                value,
                design.columns,
                matrix,
                name,
                0,
            )
        )
        for name, value in contrasts.items()
    }


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
    return compute_contrast(labels, regression_results, vector, stat_type="t")


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


def _prediction(
    labels: np.ndarray,
    regression_results: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    prediction = np.empty(shape, dtype=float)
    for label, result in regression_results.items():
        prediction[:, labels == label] = result.predicted
    return prediction


def _r2(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual_sum = np.sum((observed - predicted) ** 2, axis=0)
    total_sum = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
    values = np.full(observed.shape[1], np.nan, dtype=float)
    np.divide(residual_sum, total_sum, out=values, where=total_sum > 0)
    return 1.0 - values
