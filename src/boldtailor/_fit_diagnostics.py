"""Common diagnostics for conventional and prepared GLM fits."""

import numpy as np

TASK_DELTA_R2_DEFINITION = "full_r2 - nuisance_r2"
DIAGNOSTIC_NOISE_MODEL = "ols"
NESTED_OLS_TOLERANCE = 1e-12


def validate_nested_ols_delta(raw_delta_r2):
    if np.any(raw_delta_r2 < -NESTED_OLS_TOLERANCE):
        raise ValueError("nested OLS monotonicity violated")


def rank_warnings(rank, columns, run):
    return (
        []
        if rank == columns
        else [f"run {run} design rank is {rank} for {columns} columns"]
    )


def validate_result_dimensions(data, result, *, input_label):
    designs, run_r2 = result.design_matrices, result.run_r2
    if len(designs) != data.n_runs or len(run_r2) != data.n_runs:
        raise ValueError(f"full result run dimensions do not match {input_label}")
    for run, (signals, design, values) in enumerate(
        zip(data.signals, designs, run_r2, strict=True)
    ):
        if design.shape[0] != signals.shape[0]:
            raise ValueError(
                f"full result run {run} dimensions do not match {input_label}"
            )
        if values.shape != (data.n_features,):
            raise ValueError(
                f"full result run {run} feature dimensions do not match {input_label}"
            )
    if result.r2.shape != (data.n_features,):
        raise ValueError(f"full result feature dimensions do not match {input_label}")
    if not np.isfinite(result.r2).all():
        raise ValueError("full result r-squared values must be finite")
