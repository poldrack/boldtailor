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


def nested_ols_delta(signals, full_designs, nuisance_designs, *, allow_undefined):
    """Return (full R2, nuisance R2) from nested OLS fits; guard monotonicity."""
    from boldtailor._conventional import fit_r2_designs

    full = fit_r2_designs(signals, full_designs, DIAGNOSTIC_NOISE_MODEL)
    nuisance = fit_r2_designs(signals, nuisance_designs, DIAGNOSTIC_NOISE_MODEL)
    if not allow_undefined and not (
        np.isfinite(full).all() and np.isfinite(nuisance).all()
    ):
        raise ValueError("diagnostic fit produced nonfinite r-squared values")
    validate_nested_ols_delta(full - nuisance)
    return full, nuisance


def delta_r2_activity(
    *, name, parent_id, inferential_noise_model, nuisance_model, undefined_features
):
    """Provenance activity shared by every task delta-R2 entry point."""
    return {
        "name": name,
        "stage": "fit",
        "parent_analysis_id": parent_id,
        "definition": TASK_DELTA_R2_DEFINITION,
        "clip_below_zero": True,
        "clip_policy": "numerical_roundoff_guard",
        "diagnostic_noise_model": DIAGNOSTIC_NOISE_MODEL,
        "inferential_noise_model": inferential_noise_model,
        "nuisance_model": nuisance_model,
        "undefined_features": undefined_features,
    }
