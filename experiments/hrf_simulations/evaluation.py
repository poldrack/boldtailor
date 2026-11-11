"""Cross selection/evaluation models without conflating their contrast units."""

import numpy as np

from experiments.hrf_simulations.design import MODELS, task_design


def contrast(runs, targets, candidate, model, convention):
    """Mean run contrast, using physical design units and unpenalized OLS."""
    if candidate is None:
        return np.nan, np.nan
    observed, noiseless = [], []
    for run, target in zip(runs, targets, strict=True):
        parts = task_design(run, candidate, model, convention)
        matrix = np.column_stack([parts.task, parts.junk, parts.nuisance])
        coefficients = np.linalg.lstsq(
            matrix, np.column_stack([run.signals[:, 0], target]), rcond=None
        )[0]
        delta = coefficients[1] - coefficients[0]
        observed.append(delta[0])
        noiseless.append(delta[1])
    return float(np.mean(observed)), float(np.mean(noiseless))


def evaluate_crossed(runs, targets, candidate, truth, convention, row, cache):
    output = []
    ids = tuple(r.run_id for r in runs)
    for model in MODELS:
        key = (ids, convention, model)
        if key not in cache:
            cache[key] = contrast(runs, targets, truth, model, convention)
        known, reference = cache[key]
        selected_key = (
            key,
            None if candidate is None else (candidate.kind, candidate.parameters),
        )
        if selected_key not in cache:
            cache[selected_key] = contrast(runs, targets, candidate, model, convention)[
                0
            ]
        value = cache[selected_key]
        output.append(
            dict(
                **row,
                evaluation_model=model,
                estimated_contrast=value,
                reference_contrast=reference,
                known_hrf_contrast=known,
                contrast_error=value - reference,
                known_hrf_contrast_error=known - reference,
                selection_contrast_error=value - known,
            )
        )
    return output
