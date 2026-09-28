"""Global NSD ridge tuning with independent odd/even evaluation."""

from numbers import Integral, Real
from uuid import uuid4

import numpy as np
import pandas as pd

from boldtailor._ridge_cv import subset_runs
from boldtailor.hrf_selection import select_hrf
from boldtailor.provenance import ProvenanceRecord
from boldtailor.ridge_results import CandidateScores
from boldtailor._fractional_ridge import fraction_grid, NORM_BASIS
from boldtailor.fractional_ridge import (
    score_fraction_candidates,
    select_ridge_fractions,
)
from boldtailor.ridge_selection import (
    _alpha_grid,
    score_ridge_candidates,
    select_ridge_penalty,
)
from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
from boldtailor.trial_encoding import (
    evaluate_trial_encoding,
    encoding_metadata,
    validate_encoding_mode,
)
from .nsd_hrf import spatial_signature
from .parallel_blocks import map_blocks, validate_n_jobs
from .ridge_provenance import tuning_provenance, link_final_provenance
from .workflow_analysis import fit_beta_series
from .workflow_inputs import load_block, _trimmed_sources


def trial_predictors(runs):
    """Preserve events; unavailable behavior excludes only encoding rows."""
    tables = []
    for run in runs:
        if not {"trial_type", "response_time"}.issubset(run.events):
            raise ValueError(
                f"{run.label}: encoding needs trial_type and response_time"
            )
        table = run.events[["trial_type", "response_time"]].apply(
            pd.to_numeric, errors="raise"
        )
        present = table.trial_type.dropna()
        if not present.isin([0, 1]).all():
            raise ValueError(f"{run.label}: nonmissing trial_type must be 0 or 1")
        table.loc[
            ~np.isfinite(table.response_time) | (table.response_time <= 0),
            "response_time",
        ] = np.nan
        tables.append(table)
    return tables


def _settings(runs, library, alphas, percentile, block_size, maximum, n_jobs):
    validate_n_jobs(n_jobs)
    grid = tuple(sorted(_alpha_grid(alphas)))
    if (
        isinstance(percentile, (bool, np.bool_))
        or not isinstance(percentile, Real)
        or not np.isfinite(percentile)
        or not 0 <= percentile <= 100
    ):
        raise ValueError("percentile must be between 0 and 100")
    for name, value in (("block_size", block_size), ("max_grayordinates", maximum)):
        if value is None and name == "max_grayordinates":
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if (
        not runs
        or len({r.label for r in runs}) != len(runs)
        or len({r.number for r in runs}) != len(runs)
    ):
        raise ValueError("runs must have unique labels and numbers")
    brain = runs[0].image.header.get_axis(1)
    if any(r.image.header.get_axis(1) != brain for r in runs):
        raise ValueError("runs must have identical CIFTI BrainModel axes")
    halves = {
        "odd": [i for i, r in enumerate(runs) if r.number % 2],
        "even": [i for i, r in enumerate(runs) if not r.number % 2],
    }
    minimum = 2 if library is None else 3
    if any(len(indices) < minimum for indices in halves.values()):
        raise ValueError(f"Ridge CV needs at least {minimum} odd and even runs")
    limit = len(brain) if maximum is None else min(maximum, len(brain))
    blocks = [
        np.arange(start, min(start + block_size, limit))
        for start in range(0, limit, block_size)
    ]
    return grid, halves, blocks


def _signature(runs, indices):
    return spatial_signature(runs[0].image.header.get_axis(1), indices)


def _score_block(
    indices,
    runs,
    root,
    predictors,
    library,
    alphas,
    fractional=False,
    encoding_mode="within_run",
):
    scorer = score_fraction_candidates if fractional else score_ridge_candidates
    return scorer(
        load_block(runs, root, indices),
        predictors,
        **({"fractions": alphas} if fractional else {"alphas": alphas}),
        library=library,
        run_labels=[r.label for r in runs],
        feature_signature=_signature(runs, indices),
        encoding_mode=encoding_mode,
    )


def _global_scores(
    runs,
    root,
    predictors,
    blocks,
    library,
    alphas,
    n_jobs,
    fractional=False,
    encoding_mode="within_run",
):
    n_features = runs[0].image.shape[1]
    shape = (len(runs), len(alphas), n_features)
    sse, sst = np.full(shape, np.nan), np.full(shape, np.nan)
    scores = np.full((len(alphas), n_features), np.nan)
    assignments = np.full((len(runs), n_features), -1, dtype=int)
    records = []
    for indices, result in map_blocks(
        _score_block,
        blocks,
        args=(runs, root, predictors, library, alphas, fractional, encoding_mode),
        n_jobs=n_jobs,
    ):
        sse[:, :, indices], sst[:, :, indices] = result.fold_sse, result.fold_sst
        scores[:, indices], assignments[:, indices] = (
            result.cv_r2,
            result.fold_hrf_indices,
        )
        records.append(
            dict(
                grayordinate_indices=indices.tolist(),
                activities=result.provenance.to_dict()["activities"],
            )
        )
    indices = np.concatenate(blocks)
    provenance = ProvenanceRecord(
        execution_id=str(uuid4()),
        sources=[_trimmed_sources(r, root, indices) for r in runs],
        activities=(
            dict(
                name="nsd_global_encoding_ridge_cv",
                **encoding_metadata(encoding_mode),
                **(
                    {"fractions": list(alphas)}
                    if fractional
                    else {"alphas": list(alphas)}
                ),
                block_records=records,
                feature_signature=_signature(runs, indices),
            ),
        ),
    )
    return CandidateScores(
        regularization="fractional_ridge" if fractional else "normalized_ridge",
        grid=alphas,
        cv_r2=scores,
        fold_sse=sse,
        fold_sst=sst,
        fold_hrf_indices=assignments,
        trial_masks=result.trial_masks,
        run_labels=tuple(r.label for r in runs),
        provenance=provenance,
    )


def _tune(
    runs,
    root,
    predictors,
    blocks,
    library,
    alphas,
    percentile,
    n_jobs,
    fractional=False,
    encoding_mode="within_run",
):
    scores = _global_scores(
        runs,
        root,
        predictors,
        blocks,
        library,
        alphas,
        n_jobs,
        fractional,
        encoding_mode,
    )
    selection = (
        select_ridge_fractions(scores.cv_r2, scores.grid)
        if fractional
        else select_ridge_penalty(scores.cv_r2, scores.grid, percentile=percentile)
    )
    return dict(
        scores=scores,
        selection=selection,
        run_labels=[r.label for r in runs],
        provenance=tuning_provenance(scores, selection),
        summary_percentile=percentile,
    )


def _fit_outer_trials(data, selection, labels, signature, options):
    if selection is None:
        return fit_single_trials(data, **options, run_labels=labels)
    return fit_selected_hrfs(
        data,
        selection=selection,
        **options,
        run_labels=labels,
        feature_signature=signature,
    )


def _outer_block(
    indices,
    runs,
    root,
    predictors,
    library,
    alpha,
    train,
    test,
    fractional=False,
    encoding_mode="within_run",
):
    data = load_block(runs, root, indices)
    labels, signature = [r.label for r in runs], _signature(runs, indices)
    selection = (
        None
        if library is None
        else select_hrf(
            subset_runs(data, train),
            library=library,
            run_labels=[labels[i] for i in train],
            feature_signature=signature,
        )
    )
    options = (
        {"ridge_fraction": alpha[indices]} if fractional else {"ridge_alpha": alpha}
    )
    fitted = _fit_outer_trials(data, selection, labels, signature, options)
    ids = (
        np.zeros(len(indices), dtype=int)
        if selection is None
        else selection.hrf_indices
    )
    scoring_betas = list(fitted.run_betas)
    if fractional:
        # Retain training-selected HRFs; held-out responses only define OLS targets.
        ols_options = dict(
            ridge_fraction=np.where(np.isfinite(alpha[indices]), 1.0, np.nan)
        )
        ols = _fit_outer_trials(data, selection, labels, signature, ols_options)
        for r in test:
            scoring_betas[r] = ols.run_betas[r]
    encoded = evaluate_trial_encoding(
        scoring_betas,
        predictors,
        train_runs=train,
        test_runs=test,
        encoding_mode=encoding_mode,
    )
    return dict(
        encoding_mode=encoding_mode,
        train_run_intercepts=encoded.train_run_intercepts,
        train_run_predictor_means=encoded.train_run_predictor_means,
        scoring_offsets=encoded.scoring_offsets,
        encoding_r2=encoded.r2,
        run_sse=encoded.run_sse,
        run_sst=encoded.run_sst,
        coefficients=encoded.coefficients,
        predictor_means=encoded.predictor_means,
        predictions=encoded.predictions,
        betas=[fitted.run_betas[r] for r in test],
        targets=[scoring_betas[r] for r in test],
        hrf_indices=ids,
        trial_masks=encoded.trial_masks,
        run_ridge_alphas=fitted.run_ridge_alphas,
        provenance=fitted.provenance.to_dict(),
        selection_provenance=(
            None if selection is None else selection.provenance.to_dict()
        ),
    )


def _evaluate(
    runs,
    root,
    predictors,
    blocks,
    library,
    alpha,
    train,
    test,
    n_jobs,
    fractional=False,
    encoding_mode="within_run",
):
    n = runs[0].image.shape[1]
    result = dict(
        encoding_mode=encoding_mode,
        train_run_intercepts=np.full((len(train), n), np.nan),
        scoring_offsets=np.full((len(test), n), np.nan),
        encoding_r2=np.full(n, np.nan),
        coefficients=np.full((3, n), np.nan),
        run_sse=np.full((len(test), n), np.nan),
        run_sst=np.full((len(test), n), np.nan),
        hrf_indices=np.full(n, -1, dtype=int),
        ridge_alpha=None if fractional else alpha,
        provenance=[],
        train_run_labels=[runs[r].label for r in train],
        test_run_labels=[runs[r].label for r in test],
    )
    if fractional:
        result.update(
            ridge_fraction=np.array(alpha),
            run_ridge_alphas=np.full((len(runs), n), np.nan),
        )
    for key in ("betas", "predictions", "targets"):
        result[key] = [
            np.full((len(runs[r].events), n), np.nan, dtype=np.float32) for r in test
        ]
    args = (
        runs,
        root,
        predictors,
        library,
        alpha,
        train,
        test,
        fractional,
        encoding_mode,
    )
    for indices, block in map_blocks(_outer_block, blocks, args=args, n_jobs=n_jobs):
        for key in (
            "encoding_r2",
            "coefficients",
            "run_sse",
            "run_sst",
            "hrf_indices",
            "train_run_intercepts",
            "scoring_offsets",
        ):
            result[key][..., indices] = block[key]
        if fractional:
            result["run_ridge_alphas"][:, indices] = np.asarray(
                block["run_ridge_alphas"]
            )
        for key in ("betas", "predictions", "targets"):
            for target, values in zip(result[key], block[key], strict=True):
                target[:, indices] = values
        result["train_run_predictor_means"] = block["train_run_predictor_means"]
        result["predictor_means"], result["trial_masks"] = (
            block["predictor_means"],
            block["trial_masks"],
        )
        result["provenance"].append(
            dict(
                grayordinate_indices=indices.tolist(),
                record=block["provenance"],
                selection_record=block["selection_provenance"],
            )
        )
    return result


def _final_selection(indices, runs, root, library):
    return dict(
        all=select_hrf(
            load_block(runs, root, indices),
            library=library,
            run_labels=[r.label for r in runs],
            feature_signature=_signature(runs, indices),
        )
    )


def _final_fit(runs, root, blocks, library, alpha, n_jobs, fractional=False):
    selections = None
    if library is not None:
        selections = {
            tuple(indices): selected
            for indices, selected in map_blocks(
                _final_selection, blocks, args=(runs, root, library), n_jobs=n_jobs
            )
        }
    options = {"ridge_fraction": alpha} if fractional else {"ridge_alpha": alpha}
    result = fit_beta_series(
        runs, root, blocks, selections=selections, **options, n_jobs=n_jobs
    )
    result["hrf_selections"] = selections
    return result


def fit_cv_beta_series(
    runs,
    root,
    *,
    library,
    alphas=None,
    fractions=None,
    percentile=90.0,
    block_size=4096,
    max_grayordinates=None,
    n_jobs=1,
    encoding_mode="within_run",
):
    """Tune globally on each training scope, then evaluate or refit at its alpha.

    A library requests optimized HRFs; None uses canonical SPM. Sessions with
    optimized HRFs need three or more runs in each half. Returned outer betas,
    targets and predictions contain test runs only, in test_run_labels order.
    Betas retain the selected regularization. Fractional targets are OLS fits
    under training-selected HRFs; shared-alpha targets equal the fitted betas.
    """
    validate_encoding_mode(encoding_mode)
    if (alphas is None) == (fractions is None):
        raise ValueError("provide exactly one of alphas or fractions")
    fractional = fractions is not None
    candidates = fraction_grid(fractions) if fractional else alphas
    alphas, halves, blocks = _settings(
        runs, library, candidates, percentile, block_size, max_grayordinates, n_jobs
    )
    if fractional:
        alphas = tuple(reversed(alphas))
    predictors = trial_predictors(runs)
    result = dict(tuning={}, evaluation={}, predictors=predictors)
    mode = "canonical" if library is None else "optimized"
    for scope, train in (*halves.items(), ("all", list(range(len(runs))))):
        print(f"Ridge CV ({mode}, {scope}): tuning {len(train)} runs", flush=True)
        selected = _tune(
            [runs[i] for i in train],
            root,
            [predictors[i] for i in train],
            blocks,
            library,
            alphas,
            percentile,
            n_jobs,
            fractional,
            encoding_mode,
        )
        result["tuning"][scope] = selected
        alpha = (
            selected["selection"].ridge_fraction
            if fractional
            else selected["selection"].ridge_alpha
        )
        if scope != "all":
            target = "even" if scope == "odd" else "odd"
            result["evaluation"][f"{scope}_to_{target}"] = _evaluate(
                runs,
                root,
                predictors,
                blocks,
                library,
                alpha,
                train,
                halves[target],
                n_jobs,
                fractional,
                encoding_mode,
            )
            result["evaluation"][f"{scope}_to_{target}"][
                "tuning_analysis_fingerprint"
            ] = selected["provenance"].analysis_fingerprint
        description = (
            f"selected fractions at {np.isfinite(alpha).sum()} grayordinates"
            if fractional
            else f"selected alpha={alpha:g}"
        )
        print(f"Ridge CV ({mode}, {scope}): {description}", flush=True)
    result["final"] = _final_fit(runs, root, blocks, library, alpha, n_jobs, fractional)
    link_final_provenance(result["final"], result["tuning"]["all"]["provenance"])
    result["provenance"] = dict(
        **encoding_metadata(encoding_mode),
        objective="percentile_of_" + encoding_metadata(encoding_mode)["score"],
        percentile=percentile,
        alphas=list(alphas),
        validation_target="candidate_regularized_betas",
        hrf_model=mode,
        library_fingerprint=None if library is None else library.fingerprint,
        final_ridge_alpha=None if fractional else alpha,
        outer_splits="odd_to_even_and_even_to_odd",
    )
    if fractional:
        result["provenance"].update(
            objective="maximum_encoding_r2_per_grayordinate",
            fractions=result["provenance"].pop("alphas"),
            validation_target="fixed_ols_betas",
            percentile_role="descriptive_only",
            fraction_norm_basis=NORM_BASIS,
        )
    return result
