"""Fraction or shared-alpha tuning with independent odd/even evaluation."""

from boldtailor.workflow._records import (
    OuterEvaluation,
    OuterBlock,
    TuningResult,
    CvBetaResult,
    HrfOptions,
    SelectionBundle,
    BetaFit,
)

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import logging
from numbers import Integral, Real
from uuid import uuid4

import numpy as np
import pandas as pd

from boldtailor.hrf_selection import select_hrfs, subset_runs
from boldtailor.hrf_options import resolve_reml_options
from boldtailor.provenance import (
    ProvenanceRecord,
    analysis_fingerprint,
    extend_provenance,
)
from boldtailor.ridge_results import CandidateScores, FractionSelection
from boldtailor.fractional_ridge import (
    NORM_BASIS,
    fraction_grid,
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
    trial_predictors,
    evaluate_trial_encoding,
    encoding_metadata,
    validate_encoding_mode,
)
from boldtailor.cifti import spatial_signature
from boldtailor.parallel import map_blocks, validate_n_jobs
from boldtailor.model import TaskModel
from boldtailor.workflow import inputs
from boldtailor.workflow.analysis import fit_beta_series
from boldtailor.workflow.files import odd_even_parity
from boldtailor.workflow.inputs import _trimmed_sources, load_block

log = logging.getLogger("boldtailor.workflow")


def _settings(
    runs, library, alphas, percentile, block_size, maximum, n_jobs, hrf_method="cv"
):
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
    halves = odd_even_parity(runs)
    minimum = 2 if library is None or hrf_method == "reml" else 3
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


def _block_selection_options(options, indices, *, include_model=True):
    """Subset supplied feature correlations without changing the selection model."""
    result = dict(options or {})
    if result.get("reml_options") is not None:
        result["reml_options"] = result["reml_options"].subset(indices)
    if not include_model:
        result.pop("task_model", None)
    return result


@dataclass(frozen=True, kw_only=True)
class _CvContext:
    """Shared validated configuration for CV ridge tuning, evaluation, and refit."""

    runs: object
    root: object
    predictors: object
    blocks: object
    library: object
    alphas: tuple[float, ...]
    percentile: float
    n_jobs: int
    fractional: bool
    encoding_mode: str
    task_model: TaskModel
    selection_model: TaskModel
    selection_options: dict


def _score_block(indices, ctx, scope_runs, scope_predictors):
    options = _block_selection_options(ctx.selection_options, indices)
    options["hrf_method"] = options.pop("method", "cv")
    scorer = score_fraction_candidates if ctx.fractional else score_ridge_candidates
    return scorer(
        load_block(scope_runs, ctx.root, indices, ctx.task_model),
        scope_predictors,
        **({"fractions": ctx.alphas} if ctx.fractional else {"alphas": ctx.alphas}),
        library=ctx.library,
        run_labels=[r.label for r in scope_runs],
        feature_signature=_signature(scope_runs, indices),
        encoding_mode=ctx.encoding_mode,
        **options,
    )


def _global_scores(ctx, scope_runs, scope_predictors):
    n_features = scope_runs[0].image.shape[1]
    shape = (len(scope_runs), len(ctx.alphas), n_features)
    sse, sst = np.full(shape, np.nan), np.full(shape, np.nan)
    scores = np.full((len(ctx.alphas), n_features), np.nan)
    assignments = np.full((len(scope_runs), n_features), -1, dtype=int)
    records = []
    for indices, result in map_blocks(
        _score_block,
        ctx.blocks,
        args=(ctx, scope_runs, scope_predictors),
        n_jobs=ctx.n_jobs,
    ):
        sse[:, :, indices], sst[:, :, indices] = result.fold_sse, result.fold_sst
        scores[:, indices], assignments[:, indices] = (
            result.cv_r2,
            result.fold_hrf_indices,
        )
        records.append(
            dict(
                grayordinate_indices=indices.tolist(),
                activities=result.provenance.activity_records(),
            )
        )
    indices = np.concatenate(ctx.blocks)
    grid_record = (
        {"fractions": list(ctx.alphas)}
        if ctx.fractional
        else {"alphas": list(ctx.alphas)}
    )
    provenance = ProvenanceRecord(
        execution_id=str(uuid4()),
        sources=[_trimmed_sources(r, ctx.root, indices) for r in scope_runs],
        activities=(
            dict(
                name="global_encoding_ridge_cv",
                **encoding_metadata(ctx.encoding_mode),
                **grid_record,
                block_records=records,
                feature_signature=_signature(scope_runs, indices),
            ),
        ),
    )
    return CandidateScores(
        regularization="fractional_ridge" if ctx.fractional else "normalized_ridge",
        grid=ctx.alphas,
        cv_r2=scores,
        fold_sse=sse,
        fold_sst=sst,
        fold_hrf_indices=assignments,
        trial_masks=result.trial_masks,
        run_labels=tuple(r.label for r in scope_runs),
        provenance=provenance,
    )


def _tune(ctx, train):
    scope_runs = [ctx.runs[i] for i in train]
    scope_predictors = [ctx.predictors[i] for i in train]
    scores = _global_scores(ctx, scope_runs, scope_predictors)
    selection = (
        select_ridge_fractions(scores)
        if ctx.fractional
        else select_ridge_penalty(scores, percentile=ctx.percentile)
    )
    return TuningResult(
        scores=scores,
        selection=selection,
        run_labels=[r.label for r in scope_runs],
        provenance=tuning_provenance(
            scores, selection, encoding_mode=ctx.encoding_mode
        ),
        summary_percentile=ctx.percentile,
    )


def _fit_outer_trials(data, selection, labels, signature, options):
    if selection is None:
        return fit_single_trials(data, **options, run_labels=labels)
    return fit_selected_hrfs(
        data,
        hrf_selection=selection,
        **options,
        run_labels=labels,
        feature_signature=signature,
    )


def _outer_block(indices, ctx, alpha, train, test):
    data = load_block(ctx.runs, ctx.root, indices, ctx.task_model)
    labels, signature = [r.label for r in ctx.runs], _signature(ctx.runs, indices)
    selection = (
        None
        if ctx.library is None
        else select_hrfs(
            subset_runs(data, train),
            library=ctx.library,
            run_labels=[labels[i] for i in train],
            feature_signature=signature,
            task_model=ctx.selection_model,
            **_block_selection_options(ctx.selection_options, indices, include_model=False),
        )
    )
    options = (
        {"ridge_fraction": alpha[indices]} if ctx.fractional else {"ridge_alpha": alpha}
    )
    fitted = _fit_outer_trials(data, selection, labels, signature, options)
    ids = (
        np.zeros(len(indices), dtype=int)
        if selection is None
        else selection.hrf_indices
    )
    scoring_betas = list(fitted.run_betas)
    if ctx.fractional:
        # Retain training-selected HRFs; held-out responses only define OLS targets.
        ols_options = dict(
            ridge_fraction=np.where(np.isfinite(alpha[indices]), 1.0, np.nan)
        )
        ols = _fit_outer_trials(data, selection, labels, signature, ols_options)
        for r in test:
            scoring_betas[r] = ols.run_betas[r]
    encoded = evaluate_trial_encoding(
        scoring_betas,
        ctx.predictors,
        train_runs=train,
        test_runs=test,
        encoding_mode=ctx.encoding_mode,
    )
    return OuterBlock(
        encoding_mode=ctx.encoding_mode,
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


def _evaluate(ctx, alpha, train, test):
    n = ctx.runs[0].image.shape[1]
    result = OuterEvaluation(
        encoding_mode=ctx.encoding_mode,
        selection_task_model=ctx.selection_model.to_dict(),
        coefficient_names=["task", *ctx.predictors[0].columns],
        train_run_intercepts=np.full((len(train), n), np.nan),
        scoring_offsets=np.full((len(test), n), np.nan),
        encoding_r2=np.full(n, np.nan),
        coefficients=np.full((1 + len(ctx.predictors[0].columns), n), np.nan),
        run_sse=np.full((len(test), n), np.nan),
        run_sst=np.full((len(test), n), np.nan),
        hrf_indices=np.full(n, -1, dtype=int),
        ridge_alpha=None if ctx.fractional else alpha,
        provenance=[],
        train_run_labels=[ctx.runs[r].label for r in train],
        test_run_labels=[ctx.runs[r].label for r in test],
        betas=[],
        predictions=[],
        targets=[],
    )
    if ctx.fractional:
        result.update(
            ridge_fraction=np.array(alpha),
            run_ridge_alphas=np.full((len(ctx.runs), n), np.nan),
        )
    for key in ("betas", "predictions", "targets"):
        result[key] = [
            np.full((len(ctx.runs[r].events), n), np.nan, dtype=np.float32)
            for r in test
        ]
    for indices, block in map_blocks(
        _outer_block,
        ctx.blocks,
        args=(ctx, alpha, train, test),
        n_jobs=ctx.n_jobs,
    ):
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
        if "run_ridge_alphas" in result:
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


def _final_selection(indices, ctx):
    return SelectionBundle(
        all=select_hrfs(
            load_block(ctx.runs, ctx.root, indices, ctx.task_model),
            library=ctx.library,
            run_labels=[r.label for r in ctx.runs],
            feature_signature=_signature(ctx.runs, indices),
            task_model=ctx.selection_model,
            **_block_selection_options(ctx.selection_options, indices, include_model=False),
        )
    )


def _final_fit(ctx, alpha):
    selections = None
    if ctx.library is not None:
        selections = {
            tuple(indices): selected
            for indices, selected in map_blocks(
                _final_selection,
                ctx.blocks,
                args=(ctx,),
                n_jobs=ctx.n_jobs,
            )
        }
    options = {"ridge_fraction": alpha} if ctx.fractional else {"ridge_alpha": alpha}
    result = fit_beta_series(
        ctx.runs,
        ctx.root,
        ctx.blocks,
        task_model=ctx.task_model,
        selections=selections,
        **options,
        n_jobs=ctx.n_jobs,
    )
    result["hrf_selections"] = selections
    return result


def _build_cv_context(
    runs,
    root,
    *,
    task_model,
    library,
    alphas,
    fractions,
    percentile,
    block_size,
    max_grayordinates,
    n_jobs,
    encoding_mode,
    selection_model,
    hrf_method,
    reml_options,
):
    validate_encoding_mode(encoding_mode)
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    selection_model = task_model if selection_model is None else selection_model
    if not isinstance(selection_model, TaskModel):
        raise ValueError("selection_model must be a TaskModel")
    if (alphas is None) == (fractions is None):
        raise ValueError("provide exactly one of alphas or fractions")
    reml_options = resolve_reml_options(hrf_method, reml_options)
    if library is None and hrf_method != "cv":
        raise ValueError("HRF method requires a library")
    selection_options = HrfOptions(method=hrf_method, reml_options=reml_options)
    if hrf_method == "reml":
        selection_options["task_model"] = selection_model
    fractional = fractions is not None
    candidates = fraction_grid(fractions) if fractional else alphas
    grid, halves, blocks = _settings(
        runs,
        library,
        candidates,
        percentile,
        block_size,
        max_grayordinates,
        n_jobs,
        hrf_method,
    )
    if fractional:
        grid = tuple(reversed(grid))
    predictors = trial_predictors(
        [r.events for r in runs], task_model, [r.label for r in runs]
    )
    ctx = _CvContext(
        runs=runs,
        root=root,
        predictors=predictors,
        blocks=blocks,
        library=library,
        alphas=grid,
        percentile=percentile,
        n_jobs=n_jobs,
        fractional=fractional,
        encoding_mode=encoding_mode,
        task_model=task_model,
        selection_model=selection_model,
        selection_options=selection_options,
    )
    return ctx, halves


def _selected_penalty(selection):
    return (
        selection.ridge_fraction
        if isinstance(selection, FractionSelection)
        else selection.ridge_alpha
    )


def _cv_provenance(ctx, mode, final_penalty):
    provenance = dict(
        **encoding_metadata(ctx.encoding_mode),
        objective="percentile_of_" + encoding_metadata(ctx.encoding_mode)["score"],
        percentile=ctx.percentile,
        alphas=list(ctx.alphas),
        validation_target="candidate_regularized_betas",
        hrf_model=mode,
        library_fingerprint=None if ctx.library is None else ctx.library.fingerprint,
        final_ridge_alpha=None if ctx.fractional else final_penalty,
        outer_splits="odd_to_even_and_even_to_odd",
    )
    if ctx.fractional:
        provenance.update(
            objective="maximum_encoding_r2_per_grayordinate",
            fractions=provenance.pop("alphas"),
            validation_target="fixed_ols_betas",
            percentile_role="descriptive_only",
            fraction_norm_basis=NORM_BASIS,
        )
    return provenance


def fit_cv_beta_series(
    runs,
    root,
    *,
    task_model,
    library,
    alphas=None,
    fractions=None,
    percentile=90.0,
    block_size=4096,
    max_grayordinates=None,
    n_jobs=1,
    encoding_mode="within_run",
    selection_model=None,
    hrf_method="cv",
    reml_options=None,
):
    """Tune per-feature fractions or one shared alpha, then evaluate and refit.

    A library requests optimized HRFs; None uses canonical SPM. Sessions with
    optimized HRFs need three or more runs in each half. Returned outer betas,
    targets and predictions contain test runs only, in test_run_labels order.
    Betas retain the selected regularization. Fractional targets are OLS fits
    under training-selected HRFs; shared-alpha targets equal the fitted betas.
    The task model drives fits and encoding predictors; selection_model (default
    the task model) scores HRFs, for example without RT.
    """
    ctx, halves = _build_cv_context(
        runs,
        root,
        task_model=task_model,
        library=library,
        alphas=alphas,
        fractions=fractions,
        percentile=percentile,
        block_size=block_size,
        max_grayordinates=max_grayordinates,
        n_jobs=n_jobs,
        encoding_mode=encoding_mode,
        selection_model=selection_model,
        hrf_method=hrf_method,
        reml_options=reml_options,
    )
    result = CvBetaResult(
        tuning={},
        evaluation={},
        predictors=ctx.predictors,
        selection_task_model=ctx.selection_model.to_dict(),
    )
    mode = "canonical" if ctx.library is None else "optimized"
    for scope, train in (*halves.items(), ("all", list(range(len(runs))))):
        log.info("Ridge CV (%s, %s): tuning %d runs", mode, scope, len(train))
        selected = _tune(ctx, train)
        result["tuning"][scope] = selected
        alpha = _selected_penalty(selected["selection"])
        if scope != "all":
            target = "even" if scope == "odd" else "odd"
            evaluation = _evaluate(ctx, alpha, train, halves[target])
            evaluation["tuning_analysis_fingerprint"] = selected[
                "provenance"
            ].analysis_fingerprint
            result["evaluation"][f"{scope}_to_{target}"] = evaluation
        description = (
            f"selected fractions at {np.isfinite(alpha).sum()} grayordinates"
            if ctx.fractional
            else f"selected alpha={alpha:g}"
        )
        log.info("Ridge CV (%s, %s): %s", mode, scope, description)
    final_penalty = _selected_penalty(result["tuning"]["all"]["selection"])
    final_fit = _final_fit(ctx, final_penalty)
    result["final"] = link_final_provenance(
        final_fit, result["tuning"]["all"]["provenance"], encoding_mode=ctx.encoding_mode
    )
    result["provenance"] = _cv_provenance(ctx, mode, final_penalty)
    return result


def tuning_provenance(scores, selection, *, encoding_mode):
    """Record a tuning decision with the explicitly supplied scoring mode."""
    if isinstance(selection, FractionSelection):
        activity: dict[str, object] = dict(
            name="voxelwise_encoding_fraction_selection",
            fractions=list(selection.fractions),
            objective="maximum_encoding_r2_per_grayordinate",
            validation_target="fixed_ols_betas",
            fraction_norm_basis=NORM_BASIS,
            selected_fraction_fingerprint=sha256(
                np.asarray(selection.ridge_fraction, dtype="<f8").tobytes()
            ).hexdigest(),
            tie_rule="largest fraction within 1e-12 of the maximum score",
        )
    else:
        activity = _alpha_activity(selection)
    activity.update(encoding_metadata(encoding_mode))
    if not isinstance(selection, FractionSelection):
        activity["objective"] = (
            "percentile_of_" + encoding_metadata(encoding_mode)["score"]
        )
    activity.update(
        run_labels=list(scores.run_labels),
        scoring_mask_fingerprint=sha256(selection.scoring_mask.tobytes()).hexdigest(),
        candidate_score_fingerprint=sha256(
            np.asarray(scores.cv_r2, dtype="<f8").tobytes()
        ).hexdigest(),
    )
    record = scores.provenance
    identity = dict(scoring=_identity_activities(record), selection=activity)
    return extend_provenance(
        record,
        execution_id=str(uuid4()),
        activity=activity,
        events=record.events,
        warnings=(),
        analysis_id=analysis_fingerprint(record.metadata_fingerprint, identity),
    )


def _identity_activities(record):
    """Embedded scoring activities without environment-only software keys."""
    return [
        {key: value for key, value in activity.items() if key != "software"}
        for activity in record.activity_records()
    ]


def _alpha_activity(selection) -> dict[str, object]:
    return dict(
        name="global_encoding_ridge_selection",
        alphas=list(selection.alphas),
        percentile=selection.percentile,
        objective="percentile_of_pooled_within_run_trial_encoding_r2",
        validation_target="candidate_regularized_betas",
        tie_rule="smallest alpha within 1e-12 of the maximum objective",
        objective_scores=selection.objective_scores.tolist(),
        selected_alpha=selection.ridge_alpha,
    )


def link_final_provenance(result, decision, *, encoding_mode) -> BetaFit:
    """Return a fit with new block records referencing the all-run decision.

    Provenance is copied; numerical arrays are shared and are not modified.
    Pass an unlinked fit: linking an already linked result raises ValueError.
    """
    blocks = deepcopy(result["provenance"])
    for block in blocks:
        record = ProvenanceRecord.from_dict(block["record"])
        if any(
            activity.get("name") == "encoding_guided_ridge_refit"
            for activity in record.activity_records()
        ):
            raise ValueError("final fit provenance is already linked")
        identity = dict(
            fitting_analysis_fingerprint=record.analysis_fingerprint,
            tuning_analysis_fingerprint=decision.analysis_fingerprint,
        )
        activity = dict(
            name="encoding_guided_ridge_refit",
            **encoding_metadata(encoding_mode),
            tuning_scope="all",
            selected_alpha=result["ridge_alpha"],
            tuning_execution_id=decision.execution_id,
            **identity,
        )
        if "ridge_fraction" in result:
            activity.update(
                regularization="fractional_ridge", fraction_norm_basis=NORM_BASIS
            )
        block["record"] = extend_provenance(
            record,
            execution_id=str(uuid4()),
            activity=activity,
            events=record.events,
            warnings=(),
            analysis_id=analysis_fingerprint(record.metadata_fingerprint, identity),
        ).to_dict()
    linked: BetaFit = result.copy()
    linked["provenance"] = blocks
    return linked


@dataclass(frozen=True, kw_only=True)
class BetaModel:
    """One trial-wise beta model and, for tuned ridge, how it was chosen."""

    name: str
    hrf: str
    estimator: str
    fit: Mapping[str, object]
    tuning: Mapping[str, object] | None = None
    evaluation: Mapping[str, object] | None = None
    cv_provenance: Mapping[str, object] | None = None
    predictors: tuple | None = None

    @property
    def fractional(self):
        return "ridge_fraction" in self.fit


_ESTIMATOR = {"fixed": "Ridge", "cv": "RidgeCV", "fractional_cv": "FractionalCV"}


def _fixed_models(runs, root, blocks, settings, task_model, hrf, selected):
    prefix = hrf.capitalize() + "Trial"
    fits = {"OLS": 0.0}
    if settings.ridge_mode == "fixed":
        fits["Ridge"] = settings.ridge_alpha
    return {
        prefix
        + kind: BetaModel(
            name=prefix + kind,
            hrf=hrf,
            estimator=kind,
            fit=fit_beta_series(
                runs,
                root,
                blocks,
                task_model=task_model,
                selections=selected,
                ridge_alpha=alpha,
                n_jobs=settings.n_jobs,
            ),
        )
        for kind, alpha in fits.items()
    }


def _cv_model(runs, root, settings, library, task_model, hrf):
    fractional = settings.ridge_mode == "fractional_cv"
    grid = (
        dict(fractions=settings.ridge_fractions)
        if fractional
        else dict(alphas=settings.ridge_alphas)
    )
    result = fit_cv_beta_series(
        runs,
        root,
        task_model=task_model,
        selection_model=inputs.selection_task_model(
            task_model, settings.hrf_selection_rt
        ),
        library=library,
        hrf_method=settings.hrf_method if library is not None else "cv",
        reml_options=(
            settings.selection_options()["reml_options"]
            if library is not None
            else None
        ),
        **grid,
        encoding_mode=settings.encoding_mode,
        percentile=settings.ridge_percentile,
        block_size=settings.block_size,
        max_grayordinates=settings.max_grayordinates,
        n_jobs=settings.n_jobs,
    )
    estimator = _ESTIMATOR[settings.ridge_mode]
    return BetaModel(
        name=f"{hrf.capitalize()}Trial{estimator}",
        hrf=hrf,
        estimator=estimator,
        fit=result["final"],
        tuning=result["tuning"],
        evaluation=result["evaluation"],
        cv_provenance=result["provenance"],
        predictors=tuple(result["predictors"]),
    )


def fit_beta_models(runs, root, blocks, settings, library, selections, task_model):
    """OLS for both HRFs, plus the ridge variant the settings request."""
    models = {}
    for hrf, selected, candidates in (
        ("canonical", None, None),
        ("optimized", selections, library),
    ):
        models.update(
            _fixed_models(runs, root, blocks, settings, task_model, hrf, selected)
        )
        if settings.ridge_mode in ("cv", "fractional_cv"):
            model = _cv_model(runs, root, settings, candidates, task_model, hrf)
            models[model.name] = model
    return models
