"""Fraction or shared-alpha tuning with independent odd/even evaluation."""

from boldtailor.workflow.records import (
    OuterEvaluation,
    OuterBlock,
    TuningResult,
    CvBetaResult,
    HrfOptions,
    SelectionBundle,
    BetaFit,
)

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from numbers import Real
from uuid import uuid4

import numpy as np

from boldtailor._regularizers import FRACTIONAL_RIDGE, NORMALIZED_RIDGE, Regularizer
from boldtailor._ridge_cv import score_candidates
from boldtailor.hrf_selection import select_hrfs, subset_runs
from boldtailor.hrf_options import resolve_reml_options
from boldtailor.provenance import (
    ProvenanceRecord,
)
from boldtailor.ridge_results import CandidateScores
from boldtailor.fractional_ridge import (
    select_ridge_fractions,
)
from boldtailor.ridge_selection import (
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
from boldtailor.workflow.files import min_runs_per_half, odd_even_parity
from boldtailor.workflow.inputs import load_block, trimmed_sources
from boldtailor.workflow.ridge_provenance import (
    cv_provenance,
    link_final_provenance,
    tuning_provenance,
)

log = logging.getLogger("boldtailor.workflow")


def _validate_percentile(percentile):
    if (
        isinstance(percentile, (bool, np.bool_))
        or not isinstance(percentile, Real)
        or not np.isfinite(percentile)
        or not 0 <= percentile <= 100
    ):
        raise ValueError("percentile must be between 0 and 100")


def _halves(runs, library, hrf_method):
    """Odd/even run positions of a session whose runs share one spatial axis."""
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
    minimum = min_runs_per_half(
        hrf_method, ridge_cv=True, optimized=library is not None
    )
    if any(len(indices) < minimum for indices in halves.values()):
        raise ValueError(f"Ridge CV needs at least {minimum} odd and even runs")
    return halves


def _signature(runs, indices):
    return spatial_signature(runs[0].image.header.get_axis(1), indices)


def _block_selection_options(options, indices):
    """The HRF method, with supplied feature correlations subset to one block."""
    result = dict(options)
    if result["reml_options"] is not None:
        result["reml_options"] = result["reml_options"].subset(indices)
    return result


@dataclass(frozen=True, kw_only=True)
class _CvContext:
    """Shared validated configuration for CV ridge tuning, evaluation, and refit."""

    runs: object
    root: object
    predictors: object
    blocks: object
    library: object
    grid: tuple[float, ...]
    percentile: float
    n_jobs: int
    regularizer: Regularizer
    encoding_mode: str
    task_model: TaskModel
    selection_model: TaskModel
    selection_options: dict


def _score_block(indices, ctx, scope_runs, scope_predictors):
    options = _block_selection_options(ctx.selection_options, indices)
    return score_candidates(
        load_block(scope_runs, ctx.root, indices, ctx.task_model),
        scope_predictors,
        ctx.grid,
        ctx.library,
        [r.label for r in scope_runs],
        _signature(scope_runs, indices),
        regularizer=ctx.regularizer,
        encoding_mode=ctx.encoding_mode,
        hrf_method=options["method"],
        reml_options=options["reml_options"],
        task_model=ctx.selection_model,
    )


def _global_scores(ctx, scope_runs, scope_predictors):
    n_features = scope_runs[0].image.shape[1]
    shape = (len(scope_runs), len(ctx.grid), n_features)
    sse, sst = np.full(shape, np.nan), np.full(shape, np.nan)
    scores = np.full((len(ctx.grid), n_features), np.nan)
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
    grid_record = {ctx.regularizer.grid_name: list(ctx.grid)}
    provenance = ProvenanceRecord(
        execution_id=str(uuid4()),
        sources=[trimmed_sources(r, ctx.root, indices) for r in scope_runs],
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
        regularization=ctx.regularizer.name,
        grid=ctx.grid,
        cv_r2=scores,
        fold_sse=sse,
        fold_sst=sst,
        fold_hrf_indices=assignments,
        trial_masks=result.trial_masks,
        run_labels=tuple(r.label for r in scope_runs),
        provenance=provenance,
    )


def _select(ctx, scores):
    """Per-feature fractions, or one alpha by the percentile objective."""
    if ctx.regularizer.per_feature:
        return select_ridge_fractions(scores)
    return select_ridge_penalty(scores, percentile=ctx.percentile)


def _penalty_option(regularizer, penalty, indices=slice(None)):
    """The fit keyword for this family; per-feature penalties follow the block."""
    value = penalty[indices] if regularizer.per_feature else penalty
    return {regularizer.fit_option: value}


def _tune(ctx, train):
    scope_runs = [ctx.runs[i] for i in train]
    scope_predictors = [ctx.predictors[i] for i in train]
    scores = _global_scores(ctx, scope_runs, scope_predictors)
    selection = _select(ctx, scores)
    return TuningResult(
        scores=scores,
        selection=selection,
        run_labels=tuple(r.label for r in scope_runs),
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
            **_block_selection_options(ctx.selection_options, indices),
        )
    )
    family = ctx.regularizer
    options = _penalty_option(family, alpha, indices)
    fitted = _fit_outer_trials(data, selection, labels, signature, options)
    ids = (
        np.zeros(len(indices), dtype=int)
        if selection is None
        else selection.hrf_indices
    )
    scoring_betas = list(fitted.run_betas)
    if family.fixed_target is not None:
        # Retain training-selected HRFs; held-out responses only define OLS targets.
        fixed = np.where(np.isfinite(alpha[indices]), family.fixed_target, np.nan)
        ols = _fit_outer_trials(
            data, selection, labels, signature, {family.fit_option: fixed}
        )
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


_OUTER_MAPS = (
    "encoding_r2",
    "coefficients",
    "run_sse",
    "run_sst",
    "hrf_indices",
    "train_run_intercepts",
    "scoring_offsets",
)
_OUTER_TRIALS = ("betas", "predictions", "targets")
_OUTER_INVARIANTS = ("train_run_predictor_means", "predictor_means", "trial_masks")


def _outer_buffers(ctx, train, test):
    """NaN-filled session-wide arrays that the outer blocks fill in."""
    n = ctx.runs[0].image.shape[1]
    buffers = dict(
        train_run_intercepts=np.full((len(train), n), np.nan),
        scoring_offsets=np.full((len(test), n), np.nan),
        encoding_r2=np.full(n, np.nan),
        coefficients=np.full((1 + len(ctx.predictors[0].columns), n), np.nan),
        run_sse=np.full((len(test), n), np.nan),
        run_sst=np.full((len(test), n), np.nan),
        hrf_indices=np.full(n, -1, dtype=int),
        run_ridge_alphas=(
            np.full((len(ctx.runs), n), np.nan) if ctx.regularizer.per_feature else None
        ),
    )
    for key in _OUTER_TRIALS:
        buffers[key] = tuple(
            np.full((len(ctx.runs[r].events), n), np.nan, dtype=np.float32)
            for r in test
        )
    return buffers


def _fill_outer_block(buffers, indices, block):
    for key in _OUTER_MAPS:
        buffers[key][..., indices] = block[key]
    if buffers["run_ridge_alphas"] is not None:
        buffers["run_ridge_alphas"][:, indices] = np.asarray(block["run_ridge_alphas"])
    for key in _OUTER_TRIALS:
        for target, values in zip(buffers[key], block[key], strict=True):
            target[:, indices] = values


def _outer_record(indices, block):
    return dict(
        grayordinate_indices=indices.tolist(),
        record=block["provenance"],
        selection_record=block["selection_provenance"],
    )


def _evaluate(ctx, alpha, train, test, tuning_fingerprint):
    """Assemble one outer split; predictor statistics are the same in every block."""
    buffers, invariants, provenance = _outer_buffers(ctx, train, test), {}, []
    for indices, block in map_blocks(
        _outer_block,
        ctx.blocks,
        args=(ctx, alpha, train, test),
        n_jobs=ctx.n_jobs,
    ):
        _fill_outer_block(buffers, indices, block)
        invariants = invariants or {key: block[key] for key in _OUTER_INVARIANTS}
        provenance.append(_outer_record(indices, block))
    return OuterEvaluation(
        **buffers,
        **invariants,
        regularization=ctx.regularizer.name,
        encoding_mode=ctx.encoding_mode,
        selection_task_model=ctx.selection_model.to_dict(),
        coefficient_names=("task", *ctx.predictors[0].columns),
        train_run_labels=tuple(ctx.runs[r].label for r in train),
        test_run_labels=tuple(ctx.runs[r].label for r in test),
        ridge_alpha=None if ctx.regularizer.per_feature else alpha,
        ridge_fraction=np.array(alpha) if ctx.regularizer.per_feature else None,
        provenance=tuple(provenance),
        tuning_analysis_fingerprint=tuning_fingerprint,
    )


def _final_selection(indices, ctx):
    return SelectionBundle(
        all=select_hrfs(
            load_block(ctx.runs, ctx.root, indices, ctx.task_model),
            library=ctx.library,
            run_labels=[r.label for r in ctx.runs],
            feature_signature=_signature(ctx.runs, indices),
            task_model=ctx.selection_model,
            **_block_selection_options(ctx.selection_options, indices),
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
    options = _penalty_option(ctx.regularizer, alpha)
    return fit_beta_series(
        ctx.runs,
        ctx.root,
        ctx.blocks,
        task_model=ctx.task_model,
        selections=selections,
        **options,
        n_jobs=ctx.n_jobs,
    )


def _build_cv_context(
    runs,
    root,
    *,
    task_model,
    library,
    alphas,
    fractions,
    percentile,
    blocks,
    block_size,
    max_grayordinates,
    n_jobs,
    encoding_mode,
    selection_model,
    hrf_method,
    reml_options,
):
    validate_encoding_mode(encoding_mode)
    validate_n_jobs(n_jobs)
    _validate_percentile(percentile)
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
    regularizer = NORMALIZED_RIDGE if fractions is None else FRACTIONAL_RIDGE
    grid = regularizer.grid(alphas if fractions is None else fractions)
    halves = _halves(runs, library, hrf_method)
    if blocks is None:
        blocks = inputs.make_blocks(
            runs, block_size=block_size, max_grayordinates=max_grayordinates
        )
    predictors = trial_predictors(
        [r.events for r in runs], task_model, [r.label for r in runs]
    )
    ctx = _CvContext(
        runs=runs,
        root=root,
        predictors=predictors,
        blocks=blocks,
        library=library,
        grid=grid,
        percentile=percentile,
        n_jobs=n_jobs,
        regularizer=regularizer,
        encoding_mode=encoding_mode,
        task_model=task_model,
        selection_model=selection_model,
        selection_options=selection_options,
    )
    return ctx, halves


def fit_cv_beta_series(
    runs,
    root,
    *,
    task_model,
    library,
    alphas=None,
    fractions=None,
    percentile=90.0,
    blocks=None,
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
    the task model) scores HRFs in every inner fold, outer split and the final
    fit, for example without RT. Supply the session's blocks to share them with
    other fits; otherwise nonconstant blocks are built from block_size and
    max_grayordinates. Grayordinates outside the blocks stay undefined.
    """
    ctx, halves = _build_cv_context(
        runs,
        root,
        task_model=task_model,
        library=library,
        alphas=alphas,
        fractions=fractions,
        percentile=percentile,
        blocks=blocks,
        block_size=block_size,
        max_grayordinates=max_grayordinates,
        n_jobs=n_jobs,
        encoding_mode=encoding_mode,
        selection_model=selection_model,
        hrf_method=hrf_method,
        reml_options=reml_options,
    )
    mode = "canonical" if ctx.library is None else "optimized"
    tuning, evaluation = {}, {}
    for scope, train in (*halves.items(), ("all", list(range(len(runs))))):
        log.info("Ridge CV (%s, %s): tuning %d runs", mode, scope, len(train))
        tuning[scope] = _tune(ctx, train)
        alpha = tuning[scope].selection.penalty
        if scope != "all":
            target = "even" if scope == "odd" else "odd"
            evaluation[f"{scope}_to_{target}"] = _evaluate(
                ctx,
                alpha,
                train,
                halves[target],
                tuning[scope].provenance.analysis_fingerprint,
            )
        log.info("Ridge CV (%s, %s): %s", mode, scope, _describe(ctx, alpha))
    final_penalty = tuning["all"].selection.penalty
    return CvBetaResult(
        tuning=tuning,
        evaluation=evaluation,
        predictors=tuple(ctx.predictors),
        selection_task_model=ctx.selection_model.to_dict(),
        final=link_final_provenance(
            _final_fit(ctx, final_penalty),
            tuning["all"].provenance,
            encoding_mode=ctx.encoding_mode,
        ),
        provenance=cv_provenance(
            ctx.regularizer,
            ctx.grid,
            final_penalty,
            encoding_mode=ctx.encoding_mode,
            percentile=ctx.percentile,
            library=ctx.library,
        ),
    )


def _describe(ctx, alpha):
    if ctx.regularizer.per_feature:
        return f"selected fractions at {np.isfinite(alpha).sum()} grayordinates"
    return f"selected alpha={alpha:g}"


@dataclass(frozen=True, kw_only=True)
class BetaModel:
    """One trial-wise beta model and, for tuned ridge, how it was chosen."""

    name: str
    hrf: str
    estimator: str
    fit: BetaFit
    tuning: Mapping[str, TuningResult] | None = None
    evaluation: Mapping[str, OuterEvaluation] | None = None
    cv_provenance: Mapping[str, object] | None = None
    predictors: tuple | None = None

    @property
    def fractional(self):
        return self.fit.fractional


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


def _cv_model(runs, root, blocks, settings, library, task_model, hrf):
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
        blocks=blocks,
        n_jobs=settings.n_jobs,
    )
    estimator = _ESTIMATOR[settings.ridge_mode]
    return BetaModel(
        name=f"{hrf.capitalize()}Trial{estimator}",
        hrf=hrf,
        estimator=estimator,
        fit=result.final,
        tuning=result.tuning,
        evaluation=result.evaluation,
        cv_provenance=result.provenance,
        predictors=result.predictors,
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
            model = _cv_model(runs, root, blocks, settings, candidates, task_model, hrf)
            models[model.name] = model
    return models
