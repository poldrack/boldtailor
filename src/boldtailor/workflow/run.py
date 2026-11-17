"""Run the enabled stages in order and publish one BIDS derivative with a report."""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

from boldtailor.diagnostics import one_sample_t
from boldtailor.task_detection import detect_task_model, task_model_notes
from boldtailor.model import TaskModel
from boldtailor.reliability import curve_correlations
from boldtailor.workflow import (
    analysis,
    beta_series,
    hrf_maps,
    inputs,
    outputs,
    plots,
    report,
    summaries,
    surfaces,
)
from boldtailor.workflow.export_cifti import figure_name
from boldtailor.workflow.figures import tuning_figure
from boldtailor.workflow.files import min_runs_per_half, odd_even_parity
from boldtailor.workflow.settings import STAGES, WorkflowSettings
from boldtailor.workflow.tables import hrf_boundary_table, tuning_table

log = logging.getLogger("boldtailor.workflow")
SHORT_PARITY = "fewer than two odd or two even runs"
_DESCRIPTOR = re.compile(r"_desc-([A-Za-z0-9]+)_")
RIDGE_CV_PARITY = (
    "ridge cross-validation needs at least three odd and three even runs; "
    "use --ridge-mode off|fixed or --skip-stage betas"
)
RIDGE_CV_MODULATORS = (
    "ridge cross-validation scores trial-modulator encoding and needs at least "
    "one modulator; use --ridge-mode off|fixed or --skip-stage betas"
)


@dataclass(frozen=True, kw_only=True)
class WorkflowResult:
    settings: WorkflowSettings
    paths: tuple[Path, ...]
    report_path: Path | None
    skipped: tuple[tuple[str, str], ...]
    task_model: TaskModel
    library_fingerprint: str


@dataclass(frozen=True, kw_only=True)
class _Session:
    """Loaded inputs only; scientific stages never populate this object."""

    settings: WorkflowSettings
    runs: tuple
    task_model: TaskModel
    library: object
    blocks: list
    notes: tuple
    meshes: dict | None

    @property
    def brain(self):
        return self.runs[0].image.header.get_axis(1)

    def enabled(self, stage):
        return stage in self.settings.stages


@dataclass(frozen=True, kw_only=True)
class _Analysis:
    """Completed stage outputs, assembled once for reporting and publication."""

    selections: dict | None
    glms: dict
    glm_summary: pd.DataFrame | None
    reliability: pd.DataFrame | None
    tuning: pd.DataFrame | None
    beta_models: dict
    activation: dict | None
    curve_reliability: pd.DataFrame | None
    skipped: tuple


@contextmanager
def _input_errors():
    """Loading and validation failures are input errors (CLI exit code 2)."""
    try:
        yield
    except inputs.InputError:
        raise
    except ValueError as error:
        raise inputs.InputError(str(error)) from error


def _session(settings):
    """Trimmed runs, the task model, and notes on what detection left out."""
    with _input_errors():
        runs = inputs.load_session(settings, hrf_only=True)
        events = [r.events for r in runs]
        task_model = detect_task_model(
            events, settings.modulators, labels=[r.label for r in runs]
        )
        notes = task_model_notes(events, settings.modulators)
    return runs, task_model, notes


def describe_inputs(settings):
    """What a run would use (for ``--dry-run``), without fitting anything.

    ``problems`` lists what would stop the run, each with a ``kind``
    (``existing_results`` or ``input``) and a ``message``, in run order.
    """
    plan = dict(
        hrf_method=settings.hrf_method,
        reml_rho=settings.reml_rho,
        reml_reference_hrf_id=settings.reml_reference_hrf_id,
        output_dir=str(settings.output_dir),
        fmriprep_dir=str(settings.fmriprep_dir),
        library_candidates=len(settings.build_library().candidates),
        stages=[s for s in STAGES if s in settings.stages],
        notes=[],
        problems=[],
    )
    try:
        outputs.check_output(settings)
    except FileExistsError as error:
        plan["problems"].append(dict(kind="existing_results", message=str(error)))
    try:
        _update_plan_with_session(settings, plan)
    except (inputs.InputError, FileNotFoundError) as error:
        plan["problems"].append(dict(kind="input", message=str(error)))
    return plan


def _update_plan_with_session(settings, plan):
    """Update plan in place with session details, notes and validation results."""
    runs, task_model, notes = _session(settings)
    plan.update(runs=[r.label for r in runs], task_model=task_model.to_dict())
    plan["notes"] += notes
    _check_ridge_cv(settings, runs, task_model)
    _find_meshes(settings, plan["notes"])


def _load(settings):
    runs, task_model, notes = _session(settings)
    _check_ridge_cv(settings, runs, task_model)
    with _input_errors():
        blocks = inputs.make_blocks(
            runs,
            block_size=settings.block_size,
            max_grayordinates=settings.max_grayordinates,
        )
    notes = list(notes)
    meshes = _find_meshes(settings, notes)
    library = settings.build_library()
    log.info("Loaded %d runs; %d HRF candidates", len(runs), len(library.candidates))
    return _Session(
        settings=settings,
        runs=runs,
        task_model=task_model,
        library=library,
        blocks=blocks,
        notes=tuple(notes),
        meshes=meshes,
    )


def _find_meshes(settings, notes):
    """Meshes for surface figures; a missing explicit mesh is an input error."""
    if not settings.surface_maps:
        return None
    try:
        meshes = surfaces.find_surface_meshes(
            settings.fmriprep_dir, settings.subject, paths=settings.surface_meshes
        )
    except FileNotFoundError as error:
        raise inputs.InputError(f"surface mesh not found: {error}") from error
    except ValueError as error:
        meshes, reason = None, str(error)
    else:
        reason = "no fsLR 32k midthickness meshes found"
    if meshes is None:
        notes.append(f"Skipped surface figures: {reason}")
        log.info("Skipping surface figures: %s", reason)
    return meshes


def _splits_possible(runs, minimum=2):
    return all(len(half) >= minimum for half in odd_even_parity(runs).values())


def _check_ridge_cv(settings, runs, task_model):
    """Ridge CV needs three runs per half and a modulator; refuse before fitting."""
    cv = settings.ridge_mode in ("cv", "fractional_cv")
    if "betas" not in settings.stages or not cv:
        return
    minimum = min_runs_per_half(settings.hrf_method, ridge_cv=True)
    if not _splits_possible(runs, minimum):
        message = (
            RIDGE_CV_PARITY if minimum == 3 else RIDGE_CV_PARITY.replace("three", "two")
        )
        raise inputs.InputError(message)
    if not task_model.modulators:
        raise inputs.InputError(RIDGE_CV_MODULATORS)


def _reliability_runs(session):
    minimum = min_runs_per_half(session.settings.hrf_method)
    return session.enabled("reliability") and _splits_possible(session.runs, minimum)


def _select_hrfs(session):
    """Return selections required by GLMs and optional split reliability."""
    if not (session.enabled("glms") or _reliability_runs(session)):
        return None
    settings = session.settings
    return analysis.select_hrfs(
        session.runs,
        settings.bids_dir,
        session.blocks,
        session.library,
        task_model=inputs.selection_task_model(
            session.task_model, settings.hrf_selection_rt
        ),
        n_jobs=settings.n_jobs,
        splits=_reliability_runs(session),
        **settings.selection_options(),
    )


def _fit_glms(session, *, selections):
    """Fit both GLMs using explicitly supplied optimized-HRF selections."""
    if not selections:
        raise ValueError("GLM fitting requires HRF selections")
    settings, runs = session.settings, session.runs
    model = inputs.glm_model(runs, session.task_model)
    return {
        "CanonicalGLM": analysis.fit_glms(
            runs,
            settings.bids_dir,
            session.blocks,
            model,
            n_jobs=settings.n_jobs,
        ),
        "OptimizedGLM": analysis.fit_glms(
            runs,
            settings.bids_dir,
            session.blocks,
            model,
            selections=selections,
            n_jobs=settings.n_jobs,
        ),
    }


def _add_glm_figures(session, glms, figures):
    """Insert GLM figures into figures and return the comparison table."""
    figures["Design"] = plots.design_figure(
        session.runs[0].frame_times,
        glms["CanonicalGLM"].designs[0, 0],
        session.task_model.regressor_names,
    )
    summary, figures["GLMComparison"] = plots.glm_comparison(glms)
    return summary


def _add_reliability_figures(session, *, selections, figures):
    """Insert reliability figures into figures and return the summary tables."""
    if not selections:
        raise ValueError("reliability requires split HRF selections")
    maps = hrf_maps.selection_maps(selections, len(session.brain))
    odd, even = maps["odd"][0], maps["even"][0]
    reliability, figures["HRFReliability"] = plots.parameter_agreement(
        session.library, odd, even
    )
    curve_r = curve_correlations(session.library, odd, even)
    curves, figures["HRFCurveReliability"] = plots.curve_agreement(curve_r)
    return reliability, curves


def _fit_betas(session, *, selections):
    if not selections:
        raise ValueError("beta fitting requires HRF selections")
    settings = session.settings
    return beta_series.fit_beta_models(
        session.runs,
        settings.bids_dir,
        session.blocks,
        settings,
        session.library,
        selections,
        session.task_model,
    )


def _add_tuning_figures(beta_models, figures):
    """Insert tuning figures into figures and return the tuning table."""
    tuning = tuning_table(beta_models)
    figures["RidgeTuning"] = tuning_figure(beta_models)
    if "selected_grayordinates" in tuning:
        figures["FractionSelection"] = plots.fraction_selection_figure(tuning)
    return tuning


def _summarize_betas(*, beta_models):
    if not beta_models:
        raise ValueError("summaries require fitted beta models")
    return {name: one_sample_t(model.fit.betas) for name, model in beta_models.items()}


def _add_summary_figures(session, beta_models, activation, figures):
    """Insert summary figures into the caller-owned cleanup registry."""
    figures["BetaActivation"] = plots.activation_histogram(activation)
    rt_figure = plots.rt_check_figure(beta_models, session.runs, session.brain)
    if rt_figure is not None:
        figures["RTCheck"] = rt_figure


def _add_surface_figures(session, glms, beta_models, activation, figures):
    """Insert requested cortical figures into the supplied figures registry."""
    if session.meshes is None:
        return
    for name, maps, statistic in _surface_maps(glms, beta_models, activation):
        if maps:
            figures[name] = surfaces.surface_figure(
                maps,
                session.brain,
                session.meshes,
                statistic=statistic,
            )


def _surface_maps(glms, models, activation):
    activation = activation or {}
    rt = {k: m.fit["rt"]["all"] for k, m in models.items() if m.fit["rt"] is not None}
    return [
        ("GLMR2Surface", {k: v["r2"][0] for k, v in glms.items()}, "r2"),
        ("BetaR2Surface", {k: m.fit["r2"][2] for k, m in models.items()}, "delta_r2"),
        ("BetaActivationSurface", {k: a["t"] for k, a in activation.items()}, "t"),
        ("RTSurface", rt if activation else {}, "rt"),
    ]


def _analyze(session, figures):
    """Wire stage dependencies explicitly; figures is the cleanup registry."""
    selections = _select_hrfs(session)
    glms = _fit_glms(session, selections=selections) if session.enabled("glms") else {}
    glm_summary = _add_glm_figures(session, glms, figures) if glms else None
    reliability, curves = (
        _add_reliability_figures(session, selections=selections, figures=figures)
        if _reliability_runs(session)
        else (None, None)
    )
    skipped = ()
    if session.enabled("reliability") and not _reliability_runs(session):
        reason = (
            SHORT_PARITY
            if session.settings.hrf_method == "cv"
            else "an empty odd or even run set"
        )
        skipped = (("reliability", reason),)
        log.info("Skipping reliability: %s", reason)
    models = (
        _fit_betas(session, selections=selections) if session.enabled("betas") else {}
    )
    tuning = (
        _add_tuning_figures(models, figures)
        if any(m.tuning for m in models.values())
        else None
    )
    activation = (
        _summarize_betas(beta_models=models) if session.enabled("summaries") else None
    )
    if activation is not None:
        _add_summary_figures(session, models, activation, figures)
    _add_surface_figures(session, glms, models, activation, figures)
    return _Analysis(
        selections=selections,
        glms=glms,
        glm_summary=glm_summary,
        reliability=reliability,
        curve_reliability=curves,
        beta_models=models,
        tuning=tuning,
        activation=activation,
        skipped=skipped,
    )


def _provenance(path, stem, listed):
    """The provenance JSON sharing this file's descriptor, when one was written."""
    match = _DESCRIPTOR.search(path)
    if match is None:
        return None
    descriptor = match[1]
    for name in (descriptor, "HRF" if descriptor.startswith("HRF") else None):
        candidate = f"{stem}_desc-{name}_provenance.json"
        if name and candidate != path and candidate in listed:
            return candidate
    return None


def _manifest(paths, stem):
    listed = set(paths)
    return [(p, _provenance(p, stem, listed)) for p in paths]


def _stage_tables(session, results):
    return dict(
        selected_hrfs=summaries.selected_hrfs(
            results.selections, session.library, len(session.brain)
        ),
        curve_reliability=results.curve_reliability,
        encoding=summaries.encoding_scores(results.beta_models),
        ridge_boundary=summaries.ridge_boundary(results.beta_models),
        activation_summary=summaries.activation_summary(results.activation),
        rt_summary=summaries.rt_summary(results.activation, results.beta_models),
    )


def _render(session, results, figures, artifacts):
    """The report, embedding the figure PNGs exactly as published."""
    payloads = {a.path: a.payload for a in artifacts}
    settings = session.settings
    return report.render_report(
        settings,
        runs=session.runs,
        task_model=session.task_model,
        library=session.library,
        glm_summary=results.glm_summary,
        hrf_summary=hrf_boundary_table(results.selections),
        reml_diagnostics=summaries.reml_diagnostics(results.selections),
        reliability=results.reliability,
        tuning=results.tuning,
        figures={name: payloads[figure_name(settings, name)] for name in figures},
        skipped=results.skipped,
        notes=session.notes,
        manifest=_manifest([a.path for a in artifacts], settings.stem),
        **summaries.input_tables(session.runs, session.task_model, session.library),
        **_stage_tables(session, results),
    )


def _publish(session, results, figures):
    """Write every artifact and, last in the same publication, the report."""
    settings = session.settings
    artifacts = outputs.workflow_artifacts(
        settings,
        session.runs,
        session.task_model,
        session.library,
        results.selections,
        results.glms,
        results.beta_models,
        figures=figures,
        activation=results.activation,
        skipped=results.skipped,
        notes=session.notes,
        report=outputs.report_name(settings),
        include_hrf_splits=_reliability_runs(session),
    )
    html = _render(session, results, figures, artifacts)
    return outputs.publish_workflow(settings, session.runs, artifacts, html.encode())


def _close(figures):
    for figure in figures.values():
        plt.close(figure)


def run_workflow(settings):
    """Run enabled stages with explicit dependencies, then publish their results."""
    outputs.check_output(settings)
    session = _load(settings)
    figures = {}
    try:
        figures["Library"] = plots.library_figure(session.library)
        results = _analyze(session, figures)
        paths = _publish(session, results, figures)
    finally:
        _close(figures)
    return WorkflowResult(
        settings=settings,
        paths=tuple(paths),
        report_path=settings.output_dir / outputs.report_name(settings),
        skipped=results.skipped,
        task_model=session.task_model,
        library_fingerprint=session.library.fingerprint,
    )
