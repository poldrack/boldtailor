"""Run the enabled stages in order and publish one BIDS derivative with a report."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

from boldtailor.diagnostics import one_sample_t
from boldtailor.model import TaskModel
from boldtailor.reliability import curve_correlations
from boldtailor.workflow import (
    analysis,
    beta_series,
    inputs,
    outputs,
    plots,
    report,
    summaries,
    surfaces,
)
from boldtailor.workflow.files import odd_even_parity
from boldtailor.workflow.settings import STAGES, WorkflowSettings

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


@dataclass(kw_only=True)
class _State:
    """The session inputs and every stage's results, filled in stage order."""

    settings: WorkflowSettings
    runs: tuple
    task_model: TaskModel
    library: object
    blocks: list
    selections: dict | None = None
    glms: dict = field(default_factory=dict)
    glm_summary: pd.DataFrame | None = None
    reliability: pd.DataFrame | None = None
    tuning: pd.DataFrame | None = None
    beta_models: dict = field(default_factory=dict)
    activation: dict | None = None
    curve_reliability: pd.DataFrame | None = None
    figures: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    meshes: dict | None = None

    @property
    def brain(self):
        return self.runs[0].image.header.get_axis(1)

    def enabled(self, stage):
        return stage in self.settings.stages


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
        task_model = inputs.detect_task_model(
            events, settings.modulators, labels=[r.label for r in runs]
        )
    return runs, task_model, inputs.task_model_notes(events, settings.modulators)


def describe_inputs(settings):
    """What a run would use (for ``--dry-run``), without fitting anything.

    ``problems`` lists what would stop the run, each with a ``kind``
    (``existing_results`` or ``input``) and a ``message``, in run order.
    """
    plan = dict(
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
        _describe_session(settings, plan)
    except (inputs.InputError, FileNotFoundError) as error:
        plan["problems"].append(dict(kind="input", message=str(error)))
    return plan


def _describe_session(settings, plan):
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
    return _State(
        settings=settings,
        runs=runs,
        task_model=task_model,
        library=library,
        blocks=blocks,
        notes=notes,
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
    if not _splits_possible(runs, 3):
        raise inputs.InputError(RIDGE_CV_PARITY)
    if not task_model.modulators:
        raise inputs.InputError(RIDGE_CV_MODULATORS)


def _reliability_runs(state):
    return state.enabled("reliability") and _splits_possible(state.runs)


def _hrf_selection(state):
    """All-run selections, with odd/even splits only for the reliability stage."""
    if not (state.enabled("glms") or _reliability_runs(state)):
        return
    settings = state.settings
    state.selections = analysis.select_hrfs(
        state.runs,
        settings.bids_dir,
        state.blocks,
        state.library,
        task_model=inputs.selection_task_model(
            state.task_model, settings.hrf_selection_rt
        ),
        n_jobs=settings.n_jobs,
        splits=_reliability_runs(state),
    )


def _library_figure(state):
    state.figures["Library"] = plots.library_figure(state.library)


def _glm_stage(state):
    if not state.enabled("glms"):
        return
    settings, runs = state.settings, state.runs
    model = inputs.glm_model(runs, state.task_model)
    fit = dict(n_jobs=settings.n_jobs)
    state.glms = {
        "CanonicalGLM": analysis.fit_glms(
            runs, settings.bids_dir, state.blocks, model, **fit
        ),
        "OptimizedGLM": analysis.fit_glms(
            runs,
            settings.bids_dir,
            state.blocks,
            model,
            selections=state.selections,
            **fit,
        ),
    }
    _glm_figures(state)


def _glm_figures(state):
    state.figures["Design"] = plots.design_figure(
        state.runs[0].frame_times,
        state.glms["CanonicalGLM"]["designs"][0, 0],
        state.task_model.regressor_names,
    )
    state.glm_summary, state.figures["GLMComparison"] = plots.glm_comparison(state.glms)


def _reliability_stage(state):
    if not state.enabled("reliability"):
        return
    if not _splits_possible(state.runs):
        state.skipped.append(("reliability", SHORT_PARITY))
        log.info("Skipping reliability: %s", SHORT_PARITY)
        return
    maps = analysis.selection_maps(state.selections, len(state.brain))
    odd, even = maps["odd"][0], maps["even"][0]
    state.reliability, state.figures["HRFReliability"] = plots.parameter_agreement(
        state.library, odd, even
    )
    curve_r = curve_correlations(state.library, odd, even)
    state.curve_reliability, state.figures["HRFCurveReliability"] = (
        plots.curve_agreement(curve_r)
    )


def _beta_stage(state):
    if not state.enabled("betas"):
        return
    settings = state.settings
    state.beta_models = beta_series.fit_beta_models(
        state.runs,
        settings.bids_dir,
        state.blocks,
        settings,
        state.library,
        state.selections,
        state.task_model,
    )
    if any(m.tuning for m in state.beta_models.values()):
        _tuning_figures(state)


def _tuning_figures(state):
    state.tuning = outputs.tuning_table(state.beta_models)
    state.figures["RidgeTuning"] = outputs.tuning_figure(state.beta_models)
    if "selected_grayordinates" in state.tuning:
        state.figures["FractionSelection"] = plots.fraction_selection_figure(
            state.tuning
        )


def _summaries_stage(state):
    if not state.enabled("summaries"):
        return
    state.activation = {
        name: one_sample_t(m.fit["betas"]) for name, m in state.beta_models.items()
    }
    state.figures["BetaActivation"] = plots.activation_histogram(state.activation)
    rt_figure = plots.rt_check_figure(state.beta_models, state.runs, state.brain)
    if rt_figure is not None:
        state.figures["RTCheck"] = rt_figure


def _surface_figures(state):
    """Cortical maps, only when requested and the subject's meshes are found."""
    if state.meshes is None:
        return
    for name, maps, statistic in _surface_maps(state):
        if maps:
            state.figures[name] = surfaces.surface_figure(
                maps, state.brain, state.meshes, statistic=statistic
            )


def _surface_maps(state):
    models, activation = state.beta_models, state.activation or {}
    rt = {k: m.fit["rt"]["all"] for k, m in models.items() if m.fit["rt"] is not None}
    return [
        ("GLMR2Surface", {k: v["r2"][0] for k, v in state.glms.items()}, "r2"),
        ("BetaR2Surface", {k: m.fit["r2"][2] for k, m in models.items()}, "delta_r2"),
        ("BetaActivationSurface", {k: a["t"] for k, a in activation.items()}, "t"),
        ("RTSurface", rt if activation else {}, "rt"),
    ]


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


def _stage_tables(state):
    return dict(
        selected_hrfs=summaries.selected_hrfs(
            state.selections, state.library, len(state.brain)
        ),
        curve_reliability=state.curve_reliability,
        encoding=summaries.encoding_scores(state.beta_models),
        ridge_boundary=summaries.ridge_boundary(state.beta_models),
        activation_summary=summaries.activation_summary(state.activation),
        rt_summary=summaries.rt_summary(state.activation, state.beta_models),
    )


def _render(state, artifacts):
    """The report, embedding the figure PNGs exactly as published."""
    payloads = {a.path: a.payload for a in artifacts}
    settings = state.settings
    return report.render_report(
        settings,
        runs=state.runs,
        task_model=state.task_model,
        library=state.library,
        glm_summary=state.glm_summary,
        hrf_summary=outputs.hrf_boundary_table(state.selections),
        reliability=state.reliability,
        tuning=state.tuning,
        figures={
            name: payloads[outputs.figure_name(settings, name)]
            for name in state.figures
        },
        skipped=state.skipped,
        notes=state.notes,
        manifest=_manifest([a.path for a in artifacts], settings.stem),
        **summaries.input_tables(state.runs, state.task_model, state.library),
        **_stage_tables(state),
    )


def _publish(state):
    """Write every artifact and, last in the same publication, the report."""
    settings = state.settings
    artifacts = outputs.workflow_artifacts(
        settings,
        state.runs,
        state.task_model,
        state.library,
        state.selections,
        state.glms,
        state.beta_models,
        figures=state.figures,
        activation=state.activation,
        skipped=state.skipped,
        notes=state.notes,
        report=outputs.report_name(settings),
        include_hrf_splits=_reliability_runs(state),
    )
    html = _render(state, artifacts)
    return outputs.publish_workflow(settings, state.runs, artifacts, html.encode())


_STAGES = (
    _library_figure,
    _hrf_selection,
    _glm_stage,
    _reliability_stage,
    _beta_stage,
    _summaries_stage,
    _surface_figures,
)


def _close(figures):
    for figure in figures.values():
        plt.close(figure)


def run_workflow(settings):
    """Run the enabled stages and publish the derivative; refuse existing outputs."""
    outputs.check_output(settings)
    state = _load(settings)
    try:
        for stage in _STAGES:
            stage(state)
        paths = _publish(state)
    finally:
        _close(state.figures)
    return WorkflowResult(
        settings=settings,
        paths=tuple(paths),
        report_path=settings.output_dir / outputs.report_name(settings),
        skipped=tuple(state.skipped),
        task_model=state.task_model,
        library_fingerprint=state.library.fingerprint,
    )
