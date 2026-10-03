"""Run the enabled stages in order and publish one BIDS derivative with a report."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boldtailor.diagnostics import one_sample_t
from boldtailor.reliability import curve_correlations
from boldtailor.workflow import (
    analysis,
    beta_series,
    inputs,
    outputs,
    plots,
    report,
    surfaces,
)
from boldtailor.workflow.files import odd_even_parity
from boldtailor.workflow.settings import STAGES, WorkflowSettings

log = logging.getLogger("boldtailor.workflow")
SHORT_PARITY = "fewer than two odd or two even runs"
_DESCRIPTOR = re.compile(r"_desc-([A-Za-z0-9]+)_")


@dataclass(frozen=True, kw_only=True)
class WorkflowResult:
    settings: WorkflowSettings
    paths: tuple[Path, ...]
    report_path: Path | None
    skipped: tuple[tuple[str, str], ...]
    task_model: object
    library_fingerprint: str


@dataclass(kw_only=True)
class _State:
    """The session inputs and every stage's results, filled in stage order."""

    settings: WorkflowSettings
    runs: tuple
    task_model: object
    library: object
    blocks: list
    selections: dict | None = None
    glms: dict = field(default_factory=dict)
    glm_summary: pd.DataFrame | None = None
    reliability: pd.DataFrame | None = None
    tuning: pd.DataFrame | None = None
    beta_models: dict = field(default_factory=dict)
    activation: dict | None = None
    figures: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)

    @property
    def brain(self):
        return self.runs[0].image.header.get_axis(1)

    def enabled(self, stage):
        return stage in self.settings.stages


def describe_inputs(settings):
    """What a run would use (for ``--dry-run``), without fitting anything."""
    runs = inputs.load_session(settings, hrf_only=True)
    task_model = inputs.detect_task_model([r.events for r in runs], settings.modulators)
    return dict(
        runs=[r.label for r in runs],
        task_model=task_model.to_dict(),
        output_dir=str(settings.output_dir),
        fmriprep_dir=str(settings.fmriprep_dir),
        library_candidates=len(settings.build_library().candidates),
        stages=[s for s in STAGES if s in settings.stages],
    )


def _load(settings):
    runs = inputs.load_session(settings, hrf_only=True)
    task_model = inputs.detect_task_model([r.events for r in runs], settings.modulators)
    blocks = inputs.make_blocks(
        runs,
        block_size=settings.block_size,
        max_grayordinates=settings.max_grayordinates,
    )
    library = settings.build_library()
    log.info("Loaded %d runs; %d HRF candidates", len(runs), len(library.candidates))
    return _State(
        settings=settings,
        runs=runs,
        task_model=task_model,
        library=library,
        blocks=blocks,
    )


def _splits_possible(runs):
    return all(len(half) >= 2 for half in odd_even_parity(runs).values())


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
    _, state.figures["HRFCurveReliability"] = plots.curve_agreement(curve_r)


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
    settings = state.settings
    if not settings.surface_maps:
        return
    meshes = surfaces.find_surface_meshes(
        settings.fmriprep_dir, settings.subject, paths=settings.surface_meshes
    )
    if meshes is None:
        log.info("No fsLR surface meshes found; skipping surface figures")
        return
    for name, maps, statistic in _surface_maps(state):
        if maps:
            state.figures[name] = surfaces.surface_figure(
                maps, state.brain, meshes, statistic=statistic
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


def _median(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _activation_summary(activation):
    if not activation:
        return None
    rows = [dict(model=k, median_t=_median(v["t"])) for k, v in activation.items()]
    return pd.DataFrame(rows)


def _rt_summary(state):
    if state.activation is None:
        return None
    rows = [
        dict(model=k, scope=s, median_r=_median(m.fit["rt"][s]))
        for k, m in state.beta_models.items()
        if m.fit["rt"] is not None
        for s in ("all", "odd", "even")
    ]
    return pd.DataFrame(rows) if rows else None


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


def _render(state, paths):
    return report.render_report(
        state.settings,
        runs=state.runs,
        task_model=state.task_model,
        library=state.library,
        glm_summary=state.glm_summary,
        hrf_summary=outputs.hrf_boundary_table(state.selections),
        reliability=state.reliability,
        tuning=state.tuning,
        activation_summary=_activation_summary(state.activation),
        rt_summary=_rt_summary(state),
        figures=state.figures,
        skipped=state.skipped,
        manifest=_manifest(paths, state.settings.stem),
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
        report=outputs.report_name(settings),
        include_hrf_splits=_reliability_runs(state),
    )
    html = _render(state, [a.path for a in artifacts])
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
