"""Self-contained HTML report: one section per stage, figures embedded as PNG."""

import base64
import html
import shlex
from dataclasses import fields
from io import BytesIO

from boldtailor.workflow.settings import (
    STAGES,
    WorkflowSettings,
    resolve_fmriprep_dir,
)

SECTIONS = (
    "settings",
    "inputs",
    "glms",
    "reliability",
    "betas",
    "summaries",
    "skipped",
    "files",
)
TITLES = dict(
    settings="Settings",
    inputs="Inputs",
    glms="Canonical and optimized GLMs",
    reliability="HRF reliability",
    betas="Single-trial beta series",
    summaries="Activation and reaction time",
    skipped="Skipped stages",
    files="Files",
)
_STYLE = (
    "body{font:15px/1.5 system-ui;margin:2rem auto;max-width:72rem;padding:0 1rem}"
    "table{border-collapse:collapse;margin:1rem 0}"
    "td,th{border:1px solid #ccc;padding:.25rem .6rem;text-align:left}"
    "img{max-width:100%}section{margin:2rem 0}h2{border-bottom:1px solid #ddd}"
    ".note{color:#555}pre{background:#f5f5f5;padding:.6rem;overflow-x:auto}"
)
_VALUE_FLAGS = (
    ("space", "--space"),
    ("hrf_library", "--hrf-library"),
    ("hrf_n_samples", "--hrf-n-samples"),
    ("hrf_seed", "--hrf-seed"),
    ("ridge_mode", "--ridge-mode"),
    ("ridge_alpha", "--ridge-alpha"),
    ("ridge_percentile", "--ridge-percentile"),
    ("encoding_mode", "--encoding-mode"),
    ("n_jobs", "--n-jobs"),
    ("block_size", "--block-size"),
    ("max_grayordinates", "--max-grayordinates"),
    ("existing_results", "--existing-results"),
)
_LIST_FLAGS = (
    ("ridge_fractions", "--ridge-fractions"),
    ("ridge_alphas", "--ridge-alphas"),
)


def embed_figure(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="figure" src="data:image/png;base64,{data}">'


def _default(name):
    return next(f.default for f in fields(WorkflowSettings) if f.name == name)


def _quoted(*words):
    return [shlex.quote(f"{w:g}" if isinstance(w, float) else str(w)) for w in words]


def _path_flags(settings):
    flags = []
    try:
        implicit_prep = resolve_fmriprep_dir(settings.bids_dir)
    except ValueError:
        implicit_prep = None
    if settings.fmriprep_dir != implicit_prep:
        flags += _quoted("--fmriprep-dir", settings.fmriprep_dir)
    implicit_out = settings.bids_dir / "derivatives" / settings.output_name()
    if settings.output_dir != implicit_out:
        flags += _quoted("--output-dir", settings.output_dir)
    return flags


def _modulator_flags(settings):
    flags = []
    for mod in settings.modulators or ():
        text = mod.column + (":indicator" if mod.missing == "indicator" else "")
        flags += _quoted("--modulator", text)
    return flags


def _value_flags(settings):
    flags = []
    for name, flag in _VALUE_FLAGS:
        if getattr(settings, name) != _default(name):
            flags += _quoted(flag, getattr(settings, name))
    for name, flag in _LIST_FLAGS:
        if getattr(settings, name) != _default(name):
            flags += _quoted(flag, *getattr(settings, name))
    return flags


def _switch_flags(settings):
    flags = []
    if not settings.hrf_selection_rt:
        flags.append("--no-rt-in-hrf-selection")
    if not settings.surface_maps:
        flags.append("--no-surface-maps")
    for stage in STAGES[1:]:
        if stage not in settings.stages:
            flags += _quoted("--skip-stage", stage)
    for side, path in (settings.surface_meshes or {}).items():
        flags += _quoted("--surface-mesh", f"{side}={path}")
    return flags


def command_line(settings):
    """The `boldtailor run` command that reproduces these settings."""
    required = _quoted(
        "--bids-dir",
        settings.bids_dir,
        "--subject",
        settings.subject,
        "--session",
        settings.session,
        "--task",
        settings.task,
    )
    optional = (
        _path_flags(settings)
        + _modulator_flags(settings)
        + _value_flags(settings)
        + _switch_flags(settings)
    )
    return " ".join(["boldtailor", "run", *required, *optional])


def _table(frame):
    if frame is None or len(frame) == 0:
        return '<p class="note">No table for this stage.</p>'
    return frame.to_html(index=False, float_format=lambda v: f"{v:.4g}", border=0)


def _section(name, title, body):
    return f'<section id="{name}"><h2>{html.escape(title)}</h2>{body}</section>'


def _settings_body(settings):
    rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in settings.to_dict().items()
    )
    command = html.escape(command_line(settings))
    return f"<table>{rows}</table><h3>Equivalent command line</h3><pre>{command}</pre>"


def _inputs_body(runs, task_model, library):
    labels = html.escape(", ".join(r.label for r in runs))
    regressors = html.escape(", ".join(task_model.regressor_names))
    items = [
        f"<li>{len(runs)} runs: {labels}</li>",
        f"<li>Task regressors: {regressors}</li>",
    ]
    if library is not None:
        items.append(
            f"<li>HRF library: {len(library.candidates)} candidates, "
            f"fingerprint {library.fingerprint[:12]}</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def _figures(figures, keys):
    return "".join(
        f"<h3>{html.escape(k)}</h3>{embed_figure(figures[k])}"
        for k in keys
        if k in figures
    )


def _skipped_body(skipped):
    if not skipped:
        return '<p class="note">Every enabled stage ran.</p>'
    items = (f"<li><b>{html.escape(s)}</b>: {html.escape(r)}</li>" for s, r in skipped)
    return "<ul>" + "".join(items) + "</ul>"


def _files_body(manifest):
    rows = []
    for path, provenance in manifest:
        href = html.escape(path)
        link = (
            f'<a href="{html.escape(provenance)}">provenance</a>' if provenance else ""
        )
        rows.append(f'<tr><td><a href="{href}">{href}</a></td><td>{link}</td></tr>')
    return f"<table><tr><th>file</th><th></th></tr>{''.join(rows)}</table>"


def _note(text):
    return f'<p class="note">{html.escape(text)}</p>'


def _stage_note(name, settings, skipped):
    """One-line note replacing a stage section that did not run, else None."""
    if name in STAGES and name not in settings.stages:
        return _note("Stage disabled.")
    reasons = dict(skipped)
    if name in STAGES and name in reasons:
        return _note(f"Skipped: {reasons[name]}.")
    return None


def _bodies(settings, data, figures):
    return {
        "settings": _settings_body(settings),
        "inputs": _inputs_body(data["runs"], data["task_model"], data["library"])
        + _figures(figures, ("Design", "Library")),
        "glms": _table(data["glm_summary"])
        + _table(data["hrf_summary"])
        + _figures(figures, ("GLMComparison", "GLMR2Surface")),
        "reliability": _table(data["reliability"])
        + _figures(figures, ("HRFReliability", "HRFCurveReliability")),
        "betas": _table(data["tuning"])
        + _figures(figures, ("FractionSelection", "RidgeTuning", "BetaR2Surface")),
        "summaries": _table(data["activation_summary"])
        + _table(data["rt_summary"])
        + _figures(
            figures, ("BetaActivation", "BetaActivationSurface", "RTCheck", "RTSurface")
        ),
        "skipped": _skipped_body(data["skipped"]),
        "files": _files_body(data["manifest"]),
    }


def render_report(
    settings,
    *,
    runs,
    task_model,
    library,
    glm_summary,
    hrf_summary,
    reliability,
    tuning,
    activation_summary,
    rt_summary,
    figures,
    skipped,
    manifest,
):
    data = dict(locals())
    bodies = _bodies(settings, data, dict(figures))
    sections = "".join(
        _section(n, TITLES[n], _stage_note(n, settings, skipped) or bodies[n])
        for n in SECTIONS
    )
    title = (
        f"boldtailor report: {settings.subject} {settings.session} task-{settings.task}"
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{sections}</body></html>"
    )
