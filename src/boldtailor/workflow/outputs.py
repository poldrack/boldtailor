"""Publish every workflow stage's maps, designs, tables, and provenance together."""

from boldtailor.publication import Artifact, publish_artifact_set
from boldtailor.workflow.artifacts import dataset_description, json_artifact
from boldtailor.workflow.export_cifti import (
    activation_artifacts,
    beta_model_artifacts,
    figure_artifacts,
    glm_artifacts,
    hrf_artifacts,
    input_artifacts,
)
from boldtailor.workflow.files import input_paths
from boldtailor.workflow.manifest import metadata

LOCK_TIMEOUT = 24 * 3600.0  # another session may hold the writer lock for long


def _session_files(settings):
    """This subject/session/task's files in its func directory, and nothing else."""
    directory = settings.output_dir / settings.subject / settings.session / "func"
    pattern = f"{settings.subject}_{settings.session}_task-{settings.task}_*"
    return [p for p in directory.glob(pattern) if p.is_file() and not p.is_symlink()]


def _names(paths, shown=5):
    names = sorted(p.name for p in paths)
    more = f", ... {len(names)} files" if len(names) > shown else ""
    return ", ".join(names[:shown]) + more


def check_output(settings):
    """Refuse to replace this subject/session/task's outputs unless told to."""
    if settings.existing_results == "overwrite":
        return None
    report = settings.output_dir / report_name(settings)
    existing = _session_files(settings) + ([report] if report.is_file() else [])
    if existing:
        raise FileExistsError(
            f"existing files in {settings.output_dir} ({_names(existing)}); set "
            "existing_results to overwrite, or choose a new output_dir"
        )
    return None


def metadata_name(settings):
    """The settings file's path relative to the derivative root."""
    return f"{settings.stem}_desc-boldtailor_metadata.json"


def report_name(settings):
    """The report's path relative to the derivative root."""
    return f"{settings.subject}_{settings.session}_task-{settings.task}_report.html"


def _stage_artifacts(
    settings, runs, task_model, library, selections, glms, beta_models, **options
):
    brain = runs[0].image.header.get_axis(1)
    artifacts = glm_artifacts(settings, brain, glms, runs, task_model)
    if selections:
        artifacts.extend(
            hrf_artifacts(
                settings,
                brain,
                selections,
                library,
                include_splits=options["include_hrf_splits"],
            )
        )
    for model in beta_models.values():
        artifacts.extend(beta_model_artifacts(settings, brain, runs, model, library))
    if options["activation"]:
        artifacts.extend(activation_artifacts(settings, brain, options["activation"]))
    return artifacts + input_artifacts(settings, runs, task_model)


def workflow_artifacts(
    settings,
    runs,
    task_model,
    library,
    selections,
    glms,
    beta_models,
    *,
    figures,
    activation,
    skipped,
    report,
    include_hrf_splits=True,
    notes=(),
):
    """Everything ``save_workflow`` writes except the report and dataset description.

    ``report`` is the report name recorded in the settings file (or None);
    ``include_hrf_splits`` adds the odd/even HRF artifacts. Figures stay intact.
    """
    artifacts = _stage_artifacts(
        settings,
        runs,
        task_model,
        library,
        selections,
        glms,
        beta_models,
        activation=activation,
        include_hrf_splits=include_hrf_splits,
    ) + figure_artifacts(settings, figures)
    listed = [a.path for a in artifacts] + [metadata_name(settings)]
    settings_file = metadata(
        runs,
        library,
        settings,
        task_model,
        beta_models=beta_models,
        activation=activation,
        selections=selections,
        skipped=skipped,
        report=report,
        artifacts=listed + ([report] if report else []),
        notes=notes,
    )
    return artifacts + [json_artifact(metadata_name(settings), settings_file)]


def _root_artifacts(report_html, settings):
    """The report and the dataset description (kept if another session wrote it)."""
    report = (
        [] if report_html is None else [Artifact(report_name(settings), report_html)]
    )
    return [*report, dataset_description("boldtailor")]


def publish_workflow(settings, runs, artifacts, report_html):
    """Publish ``workflow_artifacts`` with the report and dataset description at once."""
    overwrite = settings.existing_results == "overwrite"
    sources = [p for r in runs for p in input_paths(r.inputs)]
    paths = publish_artifact_set(
        settings.output_dir,
        [*artifacts, *_root_artifacts(report_html, settings)],
        source_paths=sources,
        overwrite=overwrite,
        lock_timeout=LOCK_TIMEOUT,
        keep_existing=() if overwrite else ("dataset_description.json",),
        manifest_path=metadata_name(settings) if overwrite else None,
    )
    return paths


def save_workflow(
    settings,
    runs,
    task_model,
    library,
    selections,
    glms,
    beta_models,
    *,
    figures,
    activation,
    skipped,
    report_html,
    include_hrf_splits=True,
):
    """Publish every stage, the settings file, figures, and the report at once.

    ``include_hrf_splits`` writes the odd/even HRF artifacts; pass whether the
    reliability stage ran.
    """
    artifacts = workflow_artifacts(
        settings,
        runs,
        task_model,
        library,
        selections,
        glms,
        beta_models,
        figures=figures,
        activation=activation,
        skipped=skipped,
        report=None if report_html is None else report_name(settings),
        include_hrf_splits=include_hrf_splits,
    )
    return publish_workflow(settings, runs, artifacts, report_html)
