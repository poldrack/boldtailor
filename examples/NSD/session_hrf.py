"""Estimate or reuse one independent HRF selection for each NSD session."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from boldtailor.workflow import analysis
from boldtailor.workflow.inputs import (
    detect_task_model,
    load_session,
    make_blocks,
    selection_task_model,
)
from .nsd_settings import nsd_settings
from .session_hrf_cache import (
    MAP_NAMES,
    cache_paths,
    load_cache,
    request_id,
    request_metadata,
    save_cache,
)
from .session_hrf_import import find_workflow_estimate


@dataclass(frozen=True)
class SessionEstimate:
    session: str
    brain: object
    maps: np.ndarray
    request_id: str
    cache_path: Path
    reused_from: str | None


def fit_session(
    runs,
    root,
    library,
    *,
    task_model,
    block_size=4096,
    max_grayordinates=None,
    n_jobs=1,
):
    """All-run selection maps and per-block provenance for one session."""
    blocks = make_blocks(
        runs, block_size=block_size, max_grayordinates=max_grayordinates
    )
    selections = analysis.select_hrfs(
        runs, root, blocks, library, task_model=task_model, n_jobs=n_jobs, splits=False
    )
    maps = np.full((len(MAP_NAMES), runs[0].image.shape[1]), np.nan)
    provenance = []
    for indices, bundle in selections.items():
        result = bundle["all"]
        maps[:, list(indices)] = np.vstack(
            [
                np.where(result.hrf_indices >= 0, result.hrf_indices, np.nan),
                result.cv_r2,
                result.canonical_cv_r2,
                result.delta_cv_r2,
            ]
        )
        provenance.append(
            dict(
                grayordinate_indices=[int(i) for i in indices],
                record=result.provenance.to_dict(),
            )
        )
    return maps, provenance


def _session_settings(root, prep, output, subject, session, options):
    config = dict(bids_root=root, output_root=output, subject=subject)
    if prep is not None:
        config["fmriprep_root"] = prep
    return nsd_settings(config, session=session, **options)


def _limit(settings, brain):
    maximum = settings.max_grayordinates
    return len(brain) if maximum is None else min(maximum, len(brain))


def _request(settings, runs, library):
    """The cache request, plus the full GLM model a workflow export used."""
    brain = runs[0].image.header.get_axis(1)
    task_model = detect_task_model([r.events for r in runs], settings.modulators)
    selection = selection_task_model(task_model, settings.hrf_selection_rt)
    limit = int(_limit(settings, brain))
    request = request_metadata(runs, settings.bids_dir, library, limit, selection)
    return request, task_model, selection


def _reuse_cache(settings, roots, paths, request, brain, library):
    for directory in dict.fromkeys(roots):
        maps = load_cache(directory, paths, request, brain, library)
        if maps is not None:
            source = directory / paths["selection"]
            print(f"Reused HRF estimates: {settings.session} ({source})", flush=True)
            return SessionEstimate(
                settings.session,
                brain,
                maps,
                request_id(request),
                directory / paths["metadata"],
                str(source),
            )
    return None


def _fit_or_import(settings, runs, library, roots, request, models):
    task_model, selection = models
    imported = find_workflow_estimate(
        roots, runs, settings, library, request, task_model
    )
    if imported is not None:
        maps, provenance, source = imported
        print(f"Reused HRF estimates: {settings.session} ({source})", flush=True)
        return maps, provenance, source
    print(
        f"Fitting HRFs: {settings.session}, {len(runs)} runs, "
        f"{request['grayordinate_limit']:,} grayordinates",
        flush=True,
    )
    maps, provenance = fit_session(
        runs,
        settings.bids_dir,
        library,
        task_model=selection,
        block_size=settings.block_size,
        max_grayordinates=settings.max_grayordinates,
        n_jobs=settings.n_jobs,
    )
    return maps, provenance, None


def _session_estimate(settings, runs, library, reuse_roots):
    brain = runs[0].image.header.get_axis(1)
    request, *models = _request(settings, runs, library)
    identity = request_id(request)
    paths = cache_paths(settings.subject, settings.session, identity)
    roots = [settings.output_dir, *map(Path, reuse_roots)]
    cached = _reuse_cache(settings, roots, paths, request, brain, library)
    if cached is not None:
        return cached
    maps, provenance, source = _fit_or_import(
        settings, runs, library, roots, request, models
    )
    if _request(settings, runs, library)[0] != request:
        raise ValueError(
            "Input files changed while estimating HRFs; no cache was saved"
        )
    output = settings.output_dir
    save_cache(output, paths, request, runs, library, maps, provenance, source)
    print(f"Saved HRF estimates: {settings.session}", flush=True)
    return SessionEstimate(
        settings.session, brain, maps, identity, output / paths["metadata"], source
    )


def _check_sessions(sessions):
    if not sessions or len(sessions) != len(set(sessions)):
        raise ValueError("sessions must be nonempty, unique BIDS session labels")


def estimate_sessions(
    root,
    prep,
    output,
    *,
    library,
    sessions=tuple(f"ses-nsd{i}" for i in range(10, 20)),
    subject="sub-07",
    block_size=4096,
    max_grayordinates=None,
    n_jobs=4,
    reuse_roots=(),
    include_rt=True,
):
    """Preflight all sessions, then fit only incompatible or missing estimates.

    Sessions are processed sequentially; grayordinate blocks run in parallel.
    The cache key covers source identities, library, geometry and fit settings.
    ``prep`` may be None to use the dataset's derivatives/fmriprep* directory.
    """
    sessions = tuple(sessions)
    _check_sessions(sessions)
    options = dict(
        block_size=block_size,
        max_grayordinates=max_grayordinates,
        n_jobs=n_jobs,
        hrf_selection_rt=include_rt,
    )
    settings = [
        _session_settings(root, prep, output, subject, s, options) for s in sessions
    ]
    all_runs = [load_session(s, hrf_only=True) for s in settings]
    brain = all_runs[0][0].image.header.get_axis(1)
    if any(r.image.header.get_axis(1) != brain for runs in all_runs for r in runs):
        raise ValueError(
            "All sessions must have identical grayordinate BrainModel axes"
        )
    return [
        _session_estimate(s, runs, library, reuse_roots)
        for s, runs in zip(settings, all_runs, strict=True)
    ]
