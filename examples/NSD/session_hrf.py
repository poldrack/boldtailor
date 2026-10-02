"""Estimate or reuse one independent HRF selection for each NSD session."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import re

import numpy as np

from boldtailor.hrf_selection import select_hrf
from .nsd_hrf import spatial_signature
from .parallel_blocks import map_blocks, validate_n_jobs
from .session_hrf_cache import (
    cache_paths,
    load_cache,
    request_id,
    request_metadata,
    save_cache,
)
from .session_hrf_import import find_workflow_estimate
from .workflow_inputs import (
    NSD_TASK_MODEL,
    load_session,
    load_block,
    make_blocks,
    selection_task_model,
)


@dataclass(frozen=True)
class SessionEstimate:
    session: str
    brain: object
    maps: np.ndarray
    request_id: str
    cache_path: Path
    reused_from: str | None


def _fit_block(indices, runs, root, library, task_model):
    return select_hrf(
        load_block(runs, root, indices),
        library=library,
        run_labels=[r.label for r in runs],
        feature_signature=spatial_signature(runs[0].image.header.get_axis(1), indices),
        task_model=task_model,
    )


def fit_session(
    runs,
    root,
    library,
    *,
    block_size=4096,
    max_grayordinates=None,
    n_jobs=1,
    task_model=NSD_TASK_MODEL,
):
    blocks = make_blocks(
        runs, block_size=block_size, max_grayordinates=max_grayordinates
    )
    maps = np.full((4, runs[0].image.shape[1]), np.nan)
    provenance = []
    for indices, result in map_blocks(
        _fit_block, blocks, args=(runs, root, library, task_model), n_jobs=n_jobs
    ):
        maps[:, indices] = np.vstack(
            [
                np.where(result.hrf_indices >= 0, result.hrf_indices, np.nan),
                result.cv_r2,
                result.canonical_cv_r2,
                result.delta_cv_r2,
            ]
        )
        provenance.append(
            dict(
                grayordinate_indices=indices.tolist(),
                record=result.provenance.to_dict(),
            )
        )
    return maps, provenance


def _validate_settings(subject, sessions, block_size, max_grayordinates, n_jobs):
    if not re.fullmatch(r"sub-[A-Za-z0-9]+", subject):
        raise ValueError("subject must be a BIDS subject label")
    if (
        not sessions
        or len(sessions) != len(set(sessions))
        or any(not re.fullmatch(r"ses-[A-Za-z0-9]+", s) for s in sessions)
    ):
        raise ValueError("sessions must be nonempty, unique BIDS session labels")
    for name, value in (
        ("block_size", block_size),
        ("max_grayordinates", max_grayordinates),
    ):
        if value is None and name == "max_grayordinates":
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    validate_n_jobs(n_jobs)


def _session_estimate(
    runs, root, output, library, subject, session, reuse_roots, options
):
    brain = runs[0].image.header.get_axis(1)
    maximum = options["max_grayordinates"]
    limit = len(brain) if maximum is None else min(maximum, len(brain))
    task_model = options.get("task_model", NSD_TASK_MODEL)
    request = request_metadata(runs, root, library, int(limit), task_model)
    identity = request_id(request)
    paths = cache_paths(subject, session, identity)
    roots = [Path(output), *map(Path, reuse_roots)]
    for directory in dict.fromkeys(roots):
        maps = load_cache(directory, paths, request, brain, library)
        if maps is not None:
            source = directory / paths["selection"]
            print(f"Reused HRF estimates: {session} ({source})", flush=True)
            return SessionEstimate(
                session,
                brain,
                maps,
                identity,
                directory / paths["metadata"],
                str(source),
            )
    imported = find_workflow_estimate(
        roots, runs, root, library, request, subject, session
    )
    if imported is None:
        print(
            f"Fitting HRFs: {session}, {len(runs)} runs, {limit:,} grayordinates",
            flush=True,
        )
        maps, provenance = fit_session(runs, root, library, **options)
        source = None
    else:
        maps, provenance, source = imported
        print(f"Reused HRF estimates: {session} ({source})", flush=True)
    if request_metadata(runs, root, library, int(limit), task_model) != request:
        raise ValueError(
            "Input files changed while estimating HRFs; no cache was saved"
        )
    save_cache(output, paths, request, runs, library, maps, provenance, source)
    print(f"Saved HRF estimates: {session}", flush=True)
    return SessionEstimate(
        session, brain, maps, identity, Path(output) / paths["metadata"], source
    )


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
    Old grid/untrimmed command-line results are not mixed into this analysis.
    """
    sessions = tuple(sessions)
    _validate_settings(subject, sessions, block_size, max_grayordinates, n_jobs)
    root, prep = Path(root), Path(prep)
    all_runs = [
        load_session(root, prep, subject=subject, session=s, hrf_only=True)
        for s in sessions
    ]
    brain = all_runs[0][0].image.header.get_axis(1)
    if any(r.image.header.get_axis(1) != brain for runs in all_runs for r in runs):
        raise ValueError(
            "All sessions must have identical grayordinate BrainModel axes"
        )
    options = dict(
        block_size=block_size,
        max_grayordinates=max_grayordinates,
        n_jobs=n_jobs,
        task_model=selection_task_model(include_rt),
    )
    return [
        _session_estimate(
            runs, root, output, library, subject, session, reuse_roots, options
        )
        for session, runs in zip(sessions, all_runs, strict=True)
    ]
