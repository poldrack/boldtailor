from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunInputs:
    session: str
    events: Path
    bold: Path
    mask: Path
    confounds: Path


def discover_run_inputs(
    bids_root: Path,
    fmriprep_root: Path,
    *,
    subject: str,
    session: str,
    task: str,
    run: str,
    space: str,
    resolution: int,
) -> RunInputs:
    raw_func = Path(bids_root) / subject / session / "func"
    derivative_func = Path(fmriprep_root) / subject / session / "func"
    stem = f"{subject}_{session}_task-{task}_{run}"
    return RunInputs(
        session=session,
        events=_one(raw_func.glob(f"{stem}*_events.tsv"), "events"),
        bold=_one(
            derivative_func.glob(
                f"{stem}_space-{space}_res-{resolution}_desc-preproc_bold.nii.gz"
            ),
            "BOLD",
        ),
        mask=_one(
            derivative_func.glob(
                f"{stem}_space-{space}_res-{resolution}_desc-brain_mask.nii.gz"
            ),
            "mask",
        ),
        confounds=_one(
            derivative_func.glob(f"{stem}_desc-confounds_timeseries.tsv"),
            "confounds",
        ),
    )


def _one(paths, role: str) -> Path:
    matches = tuple(sorted(paths))
    if not matches:
        raise FileNotFoundError(f"{role} discovery expected exactly one file")
    if len(matches) != 1:
        raise ValueError(f"{role} discovery expected exactly one file")
    return matches[0]
