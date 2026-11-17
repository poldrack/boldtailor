"""Boldtailor: Optimized first-level fMRI modeling."""

from boldtailor.data import AnalysisData, from_arrays
from boldtailor.denoising import select_denoising, with_denoising
from boldtailor.fit import fit, task_delta_r2
from boldtailor.hrf_library import HrfCandidate, HrfLibrary
from boldtailor.model import ModelSpec, TaskModel
from boldtailor.reml import (
    RemlFit,
    RemlGroup,
    RemlSelection,
    fit_reml,
    select_random_effects_design,
)
from boldtailor.single_trial import fit_single_trials

__version__ = "0.1.0"

__all__ = [
    "AnalysisData",
    "HrfCandidate",
    "HrfLibrary",
    "ModelSpec",
    "RemlFit",
    "RemlGroup",
    "RemlSelection",
    "TaskModel",
    "fit",
    "fit_reml",
    "fit_single_trials",
    "from_arrays",
    "select_denoising",
    "select_random_effects_design",
    "task_delta_r2",
    "with_denoising",
]
