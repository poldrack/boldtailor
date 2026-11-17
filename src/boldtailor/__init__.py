"""Boldtailor: Optimized first-level fMRI modeling."""

from boldtailor.data import AnalysisData, from_arrays
from boldtailor.fit import fit, task_delta_r2
from boldtailor.hrf_library import HrfCandidate, HrfLibrary
from boldtailor.model import ModelSpec, TaskModel
from boldtailor.single_trial import fit_single_trials

__version__ = "0.1.0"

__all__ = [
    "AnalysisData",
    "HrfCandidate",
    "HrfLibrary",
    "ModelSpec",
    "TaskModel",
    "fit",
    "fit_single_trials",
    "from_arrays",
    "task_delta_r2",
]
