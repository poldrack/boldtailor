"""Validate identified spatial assignments shared by GLM fitting workflows."""

from hashlib import sha256

import numpy as np

from boldtailor.hrf_results import HrfSelectionResult


def validate_selection(data, selection, signature):
    if not isinstance(selection, HrfSelectionResult):
        raise ValueError("selection must be an HrfSelectionResult")
    if signature != selection.feature_signature:
        raise ValueError("feature_signature must match selection")
    ids = selection.hrf_indices
    if ids.shape != (data.n_features,):
        raise ValueError("selection feature count must match data")
    activity = selection.provenance.to_dict()["activities"][-1]
    assignment = sha256(ids.astype("<i8").tobytes()).hexdigest()
    if activity.get("library_fingerprint") != selection.library.fingerprint:
        raise ValueError("selection library fingerprint does not match identity")
    if np.any(ids < -1) or np.any(ids >= len(selection.library.candidates)):
        raise ValueError("selection contains invalid HRF IDs")
    if activity.get("hrf_assignment_fingerprint") != assignment:
        raise ValueError("selection HRF assignment identity does not match provenance")
    return assignment
