"""Deterministic in-library and off-grid ground truths."""

import numpy as np
from boldtailor.hrf_library import HrfCandidate, default_hrf_library


def truth_library(library=None):
    library = library or default_hrf_library()
    parametric = [k for k in library.candidates if k.kind == "double_gamma"]
    empirical = [k for k in library.candidates if k.kind == "glmsingle"]

    def peak(k):
        return int(np.argmax(k.kernel(0.1, 1)))

    return dict(
        canonical=library.candidates[0],
        fast=min(parametric, key=peak) if parametric else library.candidates[0],
        slow=max(parametric, key=peak) if parametric else library.candidates[0],
        empirical=empirical[9] if len(empirical) > 9 else library.candidates[0],
        offgrid_fast=HrfCandidate(
            0, "double_gamma", (3.713, 11.281, 0.917, 1.217, 5.413, 0.0, 36.0)
        ),
        offgrid_slow=HrfCandidate(
            0, "double_gamma", (7.137, 17.731, 1.173, 1.417, 6.813, 0.0, 36.0)
        ),
        offgrid=HrfCandidate(
            0, "double_gamma", (4.137, 12.731, 0.873, 1.417, 5.813, 0.0, 36.0)
        ),
    )
