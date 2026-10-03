"""Leaf home of the design constants; import them through ``_hrf_design``.

They live here only because ``hrf_library`` sits below ``_hrf_design`` in the
import graph and also needs them.
"""

OVERSAMPLING = 50
MIN_ONSET = -24.0
TIE_TOLERANCE = 1e-12
