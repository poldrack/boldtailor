"""Static regression: REML energy comparisons produce feature masks, not bools.

Run with pyright and the project's Python environment; this file is not a
runtime pytest test. Existing REML tests cover single and mixed feature masks.
"""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from boldtailor.reml import FixedProjection, _pooled_energy_exceeds_floor


def check_feature_mask(projections: list[FixedProjection]) -> None:
    """Both energy floors return arrays supporting elementwise selection."""
    for floor in ("rounding_floor", "uncertainty_floor"):
        mask = _pooled_energy_exceeds_floor(projections, floor)
        assert_type(mask, NDArray[np.bool_])
        mask.any()
