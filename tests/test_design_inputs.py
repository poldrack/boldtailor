"""Run-level input validation shared by every design builder."""

import numpy as np
import pandas as pd
import pytest

from boldtailor import _design_inputs
from boldtailor._constants import MIN_ONSET

TIMES = np.arange(10.0)


def test_onsets_inside_the_sampled_window_are_accepted():
    """Onsets from the earliest supported time up to the last frame pass."""
    _design_inputs.validate_onsets([TIMES[0] + MIN_ONSET, 0.0, 8.5], TIMES)


@pytest.mark.parametrize("onset", [TIMES[0] + MIN_ONSET - 1e-6, TIMES[-1], 50.0])
def test_onsets_outside_the_sampled_window_are_rejected(onset):
    """An onset before the supported window or at/after the last frame is an error."""
    with pytest.raises(ValueError, match="^onset has no supported sampled response$"):
        _design_inputs.validate_onsets([1.0, onset], TIMES)


def test_onset_errors_name_the_subject_supplied_by_the_caller():
    """Each design builder labels the failure with what it was convolving."""
    with pytest.raises(ValueError, match="^cue: onset has no supported sampled"):
        _design_inputs.validate_onsets([99.0], TIMES, subject="cue: onset")


def test_design_inputs_expose_event_and_nuisance_validation():
    """Event checks and the nuisance matrix have one public, leaf-level home."""
    events = pd.DataFrame({"onset": [1.0], "duration": [1.0]})
    _design_inputs.validate_events(events, TIMES, "run-1")
    nuisance = _design_inputs.nuisance_matrix(pd.DataFrame({"drift": TIMES}), 10)
    assert list(nuisance.columns) == ["drift", "constant"]
    with pytest.raises(ValueError, match="reserved trial-table names"):
        _design_inputs.validate_events(events.assign(trial_id="a"), TIMES, "run-1")
