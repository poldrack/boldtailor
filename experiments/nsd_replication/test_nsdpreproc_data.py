"""Real NSD data: the pilot session loads from nsdpreproc only."""

from pathlib import Path

import numpy as np
import pytest

from experiments.nsd_replication.config import ExperimentConfig
from experiments.nsd_replication.inputs import (
    CORTEX,
    load_nsdpreproc,
    nsdpreproc_brain,
    same_grayordinates,
)

DATA = Path("/Volumes/extdata1/NSD/BIDS")
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="NSD data volume not mounted")


def _pilot_config():
    return ExperimentConfig(
        bids_dir=DATA,
        output_dir=DATA / "derivatives/nsd-replication",
        freesurfer_dir=DATA / "derivatives/freesurfer-NSD",
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
    )


def test_pilot_session_loads_from_nsdpreproc():
    config = _pilot_config()
    assert config.nsdpreproc_dir == DATA / "derivatives/nsdpreproc"
    session = load_nsdpreproc(config, "sub-07", "ses-nsd10")
    assert session.source == "nsdpreproc"
    assert len(session.labels) == 12
    assert all(s.shape == (226, 59412) for s in session.signals)
    assert set(session.brain.name) == set(CORTEX)
    assert "73k_id" in session.events[0]
    assert list(session.confounds[0].columns) == [f"poly_{d}" for d in range(4)]
    np.testing.assert_allclose(np.diff(session.frame_times[0]), 4 / 3, rtol=1e-5)
    assert same_grayordinates(
        nsdpreproc_brain(config, "sub-07", "ses-nsd10"), session.brain
    )
    assert session.task_model is not None
