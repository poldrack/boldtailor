from dataclasses import replace

import numpy as np
import pytest

from experiments.nsd_replication import calibrate_offset
from experiments.nsd_replication.config import ExperimentConfig
from experiments.nsd_replication.ladder import fit_ladder

OFFSETS = (-1.0, 0.0, 1.0)


@pytest.fixture
def config(tmp_path):
    return ExperimentConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        freesurfer_dir=tmp_path,
        released_dir=tmp_path / "released",
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
        block_size=16,
    )


def _shifted(session, offset):
    times = tuple(t + offset for t in session.frame_times)
    return replace(session, frame_times=times)


@pytest.fixture
def patched(monkeypatch, synthetic_session):
    """ppdata at onset_offset o has frame times shifted by o; released is offset 0."""
    released = fit_ladder(synthetic_session, levels=("b1",), block_size=16)["b1"].betas
    monkeypatch.setattr(
        calibrate_offset,
        "load_ppdata",
        lambda config, subject, session: _shifted(
            synthetic_session, config.onset_offset
        ),
    )
    monkeypatch.setattr(
        calibrate_offset, "load_released", lambda path, brain, n, name: released
    )
    monkeypatch.setattr(
        calibrate_offset, "load_roi", lambda config, subject, brain: np.arange(30) < 20
    )


def test_calibrate_has_one_row_per_offset(config, patched):
    table = calibrate_offset.calibrate(config, "sub-07", "ses-nsd10", OFFSETS)
    assert list(table.columns) == ["onset_offset", "true_median", "null_median"]
    assert list(table["onset_offset"]) == list(OFFSETS)


def test_best_offset_recovers_the_true_shift(config, patched):
    table = calibrate_offset.calibrate(config, "sub-07", "ses-nsd10", OFFSETS)
    assert calibrate_offset.best_offset(table) == 0.0
    best = table.set_index("onset_offset").loc[0.0]
    assert best["true_median"] > best["null_median"]


def test_main_writes_table(config, patched, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(calibrate_offset, "load_config", lambda path: config)
    argv = ["--config", "c.toml", "--subject", "sub-07", "--session", "ses-nsd10"]
    calibrate_offset.main(argv)
    out = config.output_dir / "calibration/sub-07_ses-nsd10_onset_offset.tsv"
    assert out.is_file()
    assert "best onset_offset" in capsys.readouterr().out
