import numpy as np
import pytest

from experiments.nsd_replication import pilot_report, run
from experiments.nsd_replication.config import ExperimentConfig

LEVELS = ("b1", "b2")


@pytest.fixture
def config(tmp_path):
    return ExperimentConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        freesurfer_dir=tmp_path,
        subjects=("sub-07",),
        sessions=("ses-a", "ses-b", "ses-c"),
        block_size=16,
    )


@pytest.fixture
def fitted(config, monkeypatch, synthetic_session):
    monkeypatch.setattr(
        run, "_load", lambda config, source, subject, session: synthetic_session
    )
    monkeypatch.setattr(run, "ppdata_brain", lambda *a: synthetic_session.brain)
    monkeypatch.setattr(
        run, "load_roi", lambda config, subject, brain: np.arange(30) < 20
    )
    for session in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", session, LEVELS)


def test_variability_columns_and_finite(config, fitted):
    table = pilot_report.variability(config, "ppdata", LEVELS)
    assert list(table.columns) == ["version", "loso_sd", "split_sd", "median"]
    assert list(table["version"]) == list(LEVELS)
    assert np.isfinite(table[["loso_sd", "split_sd", "median"]].to_numpy()).all()


def test_identical_sessions_have_negligible_spread(config, fitted):
    # every synthetic session is identical, so subsets give the same median
    table = pilot_report.variability(config, "ppdata", LEVELS)
    assert (table[["loso_sd", "split_sd"]] < 0.05).all().all()


def test_cli_writes_variability_table(config, fitted, capsys):
    import pandas as pd

    toml = config.bids_dir / "config.toml"
    toml.write_text(f"""bids_dir = "{config.bids_dir}"
output_dir = "{config.output_dir}"
freesurfer_dir = "{config.freesurfer_dir}"
subjects = ["sub-07"]
sessions = ["ses-a", "ses-b", "ses-c"]
block_size = 16
""")
    argv = ["--config", str(toml), "--source", "ppdata", "--levels", *LEVELS]
    assert pilot_report.main(argv) == 0
    path = config.output_dir / "metrics/ppdata/sub-07/variability.tsv"
    table = pd.read_csv(path, sep="\t")
    assert list(table["version"]) == list(LEVELS)
    assert "loso_sd" in capsys.readouterr().out
