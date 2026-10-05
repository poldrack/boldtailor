from pathlib import Path

import pytest

from experiments.nsd_replication.config import ExperimentConfig, load_config

TOML = """
bids_dir = "/data/BIDS"
output_dir = "/data/out"
freesurfer_dir = "/data/freesurfer"
subjects = ["sub-07"]
sessions = ["ses-nsd10", "ses-nsd11"]
n_jobs = 2
"""


def test_load_config_converts_paths_and_tuples(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(TOML)
    config = load_config(path)
    assert config.bids_dir == Path("/data/BIDS")
    assert config.subjects == ("sub-07",)
    assert config.sessions == ("ses-nsd10", "ses-nsd11")
    assert config.n_jobs == 2
    assert config.image_column == "73k_id"


@pytest.mark.parametrize(
    "field, value",
    [("subjects", ()), ("subjects", ("07",)), ("sessions", ("ses-1", "ses-1"))],
)
def test_bad_labels_rejected(field, value):
    values = dict(
        bids_dir=Path("/b"),
        output_dir=Path("/o"),
        freesurfer_dir=Path("/l"),
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
    )
    values[field] = value
    with pytest.raises(ValueError, match=field):
        ExperimentConfig(**values)


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(TOML + 'colour = "red"\n')
    with pytest.raises(ValueError, match="colour"):
        load_config(path)
