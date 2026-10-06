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
    assert config.onset_offset == 0.0
    assert config.ppdata_dir is None


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


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_onset_offset_must_be_finite(value):
    with pytest.raises(ValueError, match="onset_offset"):
        ExperimentConfig(
            bids_dir=Path("/b"),
            output_dir=Path("/o"),
            freesurfer_dir=Path("/l"),
            subjects=("sub-07",),
            sessions=("ses-nsd10",),
            onset_offset=value,
        )


@pytest.mark.parametrize(
    "name,subjects",
    [
        ("primary", ("sub-01", "sub-02", "sub-03", "sub-04")),
        ("extension", ("sub-05", "sub-06", "sub-08")),
    ],
)
def test_shipped_configs(name, subjects):
    path = Path(__file__).parent / "configs" / f"{name}.toml"
    config = load_config(path)
    assert config.subjects == subjects
    assert config.sessions == tuple(f"ses-nsd{i:02d}" for i in range(1, 11))
    assert config.n_jobs == 4
    assert config.released_dir is not None
