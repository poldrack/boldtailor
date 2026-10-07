from pathlib import Path

import pytest

from experiments.nsd_replication.config import (
    ExperimentConfig,
    load_config,
    select_sessions,
    session_summary,
)

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


@pytest.fixture
def four_sessions():
    return ExperimentConfig(
        bids_dir=Path("/b"),
        output_dir=Path("/o"),
        freesurfer_dir=Path("/l"),
        subjects=("sub-07",),
        sessions=("ses-nsd01", "ses-nsd02", "ses-nsd03", "ses-nsd04"),
    )


def test_configured_sessions_default_to_sessions(four_sessions):
    assert four_sessions.configured_sessions == four_sessions.sessions


def test_select_sessions_keeps_config_order(four_sessions):
    selected = select_sessions(four_sessions, ["ses-nsd03", "ses-nsd01"])
    assert selected.sessions == ("ses-nsd01", "ses-nsd03")
    assert selected.configured_sessions == four_sessions.sessions


def test_select_sessions_without_selection_keeps_all(four_sessions):
    assert select_sessions(four_sessions, None) == four_sessions


def test_select_sessions_rejects_unknown(four_sessions):
    with pytest.raises(ValueError, match="ses-nsd09.*not in the config"):
        select_sessions(four_sessions, ["ses-nsd01", "ses-nsd09"])


def test_select_sessions_rejects_duplicates(four_sessions):
    with pytest.raises(ValueError, match="sessions"):
        select_sessions(four_sessions, ["ses-nsd01", "ses-nsd01"])


def test_selected_sessions_must_be_configured():
    with pytest.raises(ValueError, match="configured_sessions"):
        ExperimentConfig(
            bids_dir=Path("/b"),
            output_dir=Path("/o"),
            freesurfer_dir=Path("/l"),
            subjects=("sub-07",),
            sessions=("ses-nsd05",),
            configured_sessions=("ses-nsd01",),
        )


def test_configured_sessions_not_a_toml_key(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(TOML + 'configured_sessions = ["ses-nsd10"]\n')
    with pytest.raises(ValueError, match="configured_sessions"):
        load_config(path)


def test_session_summary(four_sessions):
    selected = select_sessions(four_sessions, ["ses-nsd01", "ses-nsd03"])
    assert session_summary(selected) == "sessions: ses-nsd01, ses-nsd03 (2 of 4 in config)"
    assert session_summary(four_sessions).endswith("(4 of 4 in config)")
