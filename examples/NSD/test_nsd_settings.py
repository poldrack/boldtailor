"""NSD notebook configurations map onto the package's WorkflowSettings."""

from pathlib import Path

import pytest

from boldtailor.workflow.settings import WorkflowSettings
from examples.NSD.nsd_settings import nsd_paths, nsd_settings


@pytest.fixture
def config(dataset, tmp_path):
    root, prep, *_ = dataset
    return dict(
        bids_root=str(root),
        fmriprep_root=str(prep),
        output_root=str(tmp_path / "output"),
        subject="sub-07",
        session="ses-nsd10",
        hrf_n_samples=4,
    )


def test_notebook_paths_become_settings_paths_and_the_task_defaults(config):
    settings = nsd_settings(config)
    assert isinstance(settings, WorkflowSettings)
    assert settings.bids_dir == Path(config["bids_root"])
    assert settings.fmriprep_dir == Path(config["fmriprep_root"])
    assert settings.output_dir == Path(config["output_root"])
    assert settings.task == "nsdcore"
    assert settings.hrf_n_samples == 4


def test_overrides_win_and_the_configuration_is_unchanged(config):
    original = dict(config)
    settings = nsd_settings(config, session="ses-nsd11", surface_maps=False)
    assert settings.session == "ses-nsd11"
    assert settings.surface_maps is False
    assert config == original


def test_settings_reject_unknown_configuration_keys(config):
    with pytest.raises(TypeError):
        nsd_settings({**config, "not_a_setting": 1})


def test_paths_come_from_the_configuration_then_the_environment(
    monkeypatch, tmp_path
):
    for name in ("NSD_BIDS_ROOT", "NSD_FMRIPREP_ROOT", "NSD_OUTPUT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="NSD_BIDS_ROOT"):
        nsd_paths({})
    monkeypatch.setenv("NSD_BIDS_ROOT", str(tmp_path))
    assert nsd_paths({}) == {"bids_root": str(tmp_path)}
    monkeypatch.setenv("NSD_OUTPUT_ROOT", str(tmp_path / "env-out"))
    paths = nsd_paths({"bids_root": "/data/bids", "fmriprep_root": "/data/prep"})
    assert paths == {
        "bids_root": "/data/bids",
        "fmriprep_root": "/data/prep",
        "output_root": str(tmp_path / "env-out"),
    }
