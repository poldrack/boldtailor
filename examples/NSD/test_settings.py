"""Notebook settings resolve defaults, legacy ridge options, libraries, and paths."""

from pathlib import Path

import pytest

from boldtailor import hrf_library
from boldtailor.hrf_library import HrfLibrary
from examples.NSD.settings import beta_penalties, build_hrf_library, resolve_settings


@pytest.fixture
def config(tmp_path, monkeypatch):
    for name in ("NSD_BIDS_ROOT", "NSD_FMRIPREP_ROOT", "NSD_OUTPUT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    return dict(bids_root=str(tmp_path / "bids"), output_root=str(tmp_path / "out"))


def test_default_settings_use_the_default_library(config):
    settings = resolve_settings(config)
    library = build_hrf_library(settings)
    assert len(library.candidates) == 533  # canonical + 512 timing + 20 GLMsingle
    assert settings["hrf_seed"] == 0
    assert library.fingerprint == hrf_library.default_hrf_library().fingerprint
    assert settings["hrf_selection_rt"] is True


def test_configured_paths_are_user_expanded(config):
    settings = resolve_settings({**config, "bids_root": "~/nsd-example"})
    assert settings["bids_root"] == str(Path("~/nsd-example").expanduser())


def test_sobol_settings_set_size_and_seed(config):
    settings = resolve_settings(
        {**config, "hrf_library": "sobol", "hrf_n_samples": 8, "hrf_seed": 11}
    )
    library = build_hrf_library(settings)
    assert len(library.candidates) == 9
    assert library.fingerprint == hrf_library.sobol_hrf_library(8, seed=11).fingerprint
    settings = resolve_settings({**config, "hrf_n_samples": 8, "hrf_seed": 11})
    expected = hrf_library.default_hrf_library(8, seed=11).fingerprint
    assert build_hrf_library(settings).fingerprint == expected


def test_library_can_reproduce_the_grid_or_use_custom_rows(config):
    settings = resolve_settings({**config, "hrf_library": "expanded"})
    expected = hrf_library.expanded_hrf_library().fingerprint
    assert build_hrf_library(settings).fingerprint == expected
    rows = [[3, 10, 0.5, 0.5, 2, 0, 36]]
    settings = resolve_settings({**config, "hrf_parameters": rows, "hrf_n_samples": 3})
    expected = HrfLibrary.from_parameters(rows).fingerprint
    assert build_hrf_library(settings).fingerprint == expected


def test_unknown_library_is_rejected(config):
    with pytest.raises(ValueError, match="hrf_library"):
        build_hrf_library(resolve_settings({**config, "hrf_library": "typo"}))


@pytest.mark.parametrize(
    "overrides,mode",
    [
        ({}, "fractional_cv"),
        ({"ridge_alphas": [0, 0.1]}, "cv"),
        ({"ridge_alpha": 0.2}, "fixed"),
        ({"ridge_alpha": None}, "off"),
        ({"ridge_mode": "off"}, "off"),
        ({"ridge_mode": "cv", "ridge_alpha": None}, "cv"),
    ],
)
def test_ridge_mode_resolves_new_defaults_and_legacy_options(config, overrides, mode):
    settings = resolve_settings({**config, **overrides})
    assert settings["ridge_mode"] == mode
    assert settings["ridge_percentile"] == 90.0


def test_unknown_ridge_mode_is_rejected(config):
    with pytest.raises(ValueError, match="ridge_mode"):
        resolve_settings({**config, "ridge_mode": "misspelled"})


def test_explicit_fractional_mode_keeps_its_fraction_grid(config):
    settings = resolve_settings(
        {**config, "ridge_mode": "fractional_cv", "ridge_fractions": [0.4, 1]}
    )
    assert settings["ridge_mode"] == "fractional_cv"
    assert settings["ridge_fractions"] == [0.4, 1]


def test_configuration_is_not_modified(config):
    original = {**config, "ridge_alpha": 0.2}
    resolve_settings(original)
    assert original == {**config, "ridge_alpha": 0.2}


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"ridge_alpha": 0.4}, {"OLS": 0.0, "Ridge": 0.4}),
        ({"ridge_mode": "off"}, {"OLS": 0.0}),
        ({}, {"OLS": 0.0}),
        ({"ridge_mode": "cv"}, {"OLS": 0.0}),
    ],
)
def test_beta_penalties_add_fixed_ridge_only_in_fixed_mode(config, overrides, expected):
    assert beta_penalties(resolve_settings({**config, **overrides})) == expected


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_fixed_mode_needs_a_positive_alpha(config, alpha):
    settings = resolve_settings({**config, "ridge_mode": "fixed", "ridge_alpha": alpha})
    with pytest.raises(ValueError, match="ridge_alpha must be positive"):
        beta_penalties(settings)
