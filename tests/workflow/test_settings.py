"""WorkflowSettings: defaults, derived paths, validation, round trip."""

import pytest

from boldtailor.hrf_library import default_hrf_library
from boldtailor.model import Modulator
from boldtailor.workflow.report import modulator_text
from boldtailor.workflow.settings import (
    WorkflowSettings,
    parse_modulator,
    resolve_fmriprep_dir,
)


@pytest.fixture
def bids(tmp_path):
    root = tmp_path / "bids"
    (root / "derivatives" / "fmriprep-25.2.5").mkdir(parents=True)
    return root


def required(bids, **kwargs):
    identity = dict(subject="sub-07", session="ses-nsd10", task="nsdcore")
    return WorkflowSettings(bids_dir=bids, **{**identity, **kwargs})


def test_defaults_derive_fmriprep_and_output_directories(bids):
    settings = required(bids)
    assert settings.fmriprep_dir == bids / "derivatives" / "fmriprep-25.2.5"
    assert settings.output_name() == "boldtailor_hrf-default512s0_ridge-fractionalcv"
    assert settings.output_dir == bids / "derivatives" / settings.output_name()
    assert settings.stem == "sub-07/ses-nsd10/func/sub-07_ses-nsd10_task-nsdcore"
    assert settings.space_entity == "space-fsLR_den-91k"
    assert settings.stages == frozenset({"glms", "reliability", "betas", "summaries"})


@pytest.mark.parametrize(
    "kwargs, name",
    [
        (
            dict(hrf_library="sobol", hrf_n_samples=8, hrf_seed=3, ridge_mode="cv"),
            "boldtailor_hrf-sobol8s3_ridge-cv",
        ),
        (
            dict(hrf_library="expanded", ridge_mode="fixed", ridge_alpha=0.1),
            "boldtailor_hrf-expanded_ridge-fixed0.1",
        ),
        (
            dict(hrf_library="canonical", ridge_mode="off"),
            "boldtailor_hrf-canonical_ridge-off",
        ),
    ],
)
def test_output_name_encodes_library_and_ridge_mode(bids, kwargs, name):
    assert required(bids, **kwargs).output_name() == name


def test_explicit_paths_are_expanded_and_kept(bids, tmp_path):
    out = tmp_path / "elsewhere"
    settings = required(
        bids, fmriprep_dir=bids / "derivatives" / "fmriprep-25.2.5", output_dir=out
    )
    assert settings.output_dir == out


def test_fmriprep_resolution_requires_a_unique_directory(tmp_path):
    root = tmp_path / "bids"
    (root / "derivatives").mkdir(parents=True)
    with pytest.raises(ValueError, match="no derivatives/fmriprep"):
        resolve_fmriprep_dir(root)
    (root / "derivatives" / "fmriprep-24.0").mkdir()
    assert resolve_fmriprep_dir(root) == root / "derivatives" / "fmriprep-24.0"
    (root / "derivatives" / "fmriprep-25.2.5").mkdir()
    with pytest.raises(ValueError, match="fmriprep-24.0.*fmriprep-25.2.5"):
        resolve_fmriprep_dir(root)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(space="MNI152NLin2009cAsym"), "space"),
        (dict(hrf_library="grid"), "hrf_library"),
        (dict(ridge_mode="lasso"), "ridge_mode"),
        (dict(ridge_mode="fixed", ridge_alpha=0.0), "ridge_alpha"),
        (dict(ridge_percentile=101), "ridge_percentile"),
        (dict(encoding_mode="pooled"), "encoding_mode"),
        (dict(stages=frozenset({"betas"})), "betas.*glms"),
        (dict(stages=frozenset({"glms", "summaries"})), "summaries.*betas"),
        (dict(stages=frozenset({"plots"})), "stages"),
        (dict(n_jobs=0), "n_jobs"),
        (dict(block_size=0), "block_size"),
        (dict(max_grayordinates=0), "max_grayordinates"),
        (dict(existing_results="reuse"), "existing_results"),
        (dict(hrf_n_samples=12), "power of two"),
        (dict(subject="07"), "subject"),
        (dict(session="nsd10"), "session"),
        (dict(task="nsd core"), "task"),
    ],
)
def test_invalid_settings_name_the_field(bids, kwargs, message):
    with pytest.raises(ValueError, match=message):
        required(bids, **kwargs)


def test_missing_bids_dir_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="bids_dir"):
        WorkflowSettings(
            bids_dir=tmp_path / "nope", subject="sub-01", session="ses-01", task="t"
        )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("trial_type", Modulator("trial_type")),
        ("response_time:indicator", Modulator("response_time", missing="indicator")),
        ("trial_type:categorical", Modulator("trial_type", kind="categorical")),
        (
            "trial_type:categorical,reference=face",
            Modulator("trial_type", kind="categorical", reference="face"),
        ),
        (
            "c:categorical,reference=a=b",
            Modulator("c", kind="categorical", reference="a=b"),
        ),
        (
            "cond:indicator,categorical",
            Modulator("cond", kind="categorical", missing="indicator"),
        ),
    ],
)
def test_parse_modulator_options(text, expected):
    assert parse_modulator(text) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        ":indicator",
        "rt:centered",
        "rt:indicator,indicator",
        "rt:reference=a",
        "rt:categorical,reference=",
        "rt:a:b",
        "rt:indicator,",
    ],
)
def test_parse_modulator_rejects_bad_options(bad):
    with pytest.raises(ValueError, match="modulator"):
        parse_modulator(bad)


@pytest.mark.parametrize("level", ["a,b", "a:b"])
def test_modulator_text_rejects_unexpressible_reference(level):
    mod = Modulator("cond", kind="categorical", reference=level)
    with pytest.raises(ValueError, match="modulator reference"):
        modulator_text(mod)


@pytest.mark.parametrize(
    "mod",
    [
        Modulator("c", kind="categorical", reference="a,b"),
        Modulator("c", kind="categorical", reference="a:b"),
        Modulator("a:b"),
    ],
)
def test_settings_reject_modulators_the_cli_cannot_express(bids, mod):
    with pytest.raises(ValueError, match="modulators"):
        required(bids, modulators=(mod,))


@pytest.mark.parametrize(
    "mod",
    [
        Modulator("rt"),
        Modulator("rt", missing="indicator"),
        Modulator("cond", kind="categorical", reference="face", missing="indicator"),
    ],
)
def test_modulator_text_inverts_parse_modulator(mod):
    assert parse_modulator(modulator_text(mod)) == mod


def test_round_trip_through_dict_and_library_builder(bids):
    settings = required(
        bids,
        modulators=(Modulator("response_time", missing="indicator"),),
        hrf_n_samples=8,
        stages=frozenset({"glms"}),
    )
    values = settings.to_dict()
    assert values["modulators"] == [
        {"column": "response_time", "missing": "indicator", "kind": "numeric"}
    ]
    assert values["stages"] == ["glms"]
    assert WorkflowSettings.from_dict(values) == settings
    assert (
        settings.build_library().fingerprint
        == default_hrf_library(8, seed=0).fingerprint
    )
    assert len(required(bids, hrf_library="canonical").build_library().candidates) == 1


@pytest.mark.parametrize(
    "where",
    [
        lambda bids, prep: prep,
        lambda bids, prep: prep / "boldtailor",
        lambda bids, prep: bids / "derivatives",
        lambda bids, prep: bids,
    ],
    ids=["fmriprep", "inside-fmriprep", "contains-fmriprep", "bids"],
)
def test_output_dir_may_not_overlap_the_inputs(bids, where):
    prep = bids / "derivatives" / "fmriprep-25.2.5"
    with pytest.raises(ValueError, match="output_dir"):
        required(bids, output_dir=where(bids, prep))


def test_output_dir_overlap_is_checked_on_resolved_paths(bids):
    alias = bids.parent / "alias"
    alias.symlink_to(bids / "derivatives" / "fmriprep-25.2.5")
    with pytest.raises(ValueError, match="output_dir"):
        required(bids, output_dir=alias)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ridge_fractions=()),
        dict(ridge_fractions=(0.5, 0.5)),
        dict(ridge_fractions=(0.0, 1.0)),
        dict(ridge_fractions=(0.5, 1.5)),
        dict(ridge_fractions=(float("nan"),)),
        dict(ridge_alphas=()),
        dict(ridge_alphas=(1.0, 1.0)),
        dict(ridge_alphas=(-1.0, 1.0)),
        dict(ridge_alphas=(float("inf"),)),
    ],
)
def test_ridge_grids_are_validated(bids, kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        required(bids, **kwargs)
