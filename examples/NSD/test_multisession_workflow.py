"""Missing sessions run the package workflow once; completed sessions are reused."""

import nibabel as nib
import pytest

from boldtailor.workflow.settings import WorkflowSettings
from examples.NSD import (
    multisession_analysis,
    multisession_inputs,
    multisession_outputs,
    multisession_workflow,
)


def test_read_only_mode_reports_missing_sessions_without_fitting(tmp_path):
    with pytest.raises(FileNotFoundError, match="ses-nsd10"):
        multisession_workflow.ensure_session_outputs(
            dict(bids_root=str(tmp_path), output_root=str(tmp_path / "output")),
            ["ses-nsd10", "ses-nsd11"],
            estimators=["OLS"],
            fit_missing=False,
        )


def test_multisession_export_preserves_input_files(saved_sessions):
    output, sessions, _, _ = saved_sessions
    loaded = multisession_inputs.load_sessions(
        output, "sub-07", sessions, estimators=["OLS"]
    )
    results = multisession_analysis.analyze_sessions(loaded)
    source = loaded["records"][0]["sources"][0]
    before = source.read_bytes()
    paths = multisession_outputs.save_multisession(output, loaded, results)
    assert paths and source.read_bytes() == before
    # Rerunning aggregate publication is supported without recomputing sessions.
    assert multisession_outputs.save_multisession(output, loaded, results) == paths


def test_incompatible_estimators_fail_before_fitting(saved_sessions, monkeypatch):
    root, sessions, _, _ = saved_sessions
    module = multisession_workflow

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid estimator configuration must fail before fitting")

    monkeypatch.setattr(module, "run_workflow", forbidden)
    config = dict(bids_root=str(root), output_root=str(root))
    with pytest.raises(ValueError, match="estimators.*ridge_mode"):
        module.ensure_session_outputs(
            config, sessions, estimators=["OLS", "FractionalCV"]
        )
    with pytest.raises(ValueError, match="estimators"):
        module.ensure_session_outputs(
            config, sessions, estimators=["RidgeCV", "FractionalCV"]
        )


def test_complete_sessions_are_validated_before_missing_session_fits(
    saved_sessions,
    monkeypatch,
):
    root, sessions, _, _ = saved_sessions
    module = multisession_workflow
    changed = nib.cifti2.BrainModelAxis.from_surface([2, 0, 3], 5, "CortexLeft")
    for path in (root / "sub-07" / sessions[1]).rglob("*.dscalar.nii"):
        image = nib.load(path)
        nib.save(
            nib.Cifti2Image(
                image.get_fdata(),
                header=nib.Cifti2Header.from_axes((image.header.get_axis(0), changed)),
            ),
            path,
        )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Incompatible completed results must be checked before missing fits"
        )

    monkeypatch.setattr(module, "run_workflow", forbidden)
    with pytest.raises(ValueError, match="grayordinate axes"):
        module.ensure_session_outputs(
            dict(bids_root=str(root), output_root=str(root)),
            ["ses-nsd99", *sessions],
            estimators=["OLS"],
        )


def _copy_saved_session(root, source, target):
    origin = root / "sub-07" / source / "func"
    destination = root / "sub-07" / target / "func"
    destination.mkdir(parents=True)
    for path in origin.iterdir():
        name = path.name.replace(source, target)
        (destination / name).write_bytes(path.read_bytes())


def test_missing_sessions_run_the_package_workflow_with_overwrite(
    saved_sessions, monkeypatch
):
    root, sessions, _, _ = saved_sessions
    calls = []

    def run_workflow(settings):
        calls.append(settings)
        _copy_saved_session(root, sessions[0], settings.session)

    monkeypatch.setattr(multisession_workflow, "run_workflow", run_workflow)
    config = dict(
        bids_root=str(root), fmriprep_root=str(root), output_root=str(root), n_jobs=1
    )
    status = multisession_workflow.ensure_session_outputs(
        config, [*sessions, "ses-nsd99"], estimators=["OLS"]
    )
    assert status.status.tolist() == ["reused", "reused", "reused", "fitted"]
    (settings,) = calls
    assert isinstance(settings, WorkflowSettings)
    assert (settings.subject, settings.session, settings.task) == (
        "sub-07",
        "ses-nsd99",
        "nsdcore",
    )
    assert settings.bids_dir == settings.output_dir == root
    assert settings.surface_maps is False
    assert settings.existing_results == "overwrite"
    assert settings.n_jobs == 1
    # Scientific settings inherit from the completed sessions.
    assert settings.ridge_mode == "off"


def test_conflicting_settings_stop_before_the_workflow_runs(
    saved_sessions, monkeypatch
):
    root, sessions, _, _ = saved_sessions

    def forbidden(settings):
        pytest.fail("Conflicting settings must fail before fitting")

    monkeypatch.setattr(multisession_workflow, "run_workflow", forbidden)
    config = dict(
        bids_root=str(root),
        fmriprep_root=str(root),
        output_root=str(root),
        ridge_mode="fixed",
    )
    with pytest.raises(ValueError, match="ridge_mode"):
        multisession_workflow.ensure_session_outputs(
            config, [*sessions, "ses-nsd99"], estimators=["OLS"]
        )


def test_paths_resolve_from_the_configuration_and_default_the_output_root(dataset):
    root, prep, *_ = dataset
    paths = multisession_workflow.resolve_paths(
        dict(bids_root=str(root), hrf_n_samples=4), ["ses-nsd10"]
    )
    assert paths["bids_root"] == str(root)
    expected = root / "derivatives" / "boldtailor_hrf-default4s0_ridge-fractionalcv"
    assert paths["output_root"] == str(expected)
    explicit = multisession_workflow.resolve_paths(
        dict(bids_root=str(root), output_root="/elsewhere"), ["ses-nsd10"]
    )
    assert explicit["output_root"] == "/elsewhere"
