"""REML settings reach selection, maps, reports, and published derivatives."""

from dataclasses import replace
import json
import shlex

import nibabel as nib
import numpy as np
import pytest

from boldtailor.cli import build_parser, settings_from_args
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.workflow import analysis, inputs, outputs, report
from boldtailor.workflow.run import describe_inputs, run_workflow
from boldtailor.workflow.settings import WorkflowSettings
from boldtailor.task_detection import detect_task_model


def test_reml_settings_roundtrip_and_cli(dataset, settings_for):
    """Method-specific settings survive serialization and the report command line."""
    root, *_ = dataset
    cv = settings_for(root, ridge_mode="off", hrf_library="sobol", hrf_n_samples=2)
    settings = replace(
        cv, output_dir=None, hrf_method="reml", reml_rho=0.2, reml_reference_hrf_id=1
    )
    restored = WorkflowSettings.from_dict(settings.to_dict())
    assert restored.to_dict() == settings.to_dict()
    assert settings.output_dir != cv.output_dir
    assert (
        replace(settings, output_dir=None, reml_rho=0.3).output_dir
        != settings.output_dir
    )
    assert (
        replace(settings, output_dir=None, reml_reference_hrf_id=0).output_dir
        != settings.output_dir
    )
    args = build_parser().parse_args(shlex.split(report.command_line(settings))[1:])
    parsed = settings_from_args(args)
    assert parsed.hrf_method == "reml" and parsed.reml_rho == 0.2
    assert parsed.reml_reference_hrf_id == 1
    plan = describe_inputs(settings)
    assert not plan["problems"]
    assert plan["hrf_method"] == "reml"
    assert plan["reml_rho"] == 0.2
    legacy = cv.to_dict()
    for key in ("hrf_method", "reml_rho", "reml_reference_hrf_id"):
        legacy.pop(key)
    assert WorkflowSettings.from_dict(legacy).hrf_method == "cv"


@pytest.mark.parametrize(
    "changes",
    [
        dict(reml_rho=0.2),
        dict(reml_reference_hrf_id=0),
        dict(hrf_method="invalid"),
        dict(hrf_method="reml", reml_rho=1.0),
        dict(hrf_method="reml", reml_rho=True),
        dict(hrf_method="reml", reml_rho=[0.1]),
        dict(hrf_method="reml", hrf_library="canonical", reml_reference_hrf_id=1),
    ],
)
def test_reml_settings_reject_invalid_combinations(dataset, settings_for, changes):
    """Settings reject unsupported or ignored likelihood parameters before fitting."""
    root, *_ = dataset
    with pytest.raises(ValueError, match="reml|rho|reference|method|REML"):
        settings_for(root, **changes)


def test_reml_run_minima_are_method_specific(dataset, four_runs, settings_for):
    """REML supports one run per reliability half and two per nested-ridge half."""
    root, _ = four_runs
    reml = settings_for(
        root,
        hrf_method="reml",
        hrf_library="canonical",
        ridge_mode="fractional_cv",
        ridge_fractions=(0.5, 1.0),
    )
    assert not describe_inputs(reml)["problems"]
    cv = replace(reml, hrf_method="cv", reml_rho=None, reml_reference_hrf_id=None)
    assert describe_inputs(cv)["problems"]


def test_reml_blocks_preserve_rho_and_diagnostics(
    four_runs, settings_for, small_library
):
    """Feature partitions and parallel workers cannot change likelihood selection."""
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model = detect_task_model([r.events for r in runs])
    rho = np.linspace(0.05, 0.2, runs[0].image.shape[1])
    options = dict(
        task_model=model,
        splits=False,
        method="reml",
        reml_options=RemlHrfOptions(rho=rho),
    )
    indices = np.arange(len(rho))
    whole = analysis.select_hrfs(runs, root, [indices], small_library, **options)
    parts = analysis.select_hrfs(
        runs, root, [indices[:1], indices[1:]], small_library, n_jobs=2, **options
    )
    expected = outputs._all_selection_maps(whole, len(rho))["all"]
    actual = outputs._all_selection_maps(parts, len(rho))["all"]
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-8)
    for block, bundle in parts.items():
        np.testing.assert_array_equal(
            bundle["all"].reml.selection.fit.rho, rho[list(block)]
        )


@pytest.mark.integration
def test_reml_workflow_publishes_likelihood_maps_and_report(
    four_runs, settings_for, tmp_path
):
    """A complete REML workflow exports likelihood diagnostics without CV labels."""
    root, _ = four_runs
    settings = settings_for(
        root,
        hrf_method="reml",
        reml_rho=0.1,
        hrf_library="sobol",
        hrf_n_samples=2,
        ridge_mode="fractional_cv",
        ridge_fractions=(0.5, 1.0),
        output_dir=tmp_path / "reml-out",
    )
    result = run_workflow(settings)
    selection_path = next(
        p for p in result.paths if "desc-HRFAll_stat-selection" in p.name
    )
    image = nib.load(selection_path)
    names = list(image.header.get_axis(0).name)
    assert "criterion" in names and "canonical_criterion" in names
    assert "delta_criterion" in names and "criterion_gap" in names
    assert "noise_variance" in names and "effect_variance" in names
    assert "variance_ratio" in names and "rho" in names
    assert "any_unrefined" in names and "any_at_upper" in names
    assert "unrefined" in names and "at_upper" in names
    assert "status" in names and "canonical_status" in names
    assert not any("cv_r2" in name for name in names)
    values = image.get_fdata()
    assert np.isnan(values[:, -1]).all()  # constant grayordinate is never processed
    provenance_path = next(
        p for p in result.paths if p.name.endswith("desc-HRF_provenance.json")
    )
    records = json.loads(provenance_path.read_text())
    for block in records:
        for scope in block["scopes"].values():
            assert scope["selection_provenance"]["activities"][-1]["method"] == "reml"
            assert scope["reml"]["candidate_ids"] == [0, 1, 2]
            assert "no_residual_signal" in scope["reml"]["status_codes"]
    html = result.report_path.read_text()
    assert "REML selection diagnostics" in html
    assert "provisional" in html and "unsupported" in html
    assert "--hrf-method reml" in html
    assert any("HRFOdd_stat-selection" in p.name for p in result.paths)
    assert any(
        "OptimizedTrialFractionalCV_stat-ridgefraction" in p.name for p in result.paths
    )
