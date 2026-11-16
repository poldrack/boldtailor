"""Execute the tutorial against actual synthetic workflow outputs."""

from pathlib import Path

import jupytext
import numpy as np
import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "examples/bids_workflow/nsd_workflow.py"
)


@pytest.fixture
def notebook_config(six_run_dataset, tmp_path):
    root, prep = six_run_dataset
    return dict(
        bids_root=str(root),
        fmriprep_root=str(prep),
        output_root=str(tmp_path / "results"),
        hrf_library="sobol",
        hrf_n_samples=2,
        ridge_fractions=(0.4, 1.0),
        n_jobs=1,
        block_size=2,
        surface_maps=False,
    )


def execute_notebook(config):
    namespace = {"NSD_CONFIG": config}
    for cell in jupytext.read(NOTEBOOK).cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_walkthrough_reads_maps_and_matches_trials_to_beta_rows(
    notebook_config, monkeypatch
):
    """Verify walkthrough reads maps and matches trials to beta rows."""
    from boldtailor.workflow import run, beta_series

    def opaque_workflow(*args, **kwargs):
        pytest.fail(
            "The teaching notebook must expose the stages, not call an orchestrator"
        )

    monkeypatch.setattr(run, "run_workflow", opaque_workflow)
    monkeypatch.setattr(beta_series, "fit_beta_models", opaque_workflow)
    ns = execute_notebook(notebook_config)
    assert "preflight" in ns, "show input validation before fitting"
    assert ns["preflight"]["problems"] == []
    assert len(ns["preflight"]["runs"]) == 6
    assert ns["result"].skipped == ()
    assert ns["result"].report_path.is_file()
    assert "map_summary" in ns, "demonstrate named CIFTI maps read from disk"
    assert len(ns["map_summary"]) == 3
    assert len(ns["blocks"]) == 2  # Exercise assembly across spatial blocks.
    assert (ns["map_summary"]["finite_grayordinates"] == 3).all()
    assert "beta_preview" in ns, "demonstrate beta rows joined to trial metadata"
    assert len(ns["beta_preview"]) > 0
    assert np.isfinite(ns["beta_preview"]["beta"]).any()
    assert ns["beta_preview"]["trial_id"].tolist() == list(
        ns["beta_image"].header.get_axis(0).name
    )
    assert ns["metadata"]["settings"]["ridge_mode"] == "fractional_cv"
    assert set(ns["glms"]) == {"CanonicalGLM", "OptimizedGLM"}
    assert set(ns["ols_fits"]) == {"canonical", "optimized"}
    assert set(ns["cv_results"]) == {"canonical", "optimized"}
    for bundle in ns["selections"].values():
        assert set(bundle) == {"all", "odd", "even"}
    for tuned in ns["cv_results"].values():
        assert set(tuned["tuning"]) == {"all", "odd", "even"}
        assert set(tuned["evaluation"]) == {"odd_to_even", "even_to_odd"}
    assert set(ns["activation"]) == set(ns["beta_models"])
    # The example's visible fit agrees with the saved optimized-GLM maps.
    np.testing.assert_allclose(
        ns["glms"]["OptimizedGLM"]["r2"],
        ns["glm_values"],
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )


def test_walkthrough_reports_preflight_problems_without_writing(notebook_config):
    """Verify walkthrough reports preflight problems without writing."""
    notebook_config["session"] = "ses-absent"
    with pytest.raises(ValueError, match="Preflight"):
        execute_notebook(notebook_config)
    assert not Path(notebook_config["output_root"]).exists()
