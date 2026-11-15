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
        block_size=4,
        surface_maps=False,
    )


def execute_notebook(config):
    namespace = {"NSD_CONFIG": config}
    for cell in jupytext.read(NOTEBOOK).cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_walkthrough_reads_maps_and_matches_trials_to_beta_rows(notebook_config):
    ns = execute_notebook(notebook_config)
    assert "preflight" in ns, "show input validation before fitting"
    assert ns["preflight"]["problems"] == []
    assert len(ns["preflight"]["runs"]) == 6
    assert ns["result"].skipped == ()
    assert ns["result"].report_path.is_file()
    assert "map_summary" in ns, "demonstrate named CIFTI maps read from disk"
    assert len(ns["map_summary"]) == 3
    assert (ns["map_summary"]["finite_grayordinates"] > 0).all()
    assert "beta_preview" in ns, "demonstrate beta rows joined to trial metadata"
    assert len(ns["beta_preview"]) > 0
    assert np.isfinite(ns["beta_preview"]["beta"]).any()
    assert ns["beta_preview"]["trial_id"].tolist() == list(
        ns["beta_image"].header.get_axis(0).name
    )
    assert ns["metadata"]["settings"]["ridge_mode"] == "fractional_cv"


def test_walkthrough_reports_preflight_problems_without_writing(notebook_config):
    notebook_config["session"] = "ses-absent"
    with pytest.raises(ValueError, match="Preflight"):
        execute_notebook(notebook_config)
    assert not Path(notebook_config["output_root"]).exists()
