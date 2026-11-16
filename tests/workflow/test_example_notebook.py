"""Execute the tutorial against actual synthetic workflow outputs."""

from pathlib import Path
import os

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


def _call_with_pid(block, function, args):
    return os.getpid(), function(block, *args)


def test_parallel_walkthrough_fits_hrfs_and_glms_in_workers(
    notebook_config, monkeypatch
):
    """Run HRF selection and both GLMs in child processes without changing spatial results."""
    from boldtailor import parallel

    serial = execute_notebook(notebook_config)
    real_map = parallel.map_blocks
    worker_pids = {"selection": set(), "glm": set()}

    def observed_map(function, blocks, *, args=(), n_jobs=1):
        for block, (pid, result) in real_map(
            _call_with_pid, blocks, args=(function, args), n_jobs=n_jobs
        ):
            if isinstance(result, dict):
                if {"all", "odd", "even"} <= result.keys():
                    worker_pids["selection"].add(pid)
                if {"effects", "variances", "t", "z", "r2"} <= result.keys():
                    worker_pids["glm"].add(pid)
            yield block, result

    monkeypatch.setattr(parallel, "map_blocks", observed_map)
    config = {
        **notebook_config,
        "n_jobs": 2,
        "output_root": notebook_config["output_root"] + "-parallel",
    }
    concurrent = execute_notebook(config)
    for stage, pids in worker_pids.items():
        assert pids and os.getpid() not in pids, f"{stage} must run in child processes"
    assert list(concurrent["selections"]) == list(serial["selections"])
    for name, maps in concurrent["glms"].items():
        for key in ("effects", "variances", "t", "z", "r2"):
            np.testing.assert_allclose(
                maps[key],
                serial["glms"][name][key],
                rtol=1e-7,
                atol=1e-9,
                equal_nan=True,
            )
        assert [p["grayordinate_indices"] for p in maps["provenance"]] == [
            p["grayordinate_indices"] for p in serial["glms"][name]["provenance"]
        ]
    for indices, bundle in concurrent["selections"].items():
        for split in ("all", "odd", "even"):
            left, right = bundle[split], serial["selections"][indices][split]
            if split != "all":
                left, right = left.training_selection, right.training_selection
            np.testing.assert_array_equal(left.hrf_indices, right.hrf_indices)
            np.testing.assert_allclose(left.cv_r2, right.cv_r2, rtol=1e-7, atol=1e-9)
