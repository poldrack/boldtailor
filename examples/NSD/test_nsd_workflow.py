"""The tutorial must execute real CIFTI analyses with matched GLM predictors."""

import importlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary

NOTEBOOK = Path(__file__).with_name("nsd_workflow.ipynb")


def workflow(module="workflow_inputs"):
    try:
        return importlib.import_module(f"examples.NSD.{module}")
    except ModuleNotFoundError as error:
        pytest.fail(f"The NSD notebook workflow is not implemented: {error}")


def test_nsd_task_model_centers_rt_and_keeps_trial_type_uncentered(events):
    from boldtailor.design import expand_events
    from boldtailor.model import Modulator, TaskModel

    inputs = workflow()
    assert inputs.NSD_TASK_MODEL == TaskModel(
        (
            Modulator("response_time", center=True, missing="indicator"),
            Modulator("trial_type", center=False),
        )
    )
    original = events.copy(deep=True)
    result = expand_events(events, inputs.NSD_TASK_MODEL)
    amplitudes = {
        "task": np.ones(len(events)),
        "response_time": events.response_time - events.response_time.mean(),
        "trial_type": events.trial_type,
    }
    assert list(dict.fromkeys(result.trial_type)) == list(amplitudes)
    for name, expected in amplitudes.items():
        rows = result.loc[result.trial_type == name]
        np.testing.assert_allclose(rows.modulation, expected)
        np.testing.assert_allclose(
            rows[["onset", "duration"]], events[["onset", "duration"]]
        )
    pd.testing.assert_frame_equal(events, original)


def _left_cortex_meshes(directory):
    """An eight-vertex left mesh matching the fixture axis, and any right mesh."""
    coords = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float32
    )
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1]]
        + [[2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=np.int32,
    )
    paths = {}
    for hemi in ("left", "right"):
        paths[hemi] = str(directory / f"{hemi}.surf.gii")
        arrays = [
            nib.gifti.GiftiDataArray(coords, intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"),
        ]
        nib.save(nib.gifti.GiftiImage(darrays=arrays), paths[hemi])
    return paths


def _surface_images(executed, cell_id):
    cell = next(c for c in executed.cells if c.get("id") == cell_id)
    return [o for o in cell.outputs if "image/png" in o.get("data", {})]


@pytest.mark.notebook
def test_notebook_executes_full_workflow_and_exports_reusable_artifacts(
    four_runs, tmp_path
):
    library_config = {"hrf_library": "sobol", "hrf_n_samples": 4, "hrf_seed": 7}
    candidate_count = 5
    assert NOTEBOOK.is_file(), "The full NSD workflow notebook has not been created"
    root, prep = four_runs
    output = tmp_path / "output"
    config = dict(
        bids_root=str(root),
        fmriprep_root=str(prep),
        output_root=str(output),
        block_size=2,
        n_jobs=1,
        ridge_alpha=0.1,
        surface_meshes=_left_cortex_meshes(tmp_path),
        **library_config,
    )
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(f"NSD_CONFIG = {config!r}"))
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parents[2])}},
    )
    executed = client.execute()
    nbformat.write(executed, tmp_path / "executed.ipynb")
    for cell_id in ("glm-surface-maps", "beta-surface-maps", "rt-surface-maps"):
        images = _surface_images(executed, cell_id)
        assert len(images) == 1, f"{cell_id} should display its surface figure once"
    files = list(output.rglob("*"))
    scalars = [p for p in files if p.name.endswith(".dscalar.nii")]
    assert scalars and all("desc-notebook" in p.name for p in scalars)
    by_name = {p.name: p for p in scalars}

    def read(suffix):
        return nib.load(
            next(p for n, p in by_name.items() if n.endswith(suffix))
        ).get_fdata()

    canonical = read("desc-notebookCanonicalGLM_stat-rsquared.dscalar.nii")
    optimized = read("desc-notebookOptimizedGLM_stat-rsquared.dscalar.nii")
    comparison = read("desc-notebookGLMComparison_stat-deltarsquared.dscalar.nii")
    np.testing.assert_allclose(comparison[0], optimized[0] - canonical[0], atol=1e-7)
    np.testing.assert_allclose(canonical[2], canonical[0] - canonical[1], atol=1e-7)
    for label in ("All", "Odd", "Even"):
        assert any(
            f"desc-notebookHRF{label}_stat-hrfparameters" in p.name for p in scalars
        )
    curve_path = next(
        (
            p
            for p in scalars
            if "desc-notebookHRFReliability_stat-curvecorrelation" in p.name
        ),
        None,
    )
    assert curve_path is not None, "The workflow must export full-HRF correlations"
    curve_image = nib.load(curve_path)
    assert curve_image.header.get_axis(0).name.tolist() == [
        "odd_even_r",
        "odd_canonical_r",
        "even_canonical_r",
    ]
    odd_ids = read("desc-notebookHRFOdd_stat-selection.dscalar.nii")[0]
    even_ids = read("desc-notebookHRFEven_stat-selection.dscalar.nii")[0]
    curves = np.load(next(p for p in files if p.name.endswith("_library.npz")))[
        "curves"
    ]
    expected = np.full((3, 4), np.nan)
    for i, (a, b) in enumerate(zip(odd_ids, even_ids, strict=True)):
        for row, (x, y) in enumerate(((a, b), (a, 0), (b, 0))):
            if np.isfinite(x) and np.isfinite(y):
                expected[row, i] = np.corrcoef(curves[int(x)], curves[int(y)])[0, 1]
    np.testing.assert_allclose(curve_image.get_fdata(), expected, atol=1e-7)
    assert any("HRFCurveReliability_plot.png" in p.name for p in files)
    assert len([p for p in scalars if "_betas." in p.name]) == 16
    activation_paths = [p for p in scalars if "_stat-activation." in p.name]
    assert len(activation_paths) == 4
    from scipy.stats import ttest_1samp

    for path in activation_paths:
        descriptor = path.name.split("desc-notebook")[1].split("_stat-")[0]
        images = sorted(
            p for p in scalars if f"desc-notebook{descriptor}_betas." in p.name
        )
        pooled = np.concatenate([nib.load(p).get_fdata() for p in images])
        expected = ttest_1samp(pooled[:, :3], 0, axis=0)
        actual = nib.load(path).get_fdata()
        np.testing.assert_allclose(actual[0, :3], pooled[:, :3].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(actual[1, :3], expected.statistic, rtol=1e-6)
        np.testing.assert_allclose(actual[2, :3], expected.pvalue, rtol=1e-6)
        np.testing.assert_array_equal(actual[3, :3], len(pooled))
        np.testing.assert_array_equal(actual[4, :3], len(pooled) - 1)
    assert len([p for p in files if p.name.endswith("_trials.tsv")]) == 4
    assert any(p.name.endswith("_library.tsv") for p in files)
    assert any(p.name.endswith("_designs.npz") for p in files)
    for path in scalars:
        image = nib.load(path)
        assert image.shape[1] == 4
        assert np.isnan(image.get_fdata()[:, 3]).all()
    metadata = json.loads(
        next(p for p in files if p.name.endswith("_metadata.json")).read_text()
    )
    assert metadata["regressors"] == ["task", "response_time", "trial_type"]
    assert metadata["retained_scans"] == [95, 94, 93, 95]
    assert metadata["library_candidates"] == candidate_count
    for name, value in library_config.items():
        assert metadata["settings"][name] == value
    saved_table = pd.read_csv(
        next(p for p in files if p.name.endswith("_library.tsv")),
        sep="\t",
        float_precision="round_trip",
    )
    restored = HrfLibrary.from_table(saved_table)
    assert restored.fingerprint == metadata["library_fingerprint"]
    np.testing.assert_array_equal(restored.curves, curves)
    assert metadata["noise_model"] == "ols"
    assert metadata["beta_activation"]["assume_independent_trials"] is True
    assert metadata["beta_activation"]["null_mean"] == 0
    assert metadata["beta_activation"]["multiple_comparison_correction"] is None
    assert "independent" in metadata["glm_comparison"].lower()
    assert metadata["trial_type"] == (
        "Binary codes 0/1, uncentered; the task coefficient is the response on "
        "trial_type 0 trials at the run-mean RT"
    )
    assert metadata["task"] == (
        "One unit per presentation; observed RT at its run mean, trial type 0, "
        "missing-RT indicator zero"
    )
    assert metadata["hrf_selection"] == (
        "leave-one-run-out task-model prediction over task, response_time, "
        "trial_type; missing-RT indicator profiled per run when present; pooled "
        "held-out error over confound-adjusted energy"
    )
    assert metadata["settings"]["hrf_selection_rt"] is True
    assert (
        metadata["hrf_curve_correlations"]["method"]
        == "Pearson over HRF time samples, without temporal shifting"
    )
    before = {p: p.read_bytes() for p in scalars}
    with pytest.raises(FileExistsError):
        workflow("workflow_outputs").check_output(output, "sub-07", "ses-nsd10")
    assert all(p.read_bytes() == value for p, value in before.items())


def test_metadata_and_reuse_follow_the_rt_selection_switch(four_runs, small_library):
    inputs, outputs, reuse = (
        workflow(),
        workflow("workflow_outputs"),
        workflow("workflow_reuse"),
    )
    root, prep = four_runs
    runs = inputs.load_session(root, prep)
    off = outputs._metadata(runs, small_library, {"hrf_selection_rt": False})
    assert off["hrf_selection"] == (
        "leave-one-run-out task-model prediction over task, trial_type; "
        "missing-RT indicator profiled per run when present; pooled held-out "
        "error over confound-adjusted energy"
    )
    assert "never used to select HRFs" in off["rt_check"]
    on = outputs._metadata(runs, small_library, {"hrf_selection_rt": True})
    assert "response_time" in on["hrf_selection"]
    assert "never used to select HRFs" not in on["rt_check"]
    with pytest.raises(ValueError, match="hrf_selection_rt"):
        reuse.validate_saved_settings(
            {"settings": {"hrf_selection_rt": True}}, {"hrf_selection_rt": False}
        )
