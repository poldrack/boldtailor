"""Notebook executions on the synthetic fixtures; opt in with --run-notebooks."""

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfLibrary
from boldtailor.workflow.outputs import check_output
from boldtailor.workflow.settings import WorkflowSettings
from examples.NSD import multisession_inputs, multisession_workflow

HERE = Path(__file__).parent
WORKFLOW_NOTEBOOK = HERE / "nsd_workflow.ipynb"
SESSION_NOTEBOOK = HERE / "nsd_session_hrf_reliability.ipynb"
MULTISESSION_NOTEBOOK = HERE / "nsd_multisession.ipynb"


def _execute(path, name, config, tmp_path, timeout=600):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(f"{name} = {config!r}"))
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parents[2])}},
    )
    try:
        return client.execute()
    finally:
        nbformat.write(notebook, tmp_path / f"executed-{path.name}")


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


@pytest.mark.notebook
def test_session_hrf_notebook_fits_three_sessions_exports_comparisons_and_resumes(
    session_data, tmp_path
):
    assert SESSION_NOTEBOOK.exists(), "The session HRF reliability notebook is missing"
    root, prep = session_data
    output = tmp_path / "output"
    config = dict(
        bids_root=str(root),
        fmriprep_root=str(prep),
        output_root=str(output),
        sessions=["ses-nsd10", "ses-nsd11", "ses-nsd12"],
        hrf_n_samples=4,
        hrf_seed=0,
        block_size=2,
        n_jobs=1,
        reuse_roots=[],
    )
    notebook = nbformat.read(SESSION_NOTEBOOK, as_version=4)
    notebook.cells.insert(
        0, nbformat.v4.new_code_cell(f"HRF_RELIABILITY_CONFIG = {config!r}")
    )
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(SESSION_NOTEBOOK.parents[2])}},
    )
    executed = client.execute()
    nbformat.write(executed, tmp_path / "executed.ipynb")
    files = list(output.rglob("*.dscalar.nii"))
    pairwise = nib.load(next(p for p in files if "stat-pairwise" in p.name))
    canonical = nib.load(next(p for p in files if "stat-canonical" in p.name))
    summary = nib.load(next(p for p in files if "stat-summary" in p.name))
    assert pairwise.shape == canonical.shape == (3, 4)
    assert summary.shape == (5, 4)
    assert np.isnan(pairwise.get_fdata()[:, 3]).all()
    assert np.isnan(summary.get_fdata()[:3, 3]).all()
    assert len([p for p in files if "stat-selection" in p.name]) == 3
    assert len([p for p in files if "stat-hrfparameters" in p.name]) == 3
    # Identical input copies must select identical HRFs in every session.
    np.testing.assert_allclose(pairwise.get_fdata()[:, :3], 1, atol=1e-7)
    np.testing.assert_allclose(
        summary.get_fdata()[2, :3],
        1 - canonical.get_fdata()[:, :3].mean(axis=0),
        atol=1e-7,
    )
    assert list(output.rglob("*.png"))
    assert not any("betas" in p.name for p in files)
    # Rerunning must use the completed per-session cache and update the report.
    rerun = client.execute()
    text = "\n".join(
        o.get("text", "") for c in rerun.cells for o in c.get("outputs", [])
    )
    assert text.count("Reused HRF estimates") == 3
    assert "Fitting HRFs" not in text


SESSIONS = ["ses-nsd10", "ses-nsd11"]
ANALYSIS = dict(n_jobs=1, block_size=4, hrf_n_samples=1, ridge_fractions=[0.5, 1.0])


def _drop_one_run01_rt(bids_root):
    for path in Path(bids_root).rglob("*run-01*events.tsv"):
        events = pd.read_csv(path, sep="\t")
        events.loc[1, "response_time"] = np.nan
        events.to_csv(path, sep="\t", index=False)


def _assert_missing_rt_exports(output):
    loaded = multisession_inputs.load_sessions(output, "sub-07", SESSIONS)
    for record in loaded["records"]:
        assert (
            record["metadata"]["missing_response_time"]["regressor"]
            == "missing_response_time"
        )
    betas = list(output.rglob("*run-01*Trial*_betas.dscalar.nii"))
    assert len(betas) == 8
    for path in betas:
        values = nib.load(path).get_fdata()
        assert values.shape == (6, 4)
        assert np.isfinite(values[:, :3]).all()
    designs = list(output.rglob("*GLM_designs.npz"))
    assert len(designs) == 4
    for path in designs:
        with np.load(path) as archive:
            columns = [
                archive[k].tolist()
                for k in archive.files
                if k.startswith("run-01_") and k.endswith("_columns")
            ]
            assert columns and all("missing_response_time" in c for c in columns)
    masks = list(output.rglob("*FractionalCVAll_metadata.json"))
    assert len(masks) == 4
    for path in masks:
        metadata = json.loads(path.read_text())
        assert metadata["trial_masks"][0] == [True, False, True, True, True, True]
    return loaded


def _session_difference(output, pattern, rows):
    differences = []
    for session in SESSIONS:
        values = {
            hrf: nib.load(
                next((output / "sub-07" / session).rglob(pattern.format(hrf)))
            ).get_fdata()[rows]
            for hrf in ("Canonical", "Optimized")
        }
        differences.append(values["Optimized"] - values["Canonical"])
    return np.mean(differences, axis=0)


def _assert_paired_exports(output, brain):
    files = list((output / "sub-07/func").glob("*multisession*"))
    image = nib.load(next(p for p in files if "OLS_stat-difference_mean" in p.name))
    assert image.header.get_axis(1) == brain
    names = list(image.header.get_axis(0).name)
    expected = _session_difference(output, "*{}TrialOLS_stat-activation*", 0)
    np.testing.assert_allclose(
        image.get_fdata()[names.index("mean_beta"), :3], expected[:3], rtol=1e-5
    )
    assert np.isnan(image.get_fdata()[:, 3]).all()
    assert any("HRF_stat-pairwise" in p.name for p in files)
    assert any(p.suffix == ".png" for p in files)
    glm = nib.load(next(p for p in files if "GLM_stat-difference_mean" in p.name))
    assert glm.header.get_axis(0).name.tolist() == ["task", "response_time"]
    expected = _session_difference(output, "*{}GLM_stat-effects*", slice(0, 2))
    np.testing.assert_allclose(glm.get_fdata()[:, :3], expected[:, :3], rtol=1e-5)


def _assert_reuse_without_fitting(fit_config, loaded, monkeypatch):
    paths = [p for record in loaded["records"] for p in record["sources"]]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}

    def forbidden(*args, **kwargs):
        pytest.fail("Completed sessions must not rerun the workflow")

    monkeypatch.setattr(multisession_workflow, "run_workflow", forbidden)
    status = multisession_workflow.ensure_session_outputs(fit_config, SESSIONS)
    assert status.status.tolist() == ["reused", "reused"]
    assert all(
        (p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items()
    )
    with pytest.raises(ValueError, match="hrf_seed"):
        multisession_workflow.ensure_session_outputs(
            {**fit_config, "hrf_seed": 99}, SESSIONS
        )


@pytest.mark.notebook
def test_multisession_notebook_fits_missing_sessions_pairs_and_reuses_them(
    two_raw_sessions, tmp_path, monkeypatch
):
    _drop_one_run01_rt(two_raw_sessions["bids_root"])
    output = tmp_path / "output"
    config = dict(
        **two_raw_sessions,
        output_root=str(output),
        sessions=SESSIONS,
        analysis_config={**ANALYSIS, "surface_maps": False},
        fit_missing=True,
        surface_maps=False,
        grayordinate=0,
    )
    _execute(MULTISESSION_NOTEBOOK, "NSD_MULTI_CONFIG", config, tmp_path)
    loaded = _assert_missing_rt_exports(output)
    _assert_paired_exports(output, loaded["brain"])
    fit_config = {
        **config["analysis_config"],
        **two_raw_sessions,
        "output_root": str(output),
        "subject": "sub-07",
    }
    _assert_reuse_without_fitting(fit_config, loaded, monkeypatch)


def _displayed_images(executed):
    return [
        o
        for c in executed.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if "image/png" in o.get("data", {})
    ]


def _assert_glm_and_hrf_exports(by_name, files):
    def read(suffix):
        return nib.load(
            next(p for n, p in by_name.items() if n.endswith(suffix))
        ).get_fdata()

    canonical = read("desc-CanonicalGLM_stat-rsquared.dscalar.nii")
    optimized = read("desc-OptimizedGLM_stat-rsquared.dscalar.nii")
    comparison = read("desc-GLMComparison_stat-deltarsquared.dscalar.nii")
    np.testing.assert_allclose(comparison[0], optimized[0] - canonical[0], atol=1e-7)
    np.testing.assert_allclose(canonical[2], canonical[0] - canonical[1], atol=1e-7)
    for label in ("All", "Odd", "Even"):
        assert any(f"desc-HRF{label}_stat-hrfparameters" in n for n in by_name)
    curve = nib.load(
        next(
            p
            for n, p in by_name.items()
            if "desc-HRFReliability_stat-curvecorrelation" in n
        )
    )
    assert curve.header.get_axis(0).name.tolist() == [
        "odd_even_r",
        "odd_canonical_r",
        "even_canonical_r",
    ]
    odd_ids = read("desc-HRFOdd_stat-selection.dscalar.nii")[0]
    even_ids = read("desc-HRFEven_stat-selection.dscalar.nii")[0]
    with np.load(next(p for p in files if p.name.endswith("_library.npz"))) as saved:
        curves = saved["curves"]
    expected = np.full((3, 4), np.nan)
    for i, (a, b) in enumerate(zip(odd_ids, even_ids, strict=True)):
        for row, (x, y) in enumerate(((a, b), (a, 0), (b, 0))):
            if np.isfinite(x) and np.isfinite(y):
                expected[row, i] = np.corrcoef(curves[int(x)], curves[int(y)])[0, 1]
    np.testing.assert_allclose(curve.get_fdata(), expected, atol=1e-7)
    return curves


def _assert_beta_exports(scalars, files):
    from scipy.stats import ttest_1samp

    assert len([p for p in scalars if "_betas." in p.name]) == 16
    activation_paths = [p for p in scalars if "_stat-activation." in p.name]
    assert len(activation_paths) == 4
    for path in activation_paths:
        descriptor = path.name.split("_desc-")[1].split("_stat-")[0]
        images = sorted(p for p in scalars if f"desc-{descriptor}_betas." in p.name)
        pooled = np.concatenate([nib.load(p).get_fdata() for p in images])
        expected = ttest_1samp(pooled[:, :3], 0, axis=0)
        actual = nib.load(path).get_fdata()
        np.testing.assert_allclose(actual[0, :3], pooled[:, :3].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(actual[1, :3], expected.statistic, rtol=1e-6)
        np.testing.assert_allclose(actual[2, :3], expected.pvalue, rtol=1e-6)
        np.testing.assert_array_equal(actual[3, :3], len(pooled))
        np.testing.assert_array_equal(actual[4, :3], len(pooled) - 1)
    assert len([p for p in files if p.name.endswith("_trials.tsv")]) == 4
    for path in scalars:
        image = nib.load(path)
        assert image.shape[1] == 4
        assert np.isnan(image.get_fdata()[:, 3]).all()


def _assert_metadata(files, library_config, curves):
    metadata = json.loads(
        next(
            p for p in files if p.name.endswith("_desc-boldtailor_metadata.json")
        ).read_text()
    )
    assert metadata["regressors"] == ["task", "response_time", "trial_type[1]"]
    assert metadata["retained_scans"] == [95, 94, 93, 95]
    assert metadata["library_candidates"] == 5
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
    assert metadata["settings"]["hrf_selection_rt"] is True
    return metadata


@pytest.mark.notebook
def test_workflow_notebook_runs_the_package_workflow_and_shows_its_outputs(
    four_runs, tmp_path
):
    library_config = {"hrf_library": "sobol", "hrf_n_samples": 4, "hrf_seed": 7}
    root, prep = four_runs
    output = tmp_path / "output"
    config = dict(
        bids_root=str(root),
        fmriprep_root=str(prep),
        output_root=str(output),
        block_size=2,
        n_jobs=1,
        ridge_mode="fixed",
        ridge_alpha=0.1,
        surface_meshes=_left_cortex_meshes(tmp_path),
        **library_config,
    )
    executed = _execute(WORKFLOW_NOTEBOOK, "NSD_CONFIG", config, tmp_path, 300)
    assert (output / "sub-07_ses-nsd10_task-nsdcore_report.html").is_file()
    files = [p for p in output.rglob("*") if p.is_file()]
    scalars = [p for p in files if p.name.endswith(".dscalar.nii")]
    assert scalars
    curves = _assert_glm_and_hrf_exports({p.name: p for p in scalars}, files)
    _assert_beta_exports(scalars, files)
    metadata = _assert_metadata(files, library_config, curves)
    plots = [p for p in files if p.name.endswith("_plot.png")]
    for name in ("HRFCurveReliability", "GLMR2Surface", "BetaR2Surface", "RTSurface"):
        assert any(f"desc-{name}_plot.png" in p.name for p in plots)
    # The notebook shows every saved figure once, read back from its PNG.
    assert len(_displayed_images(executed)) == len(plots)
    settings = WorkflowSettings.from_dict(metadata["settings"])
    with pytest.raises(FileExistsError):
        check_output(settings)
