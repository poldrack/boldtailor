"""Missing sessions are fitted once; the combined notebook reuses completed fits."""

from pathlib import Path
import json

import nibabel as nib
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import pytest

from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401
from examples.NSD.test_ridge_workflow import six_run_dataset  # noqa: F401
from examples.NSD.test_multisession_analysis import api, saved_sessions  # noqa: F401


@pytest.fixture
def two_raw_sessions(six_run_dataset):  # noqa: F811
    root, prep = six_run_dataset
    for directory in (root, prep):
        source = directory / "sub-07/ses-nsd10/func"
        destination = directory / "sub-07/ses-nsd11/func"
        destination.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file():
                (destination / path.name.replace("ses-nsd10", "ses-nsd11")).write_bytes(
                    path.read_bytes()
                )
    return dict(bids_root=str(root), fmriprep_root=str(prep))


@pytest.mark.notebook
def test_missing_sessions_fit_then_reuse_without_fitting(
    two_raw_sessions, tmp_path, monkeypatch
):
    module = api("workflow")
    output = tmp_path / "output"
    config = dict(
        **two_raw_sessions,
        output_root=str(output),
        n_jobs=1,
        block_size=4,
        hrf_n_samples=1,
        ridge_mode="off",
        surface_maps=False,
    )
    sessions = ["ses-nsd10", "ses-nsd11"]
    status = module.ensure_session_outputs(config, sessions, estimators=["OLS"])
    assert status.status.tolist() == ["fitted", "fitted"]
    inputs = api("inputs").load_sessions(output, "sub-07", sessions, estimators=["OLS"])
    paths = [p for record in inputs["records"] for p in record["sources"]]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}

    def forbidden(*args, **kwargs):
        pytest.fail("Completed sessions must not rerun the fitting notebook")

    monkeypatch.setattr(module, "_execute_workflow", forbidden)
    status = module.ensure_session_outputs(config, sessions, estimators=["OLS"])
    assert status.status.tolist() == ["reused", "reused"]
    assert all(
        (p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items()
    )
    with pytest.raises(ValueError, match="hrf_seed"):
        module.ensure_session_outputs(
            {**config, "hrf_seed": 99}, sessions, estimators=["OLS"]
        )


def test_read_only_mode_reports_missing_sessions_without_fitting(tmp_path):
    with pytest.raises(FileNotFoundError, match="ses-nsd10"):
        api("workflow").ensure_session_outputs(
            dict(bids_root=str(tmp_path), output_root=str(tmp_path / "output")),
            ["ses-nsd10", "ses-nsd11"],
            estimators=["OLS"],
            fit_missing=False,
        )


@pytest.mark.notebook
def test_missing_rt_sessions_fit_glms_and_retain_all_trial_betas(
    two_raw_sessions, tmp_path
):
    for path in Path(two_raw_sessions["bids_root"]).rglob("*run-01*events.tsv"):
        events = pd.read_csv(path, sep="\t")
        events.loc[1, "response_time"] = np.nan
        events.to_csv(path, sep="\t", index=False)
    output = tmp_path / "output"
    status = api("workflow").ensure_session_outputs(
        dict(
            **two_raw_sessions,
            output_root=str(output),
            n_jobs=1,
            block_size=4,
            hrf_n_samples=1,
            ridge_fractions=[0.5, 1.0],
        ),
        ["ses-nsd10", "ses-nsd11"],
    )
    assert status.status.tolist() == ["fitted", "fitted"]
    loaded = api("inputs").load_sessions(output, "sub-07", ["ses-nsd10", "ses-nsd11"])
    assert len(loaded["records"]) == 2
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


@pytest.mark.notebook
def test_multisession_notebook_executes_and_exports_paired_maps(
    saved_sessions,  # noqa: F811
    tmp_path,
):
    output, sessions, _, brain = saved_sessions
    path = Path(__file__).with_name("nsd_multisession.ipynb")
    assert path.is_file(), "Create the multi-session notebook"
    notebook = nbformat.read(path, 4)
    config = dict(
        bids_root=str(tmp_path / "bids"),
        output_root=str(output),
        sessions=sessions,
        estimators=["OLS"],
        analysis_config={"ridge_mode": "off"},
        fit_missing=False,
        surface_maps=False,
        grayordinate=0,
    )
    notebook.cells.insert(
        0, nbformat.v4.new_code_cell(f"NSD_MULTI_CONFIG = {config!r}")
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parents[2])}},
    ).execute()
    nbformat.write(notebook, tmp_path / "executed-multisession.ipynb")
    files = list((output / "sub-07/func").glob("*multisession*"))
    image = nib.load(next(p for p in files if "OLS_stat-difference_mean" in p.name))
    assert image.header.get_axis(1) == brain
    names = list(image.header.get_axis(0).name)
    np.testing.assert_allclose(image.get_fdata()[names.index("mean_beta"), :2], [2, -2])
    assert np.isnan(image.get_fdata()[:, 2]).all()
    assert any("HRF_stat-pairwise" in p.name for p in files)
    assert any(p.suffix == ".png" for p in files)
    glm = nib.load(next(p for p in files if "GLM_stat-difference_mean" in p.name))
    assert glm.header.get_axis(0).name.tolist() == ["task", "response_time"]
    np.testing.assert_allclose(
        glm.get_fdata()[:, :2], [[2, -2], [0.2, -0.4]], atol=1e-7
    )


def test_multisession_export_preserves_input_files(saved_sessions):  # noqa: F811
    output, sessions, _, _ = saved_sessions
    loaded = api("inputs").load_sessions(output, "sub-07", sessions, estimators=["OLS"])
    results = api("analysis").analyze_sessions(loaded)
    source = loaded["records"][0]["sources"][0]
    before = source.read_bytes()
    paths = api("outputs").save_multisession(output, loaded, results)
    assert paths and source.read_bytes() == before
    # Rerunning aggregate publication is supported without recomputing sessions.
    assert api("outputs").save_multisession(output, loaded, results) == paths


def test_incompatible_estimators_fail_before_fitting(
    saved_sessions, monkeypatch
):  # noqa: F811
    root, sessions, _, _ = saved_sessions
    module = api("workflow")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid estimator configuration must fail before fitting")

    monkeypatch.setattr(module, "_execute_workflow", forbidden)
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
    saved_sessions,  # noqa: F811
    monkeypatch,
):
    root, sessions, _, _ = saved_sessions
    module = api("workflow")
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

    monkeypatch.setattr(module, "_execute_workflow", forbidden)
    with pytest.raises(ValueError, match="grayordinate axes"):
        module.ensure_session_outputs(
            dict(bids_root=str(root), output_root=str(root)),
            ["ses-nsd99", *sessions],
            estimators=["OLS"],
        )
