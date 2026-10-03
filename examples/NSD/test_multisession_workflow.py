"""Missing sessions are fitted once; the combined notebook reuses completed fits."""

from pathlib import Path
import json

import nibabel as nib
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import pytest

from examples.NSD import (
    multisession_analysis,
    multisession_inputs,
    multisession_outputs,
    multisession_workflow,
)


@pytest.fixture
def two_raw_sessions(six_run_dataset):
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

    monkeypatch.setattr(module, "_execute_workflow", forbidden)
    with pytest.raises(ValueError, match="grayordinate axes"):
        module.ensure_session_outputs(
            dict(bids_root=str(root), output_root=str(root)),
            ["ses-nsd99", *sessions],
            estimators=["OLS"],
        )


SESSIONS = ["ses-nsd10", "ses-nsd11"]
ANALYSIS = dict(n_jobs=1, block_size=4, hrf_n_samples=1, ridge_fractions=[0.5, 1.0])


def _drop_one_run01_rt(bids_root):
    for path in Path(bids_root).rglob("*run-01*events.tsv"):
        events = pd.read_csv(path, sep="\t")
        events.loc[1, "response_time"] = np.nan
        events.to_csv(path, sep="\t", index=False)


def _execute_multisession(config, tmp_path):
    path = Path(__file__).with_name("nsd_multisession.ipynb")
    notebook = nbformat.read(path, 4)
    notebook.cells.insert(
        0, nbformat.v4.new_code_cell(f"NSD_MULTI_CONFIG = {config!r}")
    )
    try:
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parents[2])}},
        ).execute()
    finally:
        nbformat.write(notebook, tmp_path / "executed-multisession.ipynb")


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
        pytest.fail("Completed sessions must not rerun the fitting notebook")

    monkeypatch.setattr(multisession_workflow, "_execute_workflow", forbidden)
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
    _execute_multisession(config, tmp_path)
    loaded = _assert_missing_rt_exports(output)
    _assert_paired_exports(output, loaded["brain"])
    fit_config = {
        **config["analysis_config"],
        **two_raw_sessions,
        "output_root": str(output),
        "subject": "sub-07",
    }
    _assert_reuse_without_fitting(fit_config, loaded, monkeypatch)
