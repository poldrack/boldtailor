"""Fit only missing session HRFs and reuse verified, compatible estimates."""

import importlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import sobol_hrf_library
from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401
from examples.NSD.test_nsd_workflow import four_runs  # noqa: F401

NOTEBOOK = Path(__file__).with_name("nsd_session_hrf_reliability.ipynb")


def session_api():
    try:
        return importlib.import_module("examples.NSD.session_hrf")
    except ModuleNotFoundError:
        pytest.fail("The session HRF estimation workflow is not implemented")


@pytest.fixture
def session_data(four_runs):
    root, prep = four_runs
    for directory in (root, prep):
        source = directory / "sub-07/ses-nsd10/func"
        for session in ("ses-nsd11", "ses-nsd12"):
            target = directory / "sub-07" / session / "func"
            target.mkdir(parents=True)
            for path in source.iterdir():
                (target / path.name.replace("ses-nsd10", session)).write_bytes(
                    path.read_bytes()
                )
    return root, prep


@pytest.fixture
def library():
    return sobol_hrf_library(4, seed=0)


def run_sessions(session_data, output, library, sessions_limit=3, **options):
    root, prep = session_data
    return session_api().estimate_sessions(
        root,
        prep,
        output,
        library=library,
        sessions=["ses-nsd10", "ses-nsd11", "ses-nsd12"][:sessions_limit],
        block_size=2,
        **options,
    )


def test_session_selection_matches_public_api_and_reuses_without_fitting(
    session_data, library, tmp_path, monkeypatch
):
    from boldtailor.hrf_selection import select_hrf
    from examples.NSD.workflow_inputs import NSD_TASK_MODEL, load_session, load_block

    output = tmp_path / "output"
    estimates = run_sessions(session_data, output, library, n_jobs=2)
    assert [e.session for e in estimates] == ["ses-nsd10", "ses-nsd11", "ses-nsd12"]
    root, prep = session_data
    runs = load_session(root, prep)
    expected = select_hrf(
        load_block(runs, root, [0, 1, 2]), library=library, task_model=NSD_TASK_MODEL
    )
    np.testing.assert_array_equal(estimates[0].maps[0, :3], expected.hrf_indices)
    np.testing.assert_allclose(estimates[0].maps[1, :3], expected.cv_r2, atol=1e-7)
    assert all(np.isnan(e.maps[:, 3]).all() for e in estimates)
    assert all(e.reused_from is None for e in estimates)
    snapshot = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Compatible cached HRF estimates must not trigger fitting")

    monkeypatch.setattr(session_api(), "fit_session", forbidden)
    cached = run_sessions(session_data, output, library, n_jobs=1)
    assert all(e.reused_from is not None for e in cached)
    for a, b in zip(estimates, cached, strict=True):
        np.testing.assert_allclose(a.maps, b.maps, atol=1e-7)
        assert a.request_id == b.request_id
    assert all(p.read_bytes() == value for p, value in snapshot.items())


def test_source_library_and_coverage_changes_require_new_estimates(
    session_data, library, tmp_path
):
    output = tmp_path / "output"
    original = run_sessions(session_data, output, library, max_grayordinates=2)
    root, _ = session_data
    event_path = next((root / "sub-07/ses-nsd11").rglob("*events.tsv"))
    table = pd.read_csv(event_path, sep="\t")
    table.loc[0, "onset"] += 0.25
    table.to_csv(event_path, sep="\t", index=False)
    updated = run_sessions(session_data, output, library, max_grayordinates=2)
    assert updated[0].reused_from is not None
    assert updated[1].reused_from is None
    assert updated[1].request_id != original[1].request_id
    full = run_sessions(session_data, output, library)
    assert all(e.reused_from is None for e in full)
    assert np.isfinite(full[0].maps[0, 2])
    other = run_sessions(session_data, output, sobol_hrf_library(4, seed=1))
    assert all(e.reused_from is None for e in other)


def test_damaged_cache_is_recomputed_instead_of_silently_reused(
    session_data, library, tmp_path
):
    output = tmp_path / "output"
    original = run_sessions(session_data, output, library)
    path = next(output.rglob("*ses-nsd11*stat-selection.dscalar.nii"))
    good = path.read_bytes()
    path.write_bytes(good + b"damaged")
    recovered = run_sessions(session_data, output, library)
    assert recovered[0].reused_from is not None
    assert recovered[1].reused_from is None
    assert path.read_bytes() == good
    np.testing.assert_allclose(recovered[1].maps, original[1].maps, atol=1e-7)


@pytest.mark.parametrize("bad_metadata", [[], None])
def test_malformed_cache_manifest_is_recomputed(
    session_data, library, tmp_path, bad_metadata
):
    output = tmp_path / "output"
    original = run_sessions(session_data, output, library)
    original[0].cache_path.write_text(json.dumps(bad_metadata))
    recovered = run_sessions(session_data, output, library)
    assert recovered[0].reused_from is None
    assert recovered[1].reused_from is not None
    np.testing.assert_allclose(recovered[0].maps, original[0].maps, atol=1e-7)


@pytest.fixture
def workflow_export(session_data, library, tmp_path):
    from boldtailor.publication import publish_artifact_set
    from examples.NSD.workflow_inputs import load_session, make_blocks
    from examples.NSD.workflow_analysis import select_hrfs
    from examples.NSD.workflow_outputs import (
        _hrf_artifacts,
        _input_artifacts,
        _metadata,
        _stem,
    )
    from examples.NSD.single_trial_artifacts import json_artifact

    root, prep = session_data
    runs = load_session(root, prep)
    selections = select_hrfs(runs, root, make_blocks(runs, block_size=2), library)
    stem = _stem("sub-07", "ses-nsd10")
    artifacts = _hrf_artifacts(
        stem, runs[0].image.header.get_axis(1), selections, library
    )
    artifacts.extend(_input_artifacts(stem, runs))
    artifacts.append(
        json_artifact(
            stem + "_desc-notebook_metadata.json",
            _metadata(runs, library, {"max_grayordinates": None}),
        )
    )
    existing = tmp_path / "existing"
    publish_artifact_set(existing, artifacts)
    return existing


def test_matching_full_workflow_estimates_are_reused(
    session_data, library, tmp_path, monkeypatch, workflow_export
):
    root, prep = session_data
    existing = workflow_export
    before = {p: p.read_bytes() for p in existing.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Compatible full-workflow estimates should be imported without fitting"
        )

    monkeypatch.setattr(session_api(), "fit_session", forbidden)
    result = session_api().estimate_sessions(
        root,
        prep,
        tmp_path / "output",
        library=library,
        sessions=["ses-nsd10"],
        reuse_roots=[existing],
    )
    assert result[0].reused_from is not None
    assert str(existing) in result[0].reused_from
    assert np.isfinite(result[0].maps[0, :3]).all()
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("empty", [False, True])
def test_partial_full_workflow_exports_are_refitted(
    session_data, library, tmp_path, workflow_export, empty
):
    root, prep = session_data
    path = next(workflow_export.rglob("*desc-notebookHRF_provenance.json"))
    records = json.loads(path.read_text())
    removed = records if empty else records[-1:]
    path.write_text(json.dumps([] if empty else records[:-1]))
    selected_path = next(
        workflow_export.rglob("*desc-notebookHRFAll_stat-selection.dscalar.nii")
    )
    image = nib.load(selected_path)
    maps = image.get_fdata()
    for record in removed:
        maps[:, record["grayordinate_indices"]] = np.nan
    nib.save(nib.Cifti2Image(maps.astype(np.float32), image.header), selected_path)
    result = session_api().estimate_sessions(
        root,
        prep,
        tmp_path / "output",
        library=library,
        sessions=["ses-nsd10"],
        reuse_roots=[workflow_export],
    )
    assert result[0].reused_from is None
    assert np.isfinite(result[0].maps[0, :3]).all()


@pytest.mark.parametrize("damage", ["image", "activities"])
def test_damaged_full_workflow_exports_are_refitted(
    session_data, library, tmp_path, workflow_export, damage
):
    root, prep = session_data
    if damage == "image":
        path = next(
            workflow_export.rglob("*desc-notebookHRFAll_stat-selection.dscalar.nii")
        )
        path.write_bytes(b"truncated image")
    else:
        path = next(workflow_export.rglob("*desc-notebookHRF_provenance.json"))
        records = json.loads(path.read_text())
        records[0]["scopes"]["all"]["selection_provenance"]["activities"] = []
        path.write_text(json.dumps(records))
    result = session_api().estimate_sessions(
        root,
        prep,
        tmp_path / "output",
        library=library,
        sessions=["ses-nsd10"],
        reuse_roots=[workflow_export],
    )
    assert result[0].reused_from is None
    assert np.isfinite(result[0].maps[0, :3]).all()


def test_hrf_only_analysis_retains_trials_with_missing_reaction_times(
    session_data, library, tmp_path
):
    from examples.NSD.workflow_inputs import load_session

    root, prep = session_data
    path = next((root / "sub-07/ses-nsd10").rglob("*events.tsv"))
    events = pd.read_csv(path, sep="\t")
    events.loc[0, "response_time"] = np.nan
    events.to_csv(path, sep="\t", index=False)
    result = session_api().estimate_sessions(
        root, prep, tmp_path / "output", library=library, sessions=["ses-nsd10"]
    )
    assert np.isfinite(result[0].maps[0, :3]).all()
    runs = load_session(root, prep, hrf_only=True)
    assert len(runs[0].events) == len(events)
    assert pd.isna(runs[0].events.response_time).sum() == 1
    glm_runs = load_session(root, prep)
    pd.testing.assert_frame_equal(glm_runs[0].events, runs[0].events)


def test_hrf_only_analysis_requires_two_runs_without_odd_even_splits(
    dataset, library, tmp_path
):
    root, prep, *_ = dataset
    result = session_api().estimate_sessions(
        root, prep, tmp_path / "output", library=library, sessions=["ses-nsd10"]
    )
    assert np.isfinite(result[0].maps[0, :3]).all()


def test_incompatible_axes_or_missing_sessions_fail_before_fitting(
    session_data, library, tmp_path, monkeypatch
):
    root, prep = session_data
    path = next((prep / "sub-07/ses-nsd12").rglob("*.dtseries.nii"))
    image = nib.load(path)
    brain = nib.cifti2.BrainModelAxis.from_surface([0, 1, 3, 6], 8, name="CortexLeft")
    nib.save(
        nib.Cifti2Image(
            image.get_fdata(),
            nib.Cifti2Header.from_axes((image.header.get_axis(0), brain)),
        ),
        path,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid inputs should be rejected before expensive fitting")

    monkeypatch.setattr(session_api(), "fit_session", forbidden)
    with pytest.raises(ValueError, match="axes|axis|grayordinate"):
        run_sessions(session_data, tmp_path / "output", library)
    with pytest.raises((ValueError, FileNotFoundError)):
        session_api().estimate_sessions(
            root,
            prep,
            tmp_path / "output",
            library=library,
            sessions=["ses-nsd10", "ses-nsd99"],
        )


@pytest.mark.notebook
def test_notebook_fits_three_sessions_exports_comparisons_and_resumes(
    session_data, tmp_path
):
    assert NOTEBOOK.exists(), "The session HRF reliability notebook is missing"
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
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.insert(
        0, nbformat.v4.new_code_cell(f"HRF_RELIABILITY_CONFIG = {config!r}")
    )
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parents[2])}},
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


def test_rt_switch_changes_request_identity_and_cache_metadata(
    session_data, library, tmp_path
):
    from examples.NSD.workflow_inputs import selection_task_model

    with_rt = run_sessions(session_data, tmp_path / "rt", library, sessions_limit=1)
    without = run_sessions(
        session_data, tmp_path / "nort", library, sessions_limit=1, include_rt=False
    )
    assert with_rt[0].request_id != without[0].request_id
    metadata = json.loads(without[0].cache_path.read_text())
    assert metadata["request"]["task_model"] == selection_task_model(False).to_dict()
    assert metadata["request"]["task_model"]["regressors"] == ["task", "trial_type"]
    assert np.isfinite(without[0].maps[0, :3]).all()


def test_full_workflow_export_with_rt_is_not_imported_when_rt_is_off(
    session_data, library, tmp_path, workflow_export
):
    root, prep = session_data
    result = session_api().estimate_sessions(
        root,
        prep,
        tmp_path / "output",
        library=library,
        sessions=["ses-nsd10"],
        reuse_roots=[workflow_export],
        include_rt=False,
    )
    assert result[0].reused_from is None
    assert np.isfinite(result[0].maps[0, :3]).all()


def test_cache_request_records_peak_hrf_normalization(session_data, library, tmp_path):
    result = run_sessions(session_data, tmp_path / "out", library, sessions_limit=1)
    metadata = json.loads(result[0].cache_path.read_text())
    assert metadata["request"]["hrf_normalization"] == "peak_one"
