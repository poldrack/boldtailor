"""Execute the new default and inspect its exported fraction/alpha decisions."""

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import nibabel as nib
import numpy as np

from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401
from examples.NSD.test_ridge_workflow import six_run_dataset  # noqa: F401
from examples.NSD.test_nsd_workflow import preview_library


def test_explicit_fractional_mode_uses_fraction_grid(tmp_path):
    _, settings = preview_library(
        tmp_path, hrf_n_samples=2, ridge_mode="fractional_cv", ridge_fractions=[0.4, 1]
    )
    assert settings["ridge_mode"] == "fractional_cv"
    assert settings["ridge_fractions"] == [0.4, 1]


def test_notebook_default_executes_fractional_cv(six_run_dataset, tmp_path):
    root, prep = six_run_dataset
    output = tmp_path / "fractional-notebook"
    config = dict(
        bids_root=str(root),
        fmriprep_root=str(prep),
        output_root=str(output),
        block_size=2,
        n_jobs=1,
        ridge_fractions=[0.4, 1],
        hrf_n_samples=2,
    )
    path = Path(__file__).with_name("nsd_workflow.ipynb")
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(f"NSD_CONFIG = {config!r}"))
    try:
        NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parents[2])}},
        ).execute()
    finally:
        nbformat.write(notebook, tmp_path / "executed-fractional.ipynb")
    files = list(output.rglob("*"))
    betas = [p for p in files if "TrialFractionalCV_betas.dscalar.nii" in p.name]
    assert len(betas) == 12
    assert not any("TrialRidgeCV_betas" in p.name for p in files)
    for image in map(nib.load, betas):
        assert image.shape == (6, 4)
        assert np.isnan(image.get_fdata()[:, -1]).all()
    maps = [
        p
        for p in files
        if "FractionalCV" in p.name and "_stat-ridgefraction." in p.name
    ]
    assert len(maps) == 12  # Three tuning scopes, two outer splits, and final per mode.
    metadata = json.loads(
        next(
            p for p in files if p.name.endswith("desc-notebook_metadata.json")
        ).read_text()
    )
    assert (
        metadata["ridge_cv"]["validation_target"]
        == "candidate_fraction_regularized_betas"
    )
    assert metadata["ridge_cv"]["objective"] == "maximum_encoding_r2_per_grayordinate"
    assert metadata["ridge_cv"]["percentile_role"] == "descriptive_only"
    for mode in ("Canonical", "Optimized"):
        decision = json.loads(
            next(
                p for p in files if f"{mode}FractionalCVAll_provenance.json" in p.name
            ).read_text()
        )
        final = json.loads(
            next(
                p for p in files if f"{mode}TrialFractionalCV_provenance.json" in p.name
            ).read_text()
        )
        for block in final:
            assert (
                block["record"]["activities"][-1]["tuning_analysis_fingerprint"]
                == decision["analysis_fingerprint"]
            )
