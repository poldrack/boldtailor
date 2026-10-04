"""Shared NSD example fixtures: small real CIFTI sessions and saved results."""

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

# The examples import their siblings as examples.NSD.*; put the repo root on
# sys.path so that works under plain `pytest` (pyproject no longer sets pythonpath).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from boldtailor.cifti import scalar_artifact  # noqa: E402
from boldtailor.hrf_library import HrfLibrary  # noqa: E402
from boldtailor.publication import publish_artifact_set  # noqa: E402
from boldtailor.workflow.artifacts import (  # noqa: E402
    json_artifact,
    npz_artifact,
    table_artifact,
)
from tests.workflow import synthetic_bids  # noqa: E402


def pytest_addoption(parser):
    try:  # tests/conftest.py registers it too when both trees are collected
        parser.addoption(
            "--run-notebooks",
            action="store_true",
            default=False,
            help="execute notebook kernels (slow); off by default",
        )
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "notebook: executes a notebook kernel; needs --run-notebooks"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-notebooks"):
        return
    skip = pytest.mark.skip(reason="notebook execution needs --run-notebooks")
    for item in items:
        if "notebook" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def confounds():
    return synthetic_bids.make_confounds()


@pytest.fixture
def events():
    return synthetic_bids.make_events()


@pytest.fixture
def dataset(tmp_path, confounds, events):
    return synthetic_bids.write_dataset(tmp_path, confounds, events)


@pytest.fixture
def four_runs(dataset):
    root, prep, *_ = dataset
    return synthetic_bids.add_runs_three_and_four(root, prep)


@pytest.fixture
def six_run_dataset(dataset):
    root, prep, *_ = dataset
    return synthetic_bids.make_six_runs(root, prep)


@pytest.fixture
def cv_library():
    return HrfLibrary.from_parameters([[4, 12, 0.8, 1, 5, 0, 36]])


def _copy_session(directories, source, targets):
    for directory in directories:
        origin = directory / "sub-07" / source / "func"
        for session in targets:
            target = directory / "sub-07" / session / "func"
            target.mkdir(parents=True)
            for path in origin.iterdir():
                if path.is_file():
                    name = path.name.replace(source, session)
                    (target / name).write_bytes(path.read_bytes())


@pytest.fixture
def session_data(four_runs):
    """Three identical four-run sessions, ses-nsd10 to ses-nsd12."""
    _copy_session(four_runs, "ses-nsd10", ("ses-nsd11", "ses-nsd12"))
    return four_runs


@pytest.fixture
def two_raw_sessions(six_run_dataset):
    """Two identical six-run sessions as notebook path settings."""
    _copy_session(six_run_dataset, "ses-nsd10", ("ses-nsd11",))
    root, prep = six_run_dataset
    return dict(bids_root=str(root), fmriprep_root=str(prep))


@pytest.fixture
def saved_sessions(tmp_path):
    brain = nib.cifti2.BrainModelAxis.from_surface([0, 2, 3], 5, "CortexLeft")
    library = HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])
    sessions = ["ses-nsd10", "ses-nsd11", "ses-nsd12"]
    for i, session in enumerate(sessions):
        base = f"sub-07/{session}/func/sub-07_{session}_task-nsdcore"
        meta = dict(
            settings=dict(subject="sub-07", session=session, ridge_mode="off"),
            library_fingerprint=library.fingerprint,
            regressors=["task", "response_time", "trial_type[1]"],
        )
        artifacts = [
            json_artifact(base + "_desc-boldtailor_metadata.json", meta),
            table_artifact(base + "_desc-HRF_library.tsv", library.parameter_table),
            npz_artifact(
                base + "_desc-HRF_library.npz",
                times=library.times,
                curves=library.curves,
            ),
        ]

        def add(descriptor, stat, values, labels):
            artifacts.append(
                scalar_artifact(
                    base
                    + f"_space-fsLR_den-91k_desc-{descriptor}_stat-{stat}.dscalar.nii",
                    brain,
                    values,
                    labels,
                )
            )

        ids = [i % 2, 1, np.nan]
        add(
            "HRFAll",
            "selection",
            [ids, [0.1, 0.2, np.nan], [0.1, 0.1, np.nan], [0, 0.1, np.nan]],
            ["hrf_id", "selected_cv_r2", "canonical_cv_r2", "delta_cv_r2"],
        )
        for optimized, prefix in enumerate(("Canonical", "Optimized")):
            shift = optimized * (i + 1)
            add(
                prefix + "GLM",
                "effects",
                [
                    [10 + i + shift, -4 - i - shift, np.nan],
                    [0.1 * (i + 1 + shift), -0.2 * (i + 1 + shift), np.nan],
                    [99, -99, np.nan],
                ],
                ["task", "response_time", "trial_type[1]"],
            )
            add(
                prefix + "TrialOLS",
                "activation",
                [
                    [2 + shift, -2 - shift, np.nan],
                    [4 + shift, -4 - shift, np.nan],
                    [0.1, 0.1, np.nan],
                    [20, 20, np.nan],
                    [19, 19, np.nan],
                ],
                ["mean_beta", "t", "p_uncorrected", "n_trials", "df"],
            )
            add(
                prefix + "TrialOLS",
                "rsquared",
                [
                    [0.8, 0.8, np.nan],
                    [0.7 - 0.01 * shift, 0.7, np.nan],
                    [0.1 + 0.01 * shift, 0.1, np.nan],
                ],
                ["full_r2", "confounds_r2", "task_delta_r2"],
            )
            add(
                prefix + "TrialOLS",
                "rtcorrelation",
                [
                    [-0.2 - 0.1 * shift, 0.2, np.nan],
                    [0.1, 0.1, np.nan],
                    [0.1, 0.1, np.nan],
                ],
                ["all_runs", "odd_runs", "even_runs"],
            )
        publish_artifact_set(tmp_path, artifacts)
    return tmp_path, sessions, library, brain
