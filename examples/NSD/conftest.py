"""Shared NSD example fixtures: small real CIFTI sessions and saved results."""

import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

# The scripts import their siblings as examples.NSD.*; put the repo root on
# sys.path so that works under plain `pytest` (pyproject no longer sets pythonpath).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from nilearn.glm.first_level import compute_regressor  # noqa: E402

from boldtailor.cifti import scalar_artifact  # noqa: E402
from boldtailor.hrf_library import HrfLibrary  # noqa: E402
from boldtailor.publication import publish_artifact_set  # noqa: E402
from examples.NSD.workflow_artifacts import (  # noqa: E402
    json_artifact,
    npz_artifact,
    table_artifact,
)
from tests.oracles import peak_kernel, scaled_condition  # noqa: E402


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
    rng = np.random.default_rng(54)
    table = pd.DataFrame()
    for axis in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"):
        motion = rng.normal(size=96)
        derivative = np.r_[np.nan, np.diff(motion)]
        for suffix, values in {
            "": motion,
            "_derivative1": derivative,
            "_power2": motion**2,
            "_derivative1_power2": derivative**2,
        }.items():
            table[axis + suffix] = values
    metadata = {}
    for i in range(8):
        name = f"a_comp_cor_{i:02d}"
        table[name] = rng.normal(size=96)
        metadata[name] = {
            "Mask": "combined",
            "Retained": True,
            "VarianceExplained": (8 - i) / 100,
            "Method": "aCompCor",
        }
    # A separate mask must not displace the combined-mask components.
    table["a_comp_cor_08"] = rng.normal(size=96)
    metadata["a_comp_cor_08"] = {
        "Mask": "CSF",
        "Retained": True,
        "VarianceExplained": 0.8,
    }
    table["cosine00"] = np.cos(np.pi * (np.arange(96) + 0.5) / 96)
    table["non_steady_state_outlier00"] = np.r_[1.0, np.zeros(95)]
    return table, metadata


@pytest.fixture
def events():
    return pd.DataFrame(
        {
            "onset": [8.0, 22.0, 38.0, 60.0, 90.0, 112.0],
            "duration": [3.0] * 6,
            "trial_type": [0, 1, 0, 1, 1, 0],
            "response_time": [0.5, 1.0, 2.5, 1.5, 3.0, 0.5],
        }
    )


@pytest.fixture
def dataset(tmp_path, confounds, events):
    root = tmp_path / "bids"
    prep = root / "derivatives" / "fmriprep"
    raw_func = root / "sub-07" / "ses-nsd10" / "func"
    prep_func = prep / "sub-07" / "ses-nsd10" / "func"
    raw_func.mkdir(parents=True)
    prep_func.mkdir(parents=True)
    brain = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 2, 3, 6]),
        8,
        name="CortexLeft",
    )
    table, metadata = confounds
    times = 0.775 + np.arange(96) * 1.6
    # The oracle design uses the peak-one kernel; simulated responses keep
    # their original Nilearn-scaled amplitudes.
    task, simulated_task = (
        np.column_stack(
            [
                compute_regressor(
                    (
                        np.vstack([events.onset, events.duration, amp])
                        if hrf == "spm"
                        else scaled_condition(
                            events.onset, events.duration, amp, hrf, times
                        )
                    ),
                    hrf,
                    times,
                )[0]
                for amp in [np.ones(6), np.array([-1, -0.5, 1, 0, 1.5, -1])]
            ]
        )
        for hrf in (peak_kernel("spm"), "spm")
    )
    nuisance = np.column_stack(
        [
            table.iloc[:, :30].fillna(0),
            table.cosine00,
            table.non_steady_state_outlier00,
            np.ones(96),
        ]
    )
    full = np.column_stack([task, nuisance])
    simulated = np.column_stack([simulated_task, nuisance])
    rng = np.random.default_rng(23)
    signal_runs = []
    # Unequal variances and run means expose erroneous arithmetic/global pooling.
    for run, scale in [(1, 1), (2, 5)]:
        stem = f"sub-07_ses-nsd10_task-nsdcore_run-{run:02d}"
        events.to_csv(raw_func / f"{stem}_events.tsv", sep="\t", index=False)
        table.to_csv(
            prep_func / f"{stem}_desc-confounds_timeseries.tsv", sep="\t", index=False
        )
        (prep_func / f"{stem}_desc-confounds_timeseries.json").write_text(
            json.dumps(metadata)
        )
        y = scale * (simulated @ rng.normal(size=(35, 4)) + rng.normal(size=(96, 4)))
        y += run * 100
        y[:, -1] = 0  # Undefined R² must retain its spatial position as NaN.
        signal_runs.append(y)
        axes = (nib.cifti2.SeriesAxis(0, 1.6, 96), brain)
        image = nib.Cifti2Image(y, header=nib.Cifti2Header.from_axes(axes))
        nib.save(image, prep_func / f"{stem}_space-fsLR_den-91k_bold.dtseries.nii")
        (prep_func / f"{stem}_space-fsLR_den-91k_bold.json").write_text(
            json.dumps(
                {
                    "StartTime": 0.775,
                    "RepetitionTime": 1.6,
                    "SliceTimingCorrected": True,
                }
            )
        )
    return root, prep, signal_runs, full, nuisance, brain


@pytest.fixture
def four_runs(dataset):
    root, prep, *_ = dataset
    for directory in (root, prep):
        func = directory / "sub-07/ses-nsd10/func"
        for source in list(func.glob("*run-0[12]*")):
            target = source.with_name(
                source.name.replace("run-01", "run-03").replace("run-02", "run-04")
            )
            target.write_bytes(source.read_bytes())
    for number, dropped in enumerate((1, 2, 3, 1), 1):
        path = next(prep.rglob(f"*run-{number:02d}*confounds_timeseries.tsv"))
        table = pd.read_csv(path, sep="\t")
        for i in range(dropped):
            table[f"non_steady_state_outlier{i:02d}"] = (
                np.arange(len(table)) == i
            ).astype(int)
        table.to_csv(path, sep="\t", index=False)
    return root, prep


@pytest.fixture
def small_library():
    return HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])


@pytest.fixture
def six_run_dataset(dataset):
    root, prep, *_ = dataset
    for directory in (root, prep):
        func = directory / "sub-07/ses-nsd10/func"
        sources = list(func.glob("*run-01*"))
        for number in (3, 4, 5, 6):
            for source in sources:
                target = source.with_name(
                    source.name.replace("run-01", f"run-{number:02d}")
                )
                target.write_bytes(source.read_bytes())
    rng = np.random.default_rng(75)
    for number in range(1, 7):
        path = next(prep.rglob(f"*run-{number:02d}*dtseries.nii"))
        image = nib.load(path)
        y = image.get_fdata()
        y[:, :3] += rng.normal(0, 0.3, y[:, :3].shape)
        nib.save(nib.Cifti2Image(y, header=image.header), path)
        event_path = next((root / "sub-07").rglob(f"*run-{number:02d}*events.tsv"))
        table = pd.read_csv(event_path, sep="\t")
        table.response_time += 0.03 * number
        table["stimulus_id"] = np.arange(len(table)) + number * 100
        table.to_csv(event_path, sep="\t", index=False)
    return root, prep


@pytest.fixture
def cv_library():
    return HrfLibrary.from_parameters([[4, 12, 0.8, 1, 5, 0, 36]])


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
        )
        artifacts = [
            json_artifact(base + "_desc-notebook_metadata.json", meta),
            table_artifact(
                base + "_desc-notebookHRF_library.tsv", library.parameter_table
            ),
            npz_artifact(
                base + "_desc-notebookHRF_library.npz",
                times=library.times,
                curves=library.curves,
            ),
        ]

        def add(descriptor, stat, values, labels):
            artifacts.append(
                scalar_artifact(
                    base
                    + f"_space-fsLR_den-91k_desc-notebook{descriptor}_stat-{stat}.dscalar.nii",
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
                ["task", "response_time", "trial_type"],
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
