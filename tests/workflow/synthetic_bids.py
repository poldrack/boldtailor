"""Synthetic BIDS/fMRIPrep datasets with small real CIFTI files for workflow tests."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor

from tests.oracles import peak_kernel, scaled_condition


def make_confounds():
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


def make_events():
    return pd.DataFrame(
        {
            "onset": [8.0, 22.0, 38.0, 60.0, 90.0, 112.0],
            "duration": [3.0] * 6,
            "trial_type": [0, 1, 0, 1, 1, 0],
            "response_time": [0.5, 1.0, 2.5, 1.5, 3.0, 0.5],
        }
    )


def write_dataset(tmp_path, confounds, events):
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


def add_runs_three_and_four(root, prep):
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


def make_six_runs(root, prep):
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


def rewrite_events(root, transform):
    """Apply ``transform`` to every run's events table in place."""
    for path in sorted((root / "sub-07" / "ses-nsd10" / "func").glob("*_events.tsv")):
        transform(pd.read_csv(path, sep="\t")).to_csv(path, sep="\t", index=False)


def face_house_events(table):
    """String trial types and no reaction times, as in many BIDS datasets."""
    labels = np.where(np.arange(len(table)) % 2, "house", "face")
    return table.drop(columns="response_time").assign(trial_type=labels)


def task_only_events(table):
    return table.drop(columns=["response_time", "trial_type"])
