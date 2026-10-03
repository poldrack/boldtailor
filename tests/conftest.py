import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor._single_trial_design import compile_trial_run
from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import select_hrfs
from boldtailor.model import ModelSpec
from boldtailor.provenance import RunSources, SourceRef
from tests.oracles import (
    glm_run_sources,
    hrf_glm_oracle_design,
    peak_design_matrix,
)


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


SESSIONS = ("ses-02", "ses-04")
AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, 42.0],
        [0.0, 2.0, 0.0, 10.0],
        [0.0, 0.0, 2.0, 14.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _events(session):
    trial_types = [
        "go_success",
        "stop_success",
        "stop_failure",
        "go_failure",
        "go_success",
        "stop_success",
        "stop_failure",
        "go_success",
    ]
    if session == "ses-04":
        trial_types.remove("go_failure")
    return pd.DataFrame(
        {
            "onset": np.arange(5.0, 5.0 + 10.0 * len(trial_types), 10.0),
            "duration": np.ones(len(trial_types)),
            "trial_type": trial_types,
        }
    )


def _confounds(n_scans):
    rng = np.random.default_rng(20260808 + n_scans)
    frame = pd.DataFrame(
        rng.normal(0.0, 0.05, size=(n_scans, 6)),
        columns=("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"),
    )
    frame["framewise_displacement"] = np.abs(rng.normal(0.1, 0.02, size=n_scans))
    frame.loc[0, "framewise_displacement"] = np.nan
    return frame


@pytest.fixture
def stop_signal_bids_dataset(tmp_path):
    root = tmp_path / "rdoc_fmri"
    derivative = root / "derivatives" / "fmri_25.2.0"
    rng = np.random.default_rng(20260808)
    for index, session in enumerate(SESSIONS):
        n_scans = 80 + index * 8
        raw_func = root / "sub-s4" / session / "func"
        derivative_func = derivative / "sub-s4" / session / "func"
        raw_func.mkdir(parents=True)
        derivative_func.mkdir(parents=True)
        stem = f"sub-s4_{session}_task-stopSignal_run-01"
        _events(session).to_csv(raw_func / f"{stem}_events.tsv", sep="\t", index=False)
        shape = (7, 7, 7, n_scans)
        signal = rng.normal(1000.0, 3.0, shape).astype(np.float32)
        image = nib.Nifti1Image(signal, AFFINE)
        image.header.set_zooms((2.0, 2.0, 2.0, 1.5))
        nib.save(
            image,
            derivative_func
            / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        )
        (
            derivative_func
            / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json"
        ).write_text('{"RepetitionTime": 1.5, "StartTime": 0.75}\n')
        mask = nib.Nifti1Image(np.ones(shape[:3], dtype=np.uint8), AFFINE)
        nib.save(
            mask,
            derivative_func
            / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        )
        _confounds(n_scans).to_csv(
            derivative_func / f"{stem}_desc-confounds_timeseries.tsv",
            sep="\t",
            index=False,
        )
    (root / "dataset_description.json").write_text(
        '{"Name":"fixture","BIDSVersion":"1.11.1"}\n'
    )
    return root


@pytest.fixture
def ridge_problem():
    library = HrfLibrary.from_parameters(
        [[4, 12, 0.8, 1, 5, 0, 36], [6, 16, 1.5, 2, 8, 1, 36]]
    )
    rng = np.random.default_rng(273)
    signals, events, times, confounds, predictors = [], [], [], [], []
    for r in range(6):
        t = 0.75 + 1.5 * np.arange(86 + 3 * r)
        p = pd.DataFrame(
            dict(response_time=rng.uniform(0.3, 1.8, 8), trial_type=np.arange(8) % 2)
        )
        e = p.assign(
            onset=8 + np.arange(8) * 8.3 + 0.1 * r,
            duration=1.5,
            stimulus_id=np.arange(8) + 8 * r,
        )
        n = pd.DataFrame(
            dict(
                motion=np.sin(np.arange(len(t)) / 8 + r),
                drift=np.linspace(-1, 1, len(t)),
            )
        )
        columns = []
        for v in range(4):
            x, _, _ = compile_trial_run(
                e, t, n, f"run-{r}", hrf=library.candidates[v % 3]
            )
            beta = (
                2 + 0.8 * p.response_time - 0.6 * p.trial_type + rng.normal(0, 0.5, 8)
            )
            columns.append(x.to_numpy() @ beta + rng.normal(0, 0.04, len(t)))
        y = np.column_stack(columns) + 25 + n.motion.to_numpy()[:, None] * 0.3
        signals.append(np.column_stack([y, np.full(len(t), 25.0)]))
        if r == 0:
            p.loc[2, "response_time"] = np.nan
            e.loc[2, "response_time"] = np.nan
        predictors.append(p)
        events.append(e)
        times.append(t)
        confounds.append(n)
    return (
        from_arrays(signals, events, frame_times=times, confounds=confounds),
        predictors,
        library,
    )


@pytest.fixture
def selected_fixture(two_candidate_library):
    library = two_candidate_library
    events = []
    signals = []
    times = []
    confounds = []
    for r in range(3):
        t = 0.775 + 1.6 * np.arange(75 + r * 3)
        e = pd.DataFrame(
            dict(
                onset=[30.1 + r, 8.2 + r, 53.3 + r],
                duration=[1.2, 3.0, 2.0],
                image=[4, 4, 5],
                response_time=[1.1, np.nan, 0.7],
                details=[{"tags": [r]}, None, None],
            )
        )
        n = pd.DataFrame(dict(motion=np.linspace(-1, 1, len(t))))
        columns = []
        for cid in [1, 0, 1, 2]:
            c = library.candidates[cid]
            x = np.column_stack(
                [
                    compute_regressor(
                        np.array([[o], [d], [1.0]]), "spm" if cid == 0 else c.kernel, t
                    )[0][:, 0]
                    for o, d in zip(e.onset, e.duration, strict=True)
                ]
            )
            columns.append(x @ np.array([2.9, 3.0, 3.1]) + n.motion * (r + 1) + 50)
        y = np.column_stack([*columns, np.ones(len(t)) * 100])
        events.append(e)
        times.append(t)
        confounds.append(n)
        signals.append(y)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    selection = select_hrfs(data, library=library, feature_signature="ordered-axis")
    np.testing.assert_array_equal(selection.hrf_indices, [1, 0, 1, 2, -1])
    return data, selection


@pytest.fixture(scope="session")
def two_candidate_library():
    return HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )


_SUFFIX = {"signal": "bold", "events": "events", "confounds": "confounds"}


def _source_ref(role, run):
    stem = f"sub-01/func/sub-01_task-localizer_run-{run:02d}"
    return SourceRef(
        role=role,
        uri=f"{stem}_{_SUFFIX[role]}.tsv",
        media_type="text/tab-separated-values",
        byte_size=1024 + run,
        modified_at="2026-08-08T12:00:00Z",
    )


@pytest.fixture
def complete_sources():
    def build(n_runs):
        return [
            RunSources(
                signal=_source_ref("signal", run),
                events=_source_ref("events", run),
                confounds=_source_ref("confounds", run),
            )
            for run in range(1, n_runs + 1)
        ]

    return build


@pytest.fixture
def fail_glm(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("run_glm must not be called before contrast preflight")

    monkeypatch.setattr("boldtailor._conventional.run_glm", fail)


@pytest.fixture
def single_run_problem():
    rng = np.random.default_rng(7)
    events = pd.DataFrame(
        {
            "onset": [0.0, 8.0, 16.0, 24.0, 32.0, 40.0],
            "duration": np.ones(6),
            "trial_type": ["face", "house", "face", "house", "face", "house"],
        }
    )
    frame_times = np.arange(30) * 2.0
    design = peak_design_matrix(
        frame_times,
        events=events,
        hrf_model="glover",
        drift_model=None,
        min_onset=-24.0,
    )
    beta = np.array([[2.0, 1.0], [0.5, 1.5], [10.0, 12.0]])
    signals = design.to_numpy() @ beta + rng.normal(0.0, 0.05, (30, 2))
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model=None,
        noise_model="ols",
    )
    return signals, events, design, model


@pytest.fixture(scope="module")
def hrf_glm_problem(two_candidate_library):
    library = two_candidate_library
    ids = [1, 0, 1, 2]
    model = ModelSpec(
        contrasts={"stimulus": {"stimulus": 1}, "rt_effect": "rt"},
        confounds=("motion",),
        high_pass=0.01,
        oversampling=20,
        min_onset=-10,
        noise_model="ols",
    )
    events, times, confounds, training, signals, designs = [], [], [], [], [], {}
    rng = np.random.default_rng(734)
    for run in range(3):
        t = 0.775 + 1.6 * np.arange(85 + 5 * run)
        stimulus = pd.DataFrame(
            dict(
                onset=np.array([5.3, 21.1, 42.2, 64.4, 88.5, 110.2]) + run,
                duration=[1.2, 2.0, 0.7, 1.5, 1.1, 2.3],
                trial_type="stimulus",
                modulation=1.0,
            )
        )
        rt = stimulus.assign(
            trial_type="rt", modulation=[-0.3, 0.1, 0.5, -0.4, 0.3, -0.2]
        )
        e = pd.concat([stimulus, rt], ignore_index=True)
        n = pd.DataFrame(
            dict(motion=np.linspace(-1, 1, len(t)), unused=rng.normal(size=len(t)))
        )
        mean_columns, target_columns = [], []
        for feature, cid in enumerate(ids):
            candidate = library.candidates[cid]
            mean = compute_regressor(
                stimulus[["onset", "duration", "modulation"]].to_numpy().T,
                "spm" if cid == 0 else candidate.kernel,
                t,
            )[0][:, 0]
            mean_columns.append(3 * mean + 100 + n.motion)
            design = hrf_glm_oracle_design(e, t, n, candidate, model)
            designs[run, cid] = design
            coefficients = np.zeros(design.shape[1])
            coefficients[design.columns.get_loc("stimulus")] = 2 + feature + run
            coefficients[design.columns.get_loc("rt")] = 0.4 - feature / 4
            coefficients[design.columns.get_loc("motion")] = 1.3
            coefficients[design.columns.get_loc("constant")] = 100 + run
            noise = rng.normal(0, 0.03, len(t))
            for scan in range(1, len(t)):
                noise[scan] += 0.5 * noise[scan - 1]
            target_columns.append(design.to_numpy() @ coefficients + noise)
        events.append(e)
        times.append(t)
        confounds.append(n)
        training.append(np.column_stack([*mean_columns, np.full(len(t), 100.0)]))
        signals.append(np.column_stack([*target_columns, rng.normal(100, 1, len(t))]))
    training_data = from_arrays(
        training,
        [e.iloc[:6] for e in events],
        frame_times=times,
        confounds=[n[["motion"]] for n in confounds],
    )
    selection = select_hrfs(training_data, library=library, feature_signature="axis-v1")
    np.testing.assert_array_equal(selection.hrf_indices, [*ids, -1])
    data = from_arrays(
        signals,
        events,
        frame_times=times,
        confounds=confounds,
        sources=[glm_run_sources(r) for r in range(3)],
    )
    return data, model, selection, designs
