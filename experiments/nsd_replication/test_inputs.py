from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.inputs import (
    CORTEX,
    Session,
    analysis_data,
    cortical_indices,
    same_grayordinates,
)

DATA = Path("/Volumes/extdata1/NSD/BIDS")


def _brain(with_cortex=True):
    left = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 2]), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    right = nib.cifti2.BrainModelAxis.from_surface(
        np.array([1]), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, :] = True
    thal = nib.cifti2.BrainModelAxis.from_mask(mask, "CIFTI_STRUCTURE_THALAMUS_LEFT")
    return (left + thal + right) if with_cortex else (left + thal)


def _cortex_only():
    left = nib.cifti2.BrainModelAxis.from_surface(
        np.array([0, 2]), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    right = nib.cifti2.BrainModelAxis.from_surface(
        np.array([1]), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    return left + right


def test_cortical_indices_keep_surface_in_order():
    assert cortical_indices(_brain()).tolist() == [0, 1, 4]


def test_missing_cortical_structure_raises():
    with pytest.raises(ValueError, match="CORTEX_RIGHT"):
        cortical_indices(_brain(with_cortex=False))


def test_same_grayordinates_ignores_volume_metadata():
    full = _brain()
    assert same_grayordinates(full[cortical_indices(full)], _cortex_only())
    assert not same_grayordinates(full[[0, 1]], _cortex_only())


def test_analysis_data_selects_columns_and_keeps_runs():
    signals = (np.arange(12, dtype=np.float32).reshape(4, 3),) * 2
    events = (pd.DataFrame({"onset": [0.0], "duration": [1.0]}),) * 2
    session = Session(
        subject="sub-07",
        session="ses-nsd10",
        labels=("run-01", "run-02"),
        signals=signals,
        events=events,
        confounds=(pd.DataFrame({"c": [0.0, 1, 0, 1]}),) * 2,
        frame_times=(np.arange(4) * 1.6,) * 2,
        brain=_cortex_only(),
        task_model=None,
        source="nsdpreproc",
    )
    data = analysis_data(session, np.array([0, 2]))
    assert data.n_runs == 2
    np.testing.assert_array_equal(data.signals[0], signals[0][:, [0, 2]])


TR = 4 / 3
N_FRAMES = 40


def _cifti(path, seed):
    brain = nib.cifti2.BrainModelAxis.from_surface(
        np.arange(3), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    ) + nib.cifti2.BrainModelAxis.from_surface(
        np.arange(3), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    series = nib.cifti2.SeriesAxis(start=0.0, step=TR, size=N_FRAMES)
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 1.0, (N_FRAMES, len(brain))).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Cifti2Image(data, header=(series, brain)).to_filename(path)


def _tree_events():
    return pd.DataFrame(
        {
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [3.0] * 4,
            "trial_type": [0, 1, 0, 1],
            "response_time": [0.6, 0.8, 1.0, 0.7],
            "73k_id": [11, 12, 13, 14],
        }
    )


@pytest.fixture
def nsd_tree(tmp_path):
    """Raw events, nsdpreproc CIFTIs, and a decoy fMRIPrep CIFTI per run."""
    from experiments.nsd_replication.config import ExperimentConfig

    raw = tmp_path / "sub-07/ses-a/func"
    raw.mkdir(parents=True)
    pre = tmp_path / "derivatives/nsdpreproc/sub-07/ses-a/func"
    prep = tmp_path / "derivatives/fmriprep/sub-07/ses-a/func"
    for i, run in enumerate(("run-01", "run-02")):
        name = f"sub-07_ses-a_task-nsdcore_{run}"
        _tree_events().to_csv(raw / f"{name}_events.tsv", sep="\t", index=False)
        _cifti(pre / f"{name}_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii", i)
        _cifti(prep / f"{name}_space-fsLR_den-91k_bold.dtseries.nii", 99)
    config = ExperimentConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        freesurfer_dir=tmp_path,
        subjects=("sub-07",),
        sessions=("ses-a",),
    )
    return config, pre


def test_nsdpreproc_path_is_bids_layout(nsd_tree):
    from experiments.nsd_replication.inputs import nsdpreproc_path

    config, pre = nsd_tree
    assert nsdpreproc_path(config, "sub-07", "ses-a", "run-02") == (
        pre / "sub-07_ses-a_task-nsdcore_run-02_space-fsLR_den-32k_desc-layerB2_bold"
        ".dtseries.nii"
    )


def test_load_nsdpreproc_opens_only_nsdpreproc_files(nsd_tree, monkeypatch):
    from experiments.nsd_replication.inputs import load_nsdpreproc

    config, pre = nsd_tree
    opened, real_load = [], nib.load

    def recording_load(path, *args, **kwargs):
        opened.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", recording_load)
    session = load_nsdpreproc(config, "sub-07", "ses-a")
    monkeypatch.setattr(nib, "load", real_load)
    assert opened and all(p.is_relative_to(pre) for p in opened)
    assert session.source == "nsdpreproc"
    assert session.labels == ("run-01", "run-02")
    assert session.signals[0].shape == (N_FRAMES, 6)
    assert list(session.confounds[0].columns) == ["poly_0"]
    first = next(pre.glob("*run-01_*.dtseries.nii"))
    np.testing.assert_allclose(
        session.signals[0], real_load(first).get_fdata(), rtol=1e-6
    )


def test_missing_nsdpreproc_run_names_the_run_and_path(nsd_tree):
    from experiments.nsd_replication.inputs import load_nsdpreproc

    config, pre = nsd_tree
    next(pre.glob("*run-02*")).unlink()
    with pytest.raises(FileNotFoundError, match="run-02.*nsdpreproc"):
        load_nsdpreproc(config, "sub-07", "ses-a")


def test_missing_nsdpreproc_session_names_the_run_and_path(nsd_tree):
    import shutil

    from experiments.nsd_replication.inputs import load_nsdpreproc

    config, pre = nsd_tree
    shutil.rmtree(pre.parents[1])
    with pytest.raises(FileNotFoundError, match="run-01.*nsdpreproc/sub-07/ses-a"):
        load_nsdpreproc(config, "sub-07", "ses-a")
