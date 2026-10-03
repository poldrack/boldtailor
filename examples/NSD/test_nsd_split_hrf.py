"""Independent odd/even HRF maps retain parameter identity and spatial order."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor.hrf_library import PARAMETER_NAMES
from examples.NSD.nsd_single_trial import run_single_trial_analysis


def find(paths, fragment):
    return next(p for p in paths if fragment in p.name)


def run(fixture, output, block=2):
    return run_single_trial_analysis(
        fixture[0],
        fixture[1],
        output,
        ridge_alpha=0.1,
        block_size=block,
        hrf_library="expanded",
    )


def replace_signals(fixture, numbers, candidate_ids):
    root, prep, _, library, _ = fixture
    for number in numbers:
        events_path = next(root.glob(f"sub-07/*/func/*run-{number:02d}_events.tsv"))
        events = pd.read_csv(events_path, sep="\t")
        path = next(prep.rglob(f"*run-{number:02d}*.dtseries.nii"))
        original = nib.load(path)
        times = 0.775 + 1.6 * np.arange(original.shape[0])
        signals = np.zeros(original.shape)
        for feature, cid in enumerate(candidate_ids):
            x, _ = compute_regressor(
                np.array([events.onset, events.duration, np.ones(len(events))]),
                library.candidates[cid].kernel,
                times,
            )
            signals[:, feature] = 100 + 3 * x[:, 0]
        nib.save(nib.Cifti2Image(signals, original.header), path)


@pytest.fixture
def different_halves(hrf_nsd):
    replace_signals(hrf_nsd, [1, 3], [1, 2, 1])
    replace_signals(hrf_nsd, [2, 4], [2, 1, 2])
    return hrf_nsd


def test_split_parameters_match_each_halfs_library_winners(different_halves, tmp_path):
    paths = run(different_halves, tmp_path / "split")
    _, _, brain, library, _ = different_halves
    names = [*PARAMETER_NAMES, "peak_time"]
    expected = {"odd": [1, 2, 1], "even": [2, 1, 2]}
    for half, wanted in expected.items():
        ids = nib.load(find(paths, f"stat-{half}hrfindex."))
        np.testing.assert_array_equal(ids.get_fdata()[0, :3], wanted)
        assert np.isnan(ids.get_fdata()[0, 3])
        parameters = nib.load(find(paths, f"stat-{half}hrfparameters."))
        assert ids.header.get_axis(1) == parameters.header.get_axis(1) == brain
        assert parameters.header.get_axis(0).name.tolist() == names
        np.testing.assert_allclose(
            parameters.get_fdata()[:, :3],
            library.parameter_table.loc[wanted, names].to_numpy().T,
            rtol=0,
            atol=3e-7,
        )
        assert np.isnan(parameters.get_fdata()[:, 3]).all()
    metadata = json.loads(find(paths, "_splitmetadata.json").read_text())
    assert metadata["LibraryFingerprint"] == library.fingerprint
    assert metadata["Splits"]["odd"]["RunLabels"] == ["run-01", "run-03"]
    assert metadata["Splits"]["even"]["RunLabels"] == ["run-02", "run-04"]
    assert metadata["ParameterMaps"] == names
    provenance = json.loads(find(paths, "_splitprovenance.json").read_text())
    for half in expected:
        assert metadata["Splits"][half]["Available"] is True
        assert len(provenance[half]) == 2
        for block in provenance[half]:
            activity = block["activities"][-1]
            labels = metadata["Splits"][half]["RunLabels"]
            assert activity["run_labels"] == labels
            for fold in activity["folds"]:
                assert set(fold["train"] + fold["test"]) == set(labels)
                assert not set(fold["train"]) & set(fold["test"])
    eligibility = pd.read_csv(find(paths, "_eligibility.tsv"), sep="\t")
    assert set(eligibility.scope) == {"all", "odd", "even"}
    folds = pd.read_csv(find(paths, "_folds.tsv"), sep="\t")
    even = folds[folds.scope == "even_run_selection"]
    assert set(even.test) == {"run-02", "run-04"}
    assert set(even.train) == {"run-02", "run-04"}


@pytest.mark.parametrize(
    "unchanged,changed,runs", [("odd", "even", [2, 4]), ("even", "odd", [1, 3])]
)
def test_split_selection_cannot_use_other_halfs_signals(
    different_halves, tmp_path, unchanged, changed, runs
):
    before = run(different_halves, tmp_path / "before")
    # Force the edited half to choose the exact canonical HRF.
    replace_signals(different_halves, runs, [0, 0, 0])
    after = run(different_halves, tmp_path / "after")
    for suffix in ("hrfindex", "hrfparameters"):
        fragment = f"stat-{unchanged}{suffix}."
        np.testing.assert_array_equal(
            nib.load(find(before, fragment)).get_fdata(),
            nib.load(find(after, fragment)).get_fdata(),
        )
    np.testing.assert_array_equal(
        nib.load(find(after, f"stat-{changed}hrfindex.")).get_fdata()[0, :3],
        [0, 0, 0],
    )


def test_insufficient_half_has_nan_maps_and_explicit_reason(hrf_nsd, tmp_path):
    root, prep, *_ = hrf_nsd
    for directory in (root / "sub-07", prep):
        for path in directory.rglob("*run-04*"):
            path.unlink()
    paths = run(hrf_nsd, tmp_path / "three_runs")
    assert np.isfinite(
        nib.load(find(paths, "stat-oddhrfparameters.")).get_fdata()[:, :3]
    ).all()
    for suffix in ("hrfindex", "hrfparameters"):
        assert np.isnan(nib.load(find(paths, f"stat-even{suffix}.")).get_fdata()).all()
    meta = json.loads(find(paths, "_splitmetadata.json").read_text())
    assert meta["Splits"]["even"]["Available"] is False
    assert "two" in meta["Splits"]["even"]["Reason"]


@pytest.mark.parametrize(
    "stat", ["evenhrfindex", "oddhrfparameters", "evenhrfparameters"]
)
def test_new_map_collision_is_checked_before_fitting(
    hrf_nsd, tmp_path, monkeypatch, stat
):
    output = tmp_path / "collision"
    target = output / (
        "sub-07/ses-nsd10/func/"
        f"sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-hrfSelection_stat-{stat}.dscalar.nii"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve")

    def forbidden(*args, **kwargs):
        pytest.fail("new map collision must precede fitting")

    monkeypatch.setattr("boldtailor.hrf_selection.select_hrf", forbidden)
    with pytest.raises(FileExistsError):
        run(hrf_nsd, output)
    assert target.read_bytes() == b"preserve"
