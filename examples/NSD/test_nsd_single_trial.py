"""Single-trial imaging acceptance checks with real miniature CIFTIs."""

import importlib
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401
from boldtailor._hrf_design import hrf_model
from tests.oracles import scaled_condition


def example():
    try:
        return importlib.import_module("examples.NSD.nsd_single_trial")
    except ModuleNotFoundError:
        pytest.fail("Single-trial CIFTI runner is not implemented")


@pytest.fixture
def mini_nsd(dataset):
    root, prep, _, _, nuisance, brain = dataset
    times = 0.775 + np.arange(96) * 1.6
    rng = np.random.default_rng(92)
    signals = []
    for run in (1, 2):
        event_path = next((root / "sub-07").rglob(f"*run-{run:02d}_events.tsv"))
        table = pd.read_csv(event_path, sep="\t")
        table["73k_id"] = [4, 4, 5, 5, 4, 5]
        table.loc[1, "response_time"] = np.nan
        table.loc[5, "response_time"] = -1.0
        table.to_csv(event_path, sep="\t", index=False)
        x, simulated = (
            np.column_stack(
                [
                    compute_regressor(
                        (
                            np.array([[t], [3.0], [1.0]])
                            if hrf == "spm"
                            else scaled_condition([t], [3.0], 1.0, hrf, times)
                        ),
                        hrf,
                        times,
                    )[0][:, 0]
                    for t in table.onset
                ]
            )
            for hrf in (hrf_model("spm"), "spm")
        )
        # x is the peak-one oracle; simulated responses keep original amplitudes.
        y = simulated @ rng.normal(size=(6, 4)) + nuisance @ rng.normal(size=(33, 4))
        y += rng.normal(scale=0.1 * run, size=y.shape) + 100 * run
        y[:, -1] = 0
        path = next(prep.rglob(f"*run-{run:02d}*.dtseries.nii"))
        nib.save(nib.Cifti2Image(y, nib.load(path).header), path)
        signals.append(y)
    return root, prep, brain, signals, x, nuisance


def test_cifti_betas_and_pooled_maps_match_independent_fits(mini_nsd, tmp_path):
    root, prep, brain, signals, x, n = mini_nsd
    output = tmp_path / "output"
    output.mkdir()
    old = output / "old_desc-full_stat-rsquared.dscalar.nii"
    old.write_bytes(b"preserve conventional result")
    description = output / "dataset_description.json"
    description.write_text('{"Name": "existing derivative"}')
    paths = example().run_single_trial_analysis(
        root, prep, output, ridge_alpha=0.1, block_size=2
    )
    assert old.read_bytes() == b"preserve conventional result"
    assert description.read_text() == '{"Name": "existing derivative"}'
    for model, alpha in (("OLS", 0.0), ("Ridge", 0.1)):
        trials_path = next(
            p for p in paths if f"singletrial{model}_trials.tsv" in p.name
        )
        trials = pd.read_csv(trials_path, sep="\t")
        assert len(trials) == 12
        assert trials.event_index.tolist() == list(range(6)) * 2
        assert trials["73k_id"].tolist() == [4, 4, 5, 5, 4, 5] * 2
        assert trials.response_time.isna().sum() == 2
        sse, null_sse, sst = [], [], []
        for run, y in enumerate(signals, 1):
            path = next(
                p
                for p in paths
                if f"run-{run:02d}" in p.name and f"singletrial{model}_betas" in p.name
            )
            image = nib.load(path)
            assert image.header.get_axis(1) == brain
            assert image.header.get_axis(0).name.tolist() == [
                f"run-{run:02d}_trial-{i:04d}" for i in range(1, 7)
            ]
            xr = x - n @ np.linalg.lstsq(n, x, rcond=None)[0]
            design = np.column_stack([x, n])
            penalty = np.column_stack(
                [
                    np.diag(np.sqrt(alpha) * np.linalg.norm(xr, axis=0)),
                    np.zeros((6, 33)),
                ]
            )
            beta = np.linalg.lstsq(
                np.vstack([design, penalty]),
                np.vstack([y, np.zeros((6, 4))]),
                rcond=None,
            )[0]
            np.testing.assert_allclose(
                image.get_fdata()[:, :3], beta[:6, :3], atol=2e-6
            )
            assert np.isnan(image.get_fdata()[:, -1]).all()
            sse.append(np.sum((y - design @ beta) ** 2, axis=0))
            null_sse.append(
                np.sum((y - n @ np.linalg.lstsq(n, y, rcond=None)[0]) ** 2, axis=0)
            )
            sst.append(np.sum((y - y.mean(axis=0)) ** 2, axis=0))
        full = 1 - np.sum(sse, axis=0)[:3] / np.sum(sst, axis=0)[:3]
        null = 1 - np.sum(null_sse, axis=0)[:3] / np.sum(sst, axis=0)[:3]
        for statistic, expected in (
            ("fullrsquared", full),
            ("confoundsrsquared", null),
            ("deltarsquared", full - null),
        ):
            image = nib.load(
                next(
                    p
                    for p in paths
                    if f"singletrial{model}_stat-{statistic}.dscalar.nii" in p.name
                )
            )
            np.testing.assert_allclose(image.get_fdata()[0, :3], expected, atol=2e-7)
            assert np.isnan(image.get_fdata()[0, -1])
        count_image = nib.load(
            next(
                p
                for p in paths
                if f"singletrial{model}_stat-rtcount.dscalar.nii" in p.name
            )
        )
        assert count_image.header.get_axis(0).name.tolist() == [
            "all",
            "odd",
            "even",
            "run-01",
            "run-02",
        ]
        np.testing.assert_array_equal(count_image.get_fdata()[:, 0], [8, 4, 4, 4, 4])
        np.testing.assert_array_equal(count_image.get_fdata()[:, -1], 0)
        metadata = json.loads(
            next(
                p for p in paths if f"singletrial{model}_metadata.json" in p.name
            ).read_text()
        )
        assert metadata["ridge_alpha"] == alpha
    assert any(p.name.endswith("_scatter.png") for p in paths)


def test_collision_stops_before_fitting(mini_nsd, tmp_path, monkeypatch):
    root, prep, *_ = mini_nsd
    api = example()
    output = tmp_path / "output"
    target = (
        output
        / "sub-07/ses-nsd10/func/sub-07_ses-nsd10_task-nsdcore_desc-singletrialOLS_trials.tsv"
    )
    target.parent.mkdir(parents=True)
    target.write_text("old result")

    def unexpected_fit(*args, **kwargs):
        pytest.fail("collision must be detected before fitting")

    monkeypatch.setattr(api, "fit_single_trials", unexpected_fit)
    with pytest.raises(FileExistsError):
        api.run_single_trial_analysis(root, prep, output)
    assert target.read_text() == "old result"


@pytest.mark.parametrize("problem", ["missing", "axis"])
def test_incomplete_or_misaligned_runs_rejected(mini_nsd, tmp_path, problem):
    root, prep, *_ = mini_nsd
    path = next(prep.rglob("*run-02*.dtseries.nii"))
    if problem == "missing":
        path.unlink()
    else:
        image = nib.load(path)
        header = nib.Cifti2Header.from_axes(
            (image.header.get_axis(0), image.header.get_axis(1)[::-1])
        )
        nib.save(nib.Cifti2Image(image.get_fdata(), header), path)
    with pytest.raises(
        (ValueError, FileNotFoundError), match="run-02|grayordinate|BrainModel"
    ):
        example().run_single_trial_analysis(root, prep, tmp_path / "output")


def test_failed_publication_leaves_no_partial_single_trial_set(
    mini_nsd, tmp_path, monkeypatch
):
    import boldtailor.publication as publication

    root, prep, *_ = mini_nsd
    original = publication.os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected promotion failure")
        return original(source, target)

    monkeypatch.setattr(publication.os, "replace", fail_second)
    output = tmp_path / "output"
    with pytest.raises(publication.PublicationError):
        example().run_single_trial_analysis(root, prep, output)
    assert not list(output.rglob("*desc-singletrial*"))


def test_block_size_does_not_change_maps_or_trial_identity(mini_nsd, tmp_path):
    root, prep, *_ = mini_nsd
    small = example().run_single_trial_analysis(
        root, prep, tmp_path / "small", ridge_alpha=0.1, block_size=1
    )
    large = example().run_single_trial_analysis(
        root, prep, tmp_path / "large", ridge_alpha=0.1, block_size=4096
    )
    by_name = {p.name: p for p in large}
    for path in small:
        if path.name.endswith(".dscalar.nii"):
            a, b = nib.load(path), nib.load(by_name[path.name])
            assert a.header.get_axis(0) == b.header.get_axis(0)
            assert a.header.get_axis(1) == b.header.get_axis(1)
            np.testing.assert_allclose(
                a.get_fdata(), b.get_fdata(), atol=1e-7, equal_nan=True
            )
        elif path.name.endswith("_trials.tsv"):
            assert path.read_bytes() == by_name[path.name].read_bytes()
