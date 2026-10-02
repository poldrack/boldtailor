"""Real CIFTI acceptance tests for selection, grouped exports and isolation."""

import json
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor.hrf_library import HrfLibrary
from examples.NSD.nsd_single_trial import run_single_trial_analysis
from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401
from boldtailor._hrf_design import hrf_model


@pytest.fixture
def hrf_nsd(dataset, monkeypatch):
    root, prep, _, _, nuisance, brain = dataset
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
    # Restrict only the library boundary; exercise the real selection/fitting/I/O.
    import boldtailor.hrf_library as libraries

    monkeypatch.setattr(libraries, "expanded_hrf_library", lambda: library)
    raw = root / "sub-07/ses-nsd10/func"
    prepared = prep / "sub-07/ses-nsd10/func"
    for folder in (raw, prepared):
        originals = list(folder.glob("*run-01*"))
        for number in (3, 4):
            for path in originals:
                shutil.copyfile(
                    path, folder / path.name.replace("run-01", f"run-{number:02d}")
                )
    rng = np.random.default_rng(617)
    signals = []
    for number in range(1, 5):
        path = next(raw.glob(f"*run-{number:02d}_events.tsv"))
        e = pd.read_csv(path, sep="\t")
        e.onset += number * 0.17
        e["73k_id"] = np.arange(6) + 10 * number
        e.loc[1, "response_time"] = np.nan
        e.to_csv(path, sep="\t", index=False)
        t = 0.775 + 1.6 * np.arange(96)
        columns = []
        for cid in [1, 2, 1]:
            c = library.candidates[cid]
            x = np.column_stack(
                [
                    compute_regressor(np.array([[o], [d], [1.0]]), c.kernel, t)[0][:, 0]
                    for o, d in zip(e.onset, e.duration, strict=True)
                ]
            )
            beta = (
                3 + 0.1 * e.response_time.fillna(1).to_numpy() + rng.normal(0, 0.03, 6)
            )
            columns.append(
                x @ beta + nuisance @ rng.normal(0, 0.05, 33) + 100 + number * 7
            )
        y = np.column_stack([*columns, np.zeros(96)])
        bold = next(prepared.glob(f"*run-{number:02d}*.dtseries.nii"))
        nib.save(nib.Cifti2Image(y, nib.load(bold).header), bold)
        signals.append(y)
    return root, prep, brain, library, signals


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


def test_expanded_artifacts_reconstruct_grouped_fits(hrf_nsd, tmp_path):
    root, prep, brain, library, ys = hrf_nsd
    output = tmp_path / "output"
    output.mkdir()
    old = output / "old.dscalar.nii"
    old.write_bytes(b"preserve")
    paths = run(hrf_nsd, output)
    assert old.read_bytes() == b"preserve"
    ids = nib.load(find(paths, "desc-hrfSelection_stat-hrfindex."))
    assert ids.header.get_axis(1) == brain
    np.testing.assert_array_equal(ids.get_fdata()[0, :3], [1, 2, 1])
    assert np.isnan(ids.get_fdata()[0, 3])
    for model, alpha in [("OLS", 0.0), ("Ridge", 0.1)]:
        trials = pd.read_csv(find(paths, f"desc-hrfOpt{model}_trials.tsv"), sep="\t")
        assert len(trials) == 24
        assert trials.event_index.tolist() == list(range(6)) * 4
        sse = []
        null = []
        total = []
        for r in range(4):
            prefix = f"run-{r+1:02d}"
            beta = nib.load(
                next(
                    p
                    for p in paths
                    if prefix in p.name and f"desc-hrfOpt{model}_betas" in p.name
                )
            )
            assert beta.header.get_axis(1) == brain
            assert beta.header.get_axis(0).name.tolist() == [
                f"{prefix}_trial-{i:04d}" for i in range(1, 7)
            ]
            archive = next(
                p
                for p in paths
                if prefix in p.name and p.name.endswith("desc-hrfSelection_designs.npz")
            )
            losses = []
            nulls = []
            totals = []
            with np.load(archive, allow_pickle=False) as saved:
                for key in saved.files:
                    assert saved[key].dtype.kind != "O"
                n = saved["nuisance"]
                assert n.shape == (96, 33)
                np.testing.assert_array_equal(
                    saved["frame_times"], 0.775 + 1.6 * np.arange(96)
                )
                for v, cid in enumerate([1, 2, 1]):
                    x = saved[f"hrf_{cid}"]
                    assert x.dtype == np.float64
                    xr = x - n @ np.linalg.lstsq(n, x, rcond=None)[0]
                    matrix = np.column_stack([x, n])
                    penalty = np.column_stack(
                        [
                            np.diag(np.sqrt(alpha) * np.linalg.norm(xr, axis=0)),
                            np.zeros((6, 33)),
                        ]
                    )
                    expected = np.linalg.lstsq(
                        np.vstack([matrix, penalty]),
                        np.r_[ys[r][:, v], np.zeros(6)],
                        rcond=None,
                    )[0]
                    np.testing.assert_allclose(
                        beta.get_fdata()[:, v], expected[:6], atol=2e-6
                    )
                    losses.append(np.sum((ys[r][:, v] - matrix @ expected) ** 2))
                    nulls.append(
                        np.sum(
                            (
                                ys[r][:, v]
                                - n @ np.linalg.lstsq(n, ys[r][:, v], rcond=None)[0]
                            )
                            ** 2
                        )
                    )
                    totals.append(np.sum((ys[r][:, v] - ys[r][:, v].mean()) ** 2))
            sse.append(losses)
            null.append(nulls)
            total.append(totals)
            assert np.isnan(beta.get_fdata()[:, 3]).all()
        full = 1 - np.sum(sse, axis=0) / np.sum(total, axis=0)
        nuisance = 1 - np.sum(null, axis=0) / np.sum(total, axis=0)
        for stat, want in [
            ("fullrsquared", full),
            ("confoundsrsquared", nuisance),
            ("deltarsquared", full - nuisance),
        ]:
            actual = nib.load(find(paths, f"desc-hrfOpt{model}_stat-{stat}."))
            np.testing.assert_allclose(actual.get_fdata()[0, :3], want, atol=2e-7)
        meta = json.loads(find(paths, f"desc-hrfOpt{model}_metadata.json").read_text())
        assert meta["ridge_alpha"] == alpha
        assert "descriptive" in meta["RT"].lower()
    metadata = json.loads(find(paths, "desc-hrfSelection_metadata.json").read_text())
    assert metadata["LibraryFingerprint"] == library.fingerprint
    assert metadata["TrainingRuns"] == ["run-01", "run-03"]
    assert metadata["TestRuns"] == ["run-02", "run-04"]
    assert metadata["IndependentEvaluationAvailable"] is True
    assert find(paths, "desc-hrfSelection_library.png").stat().st_size > 1000
    assert find(paths, "desc-hrfSelection_scatter.png").stat().st_size > 1000
    assert find(paths, "desc-hrfSelection_eligibility.tsv").is_file()
    assert find(paths, "desc-hrfSelection_folds.tsv").is_file()


def test_blocks_and_even_run_edits_preserve_training_decisions(hrf_nsd, tmp_path):
    small = run(hrf_nsd, tmp_path / "small", 1)
    large = run(hrf_nsd, tmp_path / "large", 4096)
    other = {p.name: p for p in large}
    for p in small:
        if p.name.endswith(".dscalar.nii"):
            np.testing.assert_allclose(
                nib.load(p).get_fdata(),
                nib.load(other[p.name]).get_fdata(),
                atol=1e-6,
                equal_nan=True,
            )
    root, prep, *_ = hrf_nsd
    for number in [2, 4]:
        path = next(prep.rglob(f"*run-{number:02d}*.dtseries.nii"))
        image = nib.load(path)
        nib.save(
            nib.Cifti2Image(
                np.random.default_rng(number).normal(size=image.shape) * 100,
                image.header,
            ),
            path,
        )
        path = next((root / "sub-07").rglob(f"*run-{number:02d}_events.tsv"))
        events = pd.read_csv(path, sep="\t")
        events.response_time = events.response_time * 20 + 8
        events.to_csv(path, sep="\t", index=False)
    changed = run(hrf_nsd, tmp_path / "changed")
    for fragment in ["stat-oddhrfindex."]:
        np.testing.assert_array_equal(
            nib.load(find(small, fragment)).get_fdata(),
            nib.load(find(changed, fragment)).get_fdata(),
        )
    a = pd.read_csv(find(small, "_selectedvertices.tsv"), sep="\t")
    b = pd.read_csv(find(changed, "_selectedvertices.tsv"), sep="\t")
    np.testing.assert_array_equal(a.grayordinate_index, b.grayordinate_index)
    np.testing.assert_allclose(a.canonical_odd_r, b.canonical_odd_r, atol=1e-12)


def test_hrf_comparison_uses_pooled_matching_canonical_trial_fit(hrf_nsd, tmp_path):
    root, _, brain, _, signals = hrf_nsd
    paths = run(hrf_nsd, tmp_path / "comparison")
    for model, alpha in [("OLS", 0.0), ("Ridge", 0.1)]:
        comparison = nib.load(find(paths, f"desc-hrfOpt{model}_stat-hrfdeltarsquared."))
        assert comparison.header.get_axis(1) == brain
        assert comparison.header.get_axis(0).name.tolist() == [
            "optimized_full_r2_minus_canonical_full_r2"
        ]
        sse = np.zeros(3)
        sst = np.zeros(3)
        for number, signal in enumerate(signals, 1):
            events = pd.read_csv(
                next(root.glob(f"sub-07/ses-nsd10/func/*run-{number:02d}_events.tsv")),
                sep="\t",
            )
            with np.load(
                find(paths, f"run-{number:02d}_desc-hrfSelection_designs.npz"),
                allow_pickle=False,
            ) as saved:
                nuisance = saved["nuisance"]
                trial = np.column_stack(
                    [
                        compute_regressor(
                            np.array([[onset], [duration], [1.0]]),
                            hrf_model("spm"),
                            saved["frame_times"],
                        )[0][:, 0]
                        for onset, duration in zip(
                            events.onset, events.duration, strict=True
                        )
                    ]
                )
            residual_trial = (
                trial - nuisance @ np.linalg.lstsq(nuisance, trial, rcond=None)[0]
            )
            design = np.column_stack([trial, nuisance])
            penalty = np.column_stack(
                [
                    np.diag(np.sqrt(alpha) * np.linalg.norm(residual_trial, axis=0)),
                    np.zeros((trial.shape[1], nuisance.shape[1])),
                ]
            )
            coefficients = np.linalg.lstsq(
                np.vstack([design, penalty]),
                np.vstack([signal[:, :3], np.zeros((trial.shape[1], 3))]),
                rcond=None,
            )[0]
            sse += np.sum((signal[:, :3] - design @ coefficients) ** 2, axis=0)
            sst += np.sum((signal[:, :3] - signal[:, :3].mean(axis=0)) ** 2, axis=0)
        canonical = 1 - sse / sst
        optimized = nib.load(
            find(paths, f"desc-hrfOpt{model}_stat-fullrsquared.")
        ).get_fdata()[0]
        np.testing.assert_allclose(
            comparison.get_fdata()[0, :3], optimized[:3] - canonical, atol=6e-8
        )
        assert np.isnan(comparison.get_fdata()[0, 3])
        metadata = json.loads(
            find(paths, f"desc-hrfOpt{model}_stat-hrfdeltarsquared.json").read_text()
        )
        assert metadata["ridge_alpha"] == alpha
        assert "in-sample" in metadata["R2"]
        assert metadata["Formula"] == "optimized_full_r2 - canonical_full_r2"
        assert len(metadata["CanonicalFitProvenance"]) == 2


def test_too_few_odd_runs_records_unavailable_evaluation(hrf_nsd, tmp_path):
    root, prep, *_ = hrf_nsd
    for directory in (root / "sub-07", prep):
        for number in [3, 4]:
            for path in directory.rglob(f"*run-{number:02d}*"):
                path.unlink()
    paths = run(hrf_nsd, tmp_path / "output")
    metadata = json.loads(find(paths, "desc-hrfSelection_metadata.json").read_text())
    assert metadata["IndependentEvaluationAvailable"] is False
    assert metadata["IndependentEvaluationReason"]
    assert np.isnan(
        nib.load(find(paths, "desc-hrfSelection_stat-testr2.")).get_fdata()
    ).all()


def test_collision_and_publication_rollback(hrf_nsd, tmp_path, monkeypatch):
    import boldtailor.publication as publication
    import boldtailor.hrf_selection as selection

    output = tmp_path / "output"
    target = (
        output
        / "sub-07/ses-nsd10/func/sub-07_ses-nsd10_task-nsdcore_desc-hrfSelection_library.tsv"
    )
    target.parent.mkdir(parents=True)
    target.write_text("old")
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("collision must precede selection")

        patch.setattr(selection, "select_hrf", forbidden)
        with pytest.raises(FileExistsError):
            run(hrf_nsd, output)
    assert target.read_text() == "old"
    original = publication.os.replace
    count = 0

    def fail(source, target):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("injected publication failure")
        return original(source, target)

    monkeypatch.setattr(publication.os, "replace", fail)
    with pytest.raises(publication.PublicationError):
        run(hrf_nsd, tmp_path / "rollback")
    assert not list((tmp_path / "rollback").rglob("*desc-hrf*"))


def test_undefined_odd_hrf_at_rt_selected_vertex_does_not_abort(hrf_nsd, tmp_path):
    _, prep, *_ = hrf_nsd
    for number in range(1, 5):
        path = next(prep.rglob(f"*run-{number:02d}*.dtseries.nii"))
        image = nib.load(path)
        y = image.get_fdata()
        confound = pd.read_csv(
            next(prep.rglob(f"*run-{number:02d}*confounds*.tsv")), sep="\t"
        )
        y[:, 0] = 3 * confound.trans_x + 100
        nib.save(nib.Cifti2Image(y, image.header), path)
    paths = run(hrf_nsd, tmp_path / "undefined")
    ids = nib.load(find(paths, "stat-oddhrfindex.")).get_fdata()
    assert np.isnan(ids[0, 0])
    vertices = pd.read_csv(find(paths, "_selectedvertices.tsv"), sep="\t")
    if 0 in vertices.grayordinate_index.to_numpy():
        row = vertices[vertices.grayordinate_index == 0].iloc[0]
        assert np.isnan(row.odd_hrf_id)
        assert np.isnan(row.OLS_even_r)


def test_canonical_ineligible_diagnostic_does_not_abort_expanded_outputs(
    hrf_nsd, tmp_path
):
    from dataclasses import replace
    from boldtailor._single_trial_design import compile_trial_run
    from examples.NSD.nsd_cifti import discover_runs
    from examples.NSD.nsd_single_trial import _load_runs
    from examples.NSD.nsd_hrf import run_expanded_analysis

    root, prep, brain, *_ = hrf_nsd
    runs, _ = _load_runs(discover_runs(root, prep))
    altered = []
    for run in runs:
        x, _, _ = compile_trial_run(
            run.events, run.frame_times, run.confounds, run.label
        )
        nuisance = pd.concat(
            [run.confounds, x.rename(columns=lambda c: "absorbed_" + c)], axis=1
        )
        altered.append(replace(run, confounds=nuisance))
    paths = run_expanded_analysis(
        tuple(altered),
        root,
        tmp_path / "ineligible",
        brain,
        {"OLS": 0.0, "Ridge": 0.1},
        2,
        "sub-07",
        "ses-nsd10",
    )
    assert np.isfinite(
        nib.load(find(paths, "desc-hrfOptOLS_stat-fullrsquared.")).get_fdata()[0, :3]
    ).all()
    assert np.isnan(
        nib.load(find(paths, "desc-hrfSelection_stat-canonicalcvr2.")).get_fdata()
    ).all()
    assert np.isfinite(
        nib.load(find(paths, "desc-hrfSelection_stat-testr2.")).get_fdata()[0, :3]
    ).all()
    metadata = json.loads(find(paths, "desc-hrfSelection_metadata.json").read_text())
    assert metadata["IndependentRTAvailable"] is False
    assert "canonical" in metadata["IndependentRTReason"].lower()
    assert pd.read_csv(find(paths, "_selectedvertices.tsv"), sep="\t").empty
    for model in ["OLS", "Ridge"]:
        assert np.isnan(
            nib.load(
                find(paths, f"desc-hrfOpt{model}_stat-hrfdeltarsquared.")
            ).get_fdata()
        ).all()
        comparison = json.loads(
            find(paths, f"desc-hrfOpt{model}_stat-hrfdeltarsquared.json").read_text()
        )
        assert comparison["ComparisonAvailable"] is False
        assert "canonical" in comparison["ComparisonUnavailableReason"].lower()


def test_comparison_artifact_preserves_signed_difference_and_undefined(
    hrf_nsd, tmp_path
):
    from examples.NSD.hrf_artifacts import comparison_artifacts, comparison_paths
    from boldtailor.publication import publish_artifact_set

    brain = hrf_nsd[2]
    paths = comparison_paths("example_stat-fullrsquared.dscalar.nii")
    artifacts = comparison_artifacts(
        brain,
        np.array([0.8, 0.2, np.nan, 0.4]),
        np.array([0.6, 0.5, 0.7, np.nan]),
        paths,
        {"ridge_alpha": 0.0},
    )
    published = publish_artifact_set(tmp_path / "artifacts", artifacts)
    image = nib.load(find(published, "stat-hrfdeltarsquared.dscalar.nii"))
    np.testing.assert_allclose(
        image.get_fdata(), [[0.2, -0.3, np.nan, np.nan]], atol=2e-8
    )
    assert image.header.get_axis(1) == brain


@pytest.mark.parametrize("suffix", ["dscalar.nii", "json"])
def test_comparison_collisions_precede_fitting(hrf_nsd, tmp_path, monkeypatch, suffix):
    output = tmp_path / "collision"
    target = (
        output
        / f"sub-07/ses-nsd10/func/sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-hrfOptOLS_stat-hrfdeltarsquared.{suffix}"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve")

    def forbidden(*args, **kwargs):
        pytest.fail("comparison collision must be caught before selection")

    monkeypatch.setattr("boldtailor.hrf_selection.select_hrf", forbidden)
    with pytest.raises(FileExistsError):
        run(hrf_nsd, output)
    assert target.read_bytes() == b"preserve"
