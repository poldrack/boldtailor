import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication import run
from experiments.nsd_replication.comparison import LEVELS
from experiments.nsd_replication.betas import fit_dir, is_complete, read_fit, write_fit


@pytest.fixture
def config(tmp_path):
    from experiments.nsd_replication.config import ExperimentConfig

    return ExperimentConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        freesurfer_dir=tmp_path,
        subjects=("sub-07",),
        sessions=("ses-a", "ses-b"),
        block_size=16,
    )


@pytest.fixture
def patched(monkeypatch, synthetic_session):
    monkeypatch.setattr(
        run, "_load", lambda config, source, subject, session: synthetic_session
    )
    monkeypatch.setattr(run, "ppdata_brain", lambda *a: synthetic_session.brain)
    monkeypatch.setattr(
        run, "load_roi", lambda config, subject, brain: np.arange(30) < 20
    )


def test_fit_session_writes_and_skips(config, patched):
    paths = run.fit_session(config, "ppdata", "sub-07", "ses-a", ("b1",))
    assert all(is_complete(p) for p in paths)
    mtime = (paths[0] / "betas.npy").stat().st_mtime_ns
    run.fit_session(config, "ppdata", "sub-07", "ses-a", ("b1",))
    assert (paths[0] / "betas.npy").stat().st_mtime_ns == mtime


def test_metadata_is_complete_and_trials_use_session_label(config, patched):
    (path,) = run.fit_session(config, "ppdata", "sub-07", "ses-a", ("b1",))
    _, trials, meta = read_fit(path)
    assert set(trials["session"]) == {"ses-a"}
    assert meta["source"] == "ppdata" and meta["level"] == "b1"
    assert {"inputs_digest", "boldtailor_version", "git_commit", "onsets"} <= set(meta)
    assert len(meta["onsets"]) == len(trials)


def test_digest_mismatch_raises_unless_refit(config, patched):
    (path,) = run.fit_session(config, "ppdata", "sub-07", "ses-a", ("b1",))
    meta = path / "metadata.json"
    meta.write_text(
        meta.read_text().replace('"inputs_digest": "', '"inputs_digest": "x')
    )
    with pytest.raises(ValueError, match="inputs digest"):
        run.fit_session(config, "ppdata", "sub-07", "ses-a", ("b1",))
    run.fit_session(config, "ppdata", "sub-07", "ses-a", ("b1",), refit=True)


def test_unfitted_source_is_rejected(config):
    with pytest.raises(ValueError, match="source"):
        run._load(config, "released", "sub-07", "ses-a")


def test_subject_metrics_end_to_end(config, patched):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1", "b2", "b4"))
    tables = run.subject_metrics(config, "ppdata", "sub-07", ("b1", "b2", "b4"))
    r1 = tables["r1"]
    assert set(r1["version"]) == {"b1", "b2", "b4"}
    assert (config.output_dir / "metrics/ppdata/sub-07/r1.tsv").is_file()
    assert {"r1", "r1_median", "r3b", "r4_t0.0", "r4_t0.3", "r6"} <= set(tables)
    assert set(tables["r6"]["threshold"]) == {0.0, 0.1, 0.2, 0.3, 0.4}
    assert set(tables["r4_t0.3"]["lag"]) == set(range(1, 16))
    for name in tables:
        assert (config.output_dir / f"metrics/ppdata/sub-07/{name}.tsv").is_file()


def test_group_rsa_writes_agreement(config, patched):
    for subject in ("sub-07", "sub-08"):
        for ses in config.sessions:
            run.fit_session(config, "ppdata", subject, ses, ("b1",))
    table = run.group_rsa(config, "ppdata", ("sub-07", "sub-08"), ("b1",))
    assert (
        config.output_dir / "metrics/ppdata/group_sub-07_sub-08/rsa_b1.tsv"
    ).is_file()
    assert list(table.columns) == ["threshold", "subject_a", "subject_b", "r"]
    assert set(table["threshold"]) == {0.0, 0.2, 0.4}
    assert table["r"].to_numpy() == pytest.approx(1.0)


def _fit_and_measure(config, levels=("b1",)):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, levels)
    return run.subject_metrics(config, "ppdata", "sub-07", levels)


def test_discard_betas_marks_metadata_and_is_not_refit(config, patched):
    _fit_and_measure(config)
    run.discard_betas(config, "ppdata", "sub-07", ("b1",), multiple_subjects=False)
    path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b1")
    assert not (path / "betas.npy").exists()
    assert json.loads((path / "metadata.json").read_text())["betas_discarded"] is True
    assert is_complete(path)
    run.fit_session(config, "ppdata", "sub-07", "ses-a", ("b1",))
    assert not (path / "betas.npy").exists()


def test_discard_refuses_when_a_metric_is_missing(config, patched):
    _fit_and_measure(config)
    out = config.output_dir / "metrics/ppdata/sub-07"
    (out / "r6.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="r6"):
        run.discard_betas(config, "ppdata", "sub-07", ("b1",), multiple_subjects=False)
    path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b1")
    assert (path / "betas.npy").exists()


def test_main_fit_then_metrics_with_discard(config, patched):
    toml = config.bids_dir / "config.toml"
    toml.write_text(f"""bids_dir = "{config.bids_dir}"
output_dir = "{config.output_dir}"
freesurfer_dir = "{config.freesurfer_dir}"
subjects = ["sub-07"]
sessions = ["ses-a", "ses-b"]
block_size = 16
""")
    common = ["--config", str(toml), "--source", "ppdata", "--levels", "b1"]
    assert run.main(["fit", *common]) == 0
    for ses in config.sessions:
        assert is_complete(fit_dir(config.output_dir, "ppdata", "sub-07", ses, "b1"))
    assert run.main(["metrics", *common, "--discard-betas-after-metrics"]) == 0
    assert (config.output_dir / "metrics/ppdata/sub-07/r1.tsv").is_file()
    path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b1")
    assert not (path / "betas.npy").exists()


def test_main_rejects_unknown_command(config):
    with pytest.raises(SystemExit):
        run.main(["bogus", "--config", "x.toml"])


def _fit_released(config, perturb=False):
    for ses in config.sessions:
        for lv in ("b1", "b2", "b4"):
            betas, trials, _ = read_fit(
                fit_dir(config.output_dir, "ppdata", "sub-07", ses, lv)
            )
            rng = np.random.default_rng(1)
            noisy = betas + 0.1 * rng.normal(size=betas.shape)
            if perturb and ses == "ses-b":
                trials = trials.assign(image=trials["image"] + 1)
            path = fit_dir(config.output_dir, "released", "sub-07", ses, lv)
            write_fit(path, noisy, trials, {"level": lv})


def _fit_comparison(config, perturb=False):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1", "b2", "b4"))
    _fit_released(config, perturb)


def test_comparison_metrics_combined_versions(config, patched):
    _fit_comparison(config)
    tables = run.comparison_metrics(config, "sub-07")
    expected = {
        f"{s}:{lv}" for s in ("ppdata", "released") for lv in ("b1", "b2", "b4")
    }
    assert set(tables["r1"]["version"]) == expected
    assert set(tables["r1_median"]["version"]) == expected
    out = config.output_dir / "metrics/comparison/sub-07"
    assert (out / "r1.tsv").is_file() and (out / "r1_median.tsv").is_file()
    for lv in ("b1", "b2", "b4"):
        assert len(np.load(out / f"r1_difference_{lv}.npy")) == 30


def test_trial_tables_must_match(config, patched):
    _fit_comparison(config, perturb=True)
    with pytest.raises(ValueError, match="ses-b"):
        run.comparison_metrics(config, "sub-07")


def test_alignment_table_rows_and_floor(config, patched):
    _fit_comparison(config)
    table = run.alignment_table(config, "sub-07")
    assert list(table["session"]) == ["ses-a", "ses-b"]
    assert (table["run_length"] == 24).all()
    assert (table["true_median"] > 0.9).all()
    assert (config.output_dir / "metrics/comparison/sub-07/alignment.tsv").is_file()
    floored = replace(config, alignment_floor=1.5)
    with pytest.raises(ValueError, match="sub-07.*ses-a"):
        run.alignment_table(floored, "sub-07")


def test_main_metrics_released_and_comparison(config, patched):
    toml = config.bids_dir / "config.toml"
    toml.write_text(f"""bids_dir = "{config.bids_dir}"
output_dir = "{config.output_dir}"
freesurfer_dir = "{config.freesurfer_dir}"
subjects = ["sub-07"]
sessions = ["ses-a", "ses-b"]
""")
    _fit_comparison(config)
    base = ["metrics", "--config", str(toml)]
    assert run.main([*base, "--source", "released"]) == 0
    assert (config.output_dir / "metrics/released/sub-07/r1.tsv").is_file()
    assert run.main([*base, "--source", "comparison"]) == 0
    assert (config.output_dir / "metrics/comparison/sub-07/alignment.tsv").is_file()


def test_floor_failure_keeps_alignment_but_skips_comparison(config, patched):
    _fit_comparison(config)
    floored = replace(config, alignment_floor=1.5)
    with pytest.raises(ValueError, match="sub-07.*ses-a.*ses-b"):
        run.run_comparison(floored, "sub-07")
    out = config.output_dir / "metrics/comparison/sub-07"
    assert (out / "alignment.tsv").is_file()
    assert not (out / "r1.tsv").exists()


def test_comparison_runs_alignment_then_metrics(config, patched):
    _fit_comparison(config)
    run.run_comparison(config, "sub-07")
    out = config.output_dir / "metrics/comparison/sub-07"
    assert (out / "alignment.tsv").is_file() and (out / "r1.tsv").is_file()


def test_discard_requires_comparison_outputs_when_released_configured(config, patched):
    _fit_and_measure(config)
    configured = replace(config, released_dir=config.bids_dir)
    with pytest.raises(FileNotFoundError, match="metrics --source comparison"):
        run.discard_betas(configured, "ppdata", "sub-07", ("b1",))
    path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b1")
    assert (path / "betas.npy").exists()


def test_discard_succeeds_after_comparison_outputs(config, patched):
    _fit_and_measure(config)
    out = config.output_dir / "metrics/comparison/sub-07"
    _write_comparison_outputs(out)
    configured = replace(config, released_dir=config.bids_dir)
    run.discard_betas(configured, "ppdata", "sub-07", ("b1",))
    path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b1")
    assert not (path / "betas.npy").exists()


def test_trial_tables_compare_all_columns(config, patched):
    _fit_comparison(config)
    path = fit_dir(config.output_dir, "released", "sub-07", "ses-a", "b2")
    table = path / "trials.tsv"
    table.write_text(table.read_text().replace("\n", "\textra\n", 1))
    with pytest.raises(ValueError, match="ses-a"):
        run.comparison_metrics(config, "sub-07")


def _write_tsv(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def _synthetic_metrics(config):
    import pandas as pd

    out = config.output_dir / "metrics"
    curves = pd.DataFrame(
        {
            "version": ["b1", "b4"] * 2,
            "threshold": [0.0, 0.0, 0.2, 0.2],
            "n_features": 9,
            "mean_difference": [0.0, 0.1, 0.0, 0.2],
        }
    )
    for sub in config.subjects:
        _write_tsv(out / "ppdata" / sub / "r1.tsv", curves)
        versions = [f"{s}:{lv}" for s in ("ppdata", "released") for lv in LEVELS]
        median = pd.DataFrame({"version": versions, "median": np.arange(6) / 10})
        _write_tsv(out / "comparison" / sub / "r1_median.tsv", median)
    rsa = pd.DataFrame(
        {"threshold": [0.0], "subject_a": ["a"], "subject_b": ["b"], "r": [0.3]}
    )
    group = "group_" + "_".join(config.subjects)
    _write_tsv(out / "ppdata" / group / "rsa_b4.tsv", rsa)


def test_figures_command_saves_pngs_and_skips_missing(config, capsys):
    config = replace(config, subjects=("sub-01", "sub-02"))
    _synthetic_metrics(config)
    run.make_figures(config)
    figs = config.output_dir / "figures"
    assert (figs / "sub-01" / "r1.png").stat().st_size > 0
    assert (figs / "sub-02" / "r1.png").is_file()
    assert (figs / "group" / "rsa.png").is_file()
    assert (figs / "group" / "parity.png").is_file()
    assert not (figs / "sub-01" / "r6.png").exists()
    assert "skipped" in capsys.readouterr().out


def test_figures_cli(tmp_path, config):
    config_toml = tmp_path / "c.toml"
    config_toml.write_text(
        f'bids_dir="{tmp_path}"\noutput_dir="{config.output_dir}"\n'
        f'freesurfer_dir="{tmp_path}"\nsubjects=["sub-07"]\nsessions=["ses-a"]\n'
    )
    _synthetic_metrics(config)
    assert run.main(["figures", "--config", str(config_toml)]) == 0
    assert (config.output_dir / "figures" / "sub-07" / "r1.png").is_file()


def test_completed_level_survives_a_later_failure(config, patched, monkeypatch):
    from experiments.nsd_replication import ladder

    def boom(ladder_):
        raise RuntimeError("b2 failed")

    monkeypatch.setitem(ladder._FITTERS, "b2", boom)
    with pytest.raises(RuntimeError, match="b2 failed"):
        run.fit_session(config, "ppdata", "sub-07", "ses-a", ("b1", "b2"))
    assert is_complete(fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b1"))
    assert not is_complete(
        fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b2")
    )


def _metric_mtimes(config, names=run.SUBJECT_TABLES):
    out = config.output_dir / "metrics/ppdata/sub-07"
    return {n: (out / f"{n}.tsv").stat().st_mtime_ns for n in names}


def test_existing_metric_tables_skipped_unless_recompute(config, patched):
    _fit_and_measure(config)
    before = _metric_mtimes(config)
    run.subject_metrics(config, "ppdata", "sub-07", ("b1",))
    assert _metric_mtimes(config) == before
    run.subject_metrics(config, "ppdata", "sub-07", ("b1",), recompute=True)
    after = _metric_mtimes(config)
    assert all(after[n] != before[n] for n in before)


def test_metric_tables_written_before_a_later_failure(config, patched, monkeypatch):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1",))

    def boom(*args, **kwargs):
        raise RuntimeError("r6 failed")

    monkeypatch.setattr(run, "_r6_table", boom)
    with pytest.raises(RuntimeError, match="r6 failed"):
        run.subject_metrics(config, "ppdata", "sub-07", ("b1",))
    out = config.output_dir / "metrics/ppdata/sub-07"
    assert (out / "r1.tsv").is_file() and not (out / "r6.tsv").exists()


def test_main_metrics_recompute_flag(config, patched):
    toml = config.bids_dir / "config.toml"
    toml.write_text(f"""bids_dir = "{config.bids_dir}"
output_dir = "{config.output_dir}"
freesurfer_dir = "{config.freesurfer_dir}"
subjects = ["sub-07"]
sessions = ["ses-a", "ses-b"]
block_size = 16
""")
    common = ["--config", str(toml), "--source", "ppdata", "--levels", "b1"]
    run.main(["fit", *common])
    run.main(["metrics", *common])
    before = _metric_mtimes(config, ("r1",))
    run.main(["metrics", *common])
    assert _metric_mtimes(config, ("r1",)) == before
    run.main(["metrics", *common, "--recompute"])
    assert _metric_mtimes(config, ("r1",)) != before


def test_r6_same_with_one_or_two_jobs(config, patched):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1", "b2"))
    tables = {}
    for n_jobs in (1, 2):
        cfg = replace(config, n_jobs=n_jobs)
        tables[n_jobs] = run.subject_metrics(
            cfg, "ppdata", "sub-07", ("b1", "b2"), recompute=True
        )["r6"]
    import pandas as pd

    pd.testing.assert_frame_equal(tables[1], tables[2])


def test_r6_uses_config_jobs(config, patched, monkeypatch):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1",))
    seen = []
    real = run.decoding_accuracies

    def spy(reps, masks, n_jobs=1):
        seen.append(n_jobs)
        return real(reps, masks, n_jobs=1)

    monkeypatch.setattr(run, "decoding_accuracies", spy)
    run.subject_metrics(replace(config, n_jobs=3), "ppdata", "sub-07", ("b1",))
    assert seen and set(seen) == {3}


def test_subject_metrics_sliced_to_roi_with_identical_results(
    config, patched, monkeypatch
):
    from experiments.nsd_replication.betas import zscore
    from experiments.nsd_replication.metrics import threshold_curves
    from experiments.nsd_replication.trials import images_with, repetition_array

    import pandas as pd

    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1", "b2"))
    full = {}
    for lv in ("b1", "b2"):
        pairs = [
            read_fit(fit_dir(config.output_dir, "ppdata", "sub-07", s, lv))[:2]
            for s in config.sessions
        ]
        betas = np.vstack([zscore(b) for b, _ in pairs])
        trials = pd.concat([t for _, t in pairs], ignore_index=True)
        reps = repetition_array(betas, trials, images_with(trials, 3))
        full[lv] = run.voxel_reliability(reps)
    roi = np.arange(30) < 20
    expected = threshold_curves(full, roi)
    widths = []
    real = run.voxel_reliability
    monkeypatch.setattr(
        run,
        "voxel_reliability",
        lambda reps: widths.append(reps.shape[-1]) or real(reps),
    )
    r1 = run.subject_metrics(config, "ppdata", "sub-07", ("b1", "b2"))["r1"]
    assert widths and set(widths) == {20}
    pd.testing.assert_frame_equal(r1[expected.columns], expected)


def test_comparison_metrics_all_tables_share_one_composite(config, patched):
    _fit_comparison(config)
    tables = run.comparison_metrics(config, "sub-07")
    expected = {
        f"{s}:{lv}" for s in ("ppdata", "released") for lv in ("b1", "b2", "b4")
    }
    out = config.output_dir / "metrics/comparison/sub-07"
    for name in ("r3b", "r4_t0.0", "r4_t0.3", "r6"):
        assert set(tables[name]["version"]) == expected, name
        assert (out / f"{name}.tsv").is_file(), name
    # one shared composite mask: every version sees the same feature count
    r6 = tables["r6"]
    assert (r6.groupby("threshold")["n_features"].nunique() == 1).all()
    r4 = tables["r4_t0.3"]
    assert (r4.groupby("lag")["n_pairs"].nunique() == 1).all()
    assert set(r4["threshold"]) == {0.3}


def test_hrf_consistency_writes_r2_for_fitted_levels(config, patched):
    import pandas as pd

    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b2",))
    run.hrf_consistency(config, "sub-07")
    out = config.output_dir / "metrics/ppdata/sub-07"
    table = pd.read_csv(out / "r2_b2.tsv", sep="\t")
    assert len(table) == 30
    assert {"mean_pairwise_r", "mean_canonical_baseline"} <= set(table.columns)
    # identical synthetic sessions choose identical HRFs
    assert table["mean_pairwise_r"].to_numpy() == pytest.approx(1.0)
    summary = pd.read_csv(out / "r2_b2_roi.tsv", sep="\t")
    assert summary["version"].tolist() == ["b2"]
    assert summary["n_features"].tolist() == [20]
    expected = np.median(table["mean_pairwise_r"][:20])
    assert summary["mean_pairwise_r"].iloc[0] == pytest.approx(expected)
    assert not (out / "r2_b2-lib20.tsv").exists()


def test_hrf_consistency_uses_matching_library(config, patched):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b2-lib20",))
    run.hrf_consistency(config, "sub-07")
    out = config.output_dir / "metrics/ppdata/sub-07"
    assert (out / "r2_b2-lib20.tsv").is_file()
    meta = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b2-lib20")
    meta = meta / "metadata.json"
    meta.write_text(
        meta.read_text().replace(
            '"library_fingerprint": "', '"library_fingerprint": "x'
        )
    )
    with pytest.raises(ValueError, match="library"):
        run.hrf_consistency(config, "sub-07", recompute=True)


def test_main_metrics_runs_r2(config, patched):
    toml = config.bids_dir / "config.toml"
    toml.write_text(f"""bids_dir = "{config.bids_dir}"
output_dir = "{config.output_dir}"
freesurfer_dir = "{config.freesurfer_dir}"
subjects = ["sub-07"]
sessions = ["ses-a", "ses-b"]
block_size = 16
""")
    common = ["--config", str(toml), "--source", "ppdata", "--levels", "b2"]
    run.main(["fit", *common])
    run.main(["metrics", *common])
    assert (config.output_dir / "metrics/ppdata/sub-07/r2_b2_roi.tsv").is_file()


def test_figures_render_r2_summary(config):
    import pandas as pd

    out = config.output_dir / "metrics/ppdata/sub-07"
    for level in ("b2", "b2-lib20"):
        summary = pd.DataFrame(
            {
                "version": [level],
                "mean_pairwise_r": [0.5],
                "mean_canonical_baseline": [0.2],
                "n_features": [20],
            }
        )
        _write_tsv(out / f"r2_{level}_roi.tsv", summary)
    run.make_figures(config)
    assert (config.output_dir / "figures/sub-07/r2.png").stat().st_size > 0


def _write_config(config, extra=""):
    toml = config.bids_dir / "config.toml"
    toml.write_text(f"""bids_dir = "{config.bids_dir}"
output_dir = "{config.output_dir}"
freesurfer_dir = "{config.freesurfer_dir}"
subjects = ["sub-07"]
sessions = ["ses-a", "ses-b"]
block_size = 16
{extra}""")
    return toml


def test_ppdata_default_levels_include_lss(config, monkeypatch):
    from experiments.nsd_replication.ladder import LEVELS as ALL, LSS_LEVELS

    seen = {}
    monkeypatch.setattr(
        run,
        "fit_session",
        lambda c, src, sub, ses, levels, refit: seen.setdefault("fit", levels),
    )
    monkeypatch.setattr(
        run,
        "subject_metrics",
        lambda c, src, sub, levels, recompute: seen.setdefault("metrics", levels),
    )
    monkeypatch.setattr(run, "hrf_consistency", lambda *a: None)
    toml = _write_config(config)
    run.main(["fit", "--config", str(toml)])
    run.main(["metrics", "--config", str(toml)])
    assert tuple(seen["fit"]) == ALL + LSS_LEVELS
    assert tuple(seen["metrics"]) == ALL + LSS_LEVELS


def test_metric_tables_record_composite_levels(config, patched):
    import pandas as pd

    for subject in ("sub-07", "sub-08"):
        for ses in config.sessions:
            run.fit_session(config, "ppdata", subject, ses, ("b1", "b2"))
    run.subject_metrics(config, "ppdata", "sub-07", ("b1", "b2"))
    out = config.output_dir / "metrics/ppdata"
    for name in run.SUBJECT_TABLES:
        table = pd.read_csv(out / "sub-07" / f"{name}.tsv", sep="\t")
        assert set(table["composite_levels"]) == {"b1|b2"}, name
    run.group_rsa(config, "ppdata", ("sub-07", "sub-08"), ("b1", "b2"))
    rsa = pd.read_csv(out / "group_sub-07_sub-08" / "rsa_b1.tsv", sep="\t")
    assert set(rsa["composite_levels"]) == {"b1|b2"}
    assert set(rsa["sessions"]) == {"ses-a|ses-b"}
    for name in run.SUBJECT_TABLES:
        table = pd.read_csv(out / "sub-07" / f"{name}.tsv", sep="\t")
        assert set(table["sessions"]) == {"ses-a|ses-b"}, name


def test_comparison_tables_record_composite_levels(config, patched):
    import pandas as pd

    from experiments.nsd_replication.comparison import VERSIONS

    _fit_comparison(config)
    run.comparison_metrics(config, "sub-07")
    out = config.output_dir / "metrics/comparison/sub-07"
    for name in run.SUBJECT_TABLES:
        table = pd.read_csv(out / f"{name}.tsv", sep="\t")
        assert set(table["composite_levels"]) == {"|".join(VERSIONS)}, name


def test_different_level_set_requires_recompute(config, patched):
    import pandas as pd

    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1", "b2"))
    run.subject_metrics(config, "ppdata", "sub-07", ("b1",))
    before = _metric_mtimes(config)
    with pytest.raises(ValueError, match="--recompute"):
        run.subject_metrics(config, "ppdata", "sub-07", ("b1", "b2"))
    assert _metric_mtimes(config) == before
    run.subject_metrics(config, "ppdata", "sub-07", ("b1", "b2"), recompute=True)
    r1 = config.output_dir / "metrics/ppdata/sub-07/r1.tsv"
    assert set(pd.read_csv(r1, sep="\t")["composite_levels"]) == {"b1|b2"}
    # a table without the column (an older run) is not silently replaced
    pd.read_csv(r1, sep="\t").drop(columns="composite_levels").to_csv(
        r1, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="--recompute"):
        run.subject_metrics(config, "ppdata", "sub-07", ("b1", "b2"))


def test_discard_requires_tables_covering_requested_levels(config, patched):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1", "b2"))
    run.subject_metrics(config, "ppdata", "sub-07", ("b1",))
    with pytest.raises(ValueError, match="b2"):
        run.discard_betas(config, "ppdata", "sub-07", ("b1", "b2"))
    for lv in ("b1", "b2"):
        path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", lv)
        assert (path / "betas.npy").exists()
    run.subject_metrics(config, "ppdata", "sub-07", ("b1", "b2"), recompute=True)
    run.discard_betas(config, "ppdata", "sub-07", ("b1",))
    path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b1")
    assert not (path / "betas.npy").exists()


def test_subject_metrics_requires_matching_trials_across_levels(config, patched):
    import pandas as pd

    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1", "b2"))
    path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-b", "b2") / "trials.tsv"
    trials = pd.read_csv(path, sep="\t")
    trials.assign(image=trials["image"] + 1).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="sub-07.*b2"):
        run.subject_metrics(config, "ppdata", "sub-07", ("b1", "b2"))


def _write_comparison_outputs(out, skip=()):
    out.mkdir(parents=True, exist_ok=True)
    names = ["alignment.tsv", *(f"{n}.tsv" for n in run.SUBJECT_TABLES)]
    names += [f"r1_difference_{lv}.npy" for lv in ("b1", "b2", "b4")]
    for name in names:
        if name not in skip:
            (out / name).write_text("x\n")


@pytest.mark.parametrize("missing", ["r6.tsv", "r4_t0.3.tsv", "r1_difference_b4.npy"])
def test_discard_requires_every_comparison_output(config, patched, missing):
    _fit_and_measure(config)
    out = config.output_dir / "metrics/comparison/sub-07"
    _write_comparison_outputs(out, skip=(missing,))
    configured = replace(config, released_dir=config.bids_dir)
    with pytest.raises(FileNotFoundError, match=missing):
        run.discard_betas(configured, "ppdata", "sub-07", ("b1",))
    path = fit_dir(config.output_dir, "ppdata", "sub-07", "ses-a", "b1")
    assert (path / "betas.npy").exists()


def test_group_rsa_keyed_by_subject_set(config, patched):
    for subject in ("sub-01", "sub-02", "sub-05", "sub-06"):
        for ses in config.sessions:
            run.fit_session(config, "ppdata", subject, ses, ("b1",))
    run.group_rsa(config, "ppdata", ("sub-01", "sub-02"), ("b1",))
    run.group_rsa(config, "ppdata", ("sub-05", "sub-06"), ("b1",))
    out = config.output_dir / "metrics/ppdata"
    first = pd.read_csv(out / "group_sub-01_sub-02" / "rsa_b1.tsv", sep="\t")
    second = pd.read_csv(out / "group_sub-05_sub-06" / "rsa_b1.tsv", sep="\t")
    assert set(first["subject_a"]) == {"sub-01"}
    assert set(second["subject_a"]) == {"sub-05"}


def test_session_window_change_requires_recompute(config, patched):
    for ses in config.sessions:
        run.fit_session(config, "ppdata", "sub-07", ses, ("b1",))
    run.subject_metrics(config, "ppdata", "sub-07", ("b1",))
    narrowed = replace(config, sessions=("ses-a",))
    with pytest.raises(ValueError, match="sessions.*--recompute"):
        run.subject_metrics(narrowed, "ppdata", "sub-07", ("b1",))
    run.subject_metrics(narrowed, "ppdata", "sub-07", ("b1",), recompute=True)
    r1 = config.output_dir / "metrics/ppdata/sub-07/r1.tsv"
    assert set(pd.read_csv(r1, sep="\t")["sessions"]) == {"ses-a"}
