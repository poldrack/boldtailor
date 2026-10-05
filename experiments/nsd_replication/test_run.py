import json

import numpy as np
import pytest

from experiments.nsd_replication import run
from experiments.nsd_replication.betas import fit_dir, is_complete, read_fit


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
    assert (config.output_dir / "metrics/ppdata/rsa_b1.tsv").is_file()
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
