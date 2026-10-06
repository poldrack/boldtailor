import dataclasses

import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import default_hrf_library

from experiments.nsd_replication import features, ladder, run
from experiments.nsd_replication.betas import fit_dir, is_complete
from experiments.nsd_replication.features import (
    circular_shift,
    gate_experiment,
    hrf_choice_experiment,
    modulator_experiment,
    null_session,
    rt_correlation,
)

# A small library keeps the four denoising selections per gate run fast.
SMALL = default_hrf_library(32, seed=0)

N_TRIALS = 6 * 24


def _toy_events():
    return pd.DataFrame(
        {
            "onset": 8.0 + 4.0 * np.arange(24),
            "duration": 3.0,
            "row_id": np.arange(24),
            "response_time": np.linspace(0.5, 1.5, 24),
        }
    )


def _shift(original, shifted, duration):
    source = original.set_index("row_id").loc[shifted["row_id"]]
    return np.mod(shifted["onset"].to_numpy() - source["onset"].to_numpy(), duration)


def test_circular_shift_preserves_counts_and_minimum():
    events, duration = _toy_events(), 128.0
    for seed in range(100):
        shifted = circular_shift(events, duration, np.random.default_rng(seed))
        offsets = _shift(events, shifted, duration)
        assert np.allclose(offsets, offsets[0])
        assert 30.0 <= offsets[0] <= duration - 30.0
        assert len(shifted) == len(events)
        assert np.all(np.diff(shifted["onset"].to_numpy()) >= 0)
        assert np.all((shifted["onset"] >= 0) & (shifted["onset"] < duration))
        source = events.set_index("row_id").loc[shifted["row_id"]]
        np.testing.assert_array_equal(shifted["duration"], source["duration"])
        np.testing.assert_array_equal(shifted["response_time"], source["response_time"])


def test_circular_shift_does_not_modify_input():
    events = _toy_events()
    copy = events.copy()
    circular_shift(events, 128.0, np.random.default_rng(0))
    pd.testing.assert_frame_equal(events, copy)


def test_gate_experiment_structure(synthetic_session):
    table = gate_experiment(synthetic_session, seed=0, library=SMALL)
    assert list(table.columns[:5]) == [
        "condition",
        "gate",
        "n_components",
        "pcstop_count",
        "decision",
    ]
    assert len(table) == 4
    pairs = set(zip(table["condition"], table["gate"]))
    assert pairs == {("real", True), ("real", False), ("null", True), ("null", False)}
    gated = table[table["gate"]]
    ungated = table[~table["gate"]]
    assert np.all(gated["n_components"] <= gated["pcstop_count"])
    assert np.all(ungated["n_components"] == ungated["pcstop_count"])


def test_null_session_is_seeded_and_in_the_sampled_window(synthetic_session):
    a, b = null_session(synthetic_session, 0), null_session(synthetic_session, 0)
    c = null_session(synthetic_session, 1)
    for x, y in zip(a.events, b.events):
        pd.testing.assert_frame_equal(x, y)
    assert not all(x.equals(z) for x, z in zip(a.events, c.events))
    starts = [e["onset"].min() for e in a.events]
    assert len(set(starts)) > 1
    for events, times, real in zip(a.events, a.frame_times, synthetic_session.events):
        onsets = events["onset"].to_numpy()
        assert len(events) == len(real)
        assert np.all(np.diff(onsets) >= 0)
        assert np.all((onsets >= times[0] - 24.0) & (onsets < times[-1]))


def _ar(rng, shape, rho=0.3):
    shocks = rng.standard_normal(shape)
    out = np.empty_like(shocks)
    out[0] = shocks[0]
    for t in range(1, shape[0]):
        out[t] = rho * out[t - 1] + shocks[t]
    return (100.0 + out).astype(np.float32)


@pytest.fixture(scope="module")
def noise_session(synthetic_session):
    rng = np.random.default_rng(123)
    signals = tuple(_ar(rng, y.shape) for y in synthetic_session.signals)
    return dataclasses.replace(synthetic_session, signals=signals)


def test_gate_rejects_pure_noise(noise_session):
    table = gate_experiment(noise_session, seed=0, library=SMALL).set_index(
        ["condition", "gate"]
    )
    assert table.loc[("real", True), "n_components"] == 0
    assert table.loc[("null", True), "n_components"] == 0


@pytest.fixture(scope="module")
def modulator_fits(synthetic_session):
    return modulator_experiment(synthetic_session)


def test_modulator_experiment_returns_two_fits(synthetic_session, modulator_fits):
    assert set(modulator_fits) == {"b4", "b4-taskonly"}
    for level, fit in modulator_fits.items():
        assert fit.betas.shape == (N_TRIALS, 30)
        assert fit.record["level"] == level
        assert np.isfinite(fit.betas[:, :20]).all()
    assert modulator_fits["b4-taskonly"].record["task_model"] == ["task"]
    assert modulator_fits["b4"].record["task_model"] == list(
        synthetic_session.task_model.regressor_names
    )
    assert len(modulator_fits["b4"].record["task_model"]) > 1
    # Encoding ridge CV needs a modulator; both fits tune ridge on the full model.
    for fit in modulator_fits.values():
        assert fit.record["ridge_task_model"] == list(
            synthetic_session.task_model.regressor_names
        )
    assert not np.allclose(
        modulator_fits["b4"].betas, modulator_fits["b4-taskonly"].betas
    )


def test_hrf_choice_skips_without_released(synthetic_session):
    result = hrf_choice_experiment(synthetic_session, None)
    assert result == {"skipped": "released HRF index maps not available"}


def test_hrf_choice_with_released_indices(synthetic_session):
    indices = np.ones(30)
    indices[20:25] = 7
    indices[25] = 0
    indices[26] = np.nan
    result = hrf_choice_experiment(synthetic_session, indices)
    for key in ("boldtailor", "glmsingle"):
        assert result[key].shape == (N_TRIALS, 30)
        assert np.isfinite(result[key][:, :20]).all()
    assert np.isnan(result["glmsingle"][:, 25:27]).all()
    assert np.isfinite(result["glmsingle"][:, 20:25]).all()
    assert result["hrf_indices_boldtailor"].shape == (30,)
    np.testing.assert_array_equal(result["hrf_indices_glmsingle"], indices)
    assert not np.allclose(result["glmsingle"][:, :20], result["boldtailor"][:, :20])


def test_hrf_choice_glmsingle_matches_single_hrf_fit(synthetic_session):
    from boldtailor.hrf_library import glmsingle_hrf_library
    from boldtailor.single_trial import fit_single_trials

    from experiments.nsd_replication.inputs import analysis_data

    indices = np.full(30, 4.0)
    result = hrf_choice_experiment(synthetic_session, indices)
    expected = fit_single_trials(
        analysis_data(synthetic_session, np.arange(30)),
        hrf_model=glmsingle_hrf_library().candidates[4],
        run_labels=list(synthetic_session.labels),
    )
    np.testing.assert_allclose(
        result["glmsingle"], np.vstack(expected.run_betas), rtol=1e-6
    )


def test_rt_correlation_hand_computed():
    betas = np.array(
        [
            [1.0, 3.0, 100.0],
            [2.0, 2.0, -50.0],
            [4.0, 6.0, 7.0],
            [0.0, 2.0, 1.0],
            [9.0, 9.0, 9.0],
        ]
    )
    roi = np.array([True, True, False])
    events = [
        pd.DataFrame({"onset": [0.0, 4.0], "response_time": [0.6, 0.9]}),
        pd.DataFrame({"onset": [0.0, 4.0, 8.0], "response_time": [1.2, np.nan, 0.5]}),
    ]
    mean = np.array([2.0, 2.0, 5.0, 9.0])
    rt = np.array([0.6, 0.9, 1.2, 0.5])
    mx, my = mean.mean(), rt.mean()
    expected = np.sum((mean - mx) * (rt - my)) / np.sqrt(
        np.sum((mean - mx) ** 2) * np.sum((rt - my) ** 2)
    )
    assert rt_correlation(betas, events, roi) == pytest.approx(expected)


def test_rt_correlation_rejects_row_mismatch():
    events = [pd.DataFrame({"onset": [0.0], "response_time": [0.5]})]
    with pytest.raises(ValueError, match="rows"):
        rt_correlation(np.zeros((2, 3)), events, np.ones(3, dtype=bool))


@pytest.fixture
def config(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(f"""bids_dir = "{tmp_path}"
output_dir = "{tmp_path / 'out'}"
freesurfer_dir = "{tmp_path}"
subjects = ["sub-07"]
sessions = ["ses-a"]
block_size = 16
""")
    return toml


@pytest.fixture
def patched(monkeypatch, synthetic_session):
    monkeypatch.setattr(run, "_load", lambda *a: synthetic_session)
    monkeypatch.setattr(run, "ppdata_brain", lambda *a: synthetic_session.brain)
    monkeypatch.setattr(run, "load_roi", lambda *a: np.arange(30) < 20)
    monkeypatch.setattr(features, "default_hrf_library", lambda: SMALL)
    monkeypatch.setattr(ladder, "default_hrf_library", lambda: SMALL)


def test_main_features_writes_tables_and_fits(config, patched, tmp_path):
    assert run.main(["features", "--config", str(config), "--source", "ppdata"]) == 0
    metrics = tmp_path / "out/metrics/ppdata/sub-07"
    gate = pd.read_csv(metrics / "features_gate.tsv", sep="\t")
    assert len(gate) == 4
    assert set(gate["session"]) == {"ses-a"}
    # null seed is [subject number, session index, run index]
    assert set(gate["seed"]) == {"7,0"}
    rt = pd.read_csv(metrics / "features_rt.tsv", sep="\t")
    assert set(zip(rt["session"], rt["version"])) == {
        ("ses-a", "b4"),
        ("ses-a", "b4-taskonly"),
    }
    assert np.isfinite(rt["r"]).all()
    hrf = pd.read_csv(metrics / "features_hrf.tsv", sep="\t")
    assert hrf["skipped"].tolist() == ["released HRF index maps not available"]
    for level in ("b4", "b4-taskonly"):
        path = fit_dir(tmp_path / "out", "ppdata", "sub-07", "ses-a", level)
        assert is_complete(path)


def test_features_requires_ppdata(config):
    with pytest.raises(ValueError, match="ppdata"):
        run.main(["features", "--config", str(config), "--source", "released"])


def _stale_fit(out, level="b4"):
    from experiments.nsd_replication.betas import write_fit

    trials = pd.DataFrame(
        {"session": "ses-a", "run": "r", "trial": [0], "onset": [0.0], "image": [1]}
    )
    path = fit_dir(out, "ppdata", "sub-07", "ses-a", level)
    write_fit(path, np.zeros((1, 30)), trials, {"level": level, "inputs_digest": "x"})


def test_features_check_digest_before_computing(config, patched, tmp_path, monkeypatch):
    _stale_fit(tmp_path / "out")
    called = []
    monkeypatch.setattr(run, "session_features", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="inputs digest"):
        run.main(["features", "--config", str(config), "--source", "ppdata"])
    assert not called


def _two_session_config(tmp_path):
    toml = tmp_path / "two.toml"
    toml.write_text(f"""bids_dir = "{tmp_path}"
output_dir = "{tmp_path / 'out'}"
freesurfer_dir = "{tmp_path}"
subjects = ["sub-07"]
sessions = ["ses-a", "ses-b"]
block_size = 16
""")
    return toml


def _stub_features(calls, fail=()):
    from experiments.nsd_replication.ladder import LadderFit

    def stub(data, roi, seed, label, **kwargs):
        calls.append(label)
        if label in fail:
            raise RuntimeError(f"{label} failed")
        fit = LadderFit(betas=np.zeros((N_TRIALS, 30)), record={"level": "b4"})
        tag = dict(session=label)
        return dict(
            fits={"b4": fit, "b4-taskonly": fit},
            gate=pd.DataFrame({"condition": ["real"], "gate": [True]}).assign(**tag),
            rt=pd.DataFrame({"version": ["b4"], "r": [0.1]}).assign(**tag),
            hrf=pd.DataFrame([dict(tag, skipped="x")]),
        )

    return stub


def test_features_keep_finished_sessions_after_a_failure(
    patched, tmp_path, monkeypatch
):
    toml, calls = _two_session_config(tmp_path), []
    monkeypatch.setattr(run, "session_features", _stub_features(calls, fail={"ses-b"}))
    with pytest.raises(RuntimeError, match="ses-b"):
        run.main(["features", "--config", str(toml)])
    metrics = tmp_path / "out/metrics/ppdata/sub-07"
    for name in ("gate", "rt", "hrf"):
        table = pd.read_csv(metrics / f"features_{name}.tsv", sep="\t")
        assert set(table["session"]) == {"ses-a"}, name


def test_features_skip_finished_sessions_unless_recompute(
    patched, tmp_path, monkeypatch
):
    toml, calls = _two_session_config(tmp_path), []
    monkeypatch.setattr(run, "session_features", _stub_features(calls, fail={"ses-b"}))
    with pytest.raises(RuntimeError):
        run.main(["features", "--config", str(toml)])
    calls.clear()
    monkeypatch.setattr(run, "session_features", _stub_features(calls))
    run.main(["features", "--config", str(toml)])
    assert calls == ["ses-b"]
    metrics = tmp_path / "out/metrics/ppdata/sub-07"
    gate = pd.read_csv(metrics / "features_gate.tsv", sep="\t")
    assert list(gate["session"]) == ["ses-a", "ses-b"]
    calls.clear()
    run.main(["features", "--config", str(toml)])
    assert calls == []
    run.main(["features", "--config", str(toml), "--recompute"])
    assert calls == ["ses-a", "ses-b"]
    gate = pd.read_csv(metrics / "features_gate.tsv", sep="\t")
    assert list(gate["session"]) == ["ses-a", "ses-b"]


def test_features_rerun_session_missing_a_fit(patched, tmp_path, monkeypatch):
    toml, calls = _two_session_config(tmp_path), []
    monkeypatch.setattr(run, "session_features", _stub_features(calls))
    run.main(["features", "--config", str(toml)])
    path = fit_dir(tmp_path / "out", "ppdata", "sub-07", "ses-a", "b4-taskonly")
    (path / "metadata.json").unlink()
    calls.clear()
    run.main(["features", "--config", str(toml)])
    assert calls == ["ses-a"]


def test_null_session_seeds_each_run_with_the_seed_tuple(synthetic_session):
    seeded = null_session(synthetic_session, (7, 2))
    for i, (events, times) in enumerate(
        zip(synthetic_session.events, synthetic_session.frame_times)
    ):
        expected = features._shifted_run(
            events, times, np.random.default_rng([7, 2, i])
        )
        pd.testing.assert_frame_equal(seeded.events[i], expected)
    other = null_session(synthetic_session, (8, 2))
    assert not all(a.equals(b) for a, b in zip(seeded.events, other.events))


def test_subject_number():
    assert run.subject_number("sub-07") == 7
    with pytest.raises(ValueError, match="sub-ab"):
        run.subject_number("sub-ab")


def test_features_seed_by_subject_and_session(patched, tmp_path, monkeypatch):
    toml, seeds = _two_session_config(tmp_path), []
    stub = _stub_features([])
    monkeypatch.setattr(
        run,
        "session_features",
        lambda data, roi, seed, label, **k: seeds.append(seed)
        or stub(data, roi, seed, label),
    )
    run.main(["features", "--config", str(toml)])
    assert seeds == [(7, 0), (7, 1)]
