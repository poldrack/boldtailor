import importlib

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor.data import from_arrays
from boldtailor.provenance import RunSources, SourceRef


def entry(module="single_trial", name="fit_single_trials"):
    try:
        return getattr(importlib.import_module(f"boldtailor.{module}"), name)
    except ModuleNotFoundError:
        pytest.fail("Single-trial estimator is not implemented")


def solve(x, n, y, alpha):
    return entry("_single_trial_fit", "fit_trial_run")(x, n, y, alpha=alpha)


@pytest.fixture
def problem():
    rng = np.random.default_rng(52)
    times = 0.775 + np.arange(80) * 1.6
    events = pd.DataFrame(
        {
            "onset": [8.0, 18.0, 35.0, 50.0],
            "duration": [3.0] * 4,
            "response_time": [1.0, 2.0, 3.0, 1.0],
            "73k_id": [2, 2, 5, 2],
        }
    )
    x = np.column_stack(
        [
            compute_regressor(np.array([[onset], [3.0], [1.0]]), "spm", times)[0][:, 0]
            for onset in events.onset
        ]
    )
    signals, confounds, nuisance = [], [], []
    for run in range(2):
        c = pd.DataFrame({f"motion_{run}": rng.normal(size=80)})
        n = np.column_stack([c, np.ones(80)])
        y = (run + 1) * (
            x @ rng.normal(size=(4, 3))
            + n @ rng.normal(size=(2, 3))
            + rng.normal(scale=0.1, size=(80, 3))
        ) + 100 * run
        signals.append(y)
        confounds.append(c)
        nuisance.append(n)
    sources = tuple(
        RunSources(
            signal=SourceRef(
                "signal",
                f"run-{r}_bold.npy",
                byte_size=100,
                modified_at="2026-01-01T00:00:00Z",
            ),
            events=SourceRef(
                "events",
                f"run-{r}_events.tsv",
                byte_size=100,
                modified_at="2026-01-01T00:00:00Z",
            ),
            confounds=SourceRef(
                "confounds",
                f"run-{r}_confounds.tsv",
                byte_size=100,
                modified_at="2026-01-01T00:00:00Z",
            ),
        )
        for r in range(2)
    )
    data = from_arrays(
        signals,
        [events, events],
        frame_times=[times, times],
        confounds=confounds,
        sources=sources,
    )
    return data, x, nuisance


def test_normalized_ridge_leaves_nuisance_unpenalized():
    x = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -2.0]]) + 3
    n = np.ones((4, 1))
    y = 10 + x @ np.array([[2.0], [4.0]])
    fitted = solve(x, n, y, 0.1)
    expected = np.array([[2.0], [4.0]]) / 1.1
    np.testing.assert_allclose(fitted.betas, expected)
    np.testing.assert_allclose(
        fitted.nuisance_betas, np.mean(y - x @ expected, axis=0)[None]
    )


@pytest.mark.parametrize("alpha", [0.0, 0.1, 2.0])
def test_betas_and_pooled_r2_match_independent_augmented_ols(problem, alpha):
    data, x, ns = problem
    result = entry()(data, ridge_alpha=alpha)
    full_sse, nuisance_sse, sst = [], [], []
    for run, (y, n) in enumerate(zip(data.signals, ns, strict=True)):
        residual_x = x - n @ np.linalg.lstsq(n, x, rcond=None)[0]
        scale = np.linalg.norm(residual_x, axis=0)
        matrix = np.column_stack([x, n])
        penalty = np.column_stack(
            [np.diag(np.sqrt(alpha) * scale), np.zeros((4, n.shape[1]))]
        )
        beta = np.linalg.lstsq(
            np.vstack([matrix, penalty]), np.vstack([y, np.zeros((4, 3))]), rcond=None
        )[0]
        np.testing.assert_allclose(result.run_betas[run], beta[:4], atol=1e-10)
        full_sse.append(np.sum((y - matrix @ beta) ** 2, axis=0))
        nuisance_sse.append(
            np.sum((y - n @ np.linalg.lstsq(n, y, rcond=None)[0]) ** 2, axis=0)
        )
        sst.append(np.sum((y - y.mean(axis=0)) ** 2, axis=0))
        np.testing.assert_allclose(result.run_full_r2[run], 1 - full_sse[-1] / sst[-1])
    np.testing.assert_allclose(
        result.full_r2, 1 - np.sum(full_sse, axis=0) / np.sum(sst, axis=0)
    )
    np.testing.assert_allclose(
        result.nuisance_r2, 1 - np.sum(nuisance_sse, axis=0) / np.sum(sst, axis=0)
    )
    np.testing.assert_allclose(result.delta_r2, result.full_r2 - result.nuisance_r2)
    assert not np.allclose(result.full_r2, np.mean(result.run_full_r2, axis=0))
    assert result.trial_table.trial_index.tolist() == list(range(8))
    assert result.trial_table.run_index.tolist() == [0] * 4 + [1] * 4


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_rt_and_stimulus_ids_never_change_fit(problem, alpha):
    data, _, _ = problem
    first = entry()(data, ridge_alpha=alpha)
    events = [
        frame.assign(response_time=np.nan, **{"73k_id": 100}) for frame in data.events
    ]
    modified = from_arrays(
        data.signals, events, frame_times=data.frame_times, confounds=data.confounds
    )
    second = entry()(modified, ridge_alpha=alpha)
    for a, b in zip(first.run_betas, second.run_betas, strict=True):
        np.testing.assert_array_equal(a, b)
    assert len(second.trial_table) == 8


def test_constant_features_keep_nan_positions_and_zero_sums():
    x = np.array([[1.0], [-1.0], [1.0], [-1.0], [0.0]])
    n = np.ones((5, 1))
    y = np.column_stack([2 * x[:, 0] + 10, np.zeros(5), np.ones(5) * 100])
    result = solve(x, n, y, 0.0)
    np.testing.assert_allclose(result.betas[:, 0], [2.0])
    assert np.isnan(result.betas[:, 1:]).all()
    np.testing.assert_array_equal(result.full_sse[1:], [0.0, 0.0])
    np.testing.assert_array_equal(result.total_ss[1:], [0.0, 0.0])


def test_redundant_nuisance_has_same_fit():
    x = np.arange(8, dtype=float)[:, None] ** 2
    n = np.column_stack([np.ones(8), np.arange(8)])
    y = 3 * x + 2 * n[:, [1]] + 100
    a = solve(x, n, y, 0.1)
    b = solve(x, np.column_stack([n, n]), y, 0.1)
    np.testing.assert_allclose(a.betas, b.betas)
    np.testing.assert_allclose(a.full_sse, b.full_sse)


@pytest.mark.parametrize("alpha", [-1, np.inf, np.nan, True, "0.1"])
def test_invalid_alpha_rejected(problem, alpha):
    with pytest.raises(ValueError, match="alpha"):
        entry()(problem[0], ridge_alpha=alpha)


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_unidentifiable_trials_rejected(alpha):
    n = np.ones((5, 1))
    for x in [n, np.column_stack([np.arange(5), np.arange(5)])]:
        with pytest.raises(ValueError, match="rank|identif|support"):
            solve(x, n, np.arange(5)[:, None], alpha)


def test_no_residual_degrees_of_freedom_rejected():
    with pytest.raises(ValueError, match="degrees of freedom"):
        solve(np.eye(3)[:, :2], np.ones((3, 1)), np.arange(3)[:, None], 0.1)


def test_run_labels_are_unique_and_checked(problem):
    for labels in [["same", "same"], ["only-one"], ["run-1", "/private/run-2"]]:
        with pytest.raises(ValueError, match="label"):
            entry()(problem[0], run_labels=labels)


def test_results_are_immutable_and_metadata_is_copied(problem):
    result = entry()(problem[0], ridge_alpha=0.1)
    for values in [*result.run_betas, result.full_r2, result.delta_r2]:
        with pytest.raises(ValueError):
            values.setflags(write=True)
    trials = result.trial_table
    trials.loc[0, "onset"] = -999
    assert result.trial_table.loc[0, "onset"] == 8
    design = result.design_matrices[0]
    design.iloc[0, 0] = -999
    assert result.design_matrices[0].iloc[0, 0] != -999
    diagnostics = result.diagnostics
    diagnostics[0]["rank"] = -1
    assert result.diagnostics[0]["rank"] > 0


def test_provenance_tracks_design_and_penalty_without_behavior_values(problem, caplog):
    data, _, _ = problem
    first = entry()(data)
    assert first.provenance.analysis_fingerprint
    assert (
        first.provenance.analysis_fingerprint
        == entry()(data).provenance.analysis_fingerprint
    )
    assert (
        first.provenance.analysis_fingerprint
        != entry()(data, ridge_alpha=0.1).provenance.analysis_fingerprint
    )
    events = data.events
    events[0].loc[0, "onset"] += 0.2
    changed = from_arrays(
        data.signals,
        events,
        frame_times=data.frame_times,
        confounds=data.confounds,
        sources=data.provenance.sources,
    )
    assert (
        entry()(changed).provenance.analysis_fingerprint
        != first.provenance.analysis_fingerprint
    )
    assert "73k_id" not in caplog.text
    assert "/private/" not in caplog.text


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_known_rt_variability_with_ar_noise_and_nuisance_only_feature(alpha):
    rng = np.random.default_rng(842)
    times = np.arange(220) * 1.6
    rt = rng.uniform(0.4, 2.0, 30)
    onsets = 8 + np.arange(30) * 10.0
    x = np.column_stack(
        [
            compute_regressor(np.array([[t], [3.0], [1.0]]), "spm", times)[0][:, 0]
            for t in onsets
        ]
    )
    motion = rng.normal(size=len(times))
    noise = rng.normal(scale=0.01, size=len(times))
    for i in range(1, len(noise)):
        noise[i] += 0.6 * noise[i - 1]
    amplitudes = 2.0 * (rt - rt.mean()) + 1.0
    nuisance = 100 + motion * 3
    signals = np.column_stack(
        [x @ amplitudes + nuisance + noise, nuisance, np.zeros(len(times))]
    )
    events = pd.DataFrame({"onset": onsets, "duration": 3.0, "response_time": rt})
    data = from_arrays(
        [signals],
        [events],
        frame_times=[times],
        confounds=[pd.DataFrame({"motion": motion})],
    )
    result = entry()(data, ridge_alpha=alpha)
    assert np.corrcoef(result.run_betas[0][:, 0], rt)[0, 1] > 0.98
    np.testing.assert_allclose(result.run_betas[0][:, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(result.delta_r2[1], 0.0, atol=1e-12)
    assert np.isnan(result.run_betas[0][:, 2]).all()
    assert np.isnan(result.full_r2[2])


def test_nested_event_metadata_is_owned_on_every_result_access(problem):
    data, _, _ = problem
    events = data.events
    for table in events:
        table["metadata"] = [{"tags": ["original"]} for _ in range(len(table))]
    enriched = from_arrays(
        data.signals, events, frame_times=data.frame_times, confounds=data.confounds
    )
    result = entry()(enriched)
    exposed = result.trial_table
    exposed.loc[0, "metadata"]["tags"].append("changed")
    assert result.trial_table.loc[0, "metadata"] == {"tags": ["original"]}
    assert enriched.events[0].loc[0, "metadata"] == {"tags": ["original"]}
