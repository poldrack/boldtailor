"""Regularization must preserve signal, respect anatomy, and match scalar REML."""

import importlib
from dataclasses import replace
import numpy as np
import pytest
from experiments.hrf_estimation.test_library_cv import planted_runs


def api():
    return importlib.import_module("experiments.hrf_estimation.regularized_noise")


def test_shrinkage_reduces_error_without_erasing_vertex_differences():
    rng = np.random.default_rng(471)
    truth = np.repeat([0.25, 0.65], 400)
    z = np.arctanh(truth)
    samples = np.tanh(z + rng.normal(0, 0.5, (6, len(z))))
    result = api().shrink_rho(samples, 0.45)
    raw = np.tanh(np.arctanh(samples).mean(axis=0))
    assert np.mean((result["rho"] - truth) ** 2) < np.mean((raw - truth) ** 2)
    assert result["rho"][400:].mean() - result["rho"][:400].mean() > 0.2
    assert np.all((result["weight"] >= 0) & (result["weight"] <= 1))
    assert np.all(np.abs(result["rho"]) < 1)


def test_surface_smoothing_uses_mesh_paths_not_euclidean_proximity():
    # Two disconnected triangles occupy the same coordinates: no cross-mesh pooling.
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 2, dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    weights = api().surface_weights(xyz, faces, np.arange(6), 5.0)
    result = weights @ np.array([0.2] * 3 + [0.8] * 3)
    np.testing.assert_allclose(result, [0.2] * 3 + [0.8] * 3)
    mixed = weights @ np.array([1.0, 0, 0, 0, 0, 0])
    assert 0 < mixed[1] < mixed[0] < 1
    assert mixed[3] == 0


def test_corrected_noise_matches_existing_pooled_estimator(planted_runs):
    from experiments.hrf_estimation.reml_selection import estimate_noise

    runs, _ = planted_runs
    rng = np.random.default_rng(123)
    runs = tuple(
        replace(r, signals=r.signals + rng.normal(size=r.signals.shape)) for r in runs
    )
    matrix = api().run_rhos(runs[:3])
    assert matrix.shape == (3, 2)
    assert np.clip(np.nanmedian(matrix), -0.8, 0.8) == pytest.approx(
        estimate_noise(runs[:3], "amplitude")["rho"]
    )
    runs[-1].signals[:] = 1e10
    np.testing.assert_allclose(api().run_rhos(runs[:3]), matrix)


def _dense_vertex(runs, kernel, vertex):
    from experiments.hrf_estimation.reml_selection import random_design

    signals = [r.signals[:, [vertex]] for r in runs]
    fixed = [r.confounds.to_numpy(dtype=float) for r in runs]
    return signals, fixed, [random_design(r, kernel) for r in runs]


def test_grouped_reml_matches_independent_vertex_likelihoods(planted_runs):
    from tests.reml_oracles import dense_criterion, dense_fit

    runs, library = planted_runs
    rng = np.random.default_rng(53)
    runs = tuple(
        replace(r, signals=r.signals + rng.normal(0, 0.2, r.signals.shape))
        for r in runs
    )
    rho = np.array([0.2, 0.6])
    result = api().fit_grouped(runs, library, rho)
    for v in range(2):
        scores, fits = [], []
        for kernel in library.candidates:
            data = _dense_vertex(runs, kernel, v)
            fits.append(dense_fit(*data, rho[v]))
            scores.append(dense_criterion(*data, fits[-1][0][0], rho[v])[0])
        winner = int(np.argmin(scores))
        assert result.kernels[v].id == winner
        ratio, sigma2 = fits[winner]
        assert result.diagnostics["variance_ratio"][v] == pytest.approx(
            ratio[0], rel=1e-5
        )
        assert result.diagnostics["sigma2"][v] == pytest.approx(sigma2[0], rel=1e-5)
        assert result.diagnostics["candidate_gap"][v] == pytest.approx(
            sorted(scores)[1] - sorted(scores)[0], rel=1e-6
        )
    assert result.training_run_ids == tuple(r.run_id for r in runs)


def test_whitening_diagnostics_detect_correct_ar1_filter(planted_runs):
    from experiments.hrf_estimation.sensitivity import ar1_noise

    runs, _ = planted_runs
    rng = np.random.default_rng(50)
    noisy = tuple(
        replace(
            r,
            signals=np.column_stack(
                [ar1_noise(len(r.signals), 0.6, rng) for _ in range(2)]
            ),
        )
        for r in runs
    )
    diagnostic = api().whitening_diagnostics(noisy, np.array([0.6, 0.6]))
    assert diagnostic.after_abs_lag1.mean() < diagnostic.before_abs_lag1.mean() * 0.5
    assert len(diagnostic) == 8


def test_pilot_persists_paired_scores_rho_and_whitening(planted_runs, tmp_path):
    import json
    import pandas as pd

    runner = importlib.import_module("experiments.hrf_estimation.regularized_pilot")
    runs, library = planted_runs
    import nibabel as nib

    brain = nib.cifti2.BrainModelAxis.from_surface(np.array([4, 7]), 10, "CortexLeft")
    rng = np.random.default_rng(88)
    runs = tuple(
        replace(
            r, signals=r.signals + rng.normal(0, 0.3, r.signals.shape), brain_axis=brain
        )
        for r in runs
    )
    runner.run_pilot(runs, library, np.eye(2), np.arange(2), tmp_path)
    scores = pd.read_csv(tmp_path / "scores.tsv", sep="\t")
    assert len(scores) == 12
    assert set(scores.method) == {"pooled", "shrinkage", "surface"}
    assert set(scores.evaluation) == {"transfer01", "transfer02"}
    assert scores.r2_task_partial.notna().all()
    whitening = pd.read_csv(tmp_path / "whitening.tsv", sep="\t")
    assert len(whitening) == 24
    for path in tmp_path.glob("*_diagnostics.json"):
        d = json.loads(path.read_text())
        assert len(d["training_run_ids"]) == 2
        assert len(d["rho"]) == 2
    assert (tmp_path / "COMPLETE.json").exists()
