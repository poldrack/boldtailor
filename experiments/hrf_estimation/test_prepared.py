"""Fast profiling must preserve the existing scientific objective."""

from dataclasses import replace
import importlib

import numpy as np
import pytest

from experiments.hrf_estimation.continuous import profile_loss
from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.test_library_cv import planted_runs


@pytest.mark.parametrize("mode", ["amplitude", "duration"])
def test_prepared_design_and_loss_match_reference(planted_runs, mode):
    runs, library = planted_runs
    rng = np.random.default_rng(19)
    runs = tuple(replace(r, signals=r.signals + rng.normal(size=r.signals.shape),
                         confounds=r.confounds.assign(duplicate=1.0)) for r in runs)
    api = importlib.import_module("experiments.hrf_estimation.prepared")
    prepared = api.prepare_profile(runs, mode)
    for kernel in library.candidates:
        for run, item in zip(runs, prepared):
            expected = build_design(run, kernel, mode)
            actual = api.event_design(item, kernel)
            np.testing.assert_allclose(actual, np.column_stack([expected.junk, expected.task]),
                                       atol=1e-10, rtol=1e-10)
        for feature in range(2):
            assert api.prepared_loss(prepared, feature, kernel.parameters) == pytest.approx(
                profile_loss(runs, feature, kernel.parameters, mode), abs=1e-11)


def test_prepared_zero_signal_and_invalid_shape(planted_runs):
    runs, _ = planted_runs
    api = importlib.import_module("experiments.hrf_estimation.prepared")
    constant = tuple(replace(r, signals=np.ones_like(r.signals)) for r in runs)
    p = api.prepare_profile(constant, "amplitude")
    assert np.isnan(api.prepared_loss(p, 0, (6, 16, 1, 1, 6, 0, 36)))
    p = api.prepare_profile(runs, "amplitude")
    assert np.isinf(api.prepared_loss(p, 0, (10, 8, 1, 1, 6, 0, 36)))


def test_library_losses_are_batched_without_changing_values(planted_runs):
    runs, library = planted_runs
    api = importlib.import_module("experiments.hrf_estimation.prepared")
    prepared = api.prepare_profile(runs, "amplitude")
    parameters = [(*c.parameters[:5], 0, 36) for c in library.candidates]
    losses = api.library_losses(prepared, parameters)
    expected = [[profile_loss(runs, f, p) for f in range(2)] for p in parameters]
    np.testing.assert_allclose(losses, expected, atol=1e-12)


def test_fast_parallel_fit_matches_reference(offgrid):
    from experiments.hrf_estimation.continuous import fit_continuous

    runs, library, _ = offgrid
    settings = dict(library=library, **{"lambda": 0}, max_starts=2,
                    maxiter=100, maxfun=1500)
    reference = fit_continuous(runs, {**settings, "prepared": False})
    fast = fit_continuous(runs, {**settings, "prepared": True, "n_jobs": 2})
    assert fast.diagnostics["prepared"] is True
    assert fast.training_run_ids == reference.training_run_ids
    assert profile_loss(runs, 0, fast.kernels[0].parameters) < 1e-7
    np.testing.assert_allclose(fast.kernels[0].kernel(.1, 1),
                               reference.kernels[0].kernel(.1, 1), atol=2e-3)


from experiments.hrf_estimation.test_continuous import offgrid


def test_task_inside_nuisance_space_does_not_fit_projection_roundoff(planted_runs):
    import pandas as pd
    runs, library = planted_runs
    truth = library.candidates[1]
    rng = np.random.default_rng(51)
    changed = []
    for run in runs:
        parts = build_design(run, truth)
        confounds = pd.DataFrame(np.column_stack([parts.nuisance, parts.junk, parts.task]))
        changed.append(replace(run, confounds=confounds,
                               signals=rng.normal(size=run.signals.shape)))
    api = importlib.import_module("experiments.hrf_estimation.prepared")
    prepared = api.prepare_profile(tuple(changed), "amplitude")
    assert api.prepared_loss(prepared, 0, truth.parameters) == pytest.approx(1., abs=1e-12)
    np.testing.assert_allclose(api.library_losses(prepared, [truth.parameters]), 1., atol=1e-12)


def test_prepared_rejects_unsupported_event_onsets(planted_runs):
    runs, _ = planted_runs
    run = replace(runs[0], events=runs[0].events.assign(onset=-100.))
    api = importlib.import_module("experiments.hrf_estimation.prepared")
    with pytest.raises(ValueError, match="onset"):
        api.prepare_profile((run,), "amplitude")
