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
