from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.hrf_library import HrfCandidate
from experiments.hrf_estimation.inputs import RunData


@pytest.fixture
def template():
    times = np.arange(160, dtype=float)
    events = pd.DataFrame(dict(onset=[8., 31., 55., 80., 105., 125.],
        duration=3., trial_type=[0, 1, 0, 1, 0, 1],
        response_time=[.4, 1.8, 3.8, .9, np.nan, 1.3],
        is_correct=[1, 1, 1, 1, 0, 0], missing_data=[0, 0, 0, 0, 1, 0]))
    return RunData('synthetic_run-01', np.zeros((160, 1)), events, times,
                   pd.DataFrame({'intercept': np.ones(160)}), np.array([0]))


@pytest.fixture
def truth():
    return HrfCandidate(0, 'spm', (6, 16, 1, 1, 6, 0, 32))


@pytest.mark.parametrize('model,ncols', [('ConsDurNoRT', 2), ('RTDur', 2),
    ('ConsDurRTMod', 3), ('ConsDurRTDur', 3)])
def test_design_models_and_preserved_events(template, truth, model, ncols):
    from experiments.hrf_simulations.design import task_design
    before = template.events.copy(deep=True)
    parts = task_design(template, truth, model, 'stimulus')
    assert parts.task.shape == (160, ncols)
    assert parts.junk.shape == (160, 2)
    pd.testing.assert_frame_equal(before, template.events)


def test_duration_boxcars_preserve_neural_area(template, truth):
    from experiments.hrf_simulations.design import convolve
    times = np.arange(120.)
    short = convolve([5.], [1.], [1.], times, truth)
    long = convolve([5.], [4.], [1.], times, truth)
    assert long.sum() / short.sum() == pytest.approx(4., rel=.025)
    assert long.max() > short.max() * 2
    empty = convolve([], [], [], times, truth)
    np.testing.assert_array_equal(empty, np.zeros(len(times)))


@pytest.mark.parametrize('convention', ['brief', 'stimulus'])
@pytest.mark.parametrize('mechanism,model', [('none', 'ConsDurNoRT'),
    ('amplitude', 'ConsDurRTMod'), ('duration', 'RTDur'),
    ('additive_duration', 'ConsDurRTDur')])
def test_independent_synthesis_matches_design(template, truth, mechanism, model, convention):
    from experiments.hrf_simulations.design import task_design
    from experiments.hrf_simulations.simulation import signal_components
    task, junk = signal_components(template, truth, mechanism, convention, gamma=.5)
    parts = task_design(template, truth, model, convention)
    beta = np.array([1., 1., .5])[:parts.task.shape[1]]
    np.testing.assert_allclose(task, parts.task @ beta, atol=.045, rtol=.025)
    np.testing.assert_allclose(junk, parts.junk.sum(axis=1), atol=.045, rtol=.025)


def test_no_cross_run_convolution_and_missing_junk(template, truth):
    from experiments.hrf_simulations.simulation import signal_components
    late = replace(template, events=template.events.assign(onset=150.))
    a, _ = signal_components(late, truth, 'duration', 'brief')
    b, _ = signal_components(template, truth, 'duration', 'brief')
    assert np.max(np.abs(a[:145])) < 1e-12
    assert np.max(np.abs(b[:8])) < 1e-12


def test_noise_variance_acf_and_stationary_start():
    from experiments.hrf_simulations.noise import ar1
    rng = np.random.default_rng(5)
    x = ar1(150000, .65, 2.5, rng)
    assert np.var(x) == pytest.approx(2.5, rel=.025)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(.65, abs=.01)
    starts = [ar1(3, .8, 2., rng)[0] for _ in range(5000)]
    assert np.var(starts) == pytest.approx(2., rel=.06)
    with pytest.raises(ValueError):
        ar1(20, 1., 1., rng)


@pytest.mark.parametrize('n', [159, 160])
def test_phase_surrogate_preserves_spectrum(n):
    from experiments.hrf_simulations.noise import phase_surrogate
    rng = np.random.default_rng(40)
    x = rng.normal(size=n) + 4
    y = phase_surrogate(x, rng)
    assert abs(y.mean()) < 1e-14
    np.testing.assert_allclose(abs(np.fft.rfft(y))[1:], abs(np.fft.rfft(x))[1:], atol=1e-12)
    assert not np.allclose(x-x.mean(), y)


def test_invalid_convention_and_model(template, truth):
    from experiments.hrf_simulations.design import task_design
    with pytest.raises(ValueError):
        task_design(template, truth, 'bad', 'brief')
    with pytest.raises(ValueError):
        task_design(template, truth, 'RTDur', 'bad')
