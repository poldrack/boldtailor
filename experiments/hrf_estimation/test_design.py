"""Catch junk double-counting and RT convolution mistakes."""

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor.hrf_library import HrfCandidate, CANONICAL_PARAMETERS
from experiments.hrf_estimation.inputs import RunData


@pytest.fixture
def event_run():
    times = np.arange(120) * (4 / 3)
    events = pd.DataFrame(dict(onset=[8, 24, 40, 56, 72, 88], duration=[3]*6,
        trial_type=[0, 1, 0, 1, 0, 1], response_time=[0.6, 1.2, np.nan, 0.8, 1.0, 1.6],
        is_correct=[1, 1, 0, 0, 1, 1], missing_data=[0, 0, 1, 0, 0, 0]))
    return RunData('sub-07_ses-nsd11_task-nsdcore_run-01', np.zeros((120, 1)),
        events, times, pd.DataFrame({'intercept': np.ones(120)}), np.array([0]))


def api():
    return importlib.import_module('experiments.hrf_estimation.design')


def test_exclusive_trial_partition(event_run):
    table = api().partition_events(event_run.events)
    assert table.category.tolist() == ['valid', 'valid', 'missing', 'incorrect', 'valid', 'valid']
    assert 'category' not in event_run.events


@pytest.mark.parametrize('column,value', [('is_correct', 3), ('trial_type', 2),
                                         ('missing_data', 5), ('duration', -1)])
def test_unknown_event_codes_fail(event_run, column, value):
    table = event_run.events.copy()
    table.loc[0, column] = value
    with pytest.raises(ValueError):
        api().partition_events(table)


def test_design_has_one_rt_effect_and_separate_junk(event_run):
    kernel = HrfCandidate(0, 'spm', CANONICAL_PARAMETERS)
    parts = api().build_design(event_run, kernel, 'amplitude')
    assert parts.task_names == ('trial_type_0', 'trial_type_1', 'response_time')
    assert parts.junk_names == ('missing', 'incorrect')
    # Independent Nilearn convolution: constant durations give a common event scale.
    def column(onsets, modulation):
        raw, _ = compute_regressor(np.array([onsets, [3]*len(onsets), modulation]),
            kernel.kernel, event_run.frame_times, oversampling=50)
        return raw[:, 0]
    reference = column([8, 24, 72, 88], [0.6, 1.2, 1.0, 1.6])
    scale = np.dot(reference, parts.task[:, 2]) / np.dot(reference, reference)
    np.testing.assert_allclose(parts.task[:, 2], reference * scale, atol=1e-10)
    np.testing.assert_allclose(parts.task[:, 0], column([8, 72], [1, 1])*scale, atol=1e-10)
    np.testing.assert_allclose(parts.junk[:, 0], column([40], [1])*scale, atol=1e-10)
    np.testing.assert_allclose(parts.junk[:, 1], column([56], [1])*scale, atol=1e-10)


def test_absent_junk_categories_are_zero_columns(event_run):
    events = event_run.events.iloc[[0, 1, 4, 5]].copy()
    parts = api().build_design(replace(event_run, events=events),
        HrfCandidate(0, 'spm', CANONICAL_PARAMETERS), 'amplitude')
    np.testing.assert_array_equal(parts.junk, np.zeros((120, 2)))


def test_rt_duration_uses_unit_height_not_per_event_peak_scaling(event_run):
    events = event_run.events.iloc[[0, 1]].copy()
    kernel = HrfCandidate(0, 'spm', CANONICAL_PARAMETERS)
    run = replace(event_run, events=events)
    parts = api().build_design(run, kernel, 'duration')
    expected, _ = compute_regressor(np.array([events.onset, events.response_time, [1, 1]]),
        kernel.kernel, run.frame_times, oversampling=50)
    np.testing.assert_allclose(parts.task[:, 2], expected[:, 0], atol=1e-10)
