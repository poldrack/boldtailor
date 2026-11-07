"""Catch wrong motion expansion, drift cutoff, and silent scan changes."""

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def motion():
    values = np.array([1.0, 3.0, 2.0, 5.0])
    return pd.DataFrame({name: values * (i + 1) for i, name in enumerate(
        ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'))})


def build(motion, times, cutoff=80):
    module = importlib.import_module('experiments.hrf_estimation.timing')
    return module.build_nuisance(motion, times, cutoff)


def test_motion_expansion_keeps_all_scans(motion):
    result = build(motion, np.arange(4) * 1.3333334)
    assert result.shape == (4, 25)
    np.testing.assert_array_equal(result.trans_x, [1, 3, 2, 5])
    np.testing.assert_array_equal(result.trans_x_derivative1, [0, 2, -1, 3])
    np.testing.assert_array_equal(result.trans_x_power2, [1, 9, 4, 25])
    np.testing.assert_array_equal(result.trans_x_derivative1_power2, [0, 4, 1, 9])
    np.testing.assert_array_equal(result.intercept, np.ones(4))
    assert motion.columns.size == 6


def test_eighty_second_drift_basis_removes_slow_not_fast_modes():
    n = 240
    motion = pd.DataFrame(np.zeros((n, 6)), columns=[
        'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])
    result = build(motion, np.arange(n) * (4 / 3))
    cosines = result.filter(like='cosine').to_numpy()
    assert cosines.shape == (240, 8)  # 320 s run, highest removed mode = 1/80 Hz
    slow = np.cos(np.pi * (np.arange(n) + 0.5) * 3 / n)
    fast = np.cos(np.pi * (np.arange(n) + 0.5) * 12 / n)
    residual = lambda y: y - cosines @ np.linalg.lstsq(cosines, y, rcond=None)[0]
    np.testing.assert_allclose(residual(slow), 0, atol=1e-12)
    np.testing.assert_allclose(residual(fast), fast, atol=1e-12)
    np.testing.assert_allclose(cosines.mean(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize('problem', ['rows', 'nan', 'missing', 'irregular', 'cutoff'])
def test_invalid_nuisance_inputs_fail_without_imputation(motion, problem):
    times, cutoff = np.arange(4, dtype=float), 80
    if problem == 'rows':
        motion = motion.iloc[:-1]
    elif problem == 'nan':
        motion.loc[0, 'rot_z'] = np.nan
    elif problem == 'missing':
        motion = motion.drop(columns='rot_y')
    elif problem == 'irregular':
        times[-1] = 3.2
    else:
        cutoff = 0
    with pytest.raises(ValueError):
        build(motion, times, cutoff)
