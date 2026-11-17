"""Mutating helpers name their destination and preserve source ownership."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from boldtailor import _arrays, reml, trial_encoding


def fit_values(ratio, criterion, status="ok"):
    """Subset fit values in the layout the REML fit buffers use."""
    n = len(ratio)
    return reml._FitBuffers(
        ratio=np.array(ratio, dtype=float),
        noise_variance=np.ones(n),
        criterion=np.array(criterion, dtype=float),
        status=np.full(n, status, dtype=object),
        at_upper=np.zeros(n, dtype=bool),
        unrefined=np.ones(n, dtype=bool),
    )


@pytest.mark.parametrize("keep", [slice(None), np.array([True, False, True])])
def test_fit_buffers_write_updates_only_selected_feature_positions(keep):
    """Writing subset values fills the kept feature positions and leaves the source intact."""
    destination = reml._FitBuffers.empty(6)
    values = fit_values([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
    before = {name: array.copy() for name, array in vars(values).items()}
    assert destination.write(np.array([4, 0, 2]), values, keep) is None
    nan = np.nan
    wanted_ratio = (
        [20.0, nan, 30.0, nan, 10.0, nan]
        if isinstance(keep, slice)
        else [nan, nan, 30.0, nan, 10.0, nan]
    )
    written = ~np.isnan(wanted_ratio)
    np.testing.assert_array_equal(destination.ratio, wanted_ratio)
    np.testing.assert_array_equal(destination.unrefined, written)
    assert set(destination.status[written]) == {"ok"}
    assert set(destination.status[~written]) == {"no_residual_signal"}
    for name, array in vars(values).items():
        np.testing.assert_array_equal(array, before[name])


def test_fit_buffers_freeze_to_the_public_fit():
    """Buffers become a RemlFit whose effect variance is ratio times noise variance."""
    buffers = reml._FitBuffers.empty(3)
    buffers.write(np.array([1]), fit_values([4.0], [7.0]))
    buffers.mark_unresolved(np.array([2]))
    fit = buffers.freeze(np.zeros(3), 11)
    assert isinstance(fit, reml.RemlFit) and fit.restricted_dof == 11
    assert fit.status == ("no_residual_signal", "ok", "numerically_unresolved_residual")
    np.testing.assert_array_equal(fit.effect_variance, [np.nan, 4.0, np.nan])
    np.testing.assert_array_equal(fit.criterion, [np.nan, 7.0, np.nan])
    assert not fit.ratio.flags.writeable


def test_selection_buffers_rank_only_supported_fits_and_keep_the_first_tie():
    """Only "ok" fits compete, exact ties keep the earlier candidate, and the gap is runner-up minus best."""
    state, features = reml._SelectionBuffers.empty(3), np.arange(3)
    first = state.update(features, fit_values([1, 1, 1], [5.0, 5.0, 5.0]), 0)
    np.testing.assert_array_equal(first, [True, True, True])
    second = state.update(features, fit_values([1, 1, 1], [3.0, 5.0, 9.0]), 1)
    np.testing.assert_array_equal(second, [True, False, False])
    ignored = state.update(
        features, fit_values([1, 1, 1], [0.0, 0.0, 0.0], status="variance_zero"), 2
    )
    assert not ignored.any()
    np.testing.assert_array_equal(state.winner, [1, 0, 0])
    np.testing.assert_array_equal(state.gap(), [2.0, 0.0, 4.0])
    assert state.any_unrefined.all() and not state.any_at_upper.any()
    selection = state.freeze(reml._FitBuffers.empty(3).freeze(np.zeros(3), 5), 3)
    assert isinstance(selection, reml.RemlSelection) and selection.n_candidates == 3


def test_selection_gap_is_undefined_with_fewer_than_two_supported_fits():
    """A feature with one supported candidate, or none, has no gap."""
    state = reml._SelectionBuffers.empty(2)
    values = fit_values([1, 1], [5.0, 5.0])
    values.status[1] = "unidentifiable"
    state.update(np.arange(2), values, 0)
    np.testing.assert_array_equal(state.winner, [0, -1])
    assert np.isnan(state.gap()).all()


def test_censor_response_time_in_place_changes_only_invalid_times():
    """Verify censor response time in place changes only invalid times."""
    table = pd.DataFrame(
        {"response_time": [1.0, 0.0, -1.0, np.inf], "other": [1, 2, 3, 4]}
    )
    assert trial_encoding._censor_response_time_in_place(table) is None
    np.testing.assert_allclose(
        table.response_time, [1.0, np.nan, np.nan, np.nan], equal_nan=True
    )
    assert table.other.tolist() == [1, 2, 3, 4]


def test_field_mutators_replace_fields_without_aliasing_source_arrays():
    """Verify field mutators replace fields without aliasing source arrays."""
    source = np.array([1.0, 2.0])
    instance = SimpleNamespace(array=source, arrays=[source], labels=["a", "b"])
    assert _arrays.copy_array_fields_in_place(instance, ["array"]) is None
    assert _arrays.copy_array_tuple_fields_in_place(instance, ["arrays"]) is None
    assert _arrays.freeze_sequence_fields_in_place(instance, ["labels"]) is None
    assert _arrays.set_fields_in_place(instance, label="owned") is None
    source[:] = 99
    np.testing.assert_array_equal(instance.array, [1.0, 2.0])
    np.testing.assert_array_equal(instance.arrays[0], [1.0, 2.0])
    assert not instance.array.flags.writeable
    assert not instance.arrays[0].flags.writeable
    assert isinstance(instance.arrays, tuple)
    assert instance.labels == ("a", "b")
    assert instance.label == "owned"
