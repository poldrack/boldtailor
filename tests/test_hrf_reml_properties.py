"""Generated invariants of the public HRF likelihood adapter."""

from dataclasses import replace

import numpy as np
from hypothesis import given, settings, strategies as st

from boldtailor.data import from_arrays
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.hrf_selection import select_hrfs
from boldtailor.workflow.hrf_maps import selection_names, selection_values
from tests.hrf_reml_fixtures import small_reml_problem


def select(data, library, rho=0.2, **kwargs):
    return select_hrfs(
        data,
        library=library,
        method="reml",
        reml_options=RemlHrfOptions(rho=rho),
        **kwargs,
    )


def rebuilt(data, signals, order=(0, 1)):
    return from_arrays(
        [signals[i] for i in order],
        [data.events[i] for i in order],
        frame_times=[data.frame_times[i] for i in order],
    )


@settings(max_examples=6, deadline=None)
@given(
    scale=st.floats(min_value=0.3, max_value=4, allow_nan=False),
    shift=st.floats(min_value=-10, max_value=10, allow_nan=False),
)
def test_scale_and_fixed_span_invariance(scale, shift):
    """Scaling and adding a fixed-span signal preserve selection and variance ratios."""
    data, library, designs = small_reml_problem()
    base = select(data, library)
    mean = designs[0].sum(axis=1, keepdims=True)
    changed = rebuilt(data, [scale * y + shift * mean for y in data.signals])
    result = select(changed, library)
    np.testing.assert_array_equal(result.hrf_indices, base.hrf_indices)
    np.testing.assert_allclose(
        result.reml.selection.fit.ratio, base.reml.selection.fit.ratio, rtol=2e-4
    )
    np.testing.assert_allclose(
        result.reml.selection.fit.noise_variance,
        scale**2 * base.reml.selection.fit.noise_variance,
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        result.reml.delta_criterion, base.reml.delta_criterion, atol=1e-7
    )


@settings(max_examples=6, deadline=None)
@given(
    order=st.permutations((0, 1, 2)),
    reverse=st.booleans(),
    batch=st.integers(min_value=1, max_value=5),
)
def test_feature_run_permutation_and_batch_invariance(order, reverse, batch):
    """Run order, feature order, and candidate batching do not change the model."""
    data, library, _ = small_reml_problem()
    rho = np.array([0.1, 0.2, 0.3])
    base = select(data, library, rho)
    index = list(order)
    changed = rebuilt(
        data, [y[:, index] for y in data.signals], order=(1, 0) if reverse else (0, 1)
    )
    result = select(changed, library, rho[index], candidate_batch_size=batch)
    np.testing.assert_array_equal(result.hrf_indices, base.hrf_indices[index])
    np.testing.assert_allclose(
        result.reml.selection.fit.criterion,
        base.reml.selection.fit.criterion[index],
        atol=1e-8,
    )
    np.testing.assert_array_equal(
        result.reml.selection.any_unrefined, base.reml.selection.any_unrefined[index]
    )
    for f in range(3):
        single = rebuilt(data, [y[:, [f]] for y in data.signals])
        block = select(single, library, rho[f])
        np.testing.assert_allclose(
            block.reml.selection.fit.criterion,
            base.reml.selection.fit.criterion[[f]],
            atol=1e-8,
        )


def test_losing_candidate_qualifications_survive_export():
    """An unresolved losing candidate remains visible even with a refined winner."""
    data, library, _ = small_reml_problem()
    result = select(data, library)
    selected = replace(
        result.reml.selection,
        any_unrefined=np.ones(3, dtype=bool),
        any_at_upper=np.ones(3, dtype=bool),
    )
    result = replace(result, reml=replace(result.reml, selection=selected))
    values = dict(zip(selection_names("reml"), selection_values(result), strict=True))
    assert values["any_unrefined"].all() and values["any_at_upper"].all()
    np.testing.assert_array_equal(values["unrefined"], selected.fit.unrefined)
