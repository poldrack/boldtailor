"""Property-based tests for denoising pool PCA and component-count selection."""

from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd

from boldtailor.denoising import choose_count, run_components


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    scale=st.floats(
        min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)
def test_run_components_is_invariant_to_signal_scale_and_nuisance_transform(
    seed, scale
):
    """Verify pool PCA components and singular values are invariant to positive signal scaling and invertible confound transforms."""
    rng = np.random.default_rng(seed)
    n_scans, n_features = 45, 10
    confounds = pd.DataFrame(
        {"c0": rng.normal(size=n_scans), "c1": rng.normal(size=n_scans)}
    )
    latent = rng.normal(size=(n_scans, 2))
    signal = latent @ rng.normal(size=(2, n_features)) + 0.2 * rng.normal(
        size=(n_scans, n_features)
    )
    pool = np.ones(n_features, dtype=bool)

    base = run_components(signal, confounds, pool)
    scaled = run_components(signal * scale, confounds, pool)
    np.testing.assert_allclose(
        scaled.singular_values, base.singular_values, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        scaled.components, base.components, rtol=1e-10, atol=1e-12
    )

    mix = rng.normal(size=(2, 2))
    while np.linalg.cond(mix) > 50:
        mix = rng.normal(size=(2, 2))
    transformed_confounds = pd.DataFrame(
        confounds.to_numpy() @ mix, columns=["m0", "m1"]
    )
    transformed = run_components(signal, transformed_confounds, pool)
    np.testing.assert_allclose(
        transformed.singular_values, base.singular_values, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        transformed.components, base.components, rtol=1e-10, atol=1e-12
    )


@settings(max_examples=40, deadline=None)
@given(
    scores=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=12,
    ),
    pcstop=st.floats(
        min_value=1.0, max_value=1.5, allow_nan=False, allow_infinity=False
    ),
)
def test_choose_count_respects_zero_baseline_and_pcstop_one_argmax(scores, pcstop):
    """Verify choose_count picks 0 when no count beats baseline and picks the argmax when pcstop is 1."""
    counts = tuple(range(len(scores)))
    perf = np.asarray(scores, dtype=float)
    curve = perf - perf[0]
    chosen = choose_count(counts, perf, pcstop)
    if np.max(curve) <= 0:
        assert chosen == 0
    if pcstop == 1.0 and np.max(curve) > 0:
        slack = 64 * np.finfo(float).eps * max(1.0, abs(float(np.max(curve))))
        assert curve[chosen] >= np.max(curve) - slack
