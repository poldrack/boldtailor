import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from experiments.nsd_replication import figures

THRESHOLDS = [0.0, 0.2, 0.4]
VERSIONS = ["b1", "b2", "b4"]


@pytest.fixture
def curves():
    rows = [
        (v, t, 100, 0.01 * i * (1 + t))
        for i, v in enumerate(VERSIONS)
        for t in THRESHOLDS
    ]
    return pd.DataFrame(
        rows, columns=["version", "threshold", "n_features", "mean_difference"]
    )


@pytest.fixture
def lss_curves(curves):
    names = {"b1": "b4", "b2": "lss-assume", "b4": "lss-fit"}
    return curves.assign(version=curves["version"].map(names))


@pytest.fixture
def lag_tables():
    rows = [(lag, 0.5 / lag, 10) for lag in range(1, 6)]
    table = pd.DataFrame(rows, columns=["lag", "mean_r", "n_pairs"])
    return {v: table.assign(version=v, threshold=0.0) for v in VERSIONS}


@pytest.fixture
def rsa():
    rows = [(t, "sub-01", "sub-02", 0.1 + t) for t in THRESHOLDS]
    table = pd.DataFrame(rows, columns=["threshold", "subject_a", "subject_b", "r"])
    return {v: table for v in VERSIONS}


@pytest.fixture
def decoding():
    rows = [
        dict(version=v, threshold=t, accuracy=0.2 + t, chance=0.01, n_classes=100)
        for v in VERSIONS
        for t in THRESHOLDS
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def parity():
    rows = [
        dict(subject=s, level=lv, boldtailor=0.10 + i * 0.01, released=0.11)
        for i, s in enumerate(["sub-01", "sub-02", "sub-03"])
        for lv in VERSIONS
    ]
    return pd.DataFrame(rows)


def _check(fig, n_axes, xlabel=None, ylabel=None):
    assert isinstance(fig, Figure)
    assert len(fig.axes) == n_axes
    if xlabel:
        assert fig.axes[-1].get_xlabel() == xlabel
    if ylabel:
        assert ylabel in fig.axes[0].get_ylabel()
    plt.close(fig)


def test_r1_curves(curves):
    fig = figures.fig_r1_curves(curves)
    assert len(fig.axes[0].lines) == 3
    _check(fig, 1, "Voxel reliability threshold (r)", "difference from composite")


def test_r1_maps_flat(synthetic_session):
    values = np.arange(30.0)
    fig = figures.fig_r1_maps(values, synthetic_session.brain)
    assert [a.get_title() for a in fig.axes] == ["Left", "Right"]
    plt.close(fig)


def test_r1_maps_meshes_not_implemented(synthetic_session):
    with pytest.raises(NotImplementedError, match="surface"):
        figures.fig_r1_maps(np.zeros(30), synthetic_session.brain, meshes={})


def test_r3_lss(lss_curves):
    fig = figures.fig_r3_lss(lss_curves)
    assert len(fig.axes[0].lines) == 3
    _check(fig, 1, "Voxel reliability threshold (r)", "difference from composite")


def test_r4_lag(lag_tables):
    fig = figures.fig_r4_lag(lag_tables)
    assert len(fig.axes[0].lines) == 3
    assert fig.axes[0].lines[0].get_xdata()[0] == 4.0
    _check(fig, 1, "Time between trials (s)", "Mean pattern correlation")


def test_r5_rsa(rsa):
    fig = figures.fig_r5_rsa(rsa)
    assert [a.get_title() for a in fig.axes] == VERSIONS
    plt.close(fig)


def test_r6_decoding(decoding):
    fig = figures.fig_r6_decoding(decoding)
    assert len(fig.axes[0].lines) == 4  # three versions plus chance
    _check(fig, 1, "Voxel reliability threshold (r)", "accuracy")


def test_r2_hrf():
    table = pd.DataFrame(
        {"mean_pairwise_r": [0.4, 0.5], "mean_canonical_baseline": [0.1, 0.2]}
    )
    fig = figures.fig_r2_hrf(table)
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_parity_with_tost(parity):
    tost = {"b4": {"equivalent": True, "p": 0.01}}
    fig = figures.fig_parity(parity, tost)
    assert len(fig.axes) == 3
    assert "TOST" in fig.axes[2].get_title()
    assert "equivalent" in fig.axes[2].get_title()
    plt.close(fig)


def test_parity_without_tost(parity):
    fig = figures.fig_parity(parity)
    assert "TOST" not in fig.axes[0].get_title()
    plt.close(fig)


def test_gate():
    rows = [
        dict(condition=c, gate=g, n_components=n, seed=0)
        for c, g, n in [
            ("real", True, 3),
            ("real", False, 5),
            ("null", True, 0),
            ("null", False, 4),
        ]
    ]
    fig = figures.fig_gate(pd.DataFrame(rows))
    _check(fig, 1, None, "components")


def test_modulators():
    table = pd.DataFrame(
        {
            "version": ["b4", "b4-taskonly"] * 2,
            "r": [0.1, 0.2, 0.3, 0.4],
            "session": ["a", "a", "b", "b"],
        }
    )
    fig = figures.fig_modulators(table)
    _check(fig, 1, None, "RT")
