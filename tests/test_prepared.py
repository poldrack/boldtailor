from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.provenance import ProvenanceRecord


@pytest.fixture
def prepared_inputs():
    signals = [
        np.arange(24, dtype=float).reshape(8, 3),
        np.arange(30, dtype=float).reshape(10, 3),
    ]
    designs = [
        pd.DataFrame(
            {
                "face": [0, 1] * 4,
                "motion": np.linspace(0, 1, 8),
                "constant": 1.0,
            }
        ),
        pd.DataFrame(
            {
                "constant": 1.0,
                "motion": np.linspace(0, 1, 10),
                "face": [0, 1] * 5,
            }
        ),
    ]
    roles = [
        {"face": "task", "motion": "nuisance", "constant": "intercept"},
        {"constant": "intercept", "motion": "nuisance", "face": "task"},
    ]
    metadata = [
        {"subject": "01", "session": "01", "task": "faces", "run": "1"},
        {"subject": "01", "session": "02", "task": "faces", "run": "1"},
    ]
    return signals, designs, roles, metadata


def _make_prepared(
    prepared_inputs,
    *,
    tr: float | None = 2.0,
    frame_times: np.ndarray | list[np.ndarray] | None = None,
):
    signals, designs, roles, metadata = prepared_inputs
    return PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=tr,
        frame_times=frame_times,
        column_roles=roles,
        run_metadata=metadata,
    )


def _replace_design_column(case, values):
    design = case[1][0].copy()
    design["face"] = values
    case[1][0] = design


def test_prepared_analysis_owns_inputs_and_returns_defensive_state(prepared_inputs):
    signals, designs, roles, metadata = prepared_inputs
    original = deepcopy((signals, designs, roles, metadata))
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=2.0,
        column_roles=roles,
        run_metadata=metadata,
    )

    signals[0][:] = -1
    designs[0].iloc[:, :] = -1
    roles[0]["face"] = "other"
    metadata[0]["subject"] = "changed"

    assert prepared.n_runs == 2
    assert prepared.n_features == 3
    np.testing.assert_array_equal(prepared.signals[0], original[0][0])
    pd.testing.assert_frame_equal(
        prepared.design_matrices[0], original[1][0], check_dtype=False
    )
    assert all(dtype == np.float64 for dtype in prepared.design_matrices[0].dtypes)
    assert prepared.column_roles[0]["face"] == "task"
    assert prepared.run_metadata[0]["subject"] == "01"
    assert isinstance(prepared.provenance, ProvenanceRecord)


def test_prepared_analysis_accessors_return_defensive_copies(prepared_inputs):
    prepared = _make_prepared(prepared_inputs)
    designs = prepared.design_matrices
    roles = prepared.column_roles
    metadata = prepared.run_metadata

    designs[0].loc[0, "face"] = -1
    roles[0]["face"] = "other"
    metadata[0]["subject"] = "changed"

    assert prepared.design_matrices[0].loc[0, "face"] == 0.0
    assert prepared.column_roles[0]["face"] == "task"
    assert prepared.run_metadata[0]["subject"] == "01"


def test_prepared_analysis_arrays_are_strictly_immutable(prepared_inputs):
    prepared = _make_prepared(prepared_inputs)
    for values in (*prepared.signals, *prepared.frame_times):
        assert values.flags.owndata
        assert values.dtype == np.float64
        assert values.flags.c_contiguous
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda case: case[1].__setitem__(0, np.ones((8, 3))), "pandas DataFrame"),
        (
            lambda case: case[1][0].rename(columns={"face": ""}, inplace=True),
            "nonempty strings",
        ),
        (
            lambda case: setattr(case[1][0], "columns", ["face", "face", "constant"]),
            "duplicate",
        ),
        (
            lambda case: _replace_design_column(case, [True] * 8),
            "finite numeric",
        ),
        (
            lambda case: _replace_design_column(case, [np.nan] * 8),
            "finite numeric",
        ),
        (lambda case: case[2][0].pop("motion"), "one role per design column"),
        (
            lambda case: case[2][0].__setitem__("motion", "learned"),
            "invalid column role",
        ),
        (
            lambda case: case[3][0].__setitem__("path", "/private/data"),
            "path-like",
        ),
    ],
)
def test_prepared_analysis_rejects_invalid_design_inputs(
    prepared_inputs, mutator, message
):
    case = deepcopy(prepared_inputs)
    mutator(case)

    with pytest.raises(ValueError, match=message):
        _make_prepared(case)


@pytest.mark.parametrize(
    ("design", "message"),
    [
        (pd.DataFrame(columns=["face", "motion", "constant"]), "nonzero dimensions"),
        (pd.DataFrame(index=range(8)), "nonzero dimensions"),
    ],
)
def test_prepared_analysis_rejects_empty_design_dimensions(
    prepared_inputs, design, message
):
    case = deepcopy(prepared_inputs)
    case[1][0] = design

    with pytest.raises(ValueError, match=message):
        _make_prepared(case)


def test_prepared_analysis_rejects_design_signal_row_mismatch(prepared_inputs):
    case = deepcopy(prepared_inputs)
    case[1][0] = case[1][0].iloc[:-1]

    with pytest.raises(ValueError, match="8 rows"):
        _make_prepared(case)


def test_prepared_analysis_rejects_inconsistent_signal_feature_counts(prepared_inputs):
    case = deepcopy(prepared_inputs)
    case[0][1] = np.ones((10, 2))

    with pytest.raises(ValueError, match="same number of features"):
        _make_prepared(case)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (1, "design_matrices"),
        (2, "column_roles"),
        (3, "run_metadata"),
    ],
)
def test_prepared_analysis_rejects_mismatched_run_counts(
    prepared_inputs, item, message
):
    case = deepcopy(prepared_inputs)
    case[item].pop()

    with pytest.raises(ValueError, match=message):
        _make_prepared(case)


def test_prepared_analysis_rejects_nonmapping_run_metadata(prepared_inputs):
    case = deepcopy(prepared_inputs)
    case[3][0] = ["subject", "01"]

    with pytest.raises(ValueError, match="run_metadata.*mapping"):
        _make_prepared(case)


@pytest.mark.parametrize(
    ("tr", "frame_times"),
    [(None, None), (2.0, [np.arange(8.0), np.arange(10.0)])],
)
def test_prepared_analysis_requires_exactly_one_timing_source(
    prepared_inputs, tr, frame_times
):
    with pytest.raises(ValueError, match="exactly one of tr or frame_times"):
        _make_prepared(prepared_inputs, tr=tr, frame_times=frame_times)


def test_prepared_analysis_accepts_explicit_other_column_role(prepared_inputs):
    case = deepcopy(prepared_inputs)
    case[2][0]["motion"] = "other"

    prepared = _make_prepared(case)

    assert prepared.column_roles[0]["motion"] == "other"
