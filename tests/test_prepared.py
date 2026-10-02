import json
import logging
import re
from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.provenance import ProvenanceRecord, RunSources


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
    sources: tuple[RunSources, ...] | None = None,
    provenance_metadata: dict[str, object] | None = None,
):
    signals, designs, roles, metadata = prepared_inputs
    return PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=tr,
        frame_times=frame_times,
        column_roles=roles,
        run_metadata=metadata,
        sources=sources,
        provenance_metadata=provenance_metadata,
    )


def _replace_design_column(case, values):
    design = case[1][0].copy()
    design["face"] = values
    case[1][0] = design


def _structured_records(caplog):
    return [json.loads(record.getMessage()) for record in caplog.records]


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


def test_prepared_analysis_arrays_are_readonly(prepared_inputs):
    prepared = _make_prepared(prepared_inputs)
    for values in (*prepared.signals, *prepared.frame_times):
        assert values.flags.owndata
        assert values.dtype == np.float64
        assert values.flags.c_contiguous
        assert not values.flags.writeable
        assert type(values) is np.ndarray
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0


def _set_design(index, design):
    def mutate(case):
        case[1][index] = design

    return mutate


def _absolute_path_column(case):
    case[1][0] = case[1][0].rename(columns={"face": "/private/secret/design.tsv"})
    case[2][0]["/private/secret/design.tsv"] = case[2][0].pop("face")


def _drop_last_design_row(case):
    case[1][0] = case[1][0].iloc[:-1]


def _drop_run_item(item):
    def mutate(case):
        case[item].pop()

    return mutate


def _drop_role(case):
    case[2][0].pop("motion")


def _nested_path_metadata(kind):
    nested = {"safe": [{"/private/secret/metadata.json": "redacted"}]}

    def mutate(case):
        if kind == "run":
            case[3][0] = nested
            return {}
        return {"provenance_metadata": nested}

    return mutate


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
        (lambda case: _replace_design_column(case, [True] * 8), "finite numeric"),
        (lambda case: _replace_design_column(case, [np.nan] * 8), "finite numeric"),
        (_drop_role, "one role per design column"),
        (
            lambda case: case[2][0].__setitem__("motion", "learned"),
            "invalid column role",
        ),
        (
            lambda case: case[3][0].__setitem__("path", "/private/data"),
            "path-like",
        ),
        (_absolute_path_column, "path-like"),
        (
            _set_design(0, pd.DataFrame(columns=["face", "motion", "constant"])),
            "nonzero dimensions",
        ),
        (_set_design(0, pd.DataFrame(index=range(8))), "nonzero dimensions"),
        (_drop_last_design_row, "8 rows"),
        (
            lambda case: case[0].__setitem__(1, np.ones((10, 2))),
            "same number of features",
        ),
        (_drop_run_item(1), "design_matrices"),
        (_drop_run_item(2), "column_roles"),
        (_drop_run_item(3), "run_metadata"),
        (
            lambda case: case[3].__setitem__(0, ["subject", "01"]),
            "run_metadata.*mapping",
        ),
        (lambda case: {"tr": None}, "exactly one of tr or frame_times"),
        (
            lambda case: {"frame_times": [np.arange(8.0), np.arange(10.0)]},
            "exactly one of tr or frame_times",
        ),
        (_nested_path_metadata("run"), "path-like"),
        (_nested_path_metadata("provenance"), "path-like"),
    ],
)
def test_prepared_analysis_rejects_invalid_arguments(prepared_inputs, mutator, message):
    case = deepcopy(prepared_inputs)
    extra = mutator(case) or {}

    with pytest.raises(ValueError, match=message):
        _make_prepared(case, **extra)


def test_prepared_design_fingerprint_is_stable_and_value_sensitive(prepared_inputs):
    first = _make_prepared(prepared_inputs)
    reordered_metadata = deepcopy(prepared_inputs)
    reordered_metadata[3][0] = dict(reversed(tuple(reordered_metadata[3][0].items())))
    same = _make_prepared(reordered_metadata)
    changed = deepcopy(prepared_inputs)
    changed[1][0].loc[0, "motion"] += 0.25
    different = _make_prepared(changed)

    assert first.design_fingerprint == same.design_fingerprint
    assert first.run_design_fingerprints == same.run_design_fingerprints
    assert different.run_design_fingerprints[0] != first.run_design_fingerprints[0]
    assert different.design_fingerprint != first.design_fingerprint
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        for fingerprint in (*first.run_design_fingerprints, first.design_fingerprint)
    )


def test_prepared_design_fingerprint_includes_structure_and_run_order(prepared_inputs):
    original = _make_prepared(prepared_inputs)

    reordered_columns = deepcopy(prepared_inputs)
    reordered_columns[1][0] = reordered_columns[1][0][["motion", "face", "constant"]]
    changed_roles = deepcopy(prepared_inputs)
    changed_roles[2][0]["motion"] = "other"
    changed_times = _make_prepared(
        prepared_inputs,
        tr=None,
        frame_times=[
            np.array([0.0, 2.0, 4.0, 6.0, 8.25, 10.0, 12.0, 14.0]),
            np.arange(10.0) * 2.0,
        ],
    )
    reordered_runs = deepcopy(prepared_inputs)
    for values in reordered_runs:
        values.reverse()

    assert (
        _make_prepared(reordered_columns).design_fingerprint
        != original.design_fingerprint
    )
    assert (
        _make_prepared(changed_roles).design_fingerprint != original.design_fingerprint
    )
    assert changed_times.design_fingerprint != original.design_fingerprint
    assert (
        _make_prepared(reordered_runs).design_fingerprint != original.design_fingerprint
    )


def test_prepared_design_normalization_records_lifecycle_and_provenance(
    prepared_inputs, caplog, complete_sources
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    sources = complete_sources(2)

    first = _make_prepared(prepared_inputs, sources=sources)
    second = _make_prepared(prepared_inputs, sources=sources)

    records = _structured_records(caplog)
    lifecycle = [
        record
        for record in records
        if record["stage"] == "prepared_design"
        and record["event"].startswith("normalization_")
    ]
    assert [record["event"] for record in lifecycle] == [
        "normalization_started",
        "normalization_completed",
        "normalization_started",
        "normalization_completed",
    ]
    assert {record["execution_id"] for record in lifecycle[:2]} == {
        first.provenance.execution_id
    }
    assert {record["execution_id"] for record in lifecycle[2:]} == {
        second.provenance.execution_id
    }

    activity = first.provenance.activities[-1]
    assert activity == {
        "name": "normalize_prepared_design",
        "stage": "prepared_design",
        "timing_source": "tr",
        "run_count": 2,
        "feature_count": 3,
        "runs": [
            {
                "columns": ["face", "motion", "constant"],
                "roles": ["task", "nuisance", "intercept"],
            },
            {
                "columns": ["constant", "motion", "face"],
                "roles": ["intercept", "nuisance", "task"],
            },
        ],
        "run_design_fingerprints": list(first.run_design_fingerprints),
        "design_fingerprint": first.design_fingerprint,
        "run_metadata": prepared_inputs[3],
        "software_versions": {
            "boldtailor": version("boldtailor"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
        },
    }
    assert first.provenance.execution_id != second.provenance.execution_id
    assert (
        first.provenance.metadata_fingerprint == second.provenance.metadata_fingerprint
    )
    assert first.design_fingerprint == second.design_fingerprint


def test_prepared_design_provenance_retains_canonical_metadata_and_versions(
    prepared_inputs,
    complete_sources,
):
    inputs = deepcopy(prepared_inputs)
    provenance_metadata = {
        "adapter": {"name": "fitlins", "settings": ["fixed", "prepared"]},
        "node": "run",
    }
    first = _make_prepared(
        inputs,
        sources=complete_sources(2),
        provenance_metadata=provenance_metadata,
    )
    reordered = deepcopy(prepared_inputs)
    reordered[3][:] = [dict(reversed(tuple(item.items()))) for item in reordered[3]]
    second = _make_prepared(
        reordered,
        sources=complete_sources(2),
        provenance_metadata=dict(reversed(tuple(provenance_metadata.items()))),
    )
    changed_metadata = deepcopy(prepared_inputs)
    changed_metadata[3][0]["subject"] = "99"
    third = _make_prepared(
        changed_metadata,
        sources=complete_sources(2),
        provenance_metadata=provenance_metadata,
    )
    inputs[3][0]["subject"] = "mutated"
    provenance_metadata["adapter"]["name"] = "mutated"

    activity = first.provenance.to_dict()["activities"][0]
    assert activity["metadata"] == {
        "adapter": {"name": "fitlins", "settings": ["fixed", "prepared"]},
        "node": "run",
    }
    assert activity["run_metadata"] == prepared_inputs[3]
    assert activity["software_versions"] == {
        "boldtailor": version("boldtailor"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
    }
    assert activity == second.provenance.to_dict()["activities"][0]
    assert first.design_fingerprint == second.design_fingerprint
    assert first.design_fingerprint == third.design_fingerprint


def test_prepared_design_provenance_is_private_and_warns_for_anonymous_sources(
    prepared_inputs,
    complete_sources,
):
    private_inputs = deepcopy(prepared_inputs)
    private_inputs[0][0][0, 0] = 712345.5
    private_inputs[1][0].loc[0, "motion"] = 9274.25
    prepared = _make_prepared(private_inputs, sources=complete_sources(2))
    anonymous = _make_prepared(prepared_inputs)

    canonical = prepared.provenance.canonical_json()
    assert prepared.design_fingerprint in canonical
    assert all(column in canonical for column in ["face", "motion", "constant"])
    assert "712345.5" not in canonical
    assert "9274.25" not in canonical
    assert str(private_inputs[3][0]) not in canonical
    assert anonymous.provenance.metadata_fingerprint is None
    assert any(
        warning["code"] == "provenance_quality"
        for warning in anonymous.provenance.warnings
    )


@pytest.mark.parametrize("complete", [False, True])
def test_normalization_builds_one_record(
    prepared_inputs, monkeypatch, caplog, complete, complete_sources
):
    import hashlib
    import json
    import logging

    import boldtailor.prepared as module

    sources = complete_sources(2) if complete else None
    metadata = {"purpose": "normalization"}
    original = module.ProvenanceRecord
    calls = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(module, "ProvenanceRecord", record)
    caplog.set_level(logging.INFO, logger="boldtailor")
    result = _make_prepared(
        prepared_inputs, sources=sources, provenance_metadata=metadata
    )
    assert len(calls) == 1
    records = [
        json.loads(r.getMessage()) for r in caplog.records if r.name == "boldtailor"
    ]
    assert dict(result.provenance.events[-1]) == records[-1]
    if complete:
        payload = {"sources": [s.to_dict() for s in sources]}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert result.provenance.metadata_fingerprint == expected
        assert not result.provenance.warnings
    else:
        assert result.provenance.metadata_fingerprint is None
        assert [w["code"] for w in result.provenance.warnings] == ["provenance_quality"]


@pytest.mark.parametrize("outcome", ["success", "metadata", "constructor"])
def test_normalization_catches_final_failures_and_isolates_context(
    prepared_inputs, caplog, monkeypatch, outcome
):
    import json
    import logging

    import boldtailor.prepared as module
    from boldtailor.logging import bind_context, emit_event

    caplog.set_level(logging.INFO, logger="boldtailor")
    sources = None
    metadata = {"bad": object()} if outcome == "metadata" else {}
    failure = RuntimeError("private constructor failure")

    def reject(*args, **kwargs):
        raise failure

    if outcome == "constructor":
        monkeypatch.setattr(module.PreparedDesignAnalysis, "__init__", reject)
    outer = dict(
        execution_id="outer",
        data_id="outer-data",
        analysis_id="outer-analysis",
        run_index=2,
    )
    with bind_context(**outer):
        if outcome == "success":
            _make_prepared(
                prepared_inputs, sources=sources, provenance_metadata=metadata
            )
        else:
            with pytest.raises((ValueError, RuntimeError)) as caught:
                _make_prepared(
                    prepared_inputs, sources=sources, provenance_metadata=metadata
                )
            if outcome == "constructor":
                assert caught.value is failure
        restored = emit_event("after_normalization", stage="test")
    records = [
        json.loads(r.getMessage()) for r in caplog.records if r.name == "boldtailor"
    ][:-1]
    ending = "completed" if outcome == "success" else "failed"
    assert [r["event"] for r in records] == [
        "normalization_started",
        f"normalization_{ending}",
    ]
    for record in records:
        assert not {"data_id", "analysis_id", "run_index"} & record.keys()
        assert record["execution_id"] != "outer"
    if outcome != "success":
        assert records[-1]["error_code"] == (
            "invalid_input" if outcome == "metadata" else "operation_failed"
        )
    assert "private" not in caplog.text
    assert all(restored[k] == v for k, v in outer.items())
