import builtins
from dataclasses import replace
from importlib.metadata import version
import io
import json
import logging
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from nilearn.glm import compute_contrast
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.glm.first_level import run_glm

from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared, task_delta_r2_prepared
from boldtailor.provenance import RunSources, SourceRef


@pytest.fixture
def prepared_problem():
    rng = np.random.default_rng(20260811)
    n_scans = 48
    design = pd.DataFrame(
        {
            "face": rng.normal(size=n_scans),
            "house": rng.normal(size=n_scans),
            "motion": rng.normal(size=n_scans),
            "constant": np.ones(n_scans),
        }
    )
    coefficients = np.array([[2.5, 1.0], [-1.5, 0.5], [0.25, -0.25], [10.0, 12.0]])
    signals = design.to_numpy() @ coefficients
    signals += rng.normal(0.0, 0.15, signals.shape)
    prepared = _prepared(signals, design)
    contrasts = {"face_gt_house": "face - house"}

    def expected(noise_model):
        labels, regression_results = run_glm(
            signals,
            design.to_numpy(),
            noise_model=noise_model,
        )
        vector = expression_to_contrast_vector("face - house", design.columns)
        return _nilearn_t_contrast(labels, regression_results, vector)

    return prepared, contrasts, expected


def _prepared(signals, design, *, roles=None):
    if roles is None:
        roles = {
            name: "intercept" if name == "constant" else "task"
            for name in design.columns
        }
    return PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=design,
        tr=2.0,
        column_roles=roles,
    )


def _complete_sources() -> tuple[RunSources, ...]:
    return (
        RunSources(
            signal=SourceRef(
                role="signal",
                uri="sub-01/func/sub-01_task-faces_run-01_bold.tsv",
                media_type="text/tab-separated-values",
                byte_size=2048,
                modified_at="2026-08-11T12:00:00Z",
            ),
            events=SourceRef(
                role="events",
                uri="sub-01/func/sub-01_task-faces_run-01_events.tsv",
                media_type="text/tab-separated-values",
                byte_size=512,
                modified_at="2026-08-11T12:01:00Z",
            ),
        ),
    )


def _prepared_with_sources(signals, design, *, roles=None, sources=None):
    if roles is None:
        roles = {
            name: "intercept" if name == "constant" else "task"
            for name in design.columns
        }
    return PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=design,
        tr=2.0,
        column_roles=roles,
        sources=_complete_sources() if sources is None else sources,
    )


def _structured_records(caplog):
    return [json.loads(record.getMessage()) for record in caplog.records]


def _nilearn_t_contrast(labels, regression_results, vector):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^divide by zero encountered in divide$",
            category=RuntimeWarning,
            module=r"^nilearn\.glm\._utils$",
        )
        return compute_contrast(labels, regression_results, vector, stat_type="t")


@pytest.mark.parametrize("noise_model", ["ols", "ar1"])
def test_fit_prepared_matches_nilearn(noise_model, prepared_problem):
    prepared, contrasts, expected_for = prepared_problem
    expected = expected_for(noise_model)

    result = fit_prepared(
        prepared,
        contrasts=contrasts,
        noise_model=noise_model,
        model_metadata={"origin": "fitlins", "node": "run"},
    )

    np.testing.assert_allclose(result.effect("face_gt_house"), expected.effect_size())
    np.testing.assert_allclose(
        result.variance("face_gt_house"), expected.effect_variance()
    )
    np.testing.assert_allclose(result.stat("face_gt_house"), expected.stat())
    np.testing.assert_allclose(result.z_score("face_gt_house"), expected.z_score())
    np.testing.assert_allclose(
        result.one_sided_p_value("face_gt_house"), expected.p_value()
    )


def test_fit_prepared_combines_run_specific_designs_and_pools_r_squared():
    rng = np.random.default_rng(20260812)
    designs = (
        pd.DataFrame(
            {
                "face": rng.normal(size=40),
                "motion_first": rng.normal(size=40),
                "house": rng.normal(size=40),
                "constant": np.ones(40),
            }
        ),
        pd.DataFrame(
            {
                "constant": np.ones(64),
                "house": rng.normal(size=64),
                "motion_second": rng.normal(size=64),
                "face": rng.normal(size=64),
            }
        ),
    )
    signals = tuple(
        design.to_numpy()
        @ np.array(
            [
                [2.0, 1.0],
                [0.25, -0.25],
                [-1.0, 0.5],
                [10.0, 12.0],
            ]
            if design.columns[0] == "face"
            else [
                [10.0, 12.0],
                [-1.0, 0.5],
                [0.25, -0.25],
                [2.0, 1.0],
            ]
        )
        + rng.normal(0.0, 0.15, (len(design), 2))
        for design in designs
    )
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=2.0,
        column_roles=(
            {
                "face": "task",
                "motion_first": "nuisance",
                "house": "task",
                "constant": "intercept",
            },
            {
                "constant": "intercept",
                "house": "task",
                "motion_second": "nuisance",
                "face": "task",
            },
        ),
    )
    expected_contrasts = []
    residual_sums = []
    total_sums = []
    for signal, design in zip(signals, designs, strict=True):
        matrix = design.to_numpy()
        labels, regression_results = run_glm(signal, matrix, noise_model="ols")
        vector = expression_to_contrast_vector("face - house", design.columns)
        expected_contrasts.append(
            _nilearn_t_contrast(labels, regression_results, vector)
        )
        prediction = np.empty_like(signal)
        for label, fit in regression_results.items():
            prediction[:, labels == label] = matrix @ fit.theta
        residual_sums.append(np.sum((signal - prediction) ** 2, axis=0))
        total_sums.append(np.sum((signal - signal.mean(axis=0)) ** 2, axis=0))
    expected = 0.5 * (expected_contrasts[0] + expected_contrasts[1])
    expected_r2 = 1.0 - np.sum(residual_sums, axis=0) / np.sum(total_sums, axis=0)

    result = fit_prepared(
        prepared,
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        noise_model="ols",
    )

    np.testing.assert_allclose(result.effect("face_gt_house"), expected.effect_size())
    np.testing.assert_allclose(
        result.variance("face_gt_house"), expected.effect_variance()
    )
    np.testing.assert_allclose(result.r2, expected_r2)
    assert not np.allclose(result.r2, np.mean(result.run_r2, axis=0))


@pytest.mark.parametrize(
    ("contrasts", "message"),
    [
        ({}, "at least one contrast"),
        ({"face": {}}, "at least one weight"),
        ({"face": {"face": "one"}}, "weights must be numeric"),
        ({"face": {"face": np.inf}}, "weights must be finite"),
        ({"face": {"face": True}}, "weights must be numeric"),
        ({"face": {"face": 0.0}}, "nonzero weight"),
        ({"face": ""}, "must not be empty"),
        ({"face": "face + missing"}, "contrast 'face'.*invalid"),
    ],
)
def test_fit_prepared_rejects_invalid_semantic_contrasts(
    prepared_problem, contrasts, message
):
    prepared, _, _ = prepared_problem

    with pytest.raises(ValueError, match=message):
        fit_prepared(prepared, contrasts=contrasts, noise_model="ols")


def test_fit_prepared_rejects_contrast_term_missing_from_one_run_before_glm(
    monkeypatch,
):
    designs = (
        pd.DataFrame({"face": [0.0, 1.0, 0.0, 1.0], "constant": 1.0}),
        pd.DataFrame({"constant": 1.0, "house": [0.0, 1.0, 0.0, 1.0]}),
    )
    signals = tuple(design.to_numpy() @ np.array([[1.0], [5.0]]) for design in designs)
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=2.0,
        column_roles=(
            {"face": "task", "constant": "intercept"},
            {"constant": "intercept", "house": "task"},
        ),
    )

    def fail_glm(*args, **kwargs):
        pytest.fail("run_glm must not be called before contrast preflight")

    monkeypatch.setattr("boldtailor._conventional.run_glm", fail_glm)

    with pytest.raises(
        ValueError,
        match="run 1.*contrast 'face'.*missing regressor 'face'",
    ):
        fit_prepared(
            prepared,
            contrasts={"face": {"face": 1.0}},
            noise_model="ols",
        )


def test_fit_prepared_rejects_non_estimable_contrast_before_glm(monkeypatch):
    design = pd.DataFrame(
        {
            "face": [0.0, 1.0, 0.0, 1.0, 0.0],
            "duplicate": [0.0, 1.0, 0.0, 1.0, 0.0],
            "constant": 1.0,
        }
    )
    prepared = _prepared(
        design.to_numpy() @ np.array([[1.0], [1.0], [5.0]]),
        design,
    )

    def fail_glm(*args, **kwargs):
        pytest.fail("run_glm must not be called before contrast preflight")

    monkeypatch.setattr("boldtailor._conventional.run_glm", fail_glm)

    with pytest.warns(UserWarning, match="design rank"):
        with pytest.raises(ValueError, match="contrast 'difference'.*not estimable"):
            fit_prepared(
                prepared,
                contrasts={"difference": {"face": 1.0, "duplicate": -1.0}},
                noise_model="ols",
            )


def test_fit_prepared_rejects_all_zero_semantic_contrast_before_glm(
    monkeypatch,
    prepared_problem,
):
    prepared, _, _ = prepared_problem

    def fail_glm(*args, **kwargs):
        pytest.fail("run_glm must not be called before contrast preflight")

    monkeypatch.setattr("boldtailor._conventional.run_glm", fail_glm)

    with pytest.raises(
        ValueError,
        match="run 0.*contrast 'zero'.*resolves to all zeros",
    ):
        fit_prepared(
            prepared,
            contrasts={"zero": "face - face"},
            noise_model="ols",
        )


def test_fit_prepared_rejects_nonpositive_residual_degrees_of_freedom():
    design = pd.DataFrame(
        {"face": [1.0, 0.0, 0.0], "house": [0.0, 1.0, 0.0], "constant": 1.0}
    )
    prepared = _prepared(design.to_numpy() @ np.array([[1.0], [2.0], [5.0]]), design)

    with pytest.raises(ValueError, match="residual degrees of freedom 0"):
        fit_prepared(prepared, contrasts={"face": {"face": 1.0}}, noise_model="ols")


def test_fit_prepared_rejects_unsupported_noise_model(prepared_problem):
    prepared, contrasts, _ = prepared_problem

    with pytest.raises(ValueError, match="noise_model must be 'ols' or 'ar1'"):
        fit_prepared(prepared, contrasts=contrasts, noise_model="fast")


@pytest.mark.parametrize("model_metadata", [{"path": "./private"}, {"value": object()}])
def test_fit_prepared_rejects_unsafe_model_metadata(prepared_problem, model_metadata):
    prepared, contrasts, _ = prepared_problem

    with pytest.raises(ValueError):
        fit_prepared(
            prepared,
            contrasts=contrasts,
            noise_model="ols",
            model_metadata=model_metadata,
        )


def test_fit_prepared_owns_results_and_reports_prepared_design_provenance(
    prepared_problem,
):
    prepared, contrasts, _ = prepared_problem
    result = fit_prepared(prepared, contrasts=contrasts, noise_model="ols")
    returned_design = result.design_matrices[0]
    returned_design.iloc[0, 0] = -99.0

    assert result.design_matrices[0].iloc[0, 0] != -99.0
    assert result.design_provenance == (
        {
            "source": "prepared",
            "design_fingerprint": prepared.run_design_fingerprints[0],
        },
    )
    arrays = (
        result.r2,
        *result.run_r2,
        result.effect("face_gt_house"),
        result.variance("face_gt_house"),
        result.stat("face_gt_house"),
        result.z_score("face_gt_house"),
        result.one_sided_p_value("face_gt_house"),
    )
    original = tuple(values.copy() for values in arrays)
    for values, expected in zip(arrays, original, strict=True):
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)
        with pytest.raises(ValueError):
            values[...] = 0.0
        np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(result.r2, original[0])


def test_fit_prepared_records_stable_analysis_identity_and_complete_activity(
    prepared_problem,
):
    del prepared_problem
    rng = np.random.default_rng(20260815)
    design = pd.DataFrame(
        {
            "face": rng.normal(size=24),
            "house": rng.normal(size=24),
            "constant": np.ones(24),
        }
    )
    signals = design.to_numpy() @ np.array([[2.0, 1.0], [-1.0, 0.5], [5.0, 6.0]])
    prepared = _prepared_with_sources(signals, design)
    metadata = {"origin": "fitlins", "node": "run"}
    contrasts = {"face_gt_house": {"face": 1.0, "house": -1.0}}

    first = fit_prepared(
        prepared,
        contrasts=contrasts,
        noise_model="ar1",
        model_metadata=metadata,
    )
    second = fit_prepared(
        prepared,
        contrasts=contrasts,
        noise_model="ar1",
        model_metadata=metadata,
    )

    assert first.provenance.execution_id != second.provenance.execution_id
    assert (
        first.provenance.analysis_fingerprint == second.provenance.analysis_fingerprint
    )
    assert re.fullmatch(r"[0-9a-f]{64}", first.provenance.analysis_fingerprint or "")
    activity = first.provenance.activities[-1]
    assert activity["name"] == "fit_prepared"
    assert activity["stage"] == "fit"
    assert activity["numerical_backend"] == {
        "name": "nilearn",
        "version": version("nilearn"),
    }
    assert activity["model"] == {
        "kind": "prepared_design",
        "contrasts": {
            "face_gt_house": {
                "kind": "weights",
                "weights": {"face": 1.0, "house": -1.0},
            }
        },
        "noise_model": "ar1",
        "metadata": metadata,
        "design_fingerprint": prepared.design_fingerprint,
    }
    run = activity["runs"][0]
    assert run == {
        "n_scans": 24,
        "n_features": 2,
        "design_columns": ["face", "house", "constant"],
        "role_counts": {"task": 2, "nuisance": 0, "intercept": 1, "other": 0},
        "design_rank": 3,
        "residual_dof": 21,
        "run_design_fingerprint": prepared.run_design_fingerprints[0],
        "warnings": [],
    }


def test_fit_prepared_analysis_fingerprint_tracks_all_fit_inputs(prepared_problem):
    _, contrasts, _ = prepared_problem
    rng = np.random.default_rng(20260816)
    design = pd.DataFrame(
        {
            "face": rng.normal(size=24),
            "house": rng.normal(size=24),
            "constant": np.ones(24),
        }
    )
    signals = design.to_numpy() @ np.array([[2.0], [-1.0], [5.0]])
    prepared = _prepared_with_sources(signals, design)
    baseline = fit_prepared(
        prepared,
        contrasts=contrasts,
        noise_model="ols",
        model_metadata={"node": "first"},
    )
    changed_design = design.copy()
    changed_design.loc[0, "face"] += 0.25
    changed_prepared = _prepared_with_sources(signals, changed_design)
    changed_results = (
        fit_prepared(
            changed_prepared,
            contrasts=contrasts,
            noise_model="ols",
            model_metadata={"node": "first"},
        ),
        fit_prepared(
            prepared,
            contrasts={"face": {"face": 1.0}},
            noise_model="ols",
            model_metadata={"node": "first"},
        ),
        fit_prepared(
            prepared,
            contrasts=contrasts,
            noise_model="ar1",
            model_metadata={"node": "first"},
        ),
        fit_prepared(
            prepared,
            contrasts=contrasts,
            noise_model="ols",
            model_metadata={"node": "second"},
        ),
    )

    assert all(
        result.provenance.analysis_fingerprint
        != baseline.provenance.analysis_fingerprint
        for result in changed_results
    )


def test_fit_prepared_leaves_analysis_identity_unavailable_for_anonymous_sources(
    prepared_problem,
):
    prepared, contrasts, _ = prepared_problem

    result = fit_prepared(prepared, contrasts=contrasts, noise_model="ols")

    assert result.provenance.analysis_fingerprint is None
    assert any(
        warning["code"] == "provenance_quality"
        for warning in result.provenance.warnings
    )


def test_fit_prepared_logs_lifecycle_and_resets_context(caplog, prepared_problem):
    caplog.set_level(logging.INFO, logger="boldtailor")
    _, _, _ = prepared_problem
    design = pd.DataFrame(
        {
            "face": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "house": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "constant": 1.0,
        }
    )
    signals = design.to_numpy() @ np.array([[2.0], [-1.0], [5.0]])
    prepared = _prepared_with_sources(signals, design)
    contrasts = {"face_gt_house": {"face": 1.0, "house": -1.0}}

    result = fit_prepared(prepared, contrasts=contrasts, noise_model="ols")
    from boldtailor.logging import emit_event

    emit_event("after_prepared_fit", stage="test")
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]

    assert [record["event"] for record in fit_records] == [
        "fit_started",
        "fit_completed",
    ]
    assert all(
        record["execution_id"] == result.provenance.execution_id
        for record in fit_records
    )
    assert prepared.provenance.metadata_fingerprint is not None
    assert result.provenance.analysis_fingerprint is not None
    assert all(
        record["data_id"] == prepared.provenance.metadata_fingerprint
        for record in fit_records
    )
    assert all(
        record["analysis_id"] == result.provenance.analysis_fingerprint
        for record in fit_records
    )
    assert records[-1]["event"] == "after_prepared_fit"
    assert records[-1].get("execution_id") is None
    assert records[-1].get("data_id") is None
    assert records[-1].get("analysis_id") is None


def test_fit_prepared_preserves_privacy_in_failure_logs_and_provenance(
    caplog,
    monkeypatch,
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    design = pd.DataFrame(
        {
            "face": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "constant": 1.0,
        }
    )
    signals = np.full((6, 1), 712345.5)
    prepared = _prepared_with_sources(signals, design)
    result = fit_prepared(
        prepared,
        contrasts={"face": {"face": 1.0}},
        noise_model="ols",
    )
    with pytest.raises(ValueError, match="missing regressor"):
        fit_prepared(
            prepared,
            contrasts={"missing": {"missing": 1.0}},
            noise_model="ols",
        )
    from boldtailor.logging import emit_event

    emit_event("after_sanitized_failure", stage="test")
    records = _structured_records(caplog)
    canonical = result.provenance.canonical_json()
    combined = "\n".join(
        [canonical, *(record.getMessage() for record in caplog.records)]
    )

    assert "712345.5" not in combined
    assert "Traceback" not in combined
    assert not re.search(r"0x[0-9a-fA-F]+", combined)
    fit_records = [record for record in records if record["stage"] == "fit"]
    assert [record["event"] for record in fit_records[-2:]] == [
        "fit_started",
        "fit_failed",
    ]
    assert (
        fit_records[-1]["error"]
        == "run 0 contrast 'missing' references missing regressor 'missing'"
    )
    assert records[-1]["event"] == "after_sanitized_failure"
    assert records[-1].get("execution_id") is None


@pytest.mark.parametrize(
    ("contrasts", "model_metadata"),
    [
        ({"face": {"face": 1.0}}, {"/private/task-5-metadata-key": "safe"}),
        ({"/private/task-5-contrast-key": {"face": 1.0}}, None),
    ],
)
def test_fit_prepared_rejects_path_like_model_keys_without_logging_them(
    caplog,
    prepared_problem,
    contrasts,
    model_metadata,
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    prepared, _, _ = prepared_problem
    absolute_path = next(
        value
        for value in (
            "/private/task-5-metadata-key",
            "/private/task-5-contrast-key",
        )
        if value in str(contrasts) or value in str(model_metadata)
    )

    with pytest.raises(ValueError, match="path-like"):
        fit_prepared(
            prepared,
            contrasts=contrasts,
            noise_model="ols",
            model_metadata=model_metadata,
        )
    from boldtailor.logging import emit_event

    emit_event("after_path_key_failure", stage="test")
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]

    assert [record["event"] for record in fit_records] == [
        "fit_started",
        "fit_failed",
    ]
    assert absolute_path not in "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert records[-1].get("execution_id") is None


@pytest.mark.parametrize(
    "model_metadata",
    [
        {"safe": [{"/private/task-5-sequence-key": "safe"}]},
        {"safe": ({"nested": [{"/private/task-5-deep-key": "safe"}]},)},
    ],
)
def test_fit_prepared_rejects_path_like_mapping_keys_inside_metadata_sequences(
    caplog,
    prepared_problem,
    model_metadata,
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    prepared, _, _ = prepared_problem
    absolute_path = next(
        value
        for value in (
            "/private/task-5-sequence-key",
            "/private/task-5-deep-key",
        )
        if value in str(model_metadata)
    )

    with pytest.raises(ValueError, match="path-like"):
        fit_prepared(
            prepared,
            contrasts={"face": {"face": 1.0}},
            noise_model="ols",
            model_metadata=model_metadata,
        )
    from boldtailor.logging import emit_event

    emit_event("after_nested_metadata_failure", stage="test")
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]

    assert [record["event"] for record in fit_records] == [
        "fit_started",
        "fit_failed",
    ]
    assert absolute_path not in "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert records[-1].get("execution_id") is None


@pytest.mark.parametrize(
    ("contrasts", "noise_model", "model_metadata"),
    [
        ({}, "ols", None),
        ({"face": {"face": 1.0}}, "fast", None),
        ({"face": {"face": 1.0}}, "ols", {"value": object()}),
    ],
)
def test_fit_prepared_logs_invalid_specification_failures_and_resets_context(
    caplog,
    prepared_problem,
    contrasts,
    noise_model,
    model_metadata,
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    prepared, _, _ = prepared_problem

    with pytest.raises(ValueError):
        fit_prepared(
            prepared,
            contrasts=contrasts,
            noise_model=noise_model,
            model_metadata=model_metadata,
        )
    from boldtailor.logging import emit_event

    emit_event("after_invalid_fit_spec", stage="test")
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]

    assert [record["event"] for record in fit_records] == [
        "fit_started",
        "fit_failed",
    ]
    assert fit_records[0]["execution_id"] == fit_records[1]["execution_id"]
    assert fit_records[1]["level"] == "ERROR"
    assert records[-1].get("execution_id") is None
    assert records[-1].get("data_id") is None
    assert records[-1].get("analysis_id") is None


def test_fit_prepared_ignores_empty_sensitive_environment_values(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="boldtailor")
    monkeypatch.setenv("BOLDTAILOR_EMPTY_SECRET", "")
    design = pd.DataFrame({"face": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], "constant": 1.0})
    prepared = _prepared(design.to_numpy() @ np.array([[2.0], [5.0]]), design)

    with pytest.raises(ValueError, match="missing regressor"):
        fit_prepared(
            prepared,
            contrasts={"missing": {"missing": 1.0}},
            noise_model="ols",
        )

    records = _structured_records(caplog)
    failed = [record for record in records if record["event"] == "fit_failed"][-1]
    assert (
        failed["error"]
        == "run 0 contrast 'missing' references missing regressor 'missing'"
    )


def test_fit_prepared_sanitizes_injected_traceback_and_object_repr(
    caplog,
    monkeypatch,
    prepared_problem,
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    prepared, contrasts, _ = prepared_problem

    class PrivateObject:
        pass

    private_object = PrivateObject()
    raw_repr = repr(private_object)
    raw_address = re.search(r"0x[0-9a-fA-F]+", raw_repr).group(0)

    def fail_fit(*args, **kwargs):
        raise ValueError(
            "Traceback (most recent call last):\n"
            f"private failure: {private_object!r}"
        )

    monkeypatch.setattr("boldtailor.prepared_fit.fit_designs", fail_fit)

    with pytest.raises(ValueError, match="Traceback"):
        fit_prepared(prepared, contrasts=contrasts, noise_model="ols")

    combined = "\n".join(record.getMessage() for record in caplog.records)
    fit_records = [
        record for record in _structured_records(caplog) if record["stage"] == "fit"
    ]

    assert [record["event"] for record in fit_records] == [
        "fit_started",
        "fit_failed",
    ]
    assert "Traceback" not in combined
    assert raw_repr not in combined
    assert raw_address not in combined


def test_fit_prepared_never_writes_to_the_filesystem(monkeypatch, prepared_problem):
    prepared, contrasts, _ = prepared_problem
    real_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"unexpected write through open: {mode}")
        return real_open(file, mode, *args, **kwargs)

    def guarded_io_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"unexpected write through io.open: {mode}")
        return real_io_open(file, mode, *args, **kwargs)

    def fail_os_open(path, flags, *args, **kwargs):
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & write_flags:
            raise AssertionError("unexpected write through os.open")
        return real_os_open(path, flags, *args, **kwargs)

    def fail_write(*args, **kwargs):
        raise AssertionError("unexpected Path write")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(os, "open", fail_os_open)
    # Path.open delegates to the guarded io.open: allow reads, reject writes.
    monkeypatch.setattr(Path, "write_text", fail_write)
    monkeypatch.setattr(Path, "write_bytes", fail_write)

    result = fit_prepared(prepared, contrasts=contrasts, noise_model="ols")

    assert result.contrast_names == ("face_gt_house",)


@pytest.fixture
def prepared_delta_problem():
    rng = np.random.default_rng(20260817)
    designs = (
        pd.DataFrame(
            {
                "face": rng.normal(size=40),
                "motion_first": rng.normal(size=40),
                "constant": np.ones(40),
                "house": rng.normal(size=40),
            }
        ),
        pd.DataFrame(
            {
                "constant": np.ones(56),
                "house": rng.normal(size=56),
                "motion_second": rng.normal(size=56),
                "face": rng.normal(size=56),
            }
        ),
    )
    coefficients = (
        np.array([[2.0, 1.0], [0.5, -0.25], [10.0, 11.0], [-1.0, 0.5]]),
        np.array([[10.0, 11.0], [-1.0, 0.5], [0.5, -0.25], [2.0, 1.0]]),
    )
    signals = tuple(
        design.to_numpy() @ values + rng.normal(0.0, 0.1, (len(design), 2))
        for design, values in zip(designs, coefficients, strict=True)
    )
    roles = (
        {
            "face": "task",
            "motion_first": "nuisance",
            "constant": "intercept",
            "house": "task",
        },
        {
            "constant": "intercept",
            "house": "task",
            "motion_second": "nuisance",
            "face": "task",
        },
    )
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=2.0,
        column_roles=roles,
        sources=_complete_sources() * 2,
    )
    contrasts = {"face_gt_house": {"face": 1.0, "house": -1.0}}
    metadata = {"origin": "fitlins", "node": "prepared-delta"}
    full_result = fit_prepared(
        prepared,
        contrasts=contrasts,
        noise_model="ar1",
        model_metadata=metadata,
    )
    return prepared, contrasts, metadata, full_result


def _ols_r2_oracle(signals, designs):
    residual_sums = []
    total_sums = []
    for signal, design in zip(signals, designs, strict=True):
        matrix = design.to_numpy()
        labels, regression_results = run_glm(signal, matrix, noise_model="ols")
        prediction = np.empty_like(signal)
        for label, fit in regression_results.items():
            prediction[:, labels == label] = matrix @ fit.theta
        residual_sums.append(np.sum((signal - prediction) ** 2, axis=0))
        total_sums.append(np.sum((signal - signal.mean(axis=0)) ** 2, axis=0))
    return 1.0 - np.sum(residual_sums, axis=0) / np.sum(total_sums, axis=0)


def test_task_delta_r2_prepared_uses_nested_ols_and_role_selected_designs(
    prepared_delta_problem,
):
    prepared, contrasts, metadata, full_result = prepared_delta_problem

    comparison = task_delta_r2_prepared(
        prepared,
        full_result,
        contrasts=contrasts,
        noise_model="ar1",
        model_metadata=metadata,
    )

    nuisance_designs = comparison.nuisance_design_matrices
    assert [list(design.columns) for design in nuisance_designs] == [
        ["motion_first", "constant"],
        ["constant", "motion_second"],
    ]
    expected_full = _ols_r2_oracle(prepared.signals, prepared.design_matrices)
    expected_nuisance = _ols_r2_oracle(prepared.signals, nuisance_designs)
    expected_raw = expected_full - expected_nuisance
    np.testing.assert_allclose(comparison.full_r2, expected_full)
    np.testing.assert_allclose(comparison.nuisance_r2, expected_nuisance)
    np.testing.assert_allclose(comparison.raw_delta_r2, expected_raw)
    assert np.all(comparison.raw_delta_r2 >= -1e-12)
    np.testing.assert_allclose(
        comparison.delta_r2,
        np.maximum(comparison.raw_delta_r2, 0.0),
    )
    assert comparison.negative_voxel_count == np.count_nonzero(expected_raw < 0.0)
    assert comparison.raw_min == pytest.approx(expected_raw.min())


def test_task_delta_r2_prepared_returns_owned_immutable_arrays_and_copied_designs(
    prepared_delta_problem,
):
    prepared, contrasts, metadata, full_result = prepared_delta_problem
    comparison = task_delta_r2_prepared(
        prepared,
        full_result,
        contrasts=contrasts,
        noise_model="ar1",
        model_metadata=metadata,
    )
    returned_design = comparison.nuisance_design_matrices[0]
    returned_design.iloc[0, 0] = -99.0

    assert comparison.nuisance_design_matrices[0].iloc[0, 0] != -99.0
    for values in (
        comparison.full_r2,
        comparison.nuisance_r2,
        comparison.raw_delta_r2,
        comparison.delta_r2,
    ):
        assert values.flags.owndata
        assert values.flags.c_contiguous
        assert values.dtype == np.float64
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        (
            {"face": "other", "motion": "nuisance", "constant": "intercept"},
            "column roles are incomplete",
        ),
        (
            {"face": "nuisance", "motion": "nuisance", "constant": "intercept"},
            "requires at least one task column",
        ),
        (
            {"face": "task", "motion": "task", "constant": "task"},
            "requires a nuisance or intercept column",
        ),
    ],
)
def test_task_delta_r2_prepared_rejects_incomplete_role_partitions(roles, message):
    design = pd.DataFrame(
        {"face": [0.0, 1.0] * 12, "motion": np.linspace(0, 1, 24), "constant": 1.0}
    )
    prepared = _prepared_with_sources(
        design.to_numpy() @ np.array([[2.0], [0.5], [5.0]]),
        design,
        roles=roles,
    )
    full_result = fit_prepared(
        prepared,
        contrasts={"face": {"face": 1.0}},
        noise_model="ols",
    )

    with pytest.raises(ValueError, match=message):
        task_delta_r2_prepared(
            prepared,
            full_result,
            contrasts={"face": {"face": 1.0}},
            noise_model="ols",
        )


@pytest.mark.parametrize(
    "changed",
    ["sources", "design", "contrasts", "noise_model", "model_metadata"],
)
def test_task_delta_r2_prepared_rejects_changed_parent_identity(
    prepared_delta_problem,
    changed,
):
    prepared, contrasts, metadata, full_result = prepared_delta_problem
    supplied_prepared = prepared
    supplied_contrasts = contrasts
    supplied_noise_model = "ar1"
    supplied_metadata = metadata
    if changed == "sources":
        changed_sources = (
            RunSources(
                signal=SourceRef(
                    role="signal",
                    uri="sub-02/func/sub-02_task-faces_run-01_bold.tsv",
                    media_type="text/tab-separated-values",
                    byte_size=2048,
                    modified_at="2026-08-11T12:00:00Z",
                ),
                events=_complete_sources()[0].events,
            ),
            _complete_sources()[0],
        )
        supplied_prepared = PreparedDesignAnalysis.from_arrays(
            signals=prepared.signals,
            design_matrices=prepared.design_matrices,
            tr=2.0,
            column_roles=prepared.column_roles,
            sources=changed_sources,
        )
    elif changed == "design":
        designs = prepared.design_matrices
        designs[0].iloc[0, 0] += 0.1
        supplied_prepared = PreparedDesignAnalysis.from_arrays(
            signals=prepared.signals,
            design_matrices=designs,
            tr=2.0,
            column_roles=prepared.column_roles,
            sources=_complete_sources() * 2,
        )
    elif changed == "contrasts":
        supplied_contrasts = {"face": {"face": 1.0}}
    elif changed == "noise_model":
        supplied_noise_model = "ols"
    else:
        supplied_metadata = {"origin": "fitlins", "node": "changed"}

    with pytest.raises(ValueError, match="full result does not match prepared input"):
        task_delta_r2_prepared(
            supplied_prepared,
            full_result,
            contrasts=supplied_contrasts,
            noise_model=supplied_noise_model,
            model_metadata=supplied_metadata,
        )


def test_task_delta_r2_prepared_rejects_invalid_parent_dimensions(
    prepared_delta_problem,
):
    prepared, contrasts, metadata, full_result = prepared_delta_problem
    mismatched = replace(full_result, _r2=np.zeros(prepared.n_features + 1))

    with pytest.raises(ValueError, match="feature dimensions do not match prepared"):
        task_delta_r2_prepared(
            prepared,
            mismatched,
            contrasts=contrasts,
            noise_model="ar1",
            model_metadata=metadata,
        )


def test_task_delta_r2_prepared_requires_reliable_parent_identity(prepared_problem):
    prepared, contrasts, _ = prepared_problem
    full_result = fit_prepared(prepared, contrasts=contrasts, noise_model="ols")

    with pytest.raises(ValueError, match="requires fingerprintable prepared sources"):
        task_delta_r2_prepared(
            prepared,
            full_result,
            contrasts=contrasts,
            noise_model="ols",
        )


def test_task_delta_r2_prepared_rejects_materially_negative_nested_ols_difference(
    monkeypatch,
    prepared_delta_problem,
):
    prepared, contrasts, metadata, full_result = prepared_delta_problem
    calls = iter((np.array([0.1, 0.1]), np.array([0.2, 0.2])))
    monkeypatch.setattr(
        "boldtailor.prepared_fit.fit_r2_designs",
        lambda *args: next(calls),
    )

    with pytest.raises(ValueError, match="nested OLS monotonicity violated"):
        task_delta_r2_prepared(
            prepared,
            full_result,
            contrasts=contrasts,
            noise_model="ar1",
            model_metadata=metadata,
        )


def test_task_delta_r2_prepared_records_lifecycle_and_diagnostic_provenance(
    caplog,
    prepared_delta_problem,
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    prepared, contrasts, metadata, full_result = prepared_delta_problem

    comparison = task_delta_r2_prepared(
        prepared,
        full_result,
        contrasts=contrasts,
        noise_model="ar1",
        model_metadata=metadata,
    )
    from boldtailor.logging import emit_event

    emit_event("after_prepared_comparison", stage="test")

    activity = comparison.provenance.activities[-1]
    assert activity["name"] == "task_delta_r2_prepared"
    assert activity["parent_analysis_id"] == full_result.provenance.analysis_fingerprint
    assert activity["definition"] == "full_r2 - nuisance_r2"
    assert activity["diagnostic_noise_model"] == "ols"
    assert activity["inferential_noise_model"] == "ar1"
    assert activity["clip_policy"] == "numerical_roundoff_guard"
    assert activity["nuisance_rule"] == "column roles nuisance or intercept"
    assert [run["nuisance_columns"] for run in activity["runs"]] == [
        ["motion_first", "constant"],
        ["constant", "motion_second"],
    ]
    assert activity["diagnostics"] == {
        "raw_min": comparison.raw_min,
        "negative_voxel_count": comparison.negative_voxel_count,
        "mean_delta_r2": pytest.approx(float(comparison.delta_r2.mean())),
        "max_delta_r2": pytest.approx(float(comparison.delta_r2.max())),
    }
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]
    assert [record["event"] for record in fit_records] == [
        "task_delta_r2_prepared_started",
        "task_delta_r2_prepared_completed",
    ]
    assert comparison.provenance.analysis_fingerprint is not None
    assert all(
        record["execution_id"] == comparison.provenance.execution_id
        for record in fit_records
    )
    assert all(
        record["data_id"] == prepared.provenance.metadata_fingerprint
        for record in fit_records
    )
    assert all(
        record["analysis_id"] == comparison.provenance.analysis_fingerprint
        for record in fit_records
    )
    assert records[-1]["event"] == "after_prepared_comparison"
    assert records[-1].get("execution_id") is None
    assert records[-1].get("data_id") is None
    assert records[-1].get("analysis_id") is None


def test_task_delta_r2_prepared_logs_downstream_failure_with_comparison_id(
    caplog,
    monkeypatch,
    prepared_delta_problem,
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    prepared, contrasts, metadata, full_result = prepared_delta_problem
    successful = task_delta_r2_prepared(
        prepared,
        full_result,
        contrasts=contrasts,
        noise_model="ar1",
        model_metadata=metadata,
    )
    comparison_id = successful.provenance.analysis_fingerprint
    caplog.clear()

    def fail_diagnostic(*args, **kwargs):
        raise ValueError("downstream diagnostic failure")

    monkeypatch.setattr(
        "boldtailor.prepared_fit._fit_prepared_r2",
        fail_diagnostic,
    )

    with pytest.raises(ValueError, match="downstream diagnostic failure"):
        task_delta_r2_prepared(
            prepared,
            full_result,
            contrasts=contrasts,
            noise_model="ar1",
            model_metadata=metadata,
        )
    from boldtailor.logging import emit_event

    emit_event("after_prepared_comparison_failure", stage="test")
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]

    assert [record["event"] for record in fit_records] == [
        "task_delta_r2_prepared_started",
        "task_delta_r2_prepared_failed",
    ]
    assert comparison_id is not None
    assert all(record["analysis_id"] == comparison_id for record in fit_records)
    assert all(
        record["data_id"] == prepared.provenance.metadata_fingerprint
        for record in fit_records
    )
    assert len({record["execution_id"] for record in fit_records}) == 1
    assert records[-1]["event"] == "after_prepared_comparison_failure"
    assert records[-1].get("execution_id") is None
    assert records[-1].get("data_id") is None
    assert records[-1].get("analysis_id") is None


def test_task_delta_r2_prepared_logs_preidentity_failure_without_analysis_id(
    caplog,
    prepared_delta_problem,
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    prepared, contrasts, _, full_result = prepared_delta_problem
    private_key = "/private/secret/prepared-comparison.json"

    with pytest.raises(ValueError, match="path-like"):
        task_delta_r2_prepared(
            prepared,
            full_result,
            contrasts=contrasts,
            noise_model="ar1",
            model_metadata={private_key: "redacted"},
        )
    from boldtailor.logging import emit_event

    emit_event("after_preidentity_comparison_failure", stage="test")
    records = _structured_records(caplog)
    fit_records = [record for record in records if record["stage"] == "fit"]

    assert [record["event"] for record in fit_records] == [
        "task_delta_r2_prepared_started",
        "task_delta_r2_prepared_failed",
    ]
    assert all(record.get("analysis_id") is None for record in fit_records)
    assert private_key not in "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert records[-1]["event"] == "after_preidentity_comparison_failure"
    assert records[-1].get("execution_id") is None
    assert records[-1].get("data_id") is None
    assert records[-1].get("analysis_id") is None


def test_task_delta_r2_prepared_never_writes_to_the_filesystem(
    monkeypatch,
    prepared_delta_problem,
):
    prepared, contrasts, metadata, full_result = prepared_delta_problem

    def fail_write(*args, **kwargs):
        raise AssertionError("unexpected filesystem write")

    monkeypatch.setattr(Path, "open", fail_write)
    monkeypatch.setattr(Path, "write_text", fail_write)
    monkeypatch.setattr(Path, "write_bytes", fail_write)

    comparison = task_delta_r2_prepared(
        prepared,
        full_result,
        contrasts=contrasts,
        noise_model="ar1",
        model_metadata=metadata,
    )

    assert comparison.delta_r2.shape == (prepared.n_features,)
