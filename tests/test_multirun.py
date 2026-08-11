import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
    run_glm,
)

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec
from boldtailor.provenance import RunSources, SourceRef


def _problem():
    rng = np.random.default_rng(11)
    events = (
        pd.DataFrame(
            {
                "onset": [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0],
                "duration": np.ones(7),
                "trial_type": [
                    "face",
                    "house",
                    "button",
                    "face",
                    "house",
                    "face",
                    "house",
                ],
            }
        ),
        pd.DataFrame(
            {
                "onset": np.arange(0.0, 112.0, 8.0),
                "duration": np.ones(14),
                "trial_type": ["face", "house"] * 7,
            }
        ),
    )
    frame_times = (np.arange(30) * 2.0, np.arange(60) * 2.0)
    designs = tuple(
        make_first_level_design_matrix(
            times,
            events=run_events,
            hrf_model="glover",
            drift_model="cosine",
            high_pass=0.01,
            min_onset=-24.0,
        )
        for times, run_events in zip(frame_times, events)
    )
    signals = []
    for design in designs:
        beta = np.zeros((design.shape[1], 2))
        beta[design.columns.get_loc("face")] = [2.0, 1.0]
        beta[design.columns.get_loc("house")] = [0.5, 1.5]
        beta[design.columns.get_loc("constant")] = [10.0, 12.0]
        noise = rng.normal(0.0, 0.05, (len(design), 2))
        signals.append(design.to_numpy() @ beta + noise)
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model="cosine",
        noise_model="ols",
    )
    return tuple(signals), events, frame_times, designs, model


def _complete_sources() -> tuple[RunSources, ...]:
    return (
        RunSources(
            signal=SourceRef(
                role="signal",
                uri="sub-01/func/sub-01_task-localizer_run-01_bold.tsv",
                media_type="text/tab-separated-values",
                byte_size=2048,
                modified_at="2026-08-08T12:00:00Z",
            ),
            events=SourceRef(
                role="events",
                uri="sub-01/func/sub-01_task-localizer_run-01_events.tsv",
                media_type="text/tab-separated-values",
                byte_size=512,
                modified_at="2026-08-08T12:01:00Z",
            ),
        ),
        RunSources(
            signal=SourceRef(
                role="signal",
                uri="sub-01/func/sub-01_task-localizer_run-02_bold.tsv",
                media_type="text/tab-separated-values",
                byte_size=4096,
                modified_at="2026-08-08T12:02:00Z",
            ),
            events=SourceRef(
                role="events",
                uri="sub-01/func/sub-01_task-localizer_run-02_events.tsv",
                media_type="text/tab-separated-values",
                byte_size=768,
                modified_at="2026-08-08T12:03:00Z",
            ),
        ),
    )


def _as_image(signals):
    data = signals.T.reshape(2, 1, 1, signals.shape[0])
    return nib.Nifti1Image(data, np.eye(4))


def _flat_values(image):
    return image.get_fdata().reshape(-1)


def _original_space_ar1_diagnostics(signals, design):
    matrix = design.to_numpy()
    labels, regression_results = run_glm(
        signals,
        matrix,
        noise_model="ar1",
    )
    prediction = np.empty_like(signals)
    for label, result in regression_results.items():
        prediction[:, labels == label] = matrix @ result.theta
    residual_sum = np.sum((signals - prediction) ** 2, axis=0)
    total_sum = np.sum((signals - signals.mean(axis=0)) ** 2, axis=0)
    return 1.0 - residual_sum / total_sum, residual_sum, total_sum


def test_fit_allows_run_specific_designs_and_matches_first_level_model():
    signals, events, frame_times, designs, model = _problem()
    data = from_arrays(signals, events, frame_times=frame_times)

    result = fit(data, model)
    mask = nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4))
    oracle = FirstLevelModel(
        mask_img=mask,
        noise_model="ols",
        signal_scaling=False,
    )
    with pytest.warns(RuntimeWarning, match="Generation of a mask"):
        oracle.fit(
            [_as_image(run) for run in signals],
            design_matrices=list(designs),
        )
    with pytest.warns(RuntimeWarning, match="same contrast"):
        expected = oracle.compute_contrast(
            "face - house",
            output_type="all",
        )

    assert tuple(designs[0].columns) != tuple(designs[1].columns)
    np.testing.assert_allclose(
        result.effect("face_gt_house"),
        _flat_values(expected["effect_size"]),
    )
    np.testing.assert_allclose(
        result.variance("face_gt_house"),
        _flat_values(expected["effect_variance"]),
    )
    np.testing.assert_allclose(
        result.stat("face_gt_house"),
        _flat_values(expected["stat"]),
    )
    np.testing.assert_allclose(
        result.z_score("face_gt_house"),
        _flat_values(expected["z_score"]),
    )
    np.testing.assert_allclose(
        result.one_sided_p_value("face_gt_house"),
        _flat_values(expected["p_value"]),
    )
    assert len(result.run_r2) == 2


def test_fit_aggregates_r2_from_sums_not_run_means():
    signals, events, frame_times, _, model = _problem()
    result = fit(
        from_arrays(signals, events, frame_times=frame_times),
        model,
    )
    run_total = [np.sum((run - run.mean(axis=0)) ** 2, axis=0) for run in signals]
    run_residual = [
        (1.0 - score) * total for score, total in zip(result.run_r2, run_total)
    ]
    expected = 1.0 - np.sum(run_residual, axis=0) / np.sum(run_total, axis=0)

    np.testing.assert_allclose(result.r2, expected)


def test_fit_ar1_r2_uses_original_signal_space_across_runs():
    signals, events, frame_times, designs, _ = _problem()
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model="cosine",
        noise_model="ar1",
    )

    result = fit(
        from_arrays(signals, events, frame_times=frame_times),
        model,
    )
    diagnostics = tuple(
        _original_space_ar1_diagnostics(run, design)
        for run, design in zip(signals, designs, strict=True)
    )
    residual_sum = np.sum([values[1] for values in diagnostics], axis=0)
    total_sum = np.sum([values[2] for values in diagnostics], axis=0)
    aggregate_r2 = 1.0 - residual_sum / total_sum

    for actual, expected in zip(result.run_r2, diagnostics, strict=True):
        np.testing.assert_allclose(actual, expected[0])
    np.testing.assert_allclose(result.r2, aggregate_r2)


def test_fit_warns_for_rank_deficient_design():
    signals, events, frame_times, _, _ = _problem()
    confound = pd.DataFrame({"duplicate": np.ones(len(signals[0]))})
    model = ModelSpec(
        contrasts={"face": {"face": 1.0}},
        confounds=("duplicate",),
        drift_model=None,
        noise_model="ols",
    )

    with pytest.warns(UserWarning) as caught:
        fit(
            from_arrays(
                signals[0],
                events[0],
                frame_times=frame_times[0],
                confounds=confound,
            ),
            model,
        )
    messages = {str(warning.message) for warning in caught}

    assert any("design rank" in message for message in messages)
    assert "Matrix is singular at working precision, regularizing..." in messages


def test_fit_records_run_diagnostics_without_serializing_design_values():
    signals, events, frame_times, designs, model = _problem()

    result = fit(
        from_arrays(
            signals,
            events,
            frame_times=frame_times,
            sources=_complete_sources(),
        ),
        model,
    )
    runs = result.provenance.activities[-1]["runs"]
    serialized = result.provenance.canonical_json()

    assert len(runs) == 2
    for index, run in enumerate(runs):
        assert run["timing_source"] == "frame_times"
        assert run["n_scans"] == signals[index].shape[0]
        assert run["n_features"] == signals[index].shape[1]
        assert run["design_columns"] == list(result.design_matrices[index].columns)
        assert run["design_rank"] == int(
            np.linalg.matrix_rank(result.design_matrices[index].to_numpy())
        )
        assert run["residual_dof"] == designs[index].shape[0] - run["design_rank"]
        assert run["excluded_event_count"] == 0
        assert run["min_onset_cutoff"] == -24.0

    assert "design_matrix" not in serialized
    assert "residual_sum" not in serialized
    assert "total_sum" not in serialized
    assert "r2" not in serialized


def test_fit_rejects_non_estimable_contrast_before_glm(monkeypatch):
    signals, events, frame_times, designs, _ = _problem()
    confound = pd.DataFrame({"duplicate": designs[0]["face"].to_numpy()})
    model = ModelSpec(
        contrasts={"difference": {"face": 1.0, "duplicate": -1.0}},
        confounds=("duplicate",),
        drift_model="cosine",
        noise_model="ols",
    )

    def fail_glm(*args, **kwargs):
        pytest.fail("run_glm must not be called before contrast preflight")

    monkeypatch.setattr("boldtailor._conventional.run_glm", fail_glm)

    with pytest.warns(UserWarning) as caught:
        with pytest.raises(
            ValueError,
            match="run 0.*contrast 'difference'.*not estimable",
        ):
            fit(
                from_arrays(
                    signals[0],
                    events[0],
                    frame_times=frame_times[0],
                    confounds=confound,
                ),
                model,
            )
    messages = {str(warning.message) for warning in caught}

    assert any("design rank" in message for message in messages)
    assert "Matrix is singular at working precision, regularizing..." in messages


def test_fit_records_rank_deficiency_warning_in_provenance():
    signals, events, frame_times, _, _ = _problem()
    confound = pd.DataFrame({"duplicate": np.ones(len(signals[0]))})
    model = ModelSpec(
        contrasts={"face": {"face": 1.0}},
        confounds=("duplicate",),
        drift_model=None,
        noise_model="ols",
    )

    with pytest.warns(UserWarning):
        result = fit(
            from_arrays(
                signals[0],
                events[0],
                frame_times=frame_times[0],
                confounds=confound,
                sources=_complete_sources()[:1],
            ),
            model,
        )

    run = result.provenance.activities[-1]["runs"][0]

    assert run["design_rank"] < len(run["design_columns"])
    assert any("design rank" in warning for warning in run["warnings"])


def test_fit_rejects_contrast_term_missing_from_one_run_before_glm(monkeypatch):
    signals, events, frame_times, _, _ = _problem()
    model = ModelSpec(
        contrasts={"button": {"button": 1.0}},
        noise_model="ols",
    )

    def fail_glm(*args, **kwargs):
        pytest.fail("run_glm must not be called before contrast preflight")

    monkeypatch.setattr("boldtailor._conventional.run_glm", fail_glm)

    with pytest.raises(
        ValueError,
        match="run 1.*contrast 'button'.*missing regressor 'button'",
    ):
        fit(
            from_arrays(signals, events, frame_times=frame_times),
            model,
        )


def test_fit_rejects_all_zero_semantic_contrast_before_glm(monkeypatch):
    signals, events, frame_times, _, _ = _problem()
    model = ModelSpec(
        contrasts={"zero": "face - face"},
        noise_model="ols",
    )

    def fail_glm(*args, **kwargs):
        pytest.fail("run_glm must not be called before contrast preflight")

    monkeypatch.setattr("boldtailor._conventional.run_glm", fail_glm)

    with pytest.raises(
        ValueError,
        match="run 0.*contrast 'zero'.*resolves to all zeros",
    ):
        fit(
            from_arrays(signals, events, frame_times=frame_times),
            model,
        )
