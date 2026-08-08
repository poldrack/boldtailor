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
    run_total = [
        np.sum((run - run.mean(axis=0)) ** 2, axis=0) for run in signals
    ]
    run_residual = [
        (1.0 - score) * total
        for score, total in zip(result.run_r2, run_total)
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


def test_fit_rejects_non_estimable_contrast():
    signals, events, frame_times, designs, _ = _problem()
    confound = pd.DataFrame({"duplicate": designs[0]["face"].to_numpy()})
    model = ModelSpec(
        contrasts={"difference": {"face": 1.0, "duplicate": -1.0}},
        confounds=("duplicate",),
        drift_model="cosine",
        noise_model="ols",
    )

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


def test_fit_rejects_contrast_term_missing_from_one_run():
    signals, events, frame_times, _, _ = _problem()
    model = ModelSpec(
        contrasts={"button": {"button": 1.0}},
        noise_model="ols",
    )

    with pytest.raises(
        ValueError,
        match="run 1.*contrast 'button'.*missing regressor 'button'",
    ):
        fit(
            from_arrays(signals, events, frame_times=frame_times),
            model,
        )
