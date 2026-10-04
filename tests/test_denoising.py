"""Public task-guided denoising: selection result, augmentation, and fitting.

Expected PCs come from an independent least-squares residualization and SVD
of the final pool, and the downstream fit is compared with the same model fit
on data whose PCs were appended by hand.
"""

from dataclasses import FrozenInstanceError
from hashlib import sha256
import inspect
import re

import numpy as np
import pandas as pd
import pytest

from boldtailor._denoising_cv import select_component_count
from boldtailor._denoising_pool import analysis_components, onoff_r2, pool_masks
from boldtailor._mixture_threshold import mixture_threshold
from boldtailor.data import AnalysisData, from_arrays
from boldtailor.denoising import select_denoising, with_denoising
from boldtailor.denoising_results import DenoisingResult
from boldtailor.fit import fit, task_delta_r2
from boldtailor.hrf_library import default_hrf_library
from boldtailor.hrf_selection import select_hrfs, subset_runs
from boldtailor.model import Modulator, ModelSpec, TaskModel
from tests.denoising_fixtures import RT_MODEL, TR, make_denoising_fixture

COUNTS = (0, 1, 2, 4)
LABELS = ("sesA", "sesB", "sesC", "sesD")
CATEGORICAL = TaskModel(
    (
        Modulator("response_time"),
        Modulator("cond", kind="categorical", levels=("a", "b"), missing="indicator"),
    )
)


# ---- fixtures and helpers ----------------------------------------------------


@pytest.fixture(scope="module")
def fixture():
    return make_denoising_fixture()


def run_selection(fixture, data=None, **options):
    settings = dict(
        task_model=fixture.task_model,
        library=fixture.library,
        counts=COUNTS,
    )
    settings.update(options)
    return select_denoising(fixture.data if data is None else data, **settings)


@pytest.fixture(scope="module")
def result(fixture):
    return run_selection(fixture)


@pytest.fixture(scope="module")
def augmented(fixture, result):
    return with_denoising(fixture.data, result)


@pytest.fixture(scope="module")
def full_data(fixture):
    """Full-data HRFs, ON-OFF R², and auto masks computed independently."""
    selection = select_hrfs(
        fixture.data, library=fixture.library, task_model=fixture.task_model
    )
    statistic = onoff_r2(fixture.data, fixture.library)
    masks = pool_masks(statistic, "auto", hrf_indices=selection.hrf_indices)
    return selection, statistic, masks


def rebuild(data, *, signals=None, events=None, confounds=None, frame_times=None):
    return from_arrays(
        list(data.signals if signals is None else signals),
        list(data.events if events is None else events),
        frame_times=list(data.frame_times if frame_times is None else frame_times),
        confounds=list(data.confounds if confounds is None else confounds),
    )


def with_white_noise(data, n=60, seed=17):
    """Append independent white-noise features (with confound loadings)."""
    rng = np.random.default_rng(seed)
    signals = []
    for signal, frame in zip(data.signals, data.confounds):
        confounds = frame.to_numpy()
        noise = rng.normal(size=(len(signal), n)) + 100.0
        noise += confounds @ rng.normal(scale=0.8, size=(confounds.shape[1], n))
        signals.append(np.column_stack([signal, noise]))
    return rebuild(data, signals=signals)


def oracle_pcs(data, run, pool, count):
    """Leading left singular vectors of the normalized, lstsq-projected pool."""
    confounds = data.confounds[run].to_numpy()
    nuisance = np.column_stack([confounds, np.ones(len(confounds))])
    y = data.signals[run][:, pool]
    residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    residual = residual / np.linalg.norm(residual, axis=0)
    return np.linalg.svd(residual, full_matrices=False)[0][:, :count]


def projector(basis):
    return basis @ np.linalg.pinv(basis)


def snapshot(data):
    return dict(
        signals=[s.copy() for s in data.signals],
        events=data.events,
        confounds=data.confounds,
        frame_times=[t.copy() for t in data.frame_times],
        provenance=data.provenance.canonical_json(),
    )


def assert_snapshot(data, saved):
    for run in range(data.n_runs):
        np.testing.assert_array_equal(data.signals[run], saved["signals"][run])
        np.testing.assert_array_equal(data.frame_times[run], saved["frame_times"][run])
        pd.testing.assert_frame_equal(data.events[run], saved["events"][run])
        pd.testing.assert_frame_equal(data.confounds[run], saved["confounds"][run])
    assert data.provenance.canonical_json() == saved["provenance"]


def last_activity(record):
    return record.to_dict()["activities"][-1]


def digest(array, dtype="<f8"):
    return sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


# ---- input validation --------------------------------------------------------


@pytest.mark.parametrize(
    "options,match",
    [
        (dict(counts=(1, 2)), "counts"),
        (dict(counts=(0, True)), "counts"),
        (dict(counts=(0, -1)), "counts"),
        (dict(counts=(0, 1.5)), "counts"),
        (dict(pool_r2_threshold=np.nan), "pool_r2_threshold"),
        (dict(pool_r2_threshold=True), "pool_r2_threshold"),
        (dict(pool_r2_threshold="Auto"), "pool_r2_threshold"),
        (dict(pool_r2_threshold="0.1"), "pool_r2_threshold"),
        (dict(pcstop=0.99), "pcstop"),
        (dict(pcstop=np.inf), "pcstop"),
        (dict(pcstop=True), "pcstop"),
        (dict(task_model="rt"), "task_model"),
        (dict(library="default"), "library"),
        (dict(feature_signature=""), "feature_signature"),
        (dict(feature_signature=3), "feature_signature"),
        (dict(run_labels=("a", "b", "c")), "run labels"),
        (dict(run_labels=("a", "a", "b", "c")), "run labels"),
    ],
)
def test_invalid_selection_inputs_are_rejected(fixture, options, match):
    with pytest.raises(ValueError, match=match):
        run_selection(fixture, **options)


def test_selection_requires_analysis_data(fixture):
    with pytest.raises(ValueError, match="AnalysisData"):
        run_selection(fixture, data=fixture.data.signals)


def test_selection_requires_three_runs(fixture):
    with pytest.raises(ValueError, match="at least three runs"):
        run_selection(fixture, data=subset_runs(fixture.data, [0, 1]))


def test_selection_has_no_mask_parameter(fixture):
    """Anatomy-agnostic core (ruling R9): every input feature is a candidate."""
    parameters = inspect.signature(select_denoising).parameters
    assert not any("mask" in name for name in parameters)
    with pytest.raises(TypeError):
        run_selection(fixture, brain_mask=np.ones(26, bool))


def test_default_selection_settings_follow_glmsingle(fixture):
    signature = inspect.signature(select_denoising)
    assert signature.parameters["counts"].default == tuple(range(11))
    assert signature.parameters["pool_r2_threshold"].default == "auto"
    assert signature.parameters["pcstop"].default == 1.05
    assert "score_tolerance" not in signature.parameters
    assert signature.parameters["library"].default is None
    assert signature.parameters["task_model"].default == TaskModel()


# ---- count choice and final full-data result -----------------------------------


def test_hrfs_pool_and_pcs_are_full_data_by_design(fixture, result, full_data):
    selection, statistic, masks = full_data
    np.testing.assert_array_equal(result.initial_hrf_indices, selection.hrf_indices)
    np.testing.assert_array_equal(result.selection_cv_r2, selection.cv_r2)
    np.testing.assert_array_equal(result.onoff_r2, statistic)
    np.testing.assert_array_equal(result.noise_pool, masks.pool)
    np.testing.assert_array_equal(result.scoring_mask, masks.scoring)
    assert result.noise_pool_threshold == masks.threshold
    assert result.noise_pool_mixture == masks.mixture
    assert not result.scoring_fallback
    groups = fixture.groups
    expected = np.zeros(fixture.data.n_features, bool)
    expected[np.r_[groups["noise"], groups["outside_noise"]]] = True
    np.testing.assert_array_equal(result.noise_pool, expected)


def test_count_and_tables_follow_the_cross_validated_choice(fixture, result, full_data):
    selection, _, masks = full_data
    expected = select_component_count(
        fixture.data,
        hrf_indices=selection.hrf_indices,
        components=analysis_components(fixture.data, masks.pool),
        scoring=masks.scoring,
        task_model=fixture.task_model,
        library=fixture.library,
        counts=COUNTS,
        pcstop=1.05,
    )
    assert isinstance(result, DenoisingResult)
    assert result.n_components == expected.n_components
    assert result.counts == COUNTS
    assert result.pool_r2_threshold == "auto" and result.pcstop == 1.05
    scores = result.candidate_scores
    assert list(scores["count"]) == list(COUNTS)
    np.testing.assert_array_equal(scores["perf"], expected.scores["perf"])
    np.testing.assert_array_equal(scores["curve"], expected.scores["curve"])
    np.testing.assert_array_equal(scores["eligible"], expected.scores["eligible"])
    np.testing.assert_array_equal(result.perf, expected.scores["perf"])
    np.testing.assert_array_equal(result.curve, expected.scores["curve"])
    np.testing.assert_array_equal(result.scored, expected.setup.scored)
    folds = result.fold_scores
    np.testing.assert_array_equal(folds["median_r2"], expected.fold_scores["median_r2"])
    labels = [result.run_labels[v] for v in expected.fold_scores["validation_run"]]
    assert list(folds["validation_run"]) == labels


def test_selection_finds_a_positive_count_on_the_fixture(result):
    # Shared latent noise loads on the task features, so PCs help prediction.
    assert result.n_components >= 1


def test_run_components_span_the_independent_oracle_pcs(fixture, result):
    k = result.n_components
    assert len(result.run_components) == fixture.data.n_runs
    for run, components in enumerate(result.run_components):
        assert components.shape == (fixture.data.signals[run].shape[0], k)
        oracle = oracle_pcs(fixture.data, run, result.noise_pool, k)
        np.testing.assert_allclose(projector(components), projector(oracle), atol=1e-8)
        np.testing.assert_allclose(components.T @ components, np.eye(k), atol=1e-10)
        peaks = components[np.argmax(np.abs(components), axis=0), np.arange(k)]
        assert (peaks > 0).all()


def test_final_pca_diagnostics_report_rank_and_singular_values(fixture, result):
    diagnostics = result.components
    assert diagnostics.run_labels == result.run_labels
    assert diagnostics.pool_size == int(result.noise_pool.sum())
    for run in range(fixture.data.n_runs):
        confounds = fixture.data.confounds[run].to_numpy()
        nuisance = np.column_stack([confounds, np.ones(len(confounds))])
        y = fixture.data.signals[run][:, result.noise_pool]
        residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
        residual /= np.linalg.norm(residual, axis=0)
        values = np.linalg.svd(residual, compute_uv=False)
        np.testing.assert_allclose(diagnostics.singular_values[run], values, atol=1e-10)
        assert diagnostics.ranks[run] == min(len(y), y.shape[1])
    table = diagnostics.table()
    assert list(table["run_label"]) == list(result.run_labels)
    assert list(table["rank"]) == list(diagnostics.ranks)


def test_fold_diagnostics_record_runs_and_held_out_targets(fixture, result):
    assert len(result.folds) == fixture.data.n_runs
    labels = result.run_labels
    for v, fold in enumerate(result.folds):
        assert fold.validation_run == labels[v]
        assert fold.training_runs == tuple(x for i, x in enumerate(labels) if i != v)
        assert not fold.zero_target.any()
        energy = fold.target_energy
        assert np.isfinite(energy[result.scored]).all()
        assert np.isnan(energy[~result.scoring_mask]).all()
    table = result.fold_scores
    assert set(table.columns) >= {"validation_run", "count", "median_r2", "n_scored"}


def test_custom_run_labels_name_runs_in_reasons_and_tables(fixture):
    result = run_selection(fixture, counts=(0, 1, 50), run_labels=LABELS)
    assert result.run_labels == LABELS
    reasons = " ".join(result.candidate_scores["reason"])
    reasons += " " + " ".join(result.fold_scores["reason"])
    assert "exceeds pool PCA rank" in reasons
    assert "sesA" in reasons
    assert not re.search(r"\brun \d", reasons)
    assert set(result.fold_scores["validation_run"]) == set(LABELS)
    assert result.initial_selection.run_labels == LABELS


def test_categorical_task_model_is_supported(fixture):
    rng = np.random.default_rng(7)
    events = []
    for run, frame in enumerate(fixture.data.events):
        cond = np.where(np.arange(len(frame)) % 2 == 0, "a", "b").astype(object)
        cond = rng.permutation(cond)
        if run in (0, 1):
            cond[2] = np.nan
        events.append(frame.assign(cond=cond))
    data = with_white_noise(rebuild(fixture.data, events=events))
    result = run_selection(fixture, data, task_model=CATEGORICAL)
    white = np.arange(26, 86)
    assert result.noise_pool[white].mean() > 0.5
    assert not result.noise_pool[fixture.groups["task"]].any()
    activity = last_activity(result.provenance)
    assert activity["task_model_fingerprint"] == CATEGORICAL.fingerprint
    augmented = with_denoising(data, result)
    assert augmented.n_features == data.n_features


# ---- zero counts and unavailable counts ---------------------------------------


def test_zero_only_grid_returns_no_components(fixture):
    result = run_selection(fixture, counts=(0,))
    assert result.n_components == 0
    for run, components in enumerate(result.run_components):
        assert components.shape == (fixture.data.signals[run].shape[0], 0)
    assert result.component_names == ()


def test_zero_count_augmentation_preserves_numerical_inputs(fixture):
    saved = snapshot(fixture.data)
    result = run_selection(fixture, counts=(0,))
    augmented = with_denoising(fixture.data, result)
    assert_snapshot(fixture.data, saved)
    for run in range(fixture.data.n_runs):
        np.testing.assert_array_equal(augmented.signals[run], saved["signals"][run])
        pd.testing.assert_frame_equal(augmented.confounds[run], saved["confounds"][run])
        pd.testing.assert_frame_equal(augmented.events[run], saved["events"][run])
        np.testing.assert_array_equal(
            augmented.frame_times[run], saved["frame_times"][run]
        )
    assert augmented.timing_source == fixture.data.timing_source
    activity = last_activity(augmented.provenance)
    assert activity["name"] == "denoising_augmentation"
    assert activity["n_components"] == 0
    assert activity["added_columns"] == []
    assert_annotated_sources(
        fixture.data.provenance.sources, augmented.provenance.sources, result
    )


def test_empty_pool_makes_zero_the_only_eligible_count(fixture):
    result = run_selection(fixture, pool_r2_threshold=-10.0)
    assert result.n_components == 0
    assert not result.noise_pool.any()
    scores = result.candidate_scores.set_index("count")
    assert not scores.loc[[1, 2, 4], "eligible"].any()
    assert scores.loc[1, "reason"].count("empty noise pool") >= 1
    activity = last_activity(result.provenance)
    assert [row["count"] for row in activity["excluded_counts"]] == [1, 2, 4]


# ---- pool threshold, fallback, and pcstop ----------------------------------------


def test_final_pool_uses_the_mixture_threshold_of_onoff_r2(result):
    finite = np.isfinite(result.onoff_r2)
    expected = mixture_threshold(result.onoff_r2[finite])
    assert result.noise_pool_threshold == expected.threshold
    assert result.noise_pool_mixture == expected
    np.testing.assert_array_equal(
        result.noise_pool, finite & (result.onoff_r2 < expected.threshold)
    )


def test_provenance_records_the_glmsingle_rules(result):
    activity = last_activity(result.provenance)
    assert activity["pool_r2_threshold"] == "auto"
    assert activity["pool_threshold_rule"] == "gaussian_mixture_tail_threshold"
    assert activity["noise_pool_threshold"] == result.noise_pool_threshold
    assert activity["noise_pool_mixture"] == result.noise_pool_mixture.to_dict()
    assert activity["noise_pool_mixture"]["n_components"] == 2
    assert activity["pool_statistic"] == "glmsingle_onoff_r2"
    assert activity["count_rule"] == "glmsingle_pcstop"
    assert activity["pcstop"] == 1.05
    assert activity["performance"] == "median_over_scored_features_of_fold_pooled_r2"
    assert activity["scoring_fallback"] is False
    assert activity["perf"] == list(result.perf)
    assert activity["curve"] == list(result.curve)


def test_fixed_threshold_keeps_the_fixed_rule(fixture, full_data):
    result = run_selection(fixture, counts=(0, 1), pool_r2_threshold=0.01)
    assert result.pool_r2_threshold == 0.01
    assert result.noise_pool_threshold == 0.01
    assert result.noise_pool_mixture is None
    selection, statistic, _ = full_data
    masks = pool_masks(statistic, 0.01, hrf_indices=selection.hrf_indices)
    np.testing.assert_array_equal(result.noise_pool, masks.pool)
    activity = last_activity(result.provenance)
    assert activity["pool_r2_threshold"] == 0.01
    assert activity["pool_threshold_rule"] == "fixed"
    assert activity["noise_pool_threshold"] == 0.01
    assert activity["noise_pool_mixture"] is None


def test_integer_fixed_threshold_is_stored_as_float(fixture):
    result = run_selection(fixture, counts=(0,), pool_r2_threshold=0)
    assert isinstance(result.pool_r2_threshold, float)


def test_degenerate_mixture_raises_with_labels_and_a_hint(fixture, monkeypatch):
    import boldtailor.denoising as denoising

    monkeypatch.setattr(
        denoising, "onoff_r2", lambda data, library, run_labels: np.full(26, 0.2)
    )
    with pytest.raises(ValueError, match="pool_r2_threshold") as error:
        run_selection(fixture, counts=(0,), run_labels=LABELS)
    message = str(error.value)
    assert "'sesA', 'sesB', 'sesC', 'sesD'" in message and "distinct" in message
    assert "fixed pool_r2_threshold" in message


def test_best_100_fallback_scores_the_top_features_and_is_recorded(fixture):
    result = run_selection(fixture, counts=(0, 1), pool_r2_threshold=10.0)
    assert result.scoring_fallback
    candidates = np.isfinite(result.onoff_r2) & (result.initial_hrf_indices >= 0)
    np.testing.assert_array_equal(result.scoring_mask, candidates)  # 24 < 100
    np.testing.assert_array_equal(result.noise_pool, np.isfinite(result.onoff_r2))
    assert last_activity(result.provenance)["scoring_fallback"] is True


def test_pcstop_one_never_chooses_fewer_than_the_best_count(fixture, result):
    strict = run_selection(fixture, pcstop=1.0)
    scores = strict.candidate_scores
    eligible = scores[scores["eligible"]]
    best = eligible.loc[eligible["curve"].idxmax(), "count"]
    expected = 0 if eligible["curve"].max() <= 0 else best
    assert strict.n_components == expected
    assert strict.n_components >= result.n_components


# ---- ownership and immutability -------------------------------------------------


def test_result_arrays_are_read_only_and_tables_are_copies(result):
    arrays = [result.noise_pool, result.scoring_mask, result.onoff_r2, result.scored]
    arrays += [result.selection_cv_r2, result.initial_hrf_indices]
    arrays += [result.perf, result.curve]
    arrays += list(result.run_components) + list(result.components.singular_values)
    fold = result.folds[0]
    arrays += [fold.zero_target, fold.target_energy]
    for array in arrays:
        assert not array.flags.writeable
    table = result.candidate_scores
    table.loc[:, "perf"] = 99.0
    assert (result.candidate_scores["perf"] != 99.0).all()
    folds = result.fold_scores
    folds.loc[:, "reason"] = "edited"
    assert (result.fold_scores["reason"] != "edited").all()
    assert isinstance(result.run_components, tuple)
    assert isinstance(result.folds, tuple)
    with pytest.raises(FrozenInstanceError):
        result.n_components = 3
    with pytest.raises(FrozenInstanceError):
        fold.zero_target = None


def test_result_does_not_alias_caller_inputs(fixture):
    counts = [0, 1, 2]
    result = run_selection(fixture, counts=counts)
    counts.append(9)
    assert result.counts == (0, 1, 2)


def test_result_rejects_inconsistent_components(result):
    from dataclasses import replace

    with pytest.raises(ValueError, match="run_components"):
        replace(result, run_components=result.run_components[:2])
    with pytest.raises(ValueError, match="n_components"):
        replace(result, n_components=result.n_components + 1)
    with pytest.raises(ValueError, match="rows"):
        replace(result, run_components=tuple(c[:-1] for c in result.run_components))


# ---- provenance -------------------------------------------------------------------


def test_selection_provenance_records_the_identity_of_every_choice(fixture, result):
    activity = last_activity(result.provenance)
    assert activity["name"] == "denoising_selection"
    assert activity["selection_statistic"] is True
    assert activity["task_model"] == fixture.task_model.to_dict()
    assert activity["task_model_fingerprint"] == fixture.task_model.fingerprint
    assert activity["library_fingerprint"] == fixture.library.fingerprint
    assert activity["counts"] == list(COUNTS)
    assert activity["n_components"] == result.n_components
    assert activity["pool_r2_threshold"] == "auto"
    assert activity["pcstop"] == 1.05
    assert "score_tolerance" not in activity
    assert not any("brain_mask" in key for key in activity)
    assert activity["run_labels"] == list(result.run_labels)
    assert activity["folds"][1] == dict(
        train=[result.run_labels[i] for i in (0, 2, 3)], test=[result.run_labels[1]]
    )
    assert activity["noise_pool_fingerprint"] == digest(result.noise_pool, "|b1")
    assert activity["scoring_mask_fingerprint"] == digest(result.scoring_mask, "|b1")
    assert activity["scored_fingerprint"] == digest(result.scored, "|b1")
    assert activity["onoff_r2_fingerprint"] == digest(result.onoff_r2)
    assert activity["initial_hrf_assignment_fingerprint"] == digest(
        result.initial_hrf_indices, "<i8"
    )
    assert activity["component_fingerprints"] == [
        digest(c) for c in result.run_components
    ]
    baseline = activity["baseline_confounds"]
    assert [run["columns"] for run in baseline] == [
        list(frame.columns) for frame in fixture.data.confounds
    ]
    assert len({run["fingerprint"] for run in baseline}) == fixture.data.n_runs
    assert activity["excluded_counts"] == []
    assert result.provenance.sources == fixture.data.provenance.sources


FILE_FIELDS = ("role", "uri", "media_type", "byte_size", "modified_at", "sha256")


def file_fields(ref):
    return {name: getattr(ref, name) for name in FILE_FIELDS}


def assert_annotated_sources(before, after, result, role="confounds"):
    """File identity unchanged; one ref per run marks the augmentation (R6)."""
    expected = dict(
        selection_execution_id=result.provenance.execution_id,
        component_fingerprints=[digest(c) for c in result.run_components],
        n_components=result.n_components,
    )
    assert len(after) == len(before)
    for old, new in zip(before, after):
        for name in ("signal", "events", "confounds"):
            old_ref, new_ref = getattr(old, name), getattr(new, name)
            assert (old_ref is None) == (new_ref is None)
            if old_ref is None:
                continue
            assert file_fields(new_ref) == file_fields(old_ref)
            annotations = new_ref.to_dict().get("annotations", {})
            if name == role:
                assert annotations.pop("denoising_augmentation") == expected
            assert annotations == old_ref.to_dict().get("annotations", {})


def test_augmentation_extends_provenance_and_annotates_sources(
    fixture, result, augmented
):
    # Requirement change (ruling R6): sources keep file identity but record
    # the augmentation so downstream analysis ids differ.
    before = fixture.data.provenance
    after = augmented.provenance
    assert_annotated_sources(before.sources, after.sources, result)
    assert len(after.activities) == len(before.activities) + 1
    activity = last_activity(after)
    assert activity["name"] == "denoising_augmentation"
    assert activity["n_components"] == result.n_components
    assert activity["added_columns"] == list(result.component_names)
    assert activity["denoising_execution_id"] == result.provenance.execution_id
    assert activity["component_fingerprints"] == [
        digest(c) for c in result.run_components
    ]


def test_augmentation_appends_named_components_and_keeps_everything_else(
    fixture, result, augmented
):
    k = result.n_components
    names = tuple(f"denoise_pc_{i:03d}" for i in range(k))
    assert result.component_names == names
    for run in range(fixture.data.n_runs):
        original = fixture.data.confounds[run]
        frame = augmented.confounds[run]
        assert list(frame.columns) == [*original.columns, *names]
        pd.testing.assert_frame_equal(frame[list(original.columns)], original)
        np.testing.assert_array_equal(
            frame[list(names)].to_numpy(), result.run_components[run]
        )
        np.testing.assert_array_equal(augmented.signals[run], fixture.data.signals[run])
        pd.testing.assert_frame_equal(augmented.events[run], fixture.data.events[run])
        np.testing.assert_array_equal(
            augmented.frame_times[run], fixture.data.frame_times[run]
        )


# ---- rejected applications ---------------------------------------------------------


def test_applying_a_result_twice_is_rejected(fixture, result, augmented):
    with pytest.raises(ValueError, match="already applied"):
        with_denoising(augmented, result)
    zero = run_selection(fixture, counts=(0,))
    once = with_denoising(fixture.data, zero)
    with pytest.raises(ValueError, match="already applied"):
        with_denoising(once, zero)


def test_component_name_collision_is_rejected(fixture):
    rng = np.random.default_rng(3)
    confounds = [
        c.assign(denoise_pc_000=rng.normal(size=len(c))) for c in fixture.data.confounds
    ]
    data = rebuild(fixture.data, confounds=confounds)
    result = run_selection(fixture, data, counts=(0, 1))
    assert result.n_components == 1
    with pytest.raises(ValueError, match="denoise_pc_000"):
        with_denoising(data, result)


def shuffled_features(data):
    order = np.r_[1, 0, 2 : data.n_features]
    return rebuild(data, signals=[s[:, order] for s in data.signals])


def truncated_run(data):
    signals, confounds = list(data.signals), list(data.confounds)
    signals[3], confounds[3] = signals[3][:-1], confounds[3].iloc[:-1]
    times = list(data.frame_times)
    times[3] = times[3][:-1]
    return rebuild(data, signals=signals, confounds=confounds, frame_times=times)


def changed_signal(data):
    signals = [s.copy() for s in data.signals]
    signals[2][5, 7] += 1e-3
    return rebuild(data, signals=signals)


def changed_confounds(data):
    confounds = list(data.confounds)
    confounds[1] = confounds[1].assign(motion_x=confounds[1].motion_x * 1.01)
    return rebuild(data, confounds=confounds)


def changed_events(data):
    events = list(data.events)
    events[0].loc[0, "response_time"] += 0.05
    return rebuild(data, events=events)


def shifted_times(data):
    return rebuild(data, frame_times=[t + 0.5 for t in data.frame_times])


def renamed_confounds(data):
    confounds = [c.rename(columns={"drift": "trend"}) for c in data.confounds]
    return rebuild(data, confounds=confounds)


MISMATCHES = [
    (lambda d: subset_runs(d, [0, 1, 2]), "runs"),
    (lambda d: subset_runs(d, [1, 0, 2, 3]), "order"),
    (lambda d: rebuild(d, signals=[s[:, :-1] for s in d.signals]), "features"),
    (shuffled_features, "signals"),
    (truncated_run, "rows"),
    (shifted_times, "time grid"),
    (changed_signal, "signals"),
    (changed_confounds, "baseline confounds"),
    (renamed_confounds, "baseline confounds"),
    (changed_events, "events"),
]


@pytest.mark.parametrize("change,match", MISMATCHES)
def test_mismatched_analysis_is_rejected(fixture, result, change, match):
    with pytest.raises(ValueError, match=match):
        with_denoising(change(fixture.data), result)


def test_feature_signature_must_match(fixture):
    result = run_selection(fixture, feature_signature="ordered-axis")
    assert result.feature_signature == "ordered-axis"
    with pytest.raises(ValueError, match="feature_signature"):
        with_denoising(fixture.data, result)
    with pytest.raises(ValueError, match="feature_signature"):
        with_denoising(fixture.data, result, feature_signature="other-axis")
    augmented = with_denoising(fixture.data, result, feature_signature="ordered-axis")
    assert isinstance(augmented, AnalysisData)


def test_augmentation_requires_a_result_and_data(fixture, result):
    with pytest.raises(ValueError, match="DenoisingResult"):
        with_denoising(fixture.data, "result")
    with pytest.raises(ValueError, match="AnalysisData"):
        with_denoising(fixture.data.signals, result)


# ---- downstream fitting --------------------------------------------------------------


@pytest.fixture(scope="module")
def typed(fixture):
    """The fixture with an explicit trial_type, as conventional fitting expects."""
    events = [e.assign(trial_type="task") for e in fixture.data.events]
    return rebuild(fixture.data, events=events)


def appended(data, columns):
    confounds = [
        frame.assign(**dict(zip(names, values.T)))
        for frame, (names, values) in zip(data.confounds, columns)
    ]
    return from_arrays(
        list(data.signals), list(data.events), tr=TR, confounds=confounds
    )


def conventional(names=(), task_model=RT_MODEL):
    return ModelSpec(
        contrasts={"task": "task"},
        confounds=("motion_x", "drift", "cosine", *names),
        drift_model=None,
        task_model=task_model,
    )


def test_fit_on_augmented_data_matches_explicitly_appended_pcs(fixture, typed):
    saved = snapshot(typed)
    result = run_selection(fixture, typed)
    augmented = with_denoising(typed, result)
    names = list(result.component_names)
    assert names
    oracle = [
        (names, oracle_pcs(typed, run, result.noise_pool, len(names)))
        for run in range(typed.n_runs)
    ]
    denoised = fit(augmented, conventional(names))
    expected = fit(appended(typed, oracle), conventional(names))
    np.testing.assert_allclose(
        denoised.effect("task"), expected.effect("task"), rtol=1e-7, atol=1e-9
    )
    np.testing.assert_allclose(
        denoised.variance("task"), expected.variance("task"), rtol=1e-6, atol=1e-12
    )
    baseline = fit(typed, conventional())
    assert not np.allclose(baseline.effect("task"), denoised.effect("task"))
    assert_snapshot(typed, saved)


def test_augmented_data_supports_reselection_of_hrfs(fixture, augmented):
    selection = select_hrfs(
        augmented, library=fixture.library, task_model=fixture.task_model
    )
    assert selection.hrf_indices.shape == (fixture.data.n_features,)


def test_default_library_resolves_to_the_package_default(fixture, monkeypatch):
    import boldtailor.denoising as denoising

    # The real default library is slow to fit here; the resolver must be the
    # package default, and None must go through it.
    assert denoising.default_hrf_library is default_hrf_library
    calls = []

    def stand_in():
        calls.append(True)
        return fixture.library

    monkeypatch.setattr(denoising, "default_hrf_library", stand_in)
    result = select_denoising(
        fixture.data,
        task_model=fixture.task_model,
        counts=(0, 1),
    )
    assert calls
    activity = last_activity(result.provenance)
    assert activity["library_fingerprint"] == fixture.library.fingerprint


def test_augmentation_without_confounds_annotates_the_signal_source(fixture):
    data = from_arrays(list(fixture.data.signals), list(fixture.data.events), tr=TR)
    result = run_selection(fixture, data, counts=(0, 1))
    augmented = with_denoising(data, result)
    assert all(run.confounds is None for run in augmented.provenance.sources)
    assert_annotated_sources(
        data.provenance.sources, augmented.provenance.sources, result, "signal"
    )


@pytest.mark.parametrize("broken", [0, 2, 3])
def test_invalid_run_is_named_by_its_own_label(fixture, broken):
    events = list(fixture.data.events)
    events[broken] = events[broken].drop(columns="response_time")
    data = rebuild(fixture.data, events=events)
    with pytest.raises(ValueError) as error:
        run_selection(fixture, data, run_labels=LABELS)
    message = str(error.value)
    assert LABELS[broken] in message
    assert not any(label in message for i, label in enumerate(LABELS) if i != broken)


def test_distinct_augmentations_of_sourced_data_have_distinct_analysis_ids(
    fixture, typed, complete_sources
):
    data = from_arrays(
        list(typed.signals),
        list(typed.events),
        tr=TR,
        confounds=list(typed.confounds),
        sources=complete_sources(typed.n_runs),
    )
    assert data.provenance.metadata_fingerprint is not None
    # Different pool thresholds give different pools and components.
    first = run_selection(fixture, data, counts=(0, 1))
    second = run_selection(fixture, data, counts=(0, 1), pool_r2_threshold=0.01)
    assert (first.noise_pool != second.noise_pool).any()
    assert first.n_components == second.n_components == 1
    one, two = with_denoising(data, first), with_denoising(data, second)
    model = conventional(first.component_names)
    fit_one, fit_two = fit(one, model), fit(two, model)
    ids = {
        fit_one.provenance.analysis_fingerprint,
        fit_two.provenance.analysis_fingerprint,
    }
    assert None not in ids and len(ids) == 2
    # The fixture's constant features make a matched task_delta_r2 undefined,
    # so only the cross-paired parent check is exercised.
    with pytest.raises(ValueError, match="identity"):
        task_delta_r2(two, model, fit_one)
